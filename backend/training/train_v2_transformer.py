import json
import time
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

DATASET_PATH = Path(__file__).resolve().parent / "dataset_large.json"
MODEL_OUTPUT_DIR = Path(__file__).resolve().parent / "v2_distilbert"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "NON_ACTIONABLE": 0,
    "SUGGESTION": 1,
    "ACTION_ITEM": 2,
    "COMPLETED_ACTION": 3,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


class MeetingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_dataset(dataset_path: Path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for m in data.get("meetings", []):
        split = m.get("split", "train")
        for u in m.get("utterances", []):
            text = u["text"]
            lbl = LABEL_MAP.get(u["label"], 0)
            if split == "train":
                train_texts.append(text)
                train_labels.append(lbl)
            else:
                test_texts.append(text)
                test_labels.append(lbl)

    return (train_texts, train_labels), (test_texts, test_labels)


def train_v2_classifier():
    print("=" * 70)
    print("TRAINING V2 NEURAL TRANSFORMER (DistilBERT)")
    print("=" * 70)

    (train_texts, train_labels), (test_texts, test_labels) = load_dataset(DATASET_PATH)
    print(f"Total Raw Train: {len(train_texts)} | Held-Out Test: {len(test_texts)}")

    # Stratified balance for fast CPU training: keep all ACTION_ITEM & COMPLETED_ACTION, balanced samples of others
    np.random.seed(42)
    indices_by_class = {0: [], 1: [], 2: [], 3: []}
    for i, lbl in enumerate(train_labels):
        indices_by_class[lbl].append(i)

    # Class balance sampling: prioritize high-signal commitment utterances
    selected_indices = []
    selected_indices.extend(indices_by_class[2])  # All ACTION_ITEM (~728)
    selected_indices.extend(indices_by_class[3])  # All COMPLETED_ACTION (~115)
    # Sample suggestions and non_actionable proportionally
    n_samples_sugg = min(1200, len(indices_by_class[1]))
    n_samples_na = min(1800, len(indices_by_class[0]))
    selected_indices.extend(np.random.choice(indices_by_class[1], n_samples_sugg, replace=False))
    selected_indices.extend(np.random.choice(indices_by_class[0], n_samples_na, replace=False))

    np.random.shuffle(selected_indices)
    sub_train_texts = [train_texts[i] for i in selected_indices]
    sub_train_labels = [train_labels[i] for i in selected_indices]
    print(f"Balanced High-Signal Training Set: {len(sub_train_texts)} utterances")

    # Compute inverse class weights to ensure high Action-Item sensitivity
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(sub_train_labels),
        y=sub_train_labels
    )
    print(f"Computed loss weights per class: {class_weights}")

    # Load pre-trained tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    print(f"Training on device: {device}")

    train_dataset = MeetingDataset(sub_train_texts, sub_train_labels, tokenizer)
    test_dataset = MeetingDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_macro_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 25 == 0:
                print(f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - t0
        print(f"\n--- Epoch {epoch} Complete in {epoch_time:.1f}s | Avg Train Loss: {avg_train_loss:.4f} ---")

        # Evaluate on Held-Out Test Set
        model.eval()
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                preds_all.extend(preds)
                labels_all.extend(labels.numpy())

        acc = accuracy_score(labels_all, preds_all)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels_all, preds_all, average="macro", zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels_all, preds_all, average="weighted", zero_division=0)

        print(f"Validation Accuracy: {acc:.4f} | Macro F1: {f1_m:.4f} | Weighted F1: {f1_w:.4f}")

        if f1_m > best_macro_f1:
            best_macro_f1 = f1_m
            print(f"-> New Best Macro F1 ({f1_m:.4f})! Saving model to {MODEL_OUTPUT_DIR}")
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    # Detailed Final Report
    target_names = [ID2LABEL[i] for i in range(4)]
    print("\n" + "=" * 70)
    print("V2 TRANSFORMER (DistilBERT) FINAL EVALUATION REPORT")
    print("=" * 70)
    print(classification_report(labels_all, preds_all, target_names=target_names, zero_division=0))

    # Save metadata
    v2_meta = {
        "model_architecture": "DistilBertForSequenceClassification",
        "backbone": "distilbert-base-uncased",
        "parameters": 66956548,
        "epochs_trained": epochs,
        "best_macro_f1": round(best_macro_f1, 4),
        "validation_accuracy": round(acc, 4),
        "classes": target_names,
    }
    with open(MODEL_OUTPUT_DIR / "v2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(v2_meta, f, indent=2)

    print(f"V2 Model training complete! Artifacts saved in {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    train_v2_classifier()
