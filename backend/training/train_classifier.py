import json
from pathlib import Path
from typing import Dict, Any, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
from app.services.action_classifier import ActionClassifier
from app.services.preprocessing import get_spacy_nlp, ProcessedSentence

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parent / "model.joblib"


def load_dataset(dataset_path: Path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    test_utterances = []

    for meeting in data.get("meetings", []):
        split = meeting.get("split", "train")
        for u in meeting.get("utterances", []):
            text = u["text"]
            label = u["label"]
            if split == "train":
                train_texts.append(text)
                train_labels.append(label)
            else:
                test_texts.append(text)
                test_labels.append(label)
                test_utterances.append(u)

    return (train_texts, train_labels), (test_texts, test_labels), test_utterances


def train_and_evaluate():
    print(f"Loading dataset from: {DATASET_PATH}")
    (train_x, train_y), (test_x, test_y), test_utterances = load_dataset(DATASET_PATH)

    print(f"Train samples (split by meeting): {len(train_x)}")
    print(f"Test samples (held-out meetings): {len(test_x)}")

    # Build Pipeline: TF-IDF with n-grams (1, 2) + LogisticRegression
    # Calibrated probability output via LogisticRegression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("clf", LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("\nTraining ML Model (TF-IDF + LogisticRegression)...")
    pipeline.fit(train_x, train_y)

    # Save model
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Trained model saved to: {MODEL_OUTPUT_PATH}")

    # Evaluate ML Classifier on Held-Out Test Set
    ml_preds = pipeline.predict(test_x)
    print("\n" + "="*50)
    print("1. EVALUATION: Trained ML Classifier (Held-Out Test Set)")
    print("="*50)
    print(classification_report(test_y, ml_preds, zero_division=0))

    # Evaluate Rule-Based Baseline on Held-Out Test Set
    nlp = get_spacy_nlp()
    classifier = ActionClassifier(model_path=str(MODEL_OUTPUT_PATH))
    rule_preds = []

    for item in test_utterances:
        doc = nlp(item["text"])
        ps = ProcessedSentence(
            utterance_idx=0,
            sentence_idx=0,
            speaker=item.get("speaker"),
            text=item["text"],
            doc=doc,
            raw_utterance=item["text"]
        )
        rule_class, _, _ = classifier.rule_based_classify(ps)
        rule_preds.append(rule_class.value)

    print("\n" + "="*50)
    print("2. EVALUATION: Rule-Based Baseline (Held-Out Test Set)")
    print("="*50)
    print(classification_report(test_y, rule_preds, zero_division=0))

    # Binary Action Detection Comparison (ACTION_ITEM vs OTHER)
    print("\n" + "="*50)
    print("3. COMPARISON: Binary Action Detection (ACTION_ITEM vs REST)")
    print("="*50)
    binary_gold = [1 if y == "ACTION_ITEM" else 0 for y in test_y]
    binary_ml = [1 if p == "ACTION_ITEM" else 0 for p in ml_preds]
    binary_rule = [1 if p == "ACTION_ITEM" else 0 for p in rule_preds]

    p_ml, r_ml, f1_ml, _ = precision_recall_fscore_support(binary_gold, binary_ml, average="binary", zero_division=0)
    p_rb, r_rb, f1_rb, _ = precision_recall_fscore_support(binary_gold, binary_rule, average="binary", zero_division=0)

    print(f"Rule-Based Baseline -> Precision: {p_rb:.2f}, Recall: {r_rb:.2f}, F1: {f1_rb:.2f}")
    print(f"Trained ML Model    -> Precision: {p_ml:.2f}, Recall: {r_ml:.2f}, F1: {f1_ml:.2f}")

    # Evaluate Assignee and Deadline Extraction on Test Set
    from app.services.assignee_extractor import extract_assignee
    from app.services.deadline_extractor import extract_deadline
    from datetime import date

    ref_date = date(2026, 10, 27)
    correct_assignees = 0
    total_action_items = 0
    correct_deadlines = 0

    for item in test_utterances:
        if item["label"] == "ACTION_ITEM":
            total_action_items += 1
            doc = nlp(item["text"])
            ps = ProcessedSentence(
                utterance_idx=0,
                sentence_idx=0,
                speaker=item.get("speaker"),
                text=item["text"],
                doc=doc,
                raw_utterance=item["text"]
            )
            assignee_res = extract_assignee(ps)
            deadline_res = extract_deadline(ps, ref_date)

            expected_assignee = item.get("assignee")
            expected_deadline = item.get("deadline_text")

            if (assignee_res.assignee == expected_assignee) or (expected_assignee is None and assignee_res.assignee is None):
                correct_assignees += 1

            if expected_deadline:
                if deadline_res.deadline_text and (expected_deadline.lower() in deadline_res.deadline_text.lower() or deadline_res.deadline_text.lower() in expected_deadline.lower()):
                    correct_deadlines += 1
            else:
                if deadline_res.deadline_text is None:
                    correct_deadlines += 1

    if total_action_items > 0:
        print("\n" + "="*50)
        print("4. INFORMATION EXTRACTION ACCURACY ON TEST ACTION ITEMS")
        print("="*50)
        print(f"Total Action Items in Test Set: {total_action_items}")
        print(f"Assignee Accuracy: {correct_assignees}/{total_action_items} ({correct_assignees/total_action_items*100:.1f}%)")
        print(f"Deadline Accuracy: {correct_deadlines}/{total_action_items} ({correct_deadlines/total_action_items*100:.1f}%)")


if __name__ == "__main__":
    train_and_evaluate()
