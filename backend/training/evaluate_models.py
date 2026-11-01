import json
import time
import sys
from pathlib import Path

# Add backend directory to path so 'app' is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from collections import Counter
import joblib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    brier_score_loss,
)
import spacy

from app.services.action_classifier import ActionClassifier
from app.services.preprocessing import get_spacy_nlp, ProcessedSentence
from app.services.assignee_extractor import extract_assignee
from app.services.deadline_extractor import extract_deadline
from datetime import date

DATASET_PATH = Path(__file__).resolve().parent / "dataset_large.json"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parent / "model.joblib"
REPORT_OUTPUT_PATH = Path(__file__).resolve().parent / "benchmark_report.json"


def load_data(dataset_path: Path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = []
    test_data = []

    for m in data.get("meetings", []):
        split = m.get("split", "train")
        for u in m.get("utterances", []):
            item = {
                "meeting_id": m.get("meeting_id"),
                "speaker": u.get("speaker"),
                "text": u["text"],
                "label": u["label"],
                "assignee": u.get("assignee"),
                "deadline_text": u.get("deadline_text"),
            }
            if split == "train":
                train_data.append(item)
            else:
                test_data.append(item)

    return train_data, test_data


def evaluate_classifiers(train_data, test_data):
    train_texts = [d["text"] for d in train_data]
    train_labels = [d["label"] for d in train_data]

    test_texts = [d["text"] for d in test_data]
    test_labels = [d["label"] for d in test_data]

    print("\n" + "="*70)
    print("STAGE 1: BENCHMARKING ACTION CLASSIFICATION MODELS")
    print(f"Train samples: {len(train_texts)} | Held-Out Test samples: {len(test_texts)}")
    print("="*70)

    # Define model candidates
    models = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "Logistic Regression (Balanced)": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=2.5, class_weight="balanced", max_iter=1500, random_state=42)),
        ]),
    }

    results = []
    trained_pipelines = {}

    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(train_texts, train_labels)
        train_time = time.time() - t0

        t1 = time.time()
        preds = pipeline.predict(test_texts)
        inf_time = (time.time() - t1) * 1000 / len(test_texts)  # ms per sample

        acc = accuracy_score(test_labels, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, preds, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(test_labels, preds, average="weighted", zero_division=0)

        # Per-class breakdown
        report = classification_report(test_labels, preds, output_dict=True, zero_division=0)
        action_f1 = report.get("ACTION_ITEM", {}).get("f1-score", 0.0)
        sugg_f1 = report.get("SUGGESTION", {}).get("f1-score", 0.0)
        comp_f1 = report.get("COMPLETED_ACTION", {}).get("f1-score", 0.0)
        non_f1 = report.get("NON_ACTIONABLE", {}).get("f1-score", 0.0)

        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_f1": round(f1_weighted, 4),
            "action_item_f1": round(action_f1, 4),
            "suggestion_f1": round(sugg_f1, 4),
            "completed_f1": round(comp_f1, 4),
            "non_actionable_f1": round(non_f1, 4),
            "train_time_sec": round(train_time, 2),
            "latency_ms": round(inf_time, 3),
        })
        trained_pipelines[name] = pipeline

    # Benchmark Rule-Based Baseline on Held-Out Test Set
    nlp = get_spacy_nlp()
    rule_classifier = ActionClassifier(model_path=None)
    rule_preds = []
    t0 = time.time()
    for d in test_data:
        doc = nlp(d["text"])
        ps = ProcessedSentence(
            utterance_idx=0,
            sentence_idx=0,
            speaker=d.get("speaker"),
            text=d["text"],
            doc=doc,
            raw_utterance=d["text"],
        )
        cls_res, _, _ = rule_classifier.rule_based_classify(ps)
        rule_preds.append(cls_res.value)
    rule_latency = (time.time() - t0) * 1000 / len(test_data)

    acc = accuracy_score(test_labels, rule_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, rule_preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(test_labels, rule_preds, average="weighted", zero_division=0)
    rep = classification_report(test_labels, rule_preds, output_dict=True, zero_division=0)

    results.append({
        "model": "Rule-Based Baseline (spaCy + Patterns)",
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "action_item_f1": round(rep.get("ACTION_ITEM", {}).get("f1-score", 0.0), 4),
        "suggestion_f1": round(rep.get("SUGGESTION", {}).get("f1-score", 0.0), 4),
        "completed_f1": round(rep.get("COMPLETED_ACTION", {}).get("f1-score", 0.0), 4),
        "non_actionable_f1": round(rep.get("NON_ACTIONABLE", {}).get("f1-score", 0.0), 4),
        "train_time_sec": 0.0,
        "latency_ms": round(rule_latency, 3),
    })

    # Benchmark Hybrid Ensemble (Logistic Regression ML + Rule Calibration)
    log_reg_pipeline = trained_pipelines["Logistic Regression (Balanced)"]
    joblib.dump(log_reg_pipeline, MODEL_OUTPUT_PATH)
    hybrid_classifier = ActionClassifier(model_path=str(MODEL_OUTPUT_PATH))

    hybrid_preds = []
    t0 = time.time()
    for d in test_data:
        doc = nlp(d["text"])
        ps = ProcessedSentence(
            utterance_idx=0,
            sentence_idx=0,
            speaker=d.get("speaker"),
            text=d["text"],
            doc=doc,
            raw_utterance=d["text"],
        )
        res = hybrid_classifier.classify(ps, mode="hybrid")
        hybrid_preds.append(res.classification.value)
    hybrid_latency = (time.time() - t0) * 1000 / len(test_data)

    acc = accuracy_score(test_labels, hybrid_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, hybrid_preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(test_labels, hybrid_preds, average="weighted", zero_division=0)
    rep = classification_report(test_labels, hybrid_preds, output_dict=True, zero_division=0)

    results.append({
        "model": "Hybrid Ensemble (LR + Rule Calibration)",
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "action_item_f1": round(rep.get("ACTION_ITEM", {}).get("f1-score", 0.0), 4),
        "suggestion_f1": round(rep.get("SUGGESTION", {}).get("f1-score", 0.0), 4),
        "completed_f1": round(rep.get("COMPLETED_ACTION", {}).get("f1-score", 0.0), 4),
        "non_actionable_f1": round(rep.get("NON_ACTIONABLE", {}).get("f1-score", 0.0), 4),
        "train_time_sec": round(results[3]["train_time_sec"], 2),
        "latency_ms": round(hybrid_latency, 3),
    })

    return results, trained_pipelines


def evaluate_ner_and_slot_filling(test_data):
    print("\n" + "="*70)
    print("STAGE 2: EVALUATION OF NER AND SLOT FILLING (ASSIGNEE & DEADLINE)")
    print("="*70)

    nlp = get_spacy_nlp()
    ref_date = date(2026, 10, 27)

    # 1. Evaluate Assignee Extraction on samples where ground truth assignee is annotated
    assignee_samples = [d for d in test_data if d["assignee"] is not None]
    assignee_correct = 0
    spacy_person_detected = 0

    for d in assignee_samples:
        doc = nlp(d["text"])
        ps = ProcessedSentence(
            utterance_idx=0,
            sentence_idx=0,
            speaker=d.get("speaker"),
            text=d["text"],
            doc=doc,
            raw_utterance=d["text"],
        )
        res = extract_assignee(ps)
        # Check if spaCy detected any PERSON entity
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            spacy_person_detected += 1

        pred_assignee = (res.assignee or "").lower()
        true_assignee = d["assignee"].lower()

        if pred_assignee and (pred_assignee in true_assignee or true_assignee in pred_assignee):
            assignee_correct += 1

    assignee_acc = assignee_correct / len(assignee_samples) if assignee_samples else 0.0
    spacy_person_recall = spacy_person_detected / len(assignee_samples) if assignee_samples else 0.0

    # 2. Evaluate Deadline Extraction on samples where temporal phrase is present
    deadline_samples = [d for d in test_data if d["deadline_text"] is not None]
    deadline_detected = 0
    deadline_normalized_count = 0

    for d in deadline_samples:
        doc = nlp(d["text"])
        ps = ProcessedSentence(
            utterance_idx=0,
            sentence_idx=0,
            speaker=d.get("speaker"),
            text=d["text"],
            doc=doc,
            raw_utterance=d["text"],
        )
        res = extract_deadline(ps, reference_date=ref_date)
        if res.deadline_text:
            deadline_detected += 1
        if res.deadline:
            deadline_normalized_count += 1

    deadline_recall = deadline_detected / len(deadline_samples) if deadline_samples else 0.0
    normalization_rate = deadline_normalized_count / len(deadline_samples) if deadline_samples else 0.0

    ner_metrics = {
        "assignee_evaluation": {
            "test_samples_with_assignee": len(assignee_samples),
            "assignee_extraction_accuracy": round(assignee_acc, 4),
            "raw_spacy_person_recall": round(spacy_person_recall, 4),
            "methodology": "Combines spaCy PERSON entities with grammatical vocative direction and first-person speaker attribution."
        },
        "deadline_evaluation": {
            "test_samples_with_deadlines": len(deadline_samples),
            "temporal_phrase_detection_recall": round(deadline_recall, 4),
            "iso_date_normalization_success_rate": round(normalization_rate, 4),
            "methodology": "Combines dependency tree prepositional temporal attachments with dateparser relative anchoring."
        }
    }

    return ner_metrics


def run_full_benchmark():
    train_data, test_data = load_data(DATASET_PATH)
    classifier_results, trained_pipelines = evaluate_classifiers(train_data, test_data)
    ner_results = evaluate_ner_and_slot_filling(test_data)

    report = {
        "dataset_summary": {
            "name": "AMI Meeting Corpus + Enterprise Dialogue Benchmark",
            "train_utterances": len(train_data),
            "test_utterances": len(test_data),
            "total_utterances": len(train_data) + len(test_data),
        },
        "classifier_comparison": classifier_results,
        "ner_and_slot_filling": ner_results,
    }

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON TABLE (HELD-OUT TEST SET)")
    print("="*80)
    df = pd.DataFrame(classifier_results)
    cols = ["model", "accuracy", "macro_f1", "weighted_f1", "action_item_f1", "suggestion_f1", "completed_f1", "latency_ms"]
    print(df[cols].to_string(index=False))

    print("\n" + "="*80)
    print("NER & SLOT FILLING EVALUATION")
    print("="*80)
    for section, stats in ner_results.items():
        print(f"\n[{section.upper()}]")
        for k, v in stats.items():
            print(f"  {k:35s}: {v}")

    print(f"\nDetailed report saved to: {REPORT_OUTPUT_PATH}")
    print(f"Production model artifact saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    run_full_benchmark()
