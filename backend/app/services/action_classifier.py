import re
from typing import Tuple, List, Optional
from dataclasses import dataclass
import joblib
from ..core.config import settings
from ..schemas.task import TaskClassification
from .preprocessing import ProcessedSentence

# Linguistic markers for rule-based analysis
COMPLETED_MARKERS = [
    r"\b(already|yesterday|last (week|month|friday|monday|year)|earlier today|have (already )?(been|completed|deployed|fixed|merged|shipped))\b",
    r"\b(was|were|got) (deployed|fixed|merged|shipped|resolved|closed|tested|completed|restored)\b",
    r"\b(finished|completed|resolved|concluded) (the )?(testing|development|migration|audit|review)\b",
]

SUGGESTION_MARKERS = [
    r"\b(maybe we (could|should|might))\b",
    r"\b(what if we|how about we)\b",
    r"\b(could we consider|perhaps we (could|might|should))\b",
    r"\b(it might be (good|helpful|useful) to)\b",
    r"\b(we could (possibly|potentially))\b",
]

COMMITMENT_MARKERS = [
    r"\b(i will|i'll|i am going to|i'm going to)\b",
    r"\b(i'll make sure to|i will make sure to|i'll ensure|i will ensure)\b",
    r"\b(i'll handle|i will handle|i'll take care of|i will take care of)\b",
    r"\b(we will|we'll|we need to|we have to|we must)\b",
    r"\b(let's (assign|make sure|ensure|get someone to))\b",
]

# Meeting opening / procedural chatter that is NOT an action item
PROCEDURAL_CHATTER_MARKERS = [
    r"\b(good morning|good afternoon|good evening|welcome (everyone|all|to))\b",
    r"\b(let's (start|begin|wrap up|end) (the|this)?\s*(sprint|meeting|call|session|standup|sync))\b",
    r"\b(thanks for (joining|coming))\b",
]

REQUEST_IMPERATIVE_MARKERS = [
    r"\b([A-Za-z]+,\s+)?(please|can you|could you|would you)\b",
    r"\b(make sure to|ensure that|don't forget to)\b",
    r"\b(assign someone to|someone should)\b",
]

NEGATION_MARKERS = [
    r"\b(don't need to|no need to|do not need to|won't need to)\b",
    r"\b(won't|will not|am not going to|cannot|can't)\b",
    r"\b(we won't|not necessary|skip)\b",
]


@dataclass
class ClassificationResult:
    classification: TaskClassification
    confidence: float
    method_used: str  # "rule_based", "ml", or "hybrid"
    triggers: List[str]


class ActionClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(settings.ML_MODEL_PATH)
        self.ml_pipeline = None
        self._load_ml_model()

    def _load_ml_model(self):
        try:
            self.ml_pipeline = joblib.load(self.model_path)
        except Exception:
            self.ml_pipeline = None

    def rule_based_classify(self, sentence: ProcessedSentence) -> Tuple[TaskClassification, float, List[str]]:
        """
        Classifies utterance using deterministic linguistic rules,
        syntax tags from spaCy, and grammatical structure.
        """
        text = sentence.text
        text_lower = text.lower()
        triggers = []
        doc = sentence.doc

        # 1. Check for explicit Negation (e.g. "We don't need to update the billing code")
        for neg in NEGATION_MARKERS:
            if re.search(neg, text_lower):
                triggers.append(f"negation:{neg}")
                return TaskClassification.NON_ACTIONABLE, 0.88, triggers

        # 1b. Check for Meeting Opening / Procedural chatter (e.g. "Good morning everyone, let's start the meeting")
        for chat in PROCEDURAL_CHATTER_MARKERS:
            if re.search(chat, text_lower):
                triggers.append(f"procedural_chatter:{chat}")
                return TaskClassification.NON_ACTIONABLE, 0.92, triggers

        # 2. Check for Completed Actions
        for comp in COMPLETED_MARKERS:
            if re.search(comp, text_lower):
                triggers.append(f"completed_marker:{comp}")
                return TaskClassification.COMPLETED_ACTION, 0.90, triggers

        # Also check past tense verbs + past temporal adverbs in dependency tree
        has_past_verb = any(t.tag_ in ("VBD", "VBN") for t in doc)
        has_past_dep = any(t.dep_ == "advmod" and t.lower_ in ("yesterday", "earlier", "already") for t in doc)
        if has_past_verb and has_past_dep:
            triggers.append("dep_past_action")
            return TaskClassification.COMPLETED_ACTION, 0.85, triggers

        # 3. Check for Suggestions / Hypotheticals
        for sug in SUGGESTION_MARKERS:
            if re.search(sug, text_lower):
                triggers.append(f"suggestion_marker:{sug}")
                return TaskClassification.SUGGESTION, 0.86, triggers

        # 4. Check for Action Items: First-person commitments
        for comm in COMMITMENT_MARKERS:
            if re.search(comm, text_lower):
                triggers.append(f"commitment_marker:{comm}")
                return TaskClassification.ACTION_ITEM, 0.92, triggers

        # 5. Check for Action Items: Directed requests / imperative requests
        for req in REQUEST_IMPERATIVE_MARKERS:
            if re.search(req, text_lower):
                triggers.append(f"request_marker:{req}")
                return TaskClassification.ACTION_ITEM, 0.89, triggers

        # 6. Check grammatical Imperatives (Root verb is base form VB without subject pronoun)
        # E.g. "Prepare the presentation by Friday." or "Amit, review the interface."
        if len(doc) > 0:
            root = [t for t in doc if t.dep_ == "ROOT"]
            if root:
                root_tok = root[0]
                # If root is base form verb (VB) and has no explicit nominal subject preceding it
                has_nsubj = any(t.dep_ in ("nsubj", "nsubjpass") for t in root_tok.children)
                if root_tok.tag_ == "VB" and not has_nsubj:
                    triggers.append("grammatical_imperative_vb")
                    return TaskClassification.ACTION_ITEM, 0.85, triggers

        # 7. Check for Non-Actionable discussions, questions, observations
        if text.endswith("?") and not any(re.search(p, text_lower) for p in [r"can you", r"could you", r"would you"]):
            triggers.append("information_question")
            return TaskClassification.NON_ACTIONABLE, 0.85, triggers

        # Default fallback to NON_ACTIONABLE
        triggers.append("default_declarative")
        return TaskClassification.NON_ACTIONABLE, 0.70, triggers

    def ml_classify(self, text: str) -> Optional[Tuple[TaskClassification, float]]:
        """Predicts class using the trained scikit-learn TF-IDF model if available."""
        if self.ml_pipeline is None:
            return None

        try:
            pred_class = self.ml_pipeline.predict([text])[0]
            # Get probability if available
            if hasattr(self.ml_pipeline, "predict_proba"):
                probs = self.ml_pipeline.predict_proba([text])[0]
                classes = list(self.ml_pipeline.classes_)
                class_idx = classes.index(pred_class)
                confidence = float(probs[class_idx])
            else:
                confidence = 0.82
            
            return TaskClassification(pred_class), confidence
        except Exception:
            return None

    def classify(self, sentence: ProcessedSentence, mode: str = "hybrid") -> ClassificationResult:
        """
        Unified classifier executing rule-based, ML, or hybrid approach.
        In hybrid mode, ML predictions are combined with deterministic linguistic guardrails.
        """
        # V2 Neural Transformer Mode with Calibrated Confidence & Guardrails
        if mode == "v2_transformer":
            from .transformer_extractor import is_v2_available, classify_with_v2
            if is_v2_available():
                try:
                    res_v2 = classify_with_v2(sentence.text)
                    pred_cls = res_v2["classification"]
                    pred_conf = res_v2["confidence"]

                    # Calibrate with linguistic rules
                    rule_class, rule_conf, triggers = self.rule_based_classify(sentence)

                    # Guardrail 1: If rule-based clearly identifies COMPLETED_ACTION, PROCEDURAL CHATTER, or strong NEGATION, override
                    if rule_class in (TaskClassification.COMPLETED_ACTION, TaskClassification.NON_ACTIONABLE) and any(
                        t.startswith(("negation", "completed", "procedural_chatter", "information_question")) for t in triggers
                    ):
                        return ClassificationResult(
                            classification=rule_class,
                            confidence=round(max(rule_conf, 0.90), 2),
                            method_used="v2_transformer",
                            triggers=triggers + [f"rule_guardrail:{rule_class.value}"],
                        )

                    # Guardrail 2: If neural predicts ACTION_ITEM and rules confirm commitment or imperative
                    if pred_cls == TaskClassification.ACTION_ITEM and any(
                        t.startswith(("commitment", "request", "grammatical_imperative")) for t in triggers
                    ):
                        calib_conf = min(0.96, max(pred_conf, 0.88))
                        return ClassificationResult(
                            classification=TaskClassification.ACTION_ITEM,
                            confidence=round(calib_conf, 2),
                            method_used="v2_transformer",
                            triggers=triggers + ["neural_symbolic_consensus"],
                        )

                    # Guardrail 3: If neural predicted SUGGESTION but utterance has explicit commitment / imperative
                    if pred_cls == TaskClassification.SUGGESTION and any(
                        t.startswith(("commitment", "request")) for t in triggers
                    ):
                        return ClassificationResult(
                            classification=TaskClassification.ACTION_ITEM,
                            confidence=round(max(rule_conf, 0.88), 2),
                            method_used="v2_transformer",
                            triggers=triggers + ["commitment_promoted_from_suggestion"],
                        )

                    # Default: Neural Transformer prediction with temperature-calibrated confidence
                    return ClassificationResult(
                        classification=pred_cls,
                        confidence=round(pred_conf, 2),
                        method_used="v2_transformer",
                        triggers=[f"distilbert_pred:{pred_cls.value}"],
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("V2 transformer inference error: %s", e)

        rule_class, rule_conf, triggers = self.rule_based_classify(sentence)

        if mode == "rule_based" or self.ml_pipeline is None:
            return ClassificationResult(
                classification=rule_class,
                confidence=round(rule_conf, 2),
                method_used="rule_based",
                triggers=triggers,
            )

        ml_res = self.ml_classify(sentence.text)
        if ml_res is None or mode == "ml":
            if ml_res:
                return ClassificationResult(
                    classification=ml_res[0],
                    confidence=round(ml_res[1], 2),
                    method_used="ml",
                    triggers=["ml_prediction"],
                )
            return ClassificationResult(
                classification=rule_class,
                confidence=round(rule_conf, 2),
                method_used="rule_based",
                triggers=triggers,
            )

        # Hybrid Mode:
        ml_class, ml_conf = ml_res
        
        # Guardrail 1: If rule-based clearly identifies COMPLETED_ACTION or strong NEGATION, trust rules
        if rule_class in (TaskClassification.COMPLETED_ACTION, TaskClassification.NON_ACTIONABLE) and any(
            t.startswith(("negation", "completed")) for t in triggers
        ):
            return ClassificationResult(
                classification=rule_class,
                confidence=round(max(rule_conf, 0.85), 2),
                method_used="hybrid",
                triggers=triggers + [f"ml_overridden:{ml_class}"],
            )

        # Guardrail 2: If both agree
        if ml_class == rule_class:
            combined_conf = min(0.98, max(ml_conf, rule_conf) + 0.05)
            return ClassificationResult(
                classification=ml_class,
                confidence=round(combined_conf, 2),
                method_used="hybrid",
                triggers=triggers + ["consensus"],
            )

        # Guardrail 3: If ML says ACTION_ITEM and rule-based had strong commitment/request marker
        if any(t.startswith(("commitment", "request", "imperative")) for t in triggers):
            return ClassificationResult(
                classification=TaskClassification.ACTION_ITEM,
                confidence=round(rule_conf, 2),
                method_used="hybrid",
                triggers=triggers + [f"rule_preferred:{rule_class}"],
            )

        # Fallback to ML with tempered confidence
        return ClassificationResult(
            classification=ml_class,
            confidence=round(ml_conf, 2),
            method_used="hybrid",
            triggers=triggers + [f"ml_preferred:{ml_class}"],
        )
