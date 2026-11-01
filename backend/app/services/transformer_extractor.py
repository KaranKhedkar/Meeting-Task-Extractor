import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..schemas.task import TaskClassification

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "training" / "v2_distilbert"

_TOKENIZER = None
_MODEL = None


def is_v2_available() -> bool:
    """Checks if the fine-tuned V2 transformer weights are present on disk."""
    return (MODEL_DIR / "config.json").exists() and (
        (MODEL_DIR / "model.safetensors").exists() or (MODEL_DIR / "pytorch_model.bin").exists()
    )


def get_v2_model() -> Tuple[Any, Any]:
    """Loads and caches the V2 DistilBERT model and tokenizer."""
    global _TOKENIZER, _MODEL
    if _MODEL is None or _TOKENIZER is None:
        if not is_v2_available():
            raise FileNotFoundError(f"V2 Transformer model weights not found in {MODEL_DIR}")
        _TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _MODEL = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def classify_with_v2(text: str, temperature: float = 0.55) -> Dict[str, Any]:
    """
    Classifies an utterance using the fine-tuned V2 DistilBERT model.
    Applies temperature calibration (T=0.55) to sharpen softmax probabilities
    and prevent under-confident diffused predictions from 2-epoch CPU checkpoint.
    
    Returns:
      - classification: TaskClassification enum
      - confidence: float probability (0.0 - 1.0)
      - probabilities: dict of all class probabilities
    """
    tokenizer, model = get_v2_model()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        # Temperature scaling for neural calibration (Guo et al., ICML 2017)
        scaled_logits = logits / max(0.1, temperature)
        probs = torch.softmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()

    # Model class mapping: 0=NON_ACTIONABLE, 1=SUGGESTION, 2=ACTION_ITEM, 3=COMPLETED_ACTION
    id2label = {
        0: TaskClassification.NON_ACTIONABLE,
        1: TaskClassification.SUGGESTION,
        2: TaskClassification.ACTION_ITEM,
        3: TaskClassification.COMPLETED_ACTION,
    }

    pred_idx = int(probs.argmax())
    pred_class = id2label.get(pred_idx, TaskClassification.NON_ACTIONABLE)
    confidence = float(probs[pred_idx])

    prob_dict = {
        id2label[i].value: round(float(probs[i]), 4) for i in range(len(probs))
    }

    backbone_name = getattr(model.config, "model_type", "transformer")
    return {
        "classification": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "model": f"{backbone_name} (V2 Neural + Calibrated)",
    }


