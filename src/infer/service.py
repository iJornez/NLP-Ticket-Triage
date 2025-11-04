from fastapi import APIRouter
from pydantic import BaseModel
import time
from typing import Dict
from .preprocess import redact

router = APIRouter()


# --- Model loading (tries real model; falls back to mock) ---
import os, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from ..models.multitask_head import MultiTask

ART_DIR = Path("artifacts/model")
_REAL_MODEL = None
_TOKENIZER = None
TOPIC_LABELS = ["login", "billing", "bug", "feature", "shipping", "other"]
SENT_LABELS  = ["neg", "neu", "pos"]

if ART_DIR.exists():
    try:
        with open(ART_DIR / "label_maps.json") as f:
            maps = json.load(f)
        TOPIC_LABELS = maps.get("topics", TOPIC_LABELS)
        SENT_LABELS = maps.get("sentiment", SENT_LABELS)
        with open(ART_DIR / "model_config.json") as f:
            cfg = json.load(f)
        _TOKENIZER = AutoTokenizer.from_pretrained(ART_DIR)
        _REAL_MODEL = MultiTask(enc=cfg["model_name"], n_topic=len(TOPIC_LABELS), n_sent=len(SENT_LABELS))
        _REAL_MODEL.load_state_dict(torch.load(ART_DIR / "pytorch_model.bin", map_location="cpu"))
        _REAL_MODEL.eval()
    except Exception as e:
        _REAL_MODEL = None
        _TOKENIZER = None

TOPIC_LABELS = ["login", "billing", "bug", "feature", "shipping", "other"]
SENT_LABELS  = ["neg", "neu", "pos"]

class PredictIn(BaseModel):
    text: str
    top_k: int | None = None

class ClassOut(BaseModel):
    label: str
    score: float

class PredictOut(BaseModel):
    topic: ClassOut
    sentiment: ClassOut
    probs: Dict[str, Dict[str, float]]
    latency_ms: int
    model_version: str


def _mock_infer(text: str):
    """Deterministic mock until a trained model is wired.
    Topic keyed by simple heuristics; sentiment by exclamation/negation ratio.
    """
    t = text.lower()
    topic = "other"
    if any(k in t for k in ["login", "password", "reset"]):
        topic = "login"
    elif any(k in t for k in ["refund", "charge", "invoice", "billing"]):
        topic = "billing"
    elif any(k in t for k in ["bug", "error", "crash", "fail"]):
        topic = "bug"
    elif any(k in t for k in ["feature", "improve", "add"]):
        topic = "feature"
    elif any(k in t for k in ["ship", "delivery", "tracking"]):
        topic = "shipping"

    neg_terms = sum(t.count(w) for w in ["not ", "no ", "can't", "error", "fail"])
    pos_terms = sum(t.count(w) for w in ["thanks", "great", "love", "awesome"])
    excls = t.count("!")
    score = max(0.05, min(0.95, 0.5 + (pos_terms - neg_terms + excls * 0.1)))
    sent = "pos" if score > 0.6 else ("neg" if score < 0.4 else "neu")

    topic_probs = {lbl: (0.82 if lbl == topic else 0.18/(len(TOPIC_LABELS)-1)) for lbl in TOPIC_LABELS}
    sent_probs = {lbl: (0.9 if lbl == sent else 0.1/(len(SENT_LABELS)-1)) for lbl in SENT_LABELS}
    return topic, sent, topic_probs, sent_probs


@router.post("/predict", response_model=PredictOut)
async def predict(inp: PredictIn):
    t0 = time.time()
    red_text, _ = redact(inp.text)

    topic, sent, topic_probs, sent_probs = _mock_infer(red_text)
    latency = int((time.time() - t0) * 1000)

    

# Try real model if available
def _real_infer(text: str):
    if _REAL_MODEL is None or _TOKENIZER is None:
        return None
    tok = _TOKENIZER(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    with torch.no_grad():
        t_logits, s_logits = _REAL_MODEL(tok["input_ids"], tok["attention_mask"])
        import torch.nn.functional as F
        t_prob = F.softmax(t_logits, dim=-1).squeeze(0).tolist()
        s_prob = F.softmax(s_logits, dim=-1).squeeze(0).tolist()
    topic_idx = int(max(range(len(t_prob)), key=lambda i: t_prob[i]))
    sent_idx = int(max(range(len(s_prob)), key=lambda i: s_prob[i]))
    topic_probs = {lbl: float(t_prob[i]) for i, lbl in enumerate(TOPIC_LABELS)}
    sent_probs = {lbl: float(s_prob[i]) for i, lbl in enumerate(SENT_LABELS)}
    return TOPIC_LABELS[topic_idx], SENT_LABELS[sent_idx], topic_probs, sent_probs

    # pick path
    real = _real_infer(red_text)
    if real is not None:
        topic, sent, topic_probs, sent_probs = real
        latency = int((time.time() - t0) * 1000)

    return PredictOut(
        topic=ClassOut(label=topic, score=topic_probs[topic]),
        sentiment=ClassOut(label=sent, score=sent_probs[sent]),
        probs={"topic": topic_probs, "sentiment": sent_probs},
        latency_ms=latency,
        model_version="mock-0.1.0",
    )

# add this import
from pydantic import BaseModel, ConfigDict

class PredictOut(BaseModel):
    # allow fields like `model_version`
    model_config = ConfigDict(protected_namespaces=())

    topic: ClassOut
    sentiment: ClassOut
    probs: dict[str, dict[str, float]]
    latency_ms: int
    model_version: str
