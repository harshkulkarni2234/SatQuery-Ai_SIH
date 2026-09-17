"""VQA service — calls the isolated local VQA worker over HTTP.

The heavy ML stack (torch/transformers/peft) lives ONLY in the worker's
separate environment; this module stays dependency-light (requests only).
"""

import os
import time

import requests

VQA_WORKER_URL = os.getenv("VQA_WORKER_URL", "http://127.0.0.1:8001").rstrip("/")
VQA_WORKER_TIMEOUT_S = float(os.getenv("VQA_WORKER_TIMEOUT_S", "120"))

# Module-level alias so tests can mock the HTTP call without touching the
# shared requests module.
_post = requests.post

_SPECIALIST_EXCLUDE = [
    "describe", "explain", "caption", "overview", "summar", "what is visible",
    "what can you see", "show me", "find", "locate", "where is", "compare",
    "between", "change", "detect", "difference", "optical", "sar",
    "multispectral", "what changed",
]
_SPECIALIST_PRESENCE = [
    "is there", "are there", "does the image contain", "contains", "contain",
    "is any", "present in the image", "exist",
]
_SPECIALIST_COUNT = ["how many"]


def should_use_specialist(query_text: str) -> bool:
    """Deterministic, deliberately conservative gate for the experimental specialist.

    True only for obvious presence/existence or simple counting questions.
    Never for descriptions, captions, spatial reasoning, grounding,
    change detection, or optical/SAR questions. Mirrors the worker's own
    gate; keep both in sync.
    """
    query = (query_text or "").strip().lower()
    if not query or len(query.split()) > 24:
        return False
    if any(phrase in query for phrase in _SPECIALIST_EXCLUDE):
        return False
    if any(phrase in query for phrase in _SPECIALIST_COUNT):
        return len(query.split()) <= 10
    return any(phrase in query for phrase in _SPECIALIST_PRESENCE)


def _unavailable(query_text: str, use_specialist: bool, started: float, reason: str) -> dict:
    """Honest failed-worker result: no fabricated answer, no crash."""
    return {
        "answer_text": (
            "[VQA unavailable] The local SmolVLM worker is not responding. "
            f"Start it with ml/vqa-worker/run_worker.ps1. ({reason})"
        ),
        "model_version": None,
        "specialist": use_specialist,
        "adapter_used": False,
        "reason": f"worker unavailable ({reason})",
        "confidence_score": None,
        "execution_time_ms": int((time.monotonic() - started) * 1000),
        "error": True,
    }


def answer_question(image_path: str, query_text: str) -> dict:
    """VQA specialist interface: image + question -> structured answer.

    Deliberately returns confidence_score=None — there is no calibrated
    confidence method for this model yet (per Phase 8 policy).
    """
    started = time.monotonic()
    use_specialist = should_use_specialist(query_text)

    try:
        with open(image_path, "rb") as f:
            response = _post(
                f"{VQA_WORKER_URL}/vqa",
                data={
                    "question": query_text,
                    "use_specialist": "true" if use_specialist else "false",
                },
                files={"image": (os.path.basename(image_path), f)},
                timeout=VQA_WORKER_TIMEOUT_S,
            )
    except requests.RequestException as exc:
        return _unavailable(query_text, use_specialist, started, f"connection failed: {type(exc).__name__}")

    if response.status_code != 200:
        return _unavailable(query_text, use_specialist, started, f"worker error status {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        return _unavailable(query_text, use_specialist, started, "malformed worker response")

    if not isinstance(data, dict) or not isinstance(data.get("answer_text"), str) or not data["answer_text"].strip():
        return _unavailable(query_text, use_specialist, started, "malformed worker response")

    return {
        "answer_text": data["answer_text"],
        "model_version": data.get("model_version"),
        "specialist": bool(data.get("specialist", False)),
        "adapter_used": bool(data.get("adapter_used", data.get("specialist", False))),
        "reason": data.get("reason"),
        "confidence_score": None,
        "execution_time_ms": int((time.monotonic() - started) * 1000),
        "error": False,
    }