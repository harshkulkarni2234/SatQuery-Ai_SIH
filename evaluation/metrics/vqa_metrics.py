"""VQA-style scoring metrics (RSVQA/CDVQA/VRSBench answer-space conventions).

Pure functions, no dataset or network dependency, so they can be unit
tested with hand-computed tiny examples independent of which benchmark
(if any) actually gets run against them.
"""

from __future__ import annotations

import re
from typing import Optional


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/whitespace — RSVQA-style loose matching."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    """Fraction of predictions whose normalized text exactly matches the
    normalized ground truth. Raises on empty input rather than returning a
    misleading 0.0/1.0."""
    if not predictions:
        raise ValueError("exact_match_accuracy: empty predictions")
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length")
    correct = sum(
        1 for p, g in zip(predictions, ground_truths)
        if normalize_answer(p) == normalize_answer(g)
    )
    return correct / len(predictions)


def parse_categorical(text: str, vocabulary: list[str]) -> Optional[str]:
    """Extract which one of a small closed vocabulary (e.g. ["urban", "rural"])
    appears in free text. Returns None (not a guess) if none of the
    vocabulary words appear, or if more than one does (ambiguous) — added
    after a real scoring bug: exact_match_accuracy requires literal string
    equality, so a real, correct free-text answer like "It is an urban
    area." never matched a bare "urban" ground truth. This generalizes
    parse_yes_no's approach to any small fixed vocabulary."""
    normalized = normalize_answer(text)
    tokens = set(normalized.split())
    matches = [word for word in vocabulary if word in tokens]
    if len(matches) == 1:
        return matches[0]
    return None


def categorical_accuracy(predictions: list[str], ground_truths: list[str], vocabulary: list[str]) -> dict:
    """Accuracy over small-closed-vocabulary free-text answers (e.g.
    rural/urban). Predictions that can't be resolved to exactly one
    vocabulary word are excluded from the denominator, same convention as
    yes_no_accuracy — never silently scored as wrong."""
    if not predictions:
        raise ValueError("categorical_accuracy: empty predictions")
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length")
    correct = 0
    unparseable = 0
    scored = 0
    for pred, gt in zip(predictions, ground_truths):
        parsed = parse_categorical(pred, vocabulary)
        if parsed is None:
            unparseable += 1
            continue
        scored += 1
        if parsed == normalize_answer(gt):
            correct += 1
    return {
        "accuracy": (correct / scored) if scored else None,
        "n_scored": scored,
        "n_unparseable": unparseable,
        "n_total": len(predictions),
    }


_YES_WORDS = {"yes", "y", "true"}
_NO_WORDS = {"no", "n", "false"}


def parse_yes_no(text: str) -> Optional[bool]:
    """Extract a yes/no verdict from free text. Returns None (not a fabricated
    guess) when the text contains neither a clear yes nor a clear no token,
    or contains both (ambiguous)."""
    normalized = normalize_answer(text)
    tokens = set(normalized.split())
    has_yes = bool(tokens & _YES_WORDS)
    has_no = bool(tokens & _NO_WORDS)
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def yes_no_accuracy(predictions: list[str], ground_truths: list[bool]) -> dict:
    """Accuracy over yes/no questions. Predictions that can't be parsed as a
    clear yes/no are counted as 'unparseable' and excluded from the accuracy
    denominator (never silently scored as wrong or right)."""
    if not predictions:
        raise ValueError("yes_no_accuracy: empty predictions")
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length")
    correct = 0
    unparseable = 0
    scored = 0
    for pred, gt in zip(predictions, ground_truths):
        parsed = parse_yes_no(pred)
        if parsed is None:
            unparseable += 1
            continue
        scored += 1
        if parsed == gt:
            correct += 1
    return {
        "accuracy": (correct / scored) if scored else None,
        "n_scored": scored,
        "n_unparseable": unparseable,
        "n_total": len(predictions),
    }


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_count(text: str) -> Optional[float]:
    """Extract the first number found in free text. Returns None if no
    number is present rather than guessing 0."""
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    return float(match.group())


def count_accuracy_rmse(predictions: list[str], ground_truths: list[float]) -> dict:
    """RMSE + exact-match accuracy for count-style questions. Unparseable
    predictions are excluded from both, and their count is reported."""
    if not predictions:
        raise ValueError("count_accuracy_rmse: empty predictions")
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length")
    errors = []
    exact = 0
    unparseable = 0
    for pred, gt in zip(predictions, ground_truths):
        parsed = parse_count(pred)
        if parsed is None:
            unparseable += 1
            continue
        errors.append((parsed - gt) ** 2)
        if parsed == gt:
            exact += 1
    n_scored = len(errors)
    rmse = (sum(errors) / n_scored) ** 0.5 if n_scored else None
    return {
        "rmse": rmse,
        "exact_accuracy": (exact / n_scored) if n_scored else None,
        "n_scored": n_scored,
        "n_unparseable": unparseable,
        "n_total": len(predictions),
    }
