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


def parse_percentage_bucket(text: str, buckets: list[str]) -> Optional[str]:
    """Resolves free text to one of a list of bucket labels shaped like
    "0_to_10", "90_to_100", or a bare "0" for a zero bucket (CDVQA's
    change_ratio/change_ratio_types convention). Tries an exact keyword
    match first (in case the prediction literally says a bucket label),
    then falls back to extracting a bare percentage number and placing it
    in the matching bucket — added after finding that a real, correct
    answer like "approximately 5.6% of the frame changed" would otherwise
    be marked unparseable forever, since "5.6" never literally appears as
    the substring "0_to_10". Returns None if no percentage can be found
    at all, or if it falls outside every bucket's range (never guesses)."""
    keyword_match = parse_categorical(text, buckets)
    if keyword_match is not None:
        return keyword_match

    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    value = float(match.group())

    for bucket in buckets:
        if bucket == "0":
            if value == 0:
                return bucket
            continue
        parts = bucket.split("_to_")
        if len(parts) != 2:
            continue
        low, high = float(parts[0]), float(parts[1])
        if low < value <= high or (low == 0 and value == 0):
            return bucket
    return None


def percentage_bucket_accuracy(predictions: list[str], ground_truths: list[str], buckets: list[str]) -> dict:
    """Accuracy over percentage-bucket questions, resolving free-text
    percentages into their bucket before comparing (see
    parse_percentage_bucket). Same unparseable/scored convention as the
    other *_accuracy functions here."""
    if not predictions:
        raise ValueError("percentage_bucket_accuracy: empty predictions")
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length")
    correct = 0
    unparseable = 0
    scored = 0
    for pred, gt in zip(predictions, ground_truths):
        parsed = parse_percentage_bucket(pred, buckets)
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


def contains_ground_truth_accuracy(predictions: list[str], ground_truths: list[str]) -> dict:
    """Fraction of predictions whose normalized text contains the entire
    normalized ground-truth phrase as a substring. For open-vocabulary
    free-text answers (e.g. VRSBench's object color/position/category —
    no small fixed answer set to build a categorical_accuracy vocabulary
    from), a real, correct answer is often embedded in a longer sentence
    ("The vehicles appear yellow in color" contains "yellow"), so plain
    exact_match_accuracy would unfairly mark it wrong. Unlike the other
    *_accuracy functions here, this has no 'unparseable' bucket — every
    prediction is scored, since there's always something to check a
    substring against (an empty ground truth would be a data problem, not
    a parsing one)."""
    if not predictions:
        raise ValueError("contains_ground_truth_accuracy: empty predictions")
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be the same length")
    correct = sum(
        1 for p, g in zip(predictions, ground_truths)
        if normalize_answer(g) in normalize_answer(p)
    )
    return {"accuracy": correct / len(predictions), "n_scored": len(predictions), "n_total": len(predictions)}


_WORD_NUMBERS = {
    "none": 0, "zero": 0, "single": 1, "one": 1, "two": 2, "couple": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def parse_count(text: str) -> Optional[float]:
    """Extract the first number found in free text — as a digit ("3") or
    a spelled-out word ("Three", "Single") — added after a real crash:
    VRSBench's ground truth mixes both forms for the same question type
    ('3' and 'Three' both appear for "object quantity"), and a bare
    regex-only version raised ValueError on the word form instead of
    returning a clean None. Returns None if neither form is present
    rather than guessing 0."""
    match = _NUMBER_RE.search(text)
    if match is not None:
        return float(match.group())
    normalized = normalize_answer(text)
    for word in normalized.split():
        if word in _WORD_NUMBERS:
            return float(_WORD_NUMBERS[word])
    return None


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
