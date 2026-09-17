"""Text-similarity metrics for captions / change-description text.

Deliberately lightweight, pure-Python implementations of BLEU-n and
ROUGE-L, NOT the `pycocoevalcap` or `evaluate` libraries the plan
mentions as options — pycocoevalcap's METEOR/CIDEr scorers require a
Java runtime, and `evaluate` pulls in a heavy dependency chain, neither
of which was worth adding to `evaluation/requirements.txt` for a build
that (see evaluation/README.md) never had labeled caption ground truth
to score against in the first place. This module exists so the metric
*is* implemented and unit-tested against hand-computed examples per the
plan's own requirement, ready to use the moment real captioning ground
truth exists. Do not describe these as official BLEU/ROUGE reference
implementations — they use simple whitespace tokenization and no
smoothing beyond what's noted per function.
"""

from __future__ import annotations

import math
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return text.strip().lower().split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(candidate: str, reference: str, n: int = 4) -> float:
    """Modified n-gram precision (BLEU-n, single reference) with a brevity
    penalty. Returns 0.0 if the candidate has no matching n-grams or is
    empty — never fabricates a nonzero score for empty input."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens:
        return 0.0

    precisions = []
    for k in range(1, n + 1):
        cand_ngrams = _ngrams(cand_tokens, k)
        ref_ngrams = _ngrams(ref_tokens, k)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(overlap / total if total else 0.0)

    if any(p == 0.0 for p in precisions):
        geo_mean = 0.0
    else:
        product = 1.0
        for p in precisions:
            product *= p
        geo_mean = product ** (1.0 / n)

    brevity_penalty = 1.0 if len(cand_tokens) >= len(ref_tokens) else (
        math.exp(1 - len(ref_tokens) / len(cand_tokens)) if cand_tokens else 0.0
    )
    return brevity_penalty * geo_mean


def _lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(candidate: str, reference: str) -> dict:
    """ROUGE-L (longest common subsequence) precision/recall/F1."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
