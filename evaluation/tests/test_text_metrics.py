import math

import pytest

from evaluation.metrics.text_metrics import bleu_n, rouge_l


def test_bleu_n_identical_sentences_is_one():
    text = "the water body is in the upper left"
    assert bleu_n(text, text, n=4) == pytest.approx(1.0)


def test_bleu_n_empty_candidate_is_zero():
    assert bleu_n("", "the water body", n=4) == 0.0


def test_bleu_n_completely_different_is_zero():
    assert bleu_n("a b c d", "w x y z", n=4) == 0.0


def test_bleu_n_partial_overlap_hand_computed_unigram():
    # candidate (2 tokens) unigram precision is 1.0 (both tokens appear in the
    # 5-token reference), but the shorter candidate incurs a brevity penalty:
    # exp(1 - 5/2) = exp(-1.5) ~= 0.2231
    result = bleu_n("water body", "the water body is large", n=1)
    assert result == pytest.approx(math.exp(-1.5))


def test_rouge_l_identical_sentences_is_one():
    text = "the water body is in the upper left"
    result = rouge_l(text, text)
    assert result["f1"] == pytest.approx(1.0)


def test_rouge_l_empty_candidate_is_zero():
    result = rouge_l("", "the water body")
    assert result["f1"] == 0.0


def test_rouge_l_hand_computed_partial_match():
    # LCS("water is here", "water was here") -> "water" + "here" = 2 tokens
    candidate = "water is here"
    reference = "water was here"
    result = rouge_l(candidate, reference)
    # both have 3 tokens, LCS length 2 -> precision=recall=2/3
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
