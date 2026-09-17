import pytest

from evaluation.metrics.vqa_metrics import (
    categorical_accuracy,
    count_accuracy_rmse,
    exact_match_accuracy,
    normalize_answer,
    parse_categorical,
    parse_count,
    parse_yes_no,
    yes_no_accuracy,
)


def test_normalize_answer_strips_punctuation_and_case():
    assert normalize_answer("Yes!") == "yes"
    assert normalize_answer("  Water  Body  ") == "water body"


def test_exact_match_accuracy_hand_computed():
    # matches: "Yes."~"yes" (1), "no"=="no" (2), "water"!="vegetation" (miss),
    # "building"=="building" (3) -> 3 of 4 = 0.75
    preds = ["Yes.", "no", "water", "building"]
    gts = ["yes", "no", "vegetation", "building"]
    assert exact_match_accuracy(preds, gts) == 0.75


def test_exact_match_accuracy_empty_raises():
    with pytest.raises(ValueError):
        exact_match_accuracy([], [])


def test_exact_match_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        exact_match_accuracy(["a"], ["a", "b"])


def test_parse_yes_no():
    assert parse_yes_no("Yes, there is water.") is True
    assert parse_yes_no("No water visible.") is False
    assert parse_yes_no("Maybe, hard to tell.") is None
    assert parse_yes_no("yes and no") is None  # ambiguous, both present


def test_yes_no_accuracy_hand_computed():
    preds = ["Yes there is water", "No vegetation here", "unclear image"]
    gts = [True, False, True]
    result = yes_no_accuracy(preds, gts)
    assert result["n_total"] == 3
    assert result["n_unparseable"] == 1
    assert result["n_scored"] == 2
    assert result["accuracy"] == 1.0  # both parseable ones are correct


def test_parse_count():
    assert parse_count("There are 3 buildings") == 3.0
    assert parse_count("no number here") is None
    assert parse_count("-2.5 change") == -2.5


def test_parse_categorical_free_text():
    assert parse_categorical("It is an urban area.", ["urban", "rural"]) == "urban"
    assert parse_categorical("This looks rural to me", ["urban", "rural"]) == "rural"
    assert parse_categorical("I cannot tell", ["urban", "rural"]) is None
    # both present -> ambiguous, not a guess
    assert parse_categorical("urban or rural, hard to say", ["urban", "rural"]) is None


def test_categorical_accuracy_hand_computed():
    # real bug this was written to catch: exact_match_accuracy would score
    # "It is an urban area." against ground truth "urban" as WRONG (no
    # literal string equality) even though it's a correct free-text answer
    preds = ["It is an urban area.", "This is a rural region.", "unclear image"]
    gts = ["urban", "urban", "rural"]
    result = categorical_accuracy(preds, gts, ["urban", "rural"])
    assert result["n_total"] == 3
    assert result["n_unparseable"] == 1
    assert result["n_scored"] == 2
    assert result["accuracy"] == 0.5  # 1st correct (urban==urban), 2nd wrong (rural!=urban)


def test_count_accuracy_rmse_hand_computed():
    # predictions: 3, 5, unparseable; ground truth: 3, 4, 10
    preds = ["3 buildings", "about 5", "several"]
    gts = [3.0, 4.0, 10.0]
    result = count_accuracy_rmse(preds, gts)
    assert result["n_scored"] == 2
    assert result["n_unparseable"] == 1
    # errors: (3-3)^2=0, (5-4)^2=1 -> mean=0.5 -> rmse=sqrt(0.5)
    assert result["rmse"] == pytest.approx(0.5 ** 0.5)
    assert result["exact_accuracy"] == 0.5  # only the first was exact
