import pytest

from evaluation.metrics.vqa_metrics import (
    categorical_accuracy,
    contains_ground_truth_accuracy,
    count_accuracy_rmse,
    exact_match_accuracy,
    normalize_answer,
    parse_categorical,
    parse_count,
    parse_percentage_bucket,
    parse_yes_no,
    percentage_bucket_accuracy,
    yes_no_accuracy,
)

_PCT_BUCKETS = [
    "0", "0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50",
    "50_to_60", "60_to_70", "70_to_80", "80_to_90", "90_to_100",
]


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


def test_parse_count_word_form():
    # real crash this catches: VRSBench ground truth mixes '3' and 'Three'
    # for the same question type — a digit-only parser raised ValueError
    # (via a bare float()) instead of handling the word form
    assert parse_count("Three") == 3.0
    assert parse_count("Single") == 1.0
    assert parse_count("There are three vehicles visible.") == 3.0
    assert parse_count("no count word here") is None


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


def test_parse_percentage_bucket_extracts_real_percentage():
    # real bug this catches: "approximately 5.6% ... changed" must resolve
    # to bucket "0_to_10", not be marked unparseable just because the
    # literal substring "0_to_10" never appears in the text
    assert parse_percentage_bucket("approximately 5.6% of the frame changed", _PCT_BUCKETS) == "0_to_10"
    assert parse_percentage_bucket("about 95 percent changed", _PCT_BUCKETS) == "90_to_100"
    assert parse_percentage_bucket("0% changed", _PCT_BUCKETS) == "0"
    assert parse_percentage_bucket("no number here at all", _PCT_BUCKETS) is None


def test_parse_percentage_bucket_literal_keyword_still_works():
    assert parse_percentage_bucket("the bucket is 0_to_10", _PCT_BUCKETS) == "0_to_10"


def test_percentage_bucket_accuracy_hand_computed():
    preds = ["approximately 5.6% changed", "about 95 percent changed", "no idea"]
    gts = ["0_to_10", "0_to_10", "50_to_60"]
    result = percentage_bucket_accuracy(preds, gts, _PCT_BUCKETS)
    assert result["n_total"] == 3
    assert result["n_unparseable"] == 1
    assert result["n_scored"] == 2
    assert result["accuracy"] == 0.5  # 1st correct (0_to_10==0_to_10), 2nd wrong (90_to_100!=0_to_10)


def test_contains_ground_truth_accuracy_hand_computed():
    # real motivating case: a correct answer embedded in a full sentence
    # must not be scored wrong just because it's not a literal equality
    preds = ["The vehicles appear yellow in color.", "It is white.", "unrelated answer"]
    gts = ["Yellow", "white", "red"]
    result = contains_ground_truth_accuracy(preds, gts)
    assert result["n_total"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)


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
