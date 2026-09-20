"""ml/change_model/check_cdvqa_leakage.py: overlap logic + exit codes (synthetic data)."""

import importlib.util
import json
import os

import pytest

from evaluation.runners import cdvqa_adapter

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "change_model", "check_cdvqa_leakage.py")
_spec = importlib.util.spec_from_file_location("check_cdvqa_leakage", _PATH)
leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leak)


def _rows(names):
    return [{"file_name": f"{n}.png", "question_type": "change_ratio", "question": "q", "answer": "a",
             "question_id": i} for i, n in enumerate(names)]


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_name_normalisation_accepts_paths_and_extensions(tmp_path):
    p = _write(tmp_path, "n.json", ["a/b/00001.png", "00002", "00003.jpg"])
    assert leak.load_names(p) == {"00001", "00002", "00003"}


def test_report_counts_overlap_and_unseen_eval_pairs():
    r = leak.report({"1", "2"}, {"3"}, {"2", "3", "4", "5"}, {"3", "4"})
    assert r["train_overlap"] == 1 and r["val_overlap"] == 1 and r["any_overlap"] is True
    assert r["eval_pairs_seen_in_training"] == 1 and r["eval_pairs_unseen"] == ["4"]


def _images_json(tmp_path, names, inactive=()):
    items = [{"id": i, "file_name": f"{n}.png", "active": True} for i, n in enumerate(names)]
    items += [{"id": 99, "file_name": f"{n}.png", "active": False} for n in inactive]
    return _write(tmp_path, "Test_images.json", {"images": items})


def test_main_exit_codes(tmp_path):
    test_imgs = _images_json(tmp_path, ["10", "11", "10", "12"], inactive=["1"])  # dupes + inactive ignored
    val = _write(tmp_path, "val.json", ["3"])

    clean = _write(tmp_path, "train.json", ["1", "2"])  # "1" is only an INACTIVE test entry
    assert leak.main(["--train-names", clean, "--val-names", val, "--test-images", test_imgs]) == 0

    dirty = _write(tmp_path, "train_bad.json", ["1", "11"])
    assert leak.main(["--train-names", dirty, "--val-names", val, "--test-images", test_imgs]) == 1

    assert leak.main(["--train-names", str(tmp_path / "missing.json"), "--val-names", val,
                      "--test-images", test_imgs]) == 2
    assert leak.main(["--train-names", clean, "--val-names", val,
                      "--test-images", str(tmp_path / "nope.json")]) == 2


def test_eval_sample_overlap_uses_the_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(cdvqa_adapter, "load_real_test_set", lambda split="Test": _rows(["10", "11", "12"]))
    test_imgs = _images_json(tmp_path, ["10", "11", "12"])
    train = _write(tmp_path, "train.json", ["11"])
    val = _write(tmp_path, "val.json", ["3"])
    assert leak.main(["--train-names", train, "--val-names", val, "--test-images", test_imgs,
                      "--per-type", "3"]) == 1
