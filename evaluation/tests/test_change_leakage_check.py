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


def test_main_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(cdvqa_adapter, "load_real_test_set", lambda split="Test": _rows(["10", "11", "12"]))
    train = _write(tmp_path, "train.json", ["1", "2"])
    val = _write(tmp_path, "val.json", ["3"])
    assert leak.main(["--train-names", train, "--val-names", val, "--per-type", "0"]) == 0

    train_bad = _write(tmp_path, "train_bad.json", ["1", "11"])
    assert leak.main(["--train-names", train_bad, "--val-names", val, "--per-type", "3"]) == 1

    assert leak.main(["--train-names", str(tmp_path / "missing.json"), "--val-names", val]) == 2
