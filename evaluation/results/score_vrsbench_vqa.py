#!/usr/bin/env python3
"""Scores a real VRSBench VQA evaluation run against its real ground
truth. Question-type -> metric mapping (see runners/vrsbench_vqa_adapter.py):
  - object quantity -> count_accuracy_rmse (numeric)
  - object existence -> yes_no_accuracy
  - everything else (10 more types) -> contains_ground_truth_accuracy,
    since VRSBench's open-vocabulary answers (colors, positions, object
    categories, shapes, scene types, ...) have no small fixed set to
    build a categorical vocabulary from.

Usage:
    python -m evaluation.results.score_vrsbench_vqa evaluation/results/vrsbench_vqa_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from evaluation.metrics.vqa_metrics import (
    contains_ground_truth_accuracy,
    count_accuracy_rmse,
    yes_no_accuracy,
    parse_yes_no,
)
from evaluation.runners.vrsbench_vqa_adapter import NUMERIC_TYPES, YES_NO_TYPES


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(path: str) -> dict:
    rows = _load(path)
    by_type = defaultdict(list)
    n_errors = 0
    for r in rows:
        if r.get("error"):
            n_errors += 1
            continue
        meta = r.get("meta") or {}
        if not meta.get("ground_truth_available"):
            continue
        by_type[meta["question_type"]].append(r)

    results = {}
    for qtype, items in by_type.items():
        preds = [r["answer_text"] for r in items]
        gts = [r["meta"]["ground_truth_answer"] for r in items]
        if qtype in NUMERIC_TYPES:
            results[qtype] = {
                "n": len(items), "metric": "count_accuracy_rmse",
                **count_accuracy_rmse(preds, [float(g) for g in gts]),
            }
        elif qtype in YES_NO_TYPES:
            gt_bools = [parse_yes_no(g) for g in gts]
            n_gt_unparseable = sum(1 for g in gt_bools if g is None)
            paired = [(p, g) for p, g in zip(preds, gt_bools) if g is not None]
            metric_result = yes_no_accuracy([p for p, _ in paired], [g for _, g in paired]) if paired else None
            results[qtype] = {
                "n": len(items), "metric": "yes_no_accuracy",
                "n_ground_truth_unparseable": n_gt_unparseable, **(metric_result or {}),
            }
        else:
            results[qtype] = {
                "n": len(items), "metric": "contains_ground_truth_accuracy (open vocabulary)",
                **contains_ground_truth_accuracy(preds, gts),
            }

    overall_scored = sum(v.get("n_scored", v.get("n", 0)) for v in results.values())
    overall_correct = sum(
        round((v.get("accuracy") or 0) * v.get("n_scored", v.get("n", 0)))
        for v in results.values()
    )

    return {
        "benchmark": "VRSBench VQA (real official eval split)",
        "dataset": "https://huggingface.co/datasets/xiang709/VRSBench (CC-BY-4.0)",
        "model": "vqa.smolvlm_base / vqa.smolvlm_bigearthnet_lora_stage3 (whichever the registry selects)",
        "n_total_rows": len(rows),
        "n_errors": n_errors,
        "by_question_type": results,
        "approx_overall_accuracy": (overall_correct / overall_scored) if overall_scored else None,
        "date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", default="evaluation/results/vrsbench_vqa_score.json")
    args = parser.parse_args()
    result = score(args.input)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps(result["by_question_type"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
