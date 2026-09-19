#!/usr/bin/env python3
"""Scores a real CDVQA evaluation run against its real ground truth.

Question types map to metrics as follows (see runners/cdvqa_adapter.py
for the real vocabularies, taken directly from the observed answer sets
in the official test split, not guessed):
  - change_or_not / increase_or_not / decrease_or_not -> yes_no_accuracy
  - smallest_change / largest_change / change_to_what -> categorical_accuracy
    over the 6 real land-cover classes
  - change_ratio / change_ratio_types -> categorical_accuracy over the 11
    real percentage-bucket labels

Usage:
    python -m evaluation.results.score_cdvqa evaluation/results/cdvqa_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from evaluation.metrics.vqa_metrics import (
    categorical_accuracy,
    parse_yes_no,
    percentage_bucket_accuracy,
    yes_no_accuracy,
)
from evaluation.runners.cdvqa_adapter import (
    BUCKET_TYPES,
    CATEGORICAL_TYPES,
    CHANGE_RATIO_BUCKETS,
    LAND_COVER_CLASSES,
    YES_NO_TYPES,
)


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
    # Detect model from eval results
    model_name = "change.deterministic_cv"
    for r in rows:
        if r.get("error"):
            n_errors += 1
            continue
        answer = r.get("answer_text", "")
        if "Learned Siamese CNN" in answer:
            model_name = "change.semantic_model (Siamese CNN, siamese-cnn-v1)"
        meta = r.get("meta") or {}
        if not meta.get("ground_truth_available"):
            continue
        by_type[meta["question_type"]].append(r)

    results = {}
    for qtype, items in by_type.items():
        preds = [r["answer_text"] for r in items]
        gts = [r["meta"]["ground_truth_answer"] for r in items]
        if qtype in YES_NO_TYPES:
            gt_bools = [parse_yes_no(g) for g in gts]
            n_gt_unparseable = sum(1 for g in gt_bools if g is None)
            paired = [(p, g) for p, g in zip(preds, gt_bools) if g is not None]
            metric_result = yes_no_accuracy([p for p, _ in paired], [g for _, g in paired]) if paired else None
            results[qtype] = {
                "n": len(items), "metric": "yes_no_accuracy",
                "n_ground_truth_unparseable": n_gt_unparseable, **(metric_result or {}),
            }
        elif qtype in CATEGORICAL_TYPES:
            results[qtype] = {
                "n": len(items), "metric": "categorical_accuracy (6 land-cover classes)",
                **categorical_accuracy(preds, gts, LAND_COVER_CLASSES),
            }
        elif qtype in BUCKET_TYPES:
            results[qtype] = {
                "n": len(items), "metric": "percentage_bucket_accuracy (11 buckets, extracts a real % from free text)",
                **percentage_bucket_accuracy(preds, gts, CHANGE_RATIO_BUCKETS),
            }

    overall_scored = sum(v.get("n_scored", 0) for v in results.values())
    overall_correct = sum(
        round((v.get("accuracy") or 0) * v.get("n_scored", 0)) for v in results.values()
    )

    return {
        "benchmark": "CDVQA (real official test split, over SECOND dataset images)",
        "dataset": "https://github.com/YZHJessica/CDVQA (Apache-2.0 labels) + "
                   "https://captain-whu.github.io/SCD/ (images, license unstated)",
        "model": model_name,
        "note": "SatQuery's change specialist measures aggregate pixel/area change, not "
                "per-land-cover-class semantic change. Low scores on categorical question "
                "types (smallest/largest/change_to_what) are an expected, honest capability "
                "gap, not a bug — exactly what a real semantic change model (Phase B8) would "
                "need to close.",
        "n_total_rows": len(rows),
        "n_errors": n_errors,
        "by_question_type": results,
        "approx_overall_accuracy": (overall_correct / overall_scored) if overall_scored else None,
        "date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", default="evaluation/results/cdvqa_score.json")
    args = parser.parse_args()

    result = score(args.input)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps(result["by_question_type"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
