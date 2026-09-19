#!/usr/bin/env python3
"""Scores a real RSVQA-LR evaluation run against its real ground truth.

Unlike results/generate_summary.py (which deliberately refuses to report
accuracy when no ground truth exists — see testing_folder_adapter.py),
this script's whole point is that ground truth DOES exist here, so a real
accuracy number is finally reported — per question type, using the exact
metrics/ functions built in Phase B3 and unit-tested there, applied to
real labeled data for the first time in this build.

Usage:
    python -m evaluation.results.score_rsvqa_lr evaluation/results/rsvqa_lr_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from evaluation.metrics.vqa_metrics import (
    categorical_accuracy,
    count_accuracy_rmse,
    parse_yes_no,
    yes_no_accuracy,
)


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(path: str, model: str = "SmolVLM-256M-Instruct (base, no LoRA — see note)", note: str = "") -> dict:
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

    if "rural_urban" in by_type:
        # Phase B (real eval): categorical_accuracy, not exact_match_accuracy —
        # a real free-text answer like "It is an urban area." must count as
        # correct against ground truth "urban"; literal string equality
        # scored every correct free-text answer as wrong (real bug, fixed
        # here after inspecting raw predictions and finding it, not assumed).
        items = by_type["rural_urban"]
        preds = [r["answer_text"] for r in items]
        gts = [r["meta"]["ground_truth_answer"] for r in items]
        results["rural_urban"] = {
            "n": len(items),
            "metric": "categorical_accuracy (urban/rural keyword match in free text)",
            **categorical_accuracy(preds, gts, ["urban", "rural"]),
        }

    for qtype in ("presence", "comp"):
        if qtype in by_type:
            items = by_type[qtype]
            preds = [r["answer_text"] for r in items]
            gts = [parse_yes_no(r["meta"]["ground_truth_answer"]) for r in items]
            n_gt_unparseable = sum(1 for g in gts if g is None)
            paired = [(p, g) for p, g in zip(preds, gts) if g is not None]
            metric_result = yes_no_accuracy(
                [p for p, _ in paired], [g for _, g in paired]
            ) if paired else None
            results[qtype] = {
                "n": len(items),
                "n_ground_truth_unparseable": n_gt_unparseable,
                "metric": "yes_no_accuracy",
                **(metric_result or {}),
            }

    if "count" in by_type:
        items = by_type["count"]
        preds = [r["answer_text"] for r in items]
        gts = [float(r["meta"]["ground_truth_answer"]) for r in items]
        results["count"] = {
            "n": len(items),
            "metric": "count_accuracy_rmse",
            **count_accuracy_rmse(preds, gts),
        }

    overall_scored = sum(
        v.get("n_scored", v.get("n", 0)) for v in results.values()
    )
    overall_correct = sum(
        round((v.get("accuracy") or 0) * v.get("n_scored", v.get("n", 0)))
        if "accuracy" in v and v.get("accuracy") is not None
        else round((v.get("exact_accuracy") or 0) * v.get("n_scored", 0))
        for v in results.values()
    )

    return {
        "benchmark": "RSVQA-LR (real official test split)",
        "dataset": "https://zenodo.org/records/6344334 (CC-BY-4.0)",
        "model": model,
        "note": note or "Base model only, NOT fine-tuned on this dataset (that's what "
                       "the RTX 2050 training session is for). This is a real "
                       "pre-training baseline, not the model's ceiling.",
        "n_total_rows": len(rows),
        "n_errors": n_errors,
        "by_question_type": results,
        "approx_overall_accuracy": (overall_correct / overall_scored) if overall_scored else None,
        "date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to a run_rsvqa_lr_eval.py JSONL output")
    parser.add_argument("--output", default="evaluation/results/rsvqa_lr_score.json")
    parser.add_argument("--model", default=None, help="label written into the result; "
                        "defaults to the base-model baseline label")
    parser.add_argument("--note", default=None, help="note written into the result")
    args = parser.parse_args()

    result = score(args.input, model=args.model or "SmolVLM-256M-Instruct (base, no LoRA — see note)", note=args.note or "")
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps(result["by_question_type"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
