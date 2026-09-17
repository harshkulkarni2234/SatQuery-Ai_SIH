#!/usr/bin/env python3
"""Scores a real VRSBench referring/grounding evaluation run.

Reports, in priority order:
  1. task_routing: what fraction of samples were actually classified as
     GROUNDING vs VQA vs other. This is the HEADLINE finding — see
     vrsbench_referring_adapter.py's docstring for why it was expected
     (and confirmed live before this scorer was written) to be at or
     near 0% GROUNDING: SatQuery's grounding specialist supports 5
     land-cover targets; VRSBench asks about 26 DOTA-style object
     classes with zero vocabulary overlap.
  2. iou_when_grounding_ran: real box IoU (metrics/spatial_metrics),
     computed ONLY over whatever subset actually reached GROUNDING (if
     any) — reported separately and clearly labeled so it is never
     confused with "grounding accuracy on VRSBench" as a whole.

Usage:
    python -m evaluation.results.score_vrsbench_referring evaluation/results/vrsbench_referring_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone

from evaluation.metrics.spatial_metrics import box_iou
from evaluation.runners.vrsbench_referring_adapter import _real_image_size


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalized_to_pixels(box_0_100: tuple, width: int, height: int) -> tuple:
    x1, y1, x2, y2 = box_0_100
    return (x1 / 100 * width, y1 / 100 * height, x2 / 100 * width, y2 / 100 * height)


def score(path: str) -> dict:
    rows = _load(path)
    n_errors = sum(1 for r in rows if r.get("error"))
    ok_rows = [r for r in rows if not r.get("error")]

    task_routing = Counter(r.get("task_classified") for r in ok_rows)

    grounding_rows = [r for r in ok_rows if r.get("task_classified") == "GROUNDING"]
    ious = []
    for r in grounding_rows:
        meta = r.get("meta") or {}
        boxes = r.get("bounding_boxes")
        if not boxes:
            continue
        try:
            width, height = _real_image_size(meta["image_path"])
        except Exception:
            continue
        gt_box = _normalized_to_pixels(tuple(meta["ground_truth_box_normalized_0_100"]), width, height)
        best_iou = max(box_iou(tuple(pred), gt_box) for pred in boxes)
        ious.append(best_iou)

    return {
        "benchmark": "VRSBench referring/grounding (real official eval split)",
        "dataset": "https://huggingface.co/datasets/xiang709/VRSBench (CC-BY-4.0)",
        "model": "grounding.deterministic_cv (5 land-cover targets only — see note)",
        "note": "SatQuery's grounding specialist supports water/vegetation/built-up/roads/"
                "farmland only. VRSBench's referring expressions ask about individual "
                "objects (vehicle, ship, airplane, harbor, ...) with zero vocabulary "
                "overlap. task_routing below is the real, measured finding: how often "
                "the query even reached GROUNDING at all (expected near-zero). iou_stats "
                "covers ONLY that subset, if any, and must not be read as 'VRSBench "
                "grounding accuracy' overall.",
        "n_total_rows": len(rows),
        "n_errors": n_errors,
        "task_routing": dict(task_routing),
        "n_reached_grounding": len(grounding_rows),
        "iou_stats": {
            "n": len(ious),
            "mean_iou": (sum(ious) / len(ious)) if ious else None,
        },
        "date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", default="evaluation/results/vrsbench_referring_score.json")
    args = parser.parse_args()
    result = score(args.input)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps({k: v for k, v in result.items() if k != "dataset"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
