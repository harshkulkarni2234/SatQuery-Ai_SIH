#!/usr/bin/env python3
"""Scores a real VRSBench captioning evaluation run against real
reference captions, using metrics/text_metrics.py's BLEU-n/ROUGE-L —
implemented and unit-tested since Phase B3, never before run against
real labeled captions until this run.

Usage:
    python -m evaluation.results.score_vrsbench_captioning evaluation/results/vrsbench_captioning_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone

from evaluation.metrics.text_metrics import bleu_n, rouge_l


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
    n_errors = sum(1 for r in rows if r.get("error"))
    ok_rows = [r for r in rows if not r.get("error") and (r.get("meta") or {}).get("ground_truth_available")]

    bleu1_scores, bleu4_scores, rouge_l_f1_scores = [], [], []
    for r in ok_rows:
        candidate = r["answer_text"]
        reference = r["meta"]["ground_truth_answer"]
        bleu1_scores.append(bleu_n(candidate, reference, n=1))
        bleu4_scores.append(bleu_n(candidate, reference, n=4))
        rouge_l_f1_scores.append(rouge_l(candidate, reference)["f1"])

    return {
        "benchmark": "VRSBench captioning (real official eval split)",
        "dataset": "https://huggingface.co/datasets/xiang709/VRSBench (CC-BY-4.0)",
        "model": "vqa.smolvlm_base (description-style prompt, same as C9 demo scenario A)",
        "n_total_rows": len(rows),
        "n_errors": n_errors,
        "n_scored": len(ok_rows),
        "bleu1_mean": statistics.mean(bleu1_scores) if bleu1_scores else None,
        "bleu4_mean": statistics.mean(bleu4_scores) if bleu4_scores else None,
        "rouge_l_f1_mean": statistics.mean(rouge_l_f1_scores) if rouge_l_f1_scores else None,
        "date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", default="evaluation/results/vrsbench_captioning_score.json")
    args = parser.parse_args()
    result = score(args.input)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}")
    print(json.dumps({k: v for k, v in result.items() if k != "dataset"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
