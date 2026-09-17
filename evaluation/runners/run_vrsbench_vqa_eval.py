#!/usr/bin/env python3
"""CLI entry point: run a real, labeled VRSBench VQA evaluation against a
live SatQuery backend. Requires evaluation/vrsbench/raw/ (see its
README.md for exact download commands).

Usage:
    python -m evaluation.runners.run_vrsbench_vqa_eval \\
        --per-type 20 --output evaluation/results/vrsbench_vqa_eval.jsonl
"""

from __future__ import annotations

import argparse
import sys

from evaluation.runners.base import RunConfig, run_evaluation
from evaluation.runners.vrsbench_vqa_adapter import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="evaluation/results/vrsbench_vqa_eval.jsonl")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    samples = generate_samples(per_type=args.per_type, seed=args.seed)
    config = RunConfig(
        api_base_url=args.api_base_url, output_path=args.output,
        limit=args.limit, resume=not args.no_resume,
    )
    results = run_evaluation(samples, config)
    n_errors = sum(1 for r in results if r.error)
    print(f"Ran {len(results)} samples this invocation -> {args.output}")
    print(f"  errors this invocation: {n_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
