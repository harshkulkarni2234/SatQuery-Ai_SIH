#!/usr/bin/env python3
"""CLI entry point: run a real, labeled CDVQA evaluation against a live
SatQuery backend. Requires evaluation/cdvqa/raw/ (CDVQA labels) and
evaluation/second_dataset/raw/im1,im2/ (SECOND images) — see each
directory's README.md for exact download commands and license notes.

Usage:
    python -m evaluation.runners.run_cdvqa_eval \\
        --api-base-url http://127.0.0.1:8000 \\
        --per-type 30 \\
        --output evaluation/results/cdvqa_eval.jsonl
"""

from __future__ import annotations

import argparse
import sys

from evaluation.runners.base import RunConfig, run_evaluation
from evaluation.runners.cdvqa_adapter import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-type", type=int, default=30, help="stratified sample size per question type")
    parser.add_argument("--split", default="Test", choices=["Test", "Test2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="evaluation/results/cdvqa_eval.jsonl")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    samples = generate_samples(per_type=args.per_type, seed=args.seed, split=args.split)
    config = RunConfig(
        api_base_url=args.api_base_url,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
        request_timeout_s=60.0,
    )
    results = run_evaluation(samples, config)

    n_errors = sum(1 for r in results if r.error)
    print(f"Ran {len(results)} samples this invocation -> {args.output}")
    print(f"  errors this invocation: {n_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
