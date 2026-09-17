#!/usr/bin/env python3
"""CLI entry point: run the real VRSBench captioning evaluation.

Usage:
    python -m evaluation.runners.run_vrsbench_captioning_eval \\
        --n 60 --output evaluation/results/vrsbench_captioning_eval.jsonl
"""

from __future__ import annotations

import argparse
import sys

from evaluation.runners.base import RunConfig, run_evaluation
from evaluation.runners.vrsbench_captioning_adapter import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="evaluation/results/vrsbench_captioning_eval.jsonl")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    samples = generate_samples(n=args.n, seed=args.seed)
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
