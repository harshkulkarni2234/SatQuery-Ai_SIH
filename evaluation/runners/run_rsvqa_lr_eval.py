#!/usr/bin/env python3
"""CLI entry point: run a real, labeled RSVQA-LR evaluation against a live
SatQuery backend. Requires the raw dataset files under
evaluation/rsvqa_lr/raw/ (see evaluation/rsvqa_lr/README.md for the exact
official download commands) and a running backend + VQA worker.

Usage:
    python -m evaluation.runners.run_rsvqa_lr_eval \\
        --api-base-url http://127.0.0.1:8000 \\
        --per-type 60 \\
        --output evaluation/results/rsvqa_lr_eval.jsonl
"""

from __future__ import annotations

import argparse
import sys

from evaluation.runners.base import RunConfig, run_evaluation
from evaluation.runners.rsvqa_lr_adapter import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-type", type=int, default=60, help="stratified sample size per question type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="evaluation/results/rsvqa_lr_eval.jsonl")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    samples = generate_samples(per_type=args.per_type, seed=args.seed)
    config = RunConfig(
        api_base_url=args.api_base_url,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
    results = run_evaluation(samples, config)

    n_errors = sum(1 for r in results if r.error)
    print(f"Ran {len(results)} samples this invocation -> {args.output}")
    print(f"  errors this invocation: {n_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
