#!/usr/bin/env python3
"""CLI entry point: run the unlabeled testing/-folder smoke evaluation
against a live SatQuery backend.

Usage:
    python -m evaluation.runners.run_testing_folder_eval \\
        --api-base-url http://127.0.0.1:8000 \\
        --n-images 100 \\
        --limit 200 \\
        --output evaluation/results/testing_folder_smoke.jsonl

Requires a running backend (see docs/DEMO_RUNBOOK.md / scripts/start_all.sh).
"""

from __future__ import annotations

import argparse
import sys

from evaluation.runners.base import RunConfig, run_evaluation
from evaluation.runners.testing_folder_adapter import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n-images", type=int, default=100, help="how many real images to sample from testing/")
    parser.add_argument("--seed", type=int, default=42, help="seed for the deterministic sample")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many (image,prompt) samples")
    parser.add_argument("--output", default="evaluation/results/testing_folder_smoke.jsonl")
    parser.add_argument("--no-resume", action="store_true", help="overwrite output instead of resuming")
    args = parser.parse_args()

    samples = generate_samples(n_images=args.n_images, seed=args.seed)
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
