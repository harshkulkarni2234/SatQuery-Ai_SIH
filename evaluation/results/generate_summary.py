#!/usr/bin/env python3
"""Generates results/SUMMARY.md from one or more run JSONL files.

Reports only real, computed statistics from the JSONL rows — task
distribution, specialist distribution, error rate, latency percentiles,
confidence availability, warning frequency. Never writes a number that
didn't come from the results file (see AGENTS.md rule 1) — if a run has
no labeled ground truth (see runners/testing_folder_adapter.py), no
accuracy row is emitted at all, rather than a fabricated or placeholder
one.

Usage:
    python -m evaluation.results.generate_summary evaluation/results/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(path: str) -> str:
    rows = _load(path)
    n = len(rows)
    if n == 0:
        return f"## {path}\n\n_No rows._\n"

    errors = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    task_counts = Counter(r.get("task_classified") for r in ok)
    specialist_counts = Counter(r.get("specialist_id") for r in ok)
    fallback_count = sum(1 for r in ok if r.get("used_fallback"))
    confidence_available = sum(1 for r in ok if r.get("confidence_score") is not None)
    warnings_present = sum(1 for r in ok if r.get("warnings"))
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]

    has_ground_truth = any(
        isinstance(r.get("meta"), dict) and r["meta"].get("ground_truth_available")
        for r in rows
    )

    lines = [f"## {path}", ""]
    lines.append(f"- Total samples: **{n}**")
    lines.append(f"- Succeeded: **{len(ok)}** / Errored: **{len(errors)}**")
    if errors:
        error_types = Counter(e.get("error", "").split(":")[0] for e in errors)
        lines.append(f"  - Error types: {dict(error_types)}")
    lines.append(f"- Task distribution (of succeeded): {dict(task_counts)}")
    lines.append(f"- Specialist distribution (of succeeded): {dict(specialist_counts)}")
    lines.append(f"- Used fallback: **{fallback_count}** / {len(ok)}")
    lines.append(f"- Confidence available: **{confidence_available}** / {len(ok)}")
    lines.append(f"- Samples with warnings: **{warnings_present}** / {len(ok)}")
    if latencies:
        sorted_lat = sorted(latencies)
        p50 = statistics.median(sorted_lat)
        p95 = sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))]
        lines.append(f"- Latency ms: min={min(sorted_lat)} p50={p50:.0f} p95={p95} max={max(sorted_lat)}")
    lines.append("")
    if has_ground_truth:
        lines.append(
            "- **Accuracy metrics: not computed by this generator** — rows "
            "claim ground_truth_available but no scoring was wired up for "
            "this run; add it before trusting this summary for accuracy."
        )
    else:
        lines.append(
            "- **No accuracy/exact-match metrics reported** — this run's "
            "samples carry no ground-truth labels (see the adapter's "
            "docstring for why), so only real, measured system-behavior "
            "statistics above are reported. Do not infer an accuracy "
            "number from this run."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="one or more results .jsonl files")
    parser.add_argument("--output", default="evaluation/results/SUMMARY.md")
    args = parser.parse_args()

    generated_at = datetime.now(timezone.utc).isoformat()
    sections = [f"# Evaluation Results Summary\n\nGenerated: {generated_at}\n"]
    for path in args.paths:
        sections.append(summarize(path))

    with open(args.output, "w") as f:
        f.write("\n".join(sections))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
