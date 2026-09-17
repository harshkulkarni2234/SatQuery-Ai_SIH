#!/usr/bin/env python3
"""Phase B2 item 3: "re-run evaluation of the existing adapter... report
real numbers, including where the adapter is worse."

Honesty constraint that shapes this whole script: there is no labeled
held-out BigEarthNet VQA split anywhere in this repo or in testing/ (see
evaluation/README.md and ml/adaptation/MODEL_CARD.md for the full
explanation — testing/ has zero label files of any kind). So this script
cannot compute per-question-type ACCURACY the way the original plan
describes ("evaluates BASE vs ADAPTER... per question type
(accuracy/exact-match)"). What it CAN do, and does, honestly:

  - Run the SAME real images through BOTH the base model and the
    specialist (LoRA) adapter, for the same prompts, and report whether
    their answers agree (exact normalized match) or diverge — this is a
    real, measured, reproducible statistic, just not "accuracy".
  - Report each mode's real latency.
  - Report whether the specialist adapter loads and runs without error at
    all (this was previously UNVERIFIED in this build until Phase B2's
    GPU re-check — see docs/SOLO_PROGRESS.md).

Requires the ml/vqa-worker virtual environment (torch/transformers/peft) —
run with: ml/vqa-worker/venv/bin/python ml/adaptation/eval_lora.py

Usage:
    ml/vqa-worker/venv/bin/python ml/adaptation/eval_lora.py \
        --images path/to/img1.png path/to/img2.png \
        --output ml/adaptation/results/eval_run.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

VQA_WORKER_DIR = os.path.join(os.path.dirname(__file__), "..", "vqa-worker")
sys.path.insert(0, VQA_WORKER_DIR)

from model_provider import SmolVLMProvider  # noqa: E402


# Presence/count-style prompts (the only categories the specialist is
# gated to handle — see model_provider.should_use_specialist) plus a
# couple of description prompts to show the specialist deliberately isn't
# used for those.
PROMPTS = [
    ("Is there water present in the image?", "presence"),
    ("Is there vegetation in the image?", "presence"),
    ("How many buildings are visible?", "count"),
    ("What can you tell me about this image?", "description"),
]


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return "unknown"


def _dataset_hash(image_paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(image_paths):
        h.update(p.encode())
        if os.path.isfile(p):
            h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


def run_eval(image_paths: list[str], adapter_dir: str, model_dir: str) -> dict:
    provider = SmolVLMProvider(model_dir=model_dir, adapter_dir=adapter_dir)
    provider.load()

    if not provider.specialist_available:
        return {
            "error": "Specialist adapter failed to load",
            "specialist_errors": provider.specialist_errors,
        }

    rows = []
    for image_path in image_paths:
        for prompt, question_type in PROMPTS:
            base_result = provider.answer_question(image_path, prompt, use_specialist=False)
            specialist_result = provider.answer_question(image_path, prompt, use_specialist=True)
            agree = normalize(base_result["answer_text"]) == normalize(specialist_result["answer_text"])
            rows.append({
                "image": os.path.basename(image_path),
                "prompt": prompt,
                "question_type": question_type,
                "base_answer": base_result["answer_text"],
                "base_latency_ms": base_result["execution_time_ms"],
                "specialist_answer": specialist_result["answer_text"],
                "specialist_ran": specialist_result["specialist"],
                "specialist_latency_ms": specialist_result["execution_time_ms"],
                "agree_exact_normalized": agree,
            })

    n = len(rows)
    n_specialist_ran = sum(1 for r in rows if r["specialist_ran"])
    n_agree = sum(1 for r in rows if r["agree_exact_normalized"])
    by_type = {}
    for qtype in set(r["question_type"] for r in rows):
        type_rows = [r for r in rows if r["question_type"] == qtype]
        by_type[qtype] = {
            "n": len(type_rows),
            "specialist_ran": sum(1 for r in type_rows if r["specialist_ran"]),
            "agree_exact_normalized": sum(1 for r in type_rows if r["agree_exact_normalized"]),
        }

    return {
        "benchmark": "none (no labeled held-out split available — see module docstring)",
        "split": "n/a",
        "n": n,
        "metrics": {
            "note": "NOT accuracy — no ground truth exists. These are real, measured "
                    "base-vs-specialist agreement/activation statistics only.",
            "specialist_activation_rate": n_specialist_ran / n if n else None,
            "exact_agreement_rate": n_agree / n if n else None,
            "by_question_type": by_type,
        },
        "config": "ml/adaptation/configs/stage3.yaml",
        "git_commit": _git_commit(),
        "model_versions": {
            "base": "SmolVLM-256M-Instruct",
            "specialist": "smolvlm256m-ben-lora-s3-v1.0",
        },
        "dataset_hash": _dataset_hash(image_paths),
        "date": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", required=True, help="real image file paths to evaluate")
    parser.add_argument("--adapter-dir", default=os.path.join(VQA_WORKER_DIR, "..", "smolvlm", "lora_stage3"))
    parser.add_argument("--model-dir", default=None, help="defaults to MODEL_DIR env or the HF repo id")
    parser.add_argument("--output", default="ml/adaptation/results/eval_run.json")
    args = parser.parse_args()

    result = run_eval(args.images, args.adapter_dir, args.model_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    if "error" in result:
        print(f"FAILED: {result['error']}")
        return 1

    print(f"Wrote {args.output}")
    print(f"  n={result['n']}, specialist_activation_rate={result['metrics']['specialist_activation_rate']:.2f}, "
          f"exact_agreement_rate={result['metrics']['exact_agreement_rate']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
