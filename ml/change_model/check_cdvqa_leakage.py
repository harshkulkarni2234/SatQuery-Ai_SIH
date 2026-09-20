"""Check whether the change model's training images overlap the CDVQA test set.

Why this exists: CDVQA's 968 test pairs are a subset of SECOND's 2,968 public
pairs, and the change model was trained on 2,000 of those 2,968 (1,700 train /
300 val). Unless the data-preparation step excluded the CDVQA test pairs, a
large share of the CDVQA evaluation images were seen in training, and the
CDVQA scores of the learned model are then contaminated.

Run on the machine that has the training data (needs the two names files the
training run wrote, and the CDVQA raw JSONs — see evaluation/cdvqa/README.md):

    python ml/change_model/check_cdvqa_leakage.py \
        --train-names ml/change_model/trained/data/train_names.json \
        --val-names   ml/change_model/trained/data/val_names.json

Exit code 0 = no overlap, 1 = overlap found, 2 = bad input. With --per-type it
also reports the overlap for the pairs a seeded CDVQA eval actually used
(defaults match the committed v2 run: 15 questions/type, seed 42).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def load_names(path: str) -> set[str]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate {"names": [...]}
        data = data.get("names", list(data.values()))
    return {_stem(n) for n in data}


def report(train: set[str], val: set[str], test_pairs: set[str],
           eval_pairs: set[str] | None = None) -> dict:
    train_hit = train & test_pairs
    val_hit = val & test_pairs
    out = {
        "n_train": len(train),
        "n_val": len(val),
        "n_cdvqa_test_pairs": len(test_pairs),
        "train_overlap": len(train_hit),
        "val_overlap": len(val_hit),
        "any_overlap": bool(train_hit or val_hit),
    }
    if eval_pairs is not None:
        seen = (train | val) & eval_pairs
        out["n_eval_pairs"] = len(eval_pairs)
        out["eval_pairs_seen_in_training"] = len(seen)
        out["eval_pairs_unseen"] = sorted(eval_pairs - seen)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-names", required=True)
    ap.add_argument("--val-names", required=True)
    ap.add_argument("--per-type", type=int, default=15,
                    help="questions per type of the CDVQA eval sample to check (0 = skip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="Test")
    args = ap.parse_args(argv)

    try:
        train = load_names(args.train_names)
        val = load_names(args.val_names)
    except (OSError, ValueError) as exc:
        print(f"could not read names files: {exc}", file=sys.stderr)
        return 2

    try:
        from evaluation.runners import cdvqa_adapter as cd
        rows = cd.load_real_test_set(args.split)
    except Exception as exc:  # missing raw JSONs, or not run from the repo root
        print(f"could not load CDVQA {args.split} labels (run from the repo root; "
              f"see evaluation/cdvqa/README.md): {exc}", file=sys.stderr)
        return 2

    test_pairs = {_stem(r["file_name"]) for r in rows}
    eval_pairs = None
    if args.per_type:
        sample = cd.stratified_sample(rows, per_type=args.per_type, seed=args.seed)
        eval_pairs = {_stem(r["file_name"]) for r in sample}

    result = report(train, val, test_pairs, eval_pairs)
    print(json.dumps({k: v for k, v in result.items() if k != "eval_pairs_unseen"}, indent=2))
    if eval_pairs is not None:
        print(f"eval pairs NOT seen in training: {len(result['eval_pairs_unseen'])}")
    if result["any_overlap"]:
        print("OVERLAP FOUND: CDVQA scores for the learned change model are contaminated; "
              "re-run the evaluation only on the unseen pairs (or retrain excluding the CDVQA test pairs).")
        return 1
    print("No overlap between the training/validation images and the CDVQA test pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
