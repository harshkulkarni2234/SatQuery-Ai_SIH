"""Check whether the change model's training images overlap the CDVQA test set.

Why this exists: CDVQA's 968 test pairs are a subset of SECOND's 2,968 public
pairs, and the change model was trained on 2,000 of those 2,968 (1,700 train /
300 val). Unless the data-preparation step excluded the CDVQA test pairs, a
large share of the CDVQA evaluation images were seen in training, and the
CDVQA scores of the learned model are then contaminated.

Needs the two committed names files and CDVQA's Test_images.json (see
evaluation/cdvqa/README.md), nothing else:

    python ml/change_model/check_cdvqa_leakage.py \
        --train-names ml/change_model/trained/data/train_names.json \
        --val-names   ml/change_model/trained/data/val_names.json

Exit code 0 = no overlap, 1 = overlap found, 2 = bad input. Result of the run
on the committed split: 0 of 968 CDVQA test pairs appear in the 1,700 train or
300 val images (see MODEL_CARD.md).
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


DEFAULT_TEST_IMAGES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation", "cdvqa", "raw", "Test_images.json"
)


def load_test_pairs(path: str) -> set[str]:
    """The CDVQA test pair ids (SECOND file stems) from Test_images.json."""
    with open(path) as f:
        data = json.load(f)
    return {_stem(i["file_name"]) for i in data["images"] if i.get("active")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-names", required=True)
    ap.add_argument("--val-names", required=True)
    ap.add_argument("--test-images", default=DEFAULT_TEST_IMAGES,
                    help="CDVQA Test_images.json (see evaluation/cdvqa/README.md)")
    ap.add_argument("--per-type", type=int, default=0,
                    help="also report the overlap for the pairs a seeded CDVQA eval sample used "
                         "(15 = the committed v2 run); needs the full CDVQA Test_* JSONs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="Test")
    args = ap.parse_args(argv)

    try:
        train = load_names(args.train_names)
        val = load_names(args.val_names)
        test_pairs = load_test_pairs(args.test_images)
    except (OSError, ValueError, KeyError) as exc:
        print(f"could not read inputs: {exc}", file=sys.stderr)
        return 2

    eval_pairs = None
    if args.per_type:
        try:
            from evaluation.runners import cdvqa_adapter as cd
            rows = cd.load_real_test_set(args.split)
            sample = cd.stratified_sample(rows, per_type=args.per_type, seed=args.seed)
            eval_pairs = {_stem(r["file_name"]) for r in sample}
        except Exception as exc:  # missing raw JSONs, or not run from the repo root
            print(f"could not build the eval sample (run from the repo root with the full "
                  f"CDVQA Test_* JSONs): {exc}", file=sys.stderr)
            return 2

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
