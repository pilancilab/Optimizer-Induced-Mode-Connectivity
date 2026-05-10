import argparse
import json
import os
from typing import Optional

from datasets import DatasetDict, load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    # HF dataset selection
    p.add_argument("--dataset_name", type=str, default="bookcorpus",
                   help="HF dataset name. Examples: 'bookcorpus', 'rojagtap/bookcorpus', 'lucadiliello/bookcorpusopen'")
    p.add_argument("--dataset_config", type=str, default=None,
                   help="Optional config name for HF dataset (often None).")
    p.add_argument("--split", type=str, default="train",
                   help="Which split to read from the HF dataset (often only 'train').")

    # output + splitting
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to save the DatasetDict (will be created).")
    p.add_argument("--val_frac", type=float, default=0.05,
                   help="Validation fraction (0..1).")
    p.add_argument("--test_frac", type=float, default=0.05,
                   help="Test fraction (0..1).")
    p.add_argument("--method", type=str, default="contiguous",
                   choices=["contiguous", "random"],
                   help="contiguous: take first (1-val-test) as train, then val, then test. "
                        "random: shuffle then split by fractions.")
    p.add_argument("--seed", type=int, default=123,
                   help="Seed for random splitting (only used with --method random).")

    # practical knobs
    p.add_argument("--text_column", type=str, default="text",
                   help="Name of the text column in the dataset (usually 'text').")
    p.add_argument("--max_rows", type=int, default=0,
                   help="Optional cap on number of rows to use (0 = use all). Useful for quick tests.")
    return p.parse_args()


def validate_fracs(val_frac: float, test_frac: float):
    if not (0 <= val_frac < 1) or not (0 <= test_frac < 1) or (val_frac + test_frac >= 1):
        raise ValueError("val_frac and test_frac must be in [0,1) and sum to < 1.")


def main():
    args = parse_args()
    validate_fracs(args.val_frac, args.test_frac)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load HF dataset split
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
    )

    if args.text_column not in ds.column_names:
        raise KeyError(
            f"Column '{args.text_column}' not found. Available columns: {ds.column_names}. "
            "Use --text_column to set the correct field."
        )

    # Optionally cap rows for speed / debugging
    if args.max_rows and args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    n = len(ds)
    test_n = int(round(n * args.test_frac))
    val_n = int(round(n * args.val_frac))
    holdout_n = val_n + test_n
    if holdout_n >= n:
        raise ValueError("Holdout size is >= total rows; reduce val/test fractions or increase data.")

    if args.method == "contiguous":
        train = ds.select(range(0, n - holdout_n))
        val = ds.select(range(n - holdout_n, n - test_n)) if val_n > 0 else ds.select([])
        test = ds.select(range(n - test_n, n)) if test_n > 0 else ds.select([])
    else:
        # random split by rows
        holdout_frac = args.val_frac + args.test_frac
        tmp = ds.train_test_split(test_size=holdout_frac, seed=args.seed, shuffle=True)
        train = tmp["train"]
        holdout = tmp["test"]
        inner_test_size = args.test_frac / holdout_frac if holdout_frac > 0 else 0.0
        inner = holdout.train_test_split(test_size=inner_test_size, seed=args.seed, shuffle=True)
        val, test = inner["train"], inner["test"]

    out = DatasetDict({"train": train, "validation": val, "test": test})
    out.save_to_disk(args.out_dir)

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "source_split": args.split,
        "method": args.method,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed if args.method == "random" else None,
        "text_column": args.text_column,
        "max_rows": args.max_rows,
        "train_rows": len(out["train"]),
        "val_rows": len(out["validation"]),
        "test_rows": len(out["test"]),
    }
    meta_path = os.path.join(args.out_dir, "split_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved DatasetDict to: {args.out_dir}")
    print(f"   Splits: {list(out.keys())}")
    print(f"   Metadata: {meta_path}")


if __name__ == "__main__":
    main()
