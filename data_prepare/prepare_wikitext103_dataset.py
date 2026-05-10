import argparse
import json
import os
from itertools import chain

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    # WikiText-103 is available on HuggingFace
    p.add_argument("--dataset_name", type=str, default="wikitext",
                   help="HuggingFace dataset name for WikiText")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1",
                   help="Config: 'wikitext-103-raw-v1' (raw) or 'wikitext-103-v1' (tokenized)")

    p.add_argument("--out_dir", type=str, required=True, help="Where to save prepared DatasetDict.")
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer")
    p.add_argument("--block_size", type=int, default=256)

    # Speed knobs for map()
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--map_batch_size", type=int, default=1000)

    # Optional: cap rows for debugging
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--max_test_rows", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Setup tokenizer (same as LM1B)
    if not os.path.isdir(args.tokenizer_dir):
        print(f"Tokenizer dir '{args.tokenizer_dir}' not found, downloading from 'gpt2'...")
        _tok = AutoTokenizer.from_pretrained("gpt2")
        _tok.save_pretrained(args.tokenizer_dir)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.pad_token = tok.eos_token
    block_size = args.block_size

    # Load WikiText-103 (already has train/validation/test splits)
    ds = load_dataset(args.dataset_name, args.dataset_config)

    if "train" not in ds:
        raise ValueError(f"{args.dataset_name} has no 'train' split. Available: {list(ds.keys())}")
    if "validation" not in ds:
        raise ValueError(f"{args.dataset_name} has no 'validation' split. Available: {list(ds.keys())}")
    if "test" not in ds:
        raise ValueError(f"{args.dataset_name} has no 'test' split. Available: {list(ds.keys())}")

    train_raw = ds["train"]
    val_raw = ds["validation"]
    test_raw = ds["test"]

    # Filter out empty lines (WikiText has many empty lines)
    def filter_empty(example):
        return len(example["text"].strip()) > 0

    train_raw = train_raw.filter(filter_empty, num_proc=args.num_proc, desc="filter[train]")
    val_raw = val_raw.filter(filter_empty, num_proc=args.num_proc, desc="filter[validation]")
    test_raw = test_raw.filter(filter_empty, num_proc=args.num_proc, desc="filter[test]")

    # Optional caps for quick dry runs
    if args.max_train_rows and args.max_train_rows > 0:
        train_raw = train_raw.select(range(min(args.max_train_rows, len(train_raw))))
    if args.max_test_rows and args.max_test_rows > 0:
        test_raw = test_raw.select(range(min(args.max_test_rows, len(test_raw))))

    # Tokenize + concat + chunk (same as LM1B)
    def tokenize_fn(examples):
        return tok(examples["text"], return_attention_mask=True)

    def group_texts(examples):
        ids = list(chain(*examples["input_ids"]))
        masks = list(chain(*examples["attention_mask"]))
        chunk_len = (len(ids) // block_size) * block_size
        ids = ids[:chunk_len]
        masks = masks[:chunk_len]
        return {
            "input_ids": [ids[i:i + block_size] for i in range(0, chunk_len, block_size)],
            "attention_mask": [masks[i:i + block_size] for i in range(0, chunk_len, block_size)],
        }

    def prep(split, name):
        t = split.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            num_proc=args.num_proc,
            batch_size=args.map_batch_size,
            desc=f"tokenize[{name}]",
        )
        c = t.map(
            group_texts,
            batched=True,
            num_proc=args.num_proc,
            batch_size=args.map_batch_size,
            desc=f"chunk[{name}]",
        )
        c.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return c

    out = DatasetDict({
        "train": prep(train_raw, "train"),
        "validation": prep(val_raw, "validation"),
        "test": prep(test_raw, "test"),
    })

    out.save_to_disk(args.out_dir)

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
        "block_size": args.block_size,
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "train_lines": len(ds["train"]),
        "val_lines": len(ds["validation"]),
        "test_lines": len(ds["test"]),
        "train_blocks": len(out["train"]),
        "val_blocks": len(out["validation"]),
        "test_blocks": len(out["test"]),
    }
    with open(os.path.join(args.out_dir, "prepared_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Saved prepared WikiText-103 dataset to:", args.out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()