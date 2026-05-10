import argparse
import json
import os
from itertools import chain

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="spacemanidol/cc-stories",
                   help="CC-Stories (Trinh & Le, 2018) dataset. Recommended: spacemanidol/cc-stories")
    p.add_argument("--dataset_config", type=str, default=None, help="Optional HF config name (usually None).")

    p.add_argument("--out_dir", type=str, required=True, help="Where to save prepared DatasetDict.")
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer")
    p.add_argument("--block_size", type=int, default=256)

    # Build validation and test splits from train
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--split_method", type=str, default="random", choices=["random", "contiguous"])
    p.add_argument("--seed", type=int, default=123)

    # Speed knobs for map()
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--map_batch_size", type=int, default=1000)

    # Optional: cap rows for debugging (applies to full corpus before splitting)
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--max_test_rows", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isdir(args.tokenizer_dir):
        print(f"Tokenizer dir '{args.tokenizer_dir}' not found, downloading from 'gpt2'...")
        _tok = AutoTokenizer.from_pretrained("gpt2")
        _tok.save_pretrained(args.tokenizer_dir)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.pad_token = tok.eos_token  # keep consistent with your other pipelines
    block_size = args.block_size

    # Load CC-Stories
    # The HF repo uses a legacy loading script that newer `datasets` versions
    # reject, so we download the raw text file directly and load it as a
    # plain-text dataset.
    print(f"Downloading cc-stories.txt from {args.dataset_name} ...")
    txt_path = hf_hub_download(
        repo_id=args.dataset_name,
        filename="cc-stories.txt",
        repo_type="dataset",
    )
    all_data = load_dataset("text", data_files={"train": txt_path}, split="train")

    # Optional caps for quick dry runs
    if args.max_train_rows and args.max_train_rows > 0:
        all_data = all_data.select(range(min(args.max_train_rows, len(all_data))))

    # Create validation and test from train
    n = len(all_data)
    val_n = int(round(n * args.val_frac))
    test_n = int(round(n * args.test_frac))
    if val_n <= 0:
        raise ValueError("val_frac too small -> val split would be empty.")
    if test_n <= 0:
        raise ValueError("test_frac too small -> test split would be empty.")

    if args.split_method == "random":
        # First carve off test, then carve val from the remainder
        tmp = all_data.train_test_split(test_size=test_n, seed=args.seed, shuffle=True)
        remainder = tmp["train"]
        test_split = tmp["test"]

        tmp2 = remainder.train_test_split(test_size=val_n, seed=args.seed, shuffle=True)
        train_split = tmp2["train"]
        val_split = tmp2["test"]
    else:
        # contiguous: last test_n rows are test, preceding val_n rows are validation
        train_end = n - val_n - test_n
        train_split = all_data.select(range(0, train_end))
        val_split = all_data.select(range(train_end, train_end + val_n))
        test_split = all_data.select(range(train_end + val_n, n))

    # Optional cap on test rows
    if args.max_test_rows and args.max_test_rows > 0:
        test_split = test_split.select(range(min(args.max_test_rows, len(test_split))))

    # Tokenize + concat+chunk
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
        "train": prep(train_split, "train"),
        "validation": prep(val_split, "validation"),
        "test": prep(test_split, "test"),
    })

    out.save_to_disk(args.out_dir)

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
        "block_size": args.block_size,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "split_method": args.split_method,
        "seed": args.seed,
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "train_blocks": len(out["train"]),
        "val_blocks": len(out["validation"]),
        "test_blocks": len(out["test"]),
    }
    with open(os.path.join(args.out_dir, "prepared_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Saved prepared Stories dataset to:", args.out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()