import argparse
import json
import os
import zipfile
from itertools import chain
from pathlib import Path

import requests
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    # enwik8 is downloaded from Hutter Prize website (no HF dataset)
    p.add_argument("--data_url", type=str, 
                   default="http://mattmahoney.net/dc/enwik8.zip",
                   help="URL to download enwik8.zip")
    p.add_argument("--cache_dir", type=str, default="./enwik8_cache",
                   help="Directory to cache downloaded raw data")

    p.add_argument("--out_dir", type=str, required=True, help="Where to save prepared DatasetDict.")
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer")
    p.add_argument("--block_size", type=int, default=256)

    # enwik8 has standard splits: 90M train, 5M val, 5M test
    # val_frac and split_method don't apply here (fixed split)
    p.add_argument("--train_bytes", type=int, default=90_000_000)
    p.add_argument("--val_bytes", type=int, default=5_000_000)
    p.add_argument("--test_bytes", type=int, default=5_000_000)

    # Speed knobs for map()
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--map_batch_size", type=int, default=1000)

    # Optional: cap chars for debugging
    p.add_argument("--max_train_chars", type=int, default=0)
    p.add_argument("--max_test_chars", type=int, default=0)

    return p.parse_args()


def download_enwik8(url: str, cache_dir: str) -> str:
    """Download and extract enwik8 if not already cached. Returns path to raw text file."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = cache_path / "enwik8.zip"
    txt_path = cache_path / "enwik8"
    
    if txt_path.exists():
        print(f"Using cached enwik8 at {txt_path}")
        return str(txt_path)
    
    if not zip_path.exists():
        print(f"Downloading enwik8 from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {zip_path}")
    
    print("Extracting enwik8.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_path)
    
    print(f"Extracted to {txt_path}")
    return str(txt_path)


def load_enwik8_splits(txt_path: str, train_bytes: int, val_bytes: int, test_bytes: int) -> dict:
    """Load enwik8 and split into train/val/test according to standard splits."""
    with open(txt_path, "rb") as f:
        data = f.read()
    
    total_bytes = train_bytes + val_bytes + test_bytes
    if len(data) < total_bytes:
        raise ValueError(f"enwik8 file has {len(data)} bytes, but requested {total_bytes} total")
    
    # Standard enwik8 split: first 90M train, next 5M val, last 5M test
    train_data = data[:train_bytes].decode("utf-8", errors="replace")
    val_data = data[train_bytes:train_bytes + val_bytes].decode("utf-8", errors="replace")
    test_data = data[train_bytes + val_bytes:train_bytes + val_bytes + test_bytes].decode("utf-8", errors="replace")
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }


def text_to_chunks(text: str, chunk_size: int = 50000) -> list:
    """Split text into chunks for creating a Dataset."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append({"text": text[i:i + chunk_size]})
    return chunks


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

    # Download/load enwik8
    txt_path = download_enwik8(args.data_url, args.cache_dir)
    splits = load_enwik8_splits(txt_path, args.train_bytes, args.val_bytes, args.test_bytes)
    
    # Apply optional character limits for debugging
    if args.max_train_chars and args.max_train_chars > 0:
        splits["train"] = splits["train"][:args.max_train_chars]
    if args.max_test_chars and args.max_test_chars > 0:
        splits["test"] = splits["test"][:args.max_test_chars]

    # Convert text splits to Datasets
    train_ds = Dataset.from_list(text_to_chunks(splits["train"]))
    val_ds = Dataset.from_list(text_to_chunks(splits["validation"]))
    test_ds = Dataset.from_list(text_to_chunks(splits["test"]))

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

    def prep(split_ds, name):
        t = split_ds.map(
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
        "train": prep(train_ds, "train"),
        "validation": prep(val_ds, "validation"),
        "test": prep(test_ds, "test"),
    })

    out.save_to_disk(args.out_dir)

    meta = {
        "data_url": args.data_url,
        "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
        "block_size": args.block_size,
        "train_bytes": args.train_bytes,
        "val_bytes": args.val_bytes,
        "test_bytes": args.test_bytes,
        "max_train_chars": args.max_train_chars,
        "max_test_chars": args.max_test_chars,
        "train_blocks": len(out["train"]),
        "val_blocks": len(out["validation"]),
        "test_blocks": len(out["test"]),
    }
    with open(os.path.join(args.out_dir, "prepared_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Saved prepared enwik8 dataset to:", args.out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()