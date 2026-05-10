import argparse, os, json
from itertools import chain
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer_dir", default="./gpt2_tokenizer")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--num_proc", type=int, default=8)   # speed up mapping
    p.add_argument("--batch_size", type=int, default=1000)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_from_disk(args.splits_dir)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tok.pad_token = tok.eos_token

    block_size = args.block_size

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

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        print(f"== Processing {split} ==")
        t = ds[split].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            desc=f"tokenize[{split}]",
        )
        c = t.map(
            group_texts,
            batched=True,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            desc=f"chunk[{split}]",
        )
        out[split] = c

    out.save_to_disk(args.out_dir)

    meta = {
        "source_splits_dir": os.path.abspath(args.splits_dir),
        "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
        "block_size": args.block_size,
        "train_blocks": len(out["train"]),
        "val_blocks": len(out["validation"]),
        "test_blocks": len(out["test"]),
    }
    with open(os.path.join(args.out_dir, "prepared_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved prepared dataset to:", args.out_dir)
    print(meta)

if __name__ == "__main__":
    main()
