"""
train_merger_polychain_ood.py

Combines:
  - Polychain interpolation (from train_merger__7_.py / merger_1.py)
  - OOD evaluation sweep (from train_merger2.py)

Trains a GPTMerger with a learned bend-point (polygonal-chain path), then
evaluates both in-distribution and OOD loss barriers.

Typical usage:
    torchrun --standalone --nproc_per_node=8 train_merger_polychain_ood.py \
      --model_dir_0 ./enwik8_train/adamw_seed0 \
      --model_dir_1 ./enwik8_train/muon_seed0 \
      --tokenizer_dir ./gpt2_tokenizer \
      --output_dir ./enwik8_merge_ood_polychain/adamw_muon_seed0 \
      --splits_dir splits_enwik8_chunked256 \
      --ood_splits_dir splits_stories_contig_chunked256 \
      --epochs 10 --batch_size 32 --fp16 --early_stop \
      --sampler NARROW_UNIFORM \
      --eval_steps 100 \
      --curve polychain \
      --wandb --wandb_project gpt2-merging-demo --wandb_group stories-polychain
"""

import argparse
import json
import os
from itertools import chain

from datasets import load_from_disk

import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed,
)
from enums import SamplerType

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Polychain-capable merger lives in optimizer_lmc.merger_1
from merger import GPTMerger, GPTMergerWrapper
import wandb


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train polychain GPTMerger and evaluate on ID + OOD test sets."
    )

    # Model paths
    p.add_argument("--model_dir_0", type=str, required=True)
    p.add_argument("--model_dir_1", type=str, required=True)

    # Data
    p.add_argument("--splits_dir", type=str, required=True,
                   help="DatasetDict directory for in-distribution data (train/val/test).")
    p.add_argument("--ood_splits_dir", type=str, default=None,
                   help="DatasetDict directory for OOD data. "
                        "If provided, runs an additional sweep on its test split.")

    # Tokenizer / output
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer")
    p.add_argument("--output_dir", type=str, default="gpt2_lmc_polychain_merge")
    p.add_argument("--results_name", type=str, default="merged_sampler_losses.json",
                   help="Filename for in-distribution sweep results.")
    p.add_argument("--ood_results_name", type=str, default="merged_sampler_losses_ood.json",
                   help="Filename for OOD sweep results.")

    # Training knobs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", type=str, default="NARROW_UNIFORM")
    p.add_argument("--permutations_only", action="store_true",
                   help="Use only permutation matrices (no orthogonal alignment).")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--fp16", action="store_true")

    # Early stopping
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=10)

    # Coefficient sweep
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--coeff_start", type=float, default=0.0)
    p.add_argument("--coeff_end", type=float, default=1.0)
    p.add_argument("--coeff_step", type=float, default=0.1)

    # Token frequency prior
    p.add_argument("--token_freqs_path", type=str, default="tinyshakespeare_token_freqs.pt")

    # ---- Curve type ----
    p.add_argument("--curve", type=str, default="polychain",
                   choices=["linear", "polychain"],
                   help="Interpolation path: 'linear' (straight) or 'polychain' (one learned bend point).")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="gpt2-merging-demo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="merge,polychain")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _is_artifact_ref(s: str) -> bool:
    return (":" in s) and (not os.path.isdir(s))


def _ensure_local_model_dir(run, spec_or_dir: str) -> str:
    if run is None:
        return spec_or_dir
    if _is_artifact_ref(spec_or_dir):
        art = run.use_artifact(spec_or_dir, type="model")
        return art.download()
    name = f"input-{os.path.basename(os.path.abspath(spec_or_dir))}-{run.id}"
    art = wandb.Artifact(name=name, type="model",
                         metadata={"source": "local_path",
                                   "path": os.path.abspath(spec_or_dir)})
    art.add_dir(spec_or_dir)
    run.log_artifact(art, aliases=["input"])
    return spec_or_dir


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_merger_wrapper(model_dir_0, model_dir_1, sampler_type, permutations_only,
                         token_freqs_path=None, use_polychain=False):
    model0 = GPT2LMHeadModel.from_pretrained(model_dir_0)
    model1 = GPT2LMHeadModel.from_pretrained(model_dir_1)
    model0.eval()
    model1.eval()

    for m in (model0, model1):
        try:
            m.config.attn_implementation = "eager"
            m._attn_implementation = "eager"
        except Exception:
            pass

    token_freqs = None
    if token_freqs_path and os.path.isfile(token_freqs_path):
        token_freqs = torch.load(token_freqs_path, map_location="cpu")

    merger_model = GPTMerger(
        model0, model1,
        token_freqs=token_freqs,
        permutations_only=permutations_only,
        use_polychain=use_polychain,
    )
    merger_model.set_sampler(sampler_type=sampler_type)

    if use_polychain:
        n_bend = sum(p.numel() for n, p in merger_model.named_parameters() if "bend" in n)
        n_proj = sum(p.numel() for n, p in merger_model.named_parameters() if "proj." in n)
        print(f"[polychain] Bend-point params: {n_bend:,}  |  Alignment params: {n_proj:,}")

    return GPTMergerWrapper(config=model0.config, merger_model=merger_model)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_datasets_from_splits(splits_dir, tokenizer, block_size):
    """
    Supports:
      (A) pre-tokenized splits with 'input_ids' + 'attention_mask'
      (B) raw-text splits with column 'text'
    """
    ds = load_from_disk(splits_dir)
    raw_train, raw_val, raw_test = ds["train"], ds["validation"], ds["test"]
    cols = set(raw_train.column_names)

    # Already tokenized
    if "input_ids" in cols and "attention_mask" in cols and "text" not in cols:
        for split in (raw_train, raw_val, raw_test):
            split.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return raw_train, raw_val, raw_test

    if "text" not in cols:
        raise ValueError(
            f"Expected 'text' or ('input_ids','attention_mask') columns. Got: {raw_train.column_names}"
        )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=True)

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

    def prep(split):
        t = split.map(tokenize_fn, batched=True, remove_columns=["text"])
        c = t.map(group_texts, batched=True)
        c.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return c

    return prep(raw_train), prep(raw_val), prep(raw_test)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training_and_save(args, tokenizer, train_ds, val_ds):
    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_polychain = (args.curve == "polychain")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        report_to=("wandb" if args.wandb else "none"),
    )

    model = build_merger_wrapper(
        args.model_dir_0, args.model_dir_1,
        sampler_type=args.sampler,
        permutations_only=args.permutations_only,
        token_freqs_path=args.token_freqs_path,
        use_polychain=use_polychain,
    )

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    best_ckpt = trainer.state.best_model_checkpoint or args.output_dir
    with open(os.path.join(args.output_dir, "BEST_CHECKPOINT.txt"), "w") as f:
        f.write(best_ckpt + "\n")

    return trainer.model


# ---------------------------------------------------------------------------
# Evaluation sweep (reusable for ID and OOD)
# ---------------------------------------------------------------------------

def evaluate_sweep(model, tokenizer, test_ds, args, run=None,
                   results_name=None, log_prefix="merge"):
    """
    Sweep the interpolation coefficient over [coeff_start, coeff_end] and
    record the empirical loss barrier:

        B_λ = L(path(λ)) − [λ·L(endpoint_1) + (1−λ)·L(endpoint_0)]

    For polychain the 'path' is W0→bend→W1 parameterised by t in [0,1],
    so the barrier measures how far the path deviates from the chord between
    the two endpoints — exactly the quantity plotted in Figure 7.

    Parameters
    ----------
    results_name : str, optional
        Output JSON filename override (defaults to args.results_name).
    log_prefix : str
        Prefix for W&B metric keys, e.g. "merge" or "merge_ood".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    eval_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "eval_tmp"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_drop_last=False,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    # Build coefficient grid
    coeffs = []
    c = args.coeff_start
    while c <= args.coeff_end + 1e-9:
        coeffs.append(float(round(c, 10)))
        c += args.coeff_step

    # Evaluate at each coefficient
    coeff_losses = {}
    for coeff in coeffs:
        if hasattr(model, "merger_model") and hasattr(model.merger_model, "set_sampler"):
            model.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        elif hasattr(model, "set_sampler"):
            model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))

        metrics = eval_trainer.evaluate()
        loss = float(metrics.get("eval_loss", float("nan")))
        coeff_key = f"{coeff:.6f}"
        coeff_losses[coeff_key] = loss
        print(f"[{log_prefix}] coeff={coeff_key} -> loss={loss:.6f}")

    if args.coeff_end == args.coeff_start:
        raise ValueError("coeff_start and coeff_end must differ to define a barrier.")

    start_key = f"{float(args.coeff_start):.6f}"
    end_key   = f"{float(args.coeff_end):.6f}"

    L_start = coeff_losses[start_key]
    L_end   = coeff_losses[end_key]
    span    = args.coeff_end - args.coeff_start

    coeff_barriers = {}
    max_barrier   = float("-inf")
    coeff_at_max  = None

    for coeff in coeffs:
        key      = f"{coeff:.6f}"
        L_lambda = coeff_losses[key]
        lam      = (coeff - args.coeff_start) / span
        # Chord between the two endpoint losses
        expected = lam * L_end + (1.0 - lam) * L_start
        B_lambda = L_lambda - expected
        coeff_barriers[key] = B_lambda
        if B_lambda > max_barrier:
            max_barrier  = B_lambda
            coeff_at_max = coeff

    print(f"\n=== Empirical loss barrier ({log_prefix}) ===")
    print(f"Max barrier B = {max_barrier:.6f} at coeff = {coeff_at_max:.6f}")

    fname    = results_name or args.results_name
    out_path = os.path.join(args.output_dir, fname)
    payload  = {
        "coeff_losses":   coeff_losses,
        "coeff_barriers": coeff_barriers,
        "max_barrier":    max_barrier,
        "coeff_at_max":   coeff_at_max,
        "coeff_start":    args.coeff_start,
        "coeff_end":      args.coeff_end,
        "coeff_step":     args.coeff_step,
        "curve":          args.curve,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"✅ Wrote sweep losses/barriers to: {out_path}")

    if run is not None:
        run.log({
            f"{log_prefix}/max_barrier":       max_barrier,
            f"{log_prefix}/max_barrier_coeff": coeff_at_max,
        })

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metadata for reproducibility
    meta = {
        "model_dir_0":      args.model_dir_0,
        "model_dir_1":      args.model_dir_1,
        "permutations_only": args.permutations_only,
        "token_freqs_path": args.token_freqs_path,
        "sampler":          args.sampler,
        "tokenizer_dir":    args.tokenizer_dir,
        "splits_dir":       args.splits_dir,
        "ood_splits_dir":   args.ood_splits_dir,
        "curve":            args.curve,
    }
    with open(os.path.join(args.output_dir, "merge_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    set_seed(args.seed)

    # W&B
    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="merge",
            config=vars(args),
            tags=[t for t in args.wandb_tags.split(",") if t],
            name=args.wandb_run_name,
        )
        args.model_dir_0 = _ensure_local_model_dir(run, args.model_dir_0)
        args.model_dir_1 = _ensure_local_model_dir(run, args.model_dir_1)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # In-distribution data
    train_ds, val_ds, test_ds = build_datasets_from_splits(
        splits_dir=args.splits_dir,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )

    # Train (optimises alignment + bend point on ID train, early-stops on ID val)
    trained_model = run_training_and_save(args, tokenizer, train_ds, val_ds)

    # ---- In-distribution sweep ----
    print("\n" + "=" * 60)
    print("  IN-DISTRIBUTION SWEEP")
    print("=" * 60)
    results_path = evaluate_sweep(
        trained_model, tokenizer, test_ds, args, run,
        results_name=args.results_name,
        log_prefix="merge",
    )

    # ---- OOD sweep ----
    ood_results_path = None
    if args.ood_splits_dir:
        print("\n" + "=" * 60)
        print(f"  OOD SWEEP  ({args.ood_splits_dir})")
        print("=" * 60)
        # Load OOD data (only need test split for evaluation)
        _, _, ood_test_ds = build_datasets_from_splits(
            splits_dir=args.ood_splits_dir,
            tokenizer=tokenizer,
            block_size=args.block_size,
        )
        ood_results_path = evaluate_sweep(
            trained_model, tokenizer, ood_test_ds, args, run,
            results_name=args.ood_results_name,
            log_prefix="merge_ood",
        )

    # Log final artifact
    if run is not None:
        merged_art = wandb.Artifact(
            name=f"gpt2-merged-polychain-{run.id}",
            type="model",
            metadata={
                "parents": {
                    "model_dir_0": os.path.abspath(args.model_dir_0),
                    "model_dir_1": os.path.abspath(args.model_dir_1),
                },
                "results_file":     os.path.abspath(results_path),
                "ood_results_file": os.path.abspath(ood_results_path) if ood_results_path else None,
                "curve":            args.curve,
            },
        )
        merged_art.add_dir(args.output_dir)
        run.log_artifact(merged_art, aliases=["latest"])
        run.finish()


if __name__ == "__main__":
    main()