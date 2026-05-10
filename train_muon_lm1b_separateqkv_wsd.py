from itertools import chain
import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import get_cosine_schedule_with_warmup
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorForLanguageModeling,
)

from fix_muon_qkv import split_muon_params_fixed, MuonWithAuxAdamFixed

from torch.optim.lr_scheduler import LambdaLR

import wandb


def parse_args():
    p = argparse.ArgumentParser()
    # REQUIRED: load precomputed splits saved via save_to_disk (must contain train/validation/test)
    p.add_argument("--splits_dir", type=str, required=True,
                   help="Directory of a saved DatasetDict with train/validation/test (from save_to_disk).")
    p.add_argument("--tokenizer_dir", type=str, default="./gpt2_tokenizer",
                   help="Directory of a fixed GPT-2 tokenizer (same across seeds).")
    p.add_argument("--seed", type=int, default=1, help="Random seed for this run.")
    p.add_argument("--block_size", type=int, default=256, help="Context length.")
    p.add_argument("--n_layer", type=int, default=6, help="Transformer layers.")
    p.add_argument("--n_embd", type=int, default=256, help="Embedding/hidden size.")
    p.add_argument("--n_inner", type=int, default=1024, help="Inner dimension of the FFN.")
    p.add_argument("--n_head", type=int, default=4, help="Attention heads (must divide n_embd).")
    p.add_argument("--tie_word_embeddings", action="store_true", help="Tie word embeddings.")
    p.add_argument("--batch_size", type=int, default=32, help="Per-device train batch size.")
    p.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    p.add_argument("--lr", type=float, default=6e-4, help="Learning rate.")
    p.add_argument("--muon_lr", type=float, default=9e-2, help="Learning rate.")
    p.add_argument("--adam_beta1", type=float, default=0.85)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio.")
    p.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    p.add_argument("--eval_steps", type=int, default=50, help="Eval every N steps.")
    p.add_argument("--logging_steps", type=int, default=25, help="Log every N steps.")
    p.add_argument("--save_total_limit", type=int, default=2, help="Limit saved checkpoints.")
    p.add_argument("--early_stop", action="store_true", help="Enable early stopping (on validation).")
    p.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (in evals).")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if CUDA is available.")
    p.add_argument("--output_dir", type=str, default=None, help="Where to save outputs.")
    # W&B
    p.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="gpt2-merging-demo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="merge")
    p.add_argument("--max_steps", type=int, default=-1,
              help="If >0, overrides epochs and trains for exactly this many optimizer steps.")
    return p.parse_args()


def split_muon_params(model):
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # "hidden layer matrices" in GPT-2 live under transformer.h.*
        # (attn/MLP projections). Keep embeddings + lm_head + layernorm/bias in AdamW.
        is_hidden_block = name.startswith("transformer.h.")
        is_matrix = (p.ndim >= 2)

        if is_hidden_block and is_matrix:
            muon_params.append(p)
        else:
            adam_params.append(p)

    return muon_params, adam_params


def get_wsd_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,       # alternative to warmup_steps
    decay_ratio: float = 0.2,         # fraction of training spent decaying
    min_lr_ratio: float = 0.0,        # final LR as fraction of peak
    decay_type: str = "linear",       # "linear" or "cosine"
):
    """
    Warmup-Stable-Decay (WSD) learning rate schedule.
    
    Phases:
      1. Warmup:  linear ramp from 0 → peak over warmup steps
      2. Stable:  constant at peak LR
      3. Decay:   linear or cosine decay from peak → min_lr_ratio * peak
    """
    import math

    if warmup_steps == 0 and warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)
    
    decay_steps = int(total_steps * decay_ratio)
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(current_step):
        # Phase 1: Warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        
        # Phase 2: Stable
        if current_step < warmup_steps + stable_steps:
            return 1.0
        
        # Phase 3: Decay
        steps_into_decay = current_step - warmup_steps - stable_steps
        progress = steps_into_decay / max(1, decay_steps)
        progress = min(progress, 1.0)

        if decay_type == "cosine":
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        else:  # linear
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)



def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.output_dir or f"./lm1b_lmc/muon_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    # --- init W&B (optional) ---
    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="train",
            config=vars(args),  # all CLI args
            tags=[t for t in args.wandb_tags.split(",") if t],
            name=f"gpt2-train-seed{args.seed}"
        )

    # 1) Load precomputed splits (train/validation/test)
    # ds = load_from_disk(args.splits_dir)
    # raw_train, raw_val, raw_test = ds["train"], ds["validation"], ds["test"]

    # 1) Load prepared splits (already has input_ids + attention_mask)
    ds = load_from_disk(args.splits_dir)

    # Expect tokenized/chunked dataset
    train_cols = ds["train"].column_names
    if "input_ids" not in train_cols or "attention_mask" not in train_cols:
        raise ValueError(
            f"{args.splits_dir} is not tokenized/chunked. Columns are {train_cols}. "
            "Point --splits_dir to the prepared dir (e.g. splits_lm1b_contig_chunked256)."
        )

    chunked_train = ds["train"]
    chunked_val   = ds["validation"]
    chunked_test  = ds["test"]

    for dset in (chunked_train, chunked_val, chunked_test):
        dset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 2) Fixed tokenizer across runs
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token  # ensure padding token exists for collator
    block_size = args.block_size

    # 4) Collator (causal LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) Model config
    configuration = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        tie_word_embeddings=args.tie_word_embeddings,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(configuration)

    # 6) TrainingArguments: validate/early-stop on VALIDATION
    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,                 # save whenever we evaluate
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,                # keep best eval_loss (on validation)
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        report_to=("wandb" if args.wandb else "none"),
        run_name=f"gpt2-lm1b-seed{args.seed}-n_embd{args.n_embd}-tuned-muon-{args.max_steps}",
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
    )

    callbacks = []
    if args.early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    muon_params, adam_params, cattn_params = split_muon_params_fixed(model)

    param_groups = [
        dict(params=muon_params, use_muon=True, is_cattn=False,
            lr=args.muon_lr, weight_decay=args.weight_decay),
        dict(params=cattn_params, use_muon=True, is_cattn=True,
            lr=args.muon_lr, weight_decay=args.weight_decay, n_embd=args.n_embd),
        dict(params=adam_params, use_muon=False, is_cattn=False,
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=0.1),
    ]
    
    optimizer = MuonWithAuxAdamFixed(param_groups)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = training_args.gradient_accumulation_steps  # currently 1

    steps_per_epoch = len(chunked_train) // (args.batch_size * world_size * grad_accum)

    total_steps = (
        args.max_steps if args.max_steps > 0
        else steps_per_epoch * args.epochs
    )

    scheduler = get_wsd_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_ratio=0.0,    
                decay_ratio=0.2,      # last 20% is decay
                min_lr_ratio=0.0,     # decay to 0
                decay_type="linear",  
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=chunked_train,
        eval_dataset=chunked_val,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
    )

    # 7) Train
    trainer.train()

    trainer.save_model(out_dir)         
    tokenizer.save_pretrained(out_dir)

    # Record best checkpoint path
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is not None:
        with open(os.path.join(out_dir, "BEST_CHECKPOINT.txt"), "w") as f:
            f.write(best_ckpt + "\n")

    # 8) One-shot final evaluation on TEST (held out)
    test_metrics = trainer.evaluate(eval_dataset=chunked_test, metric_key_prefix="test")
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("TEST metrics:", test_metrics)

    # --- W&B: record final metrics & log artifact ---
    if run is not None:
        art = wandb.Artifact(
            name=f"gpt2-model-seed{args.seed}",
            type="model",
            metadata={
                "seed": args.seed,
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "splits_dir": args.splits_dir,
                "tokenizer_dir": args.tokenizer_dir,
                "best_checkpoint": best_ckpt,
            },
        )
        art.add_dir(out_dir)
        aliases = ["latest", f"seed-{args.seed}"]
        if best_ckpt: aliases.append("best")
        run.log_artifact(art, aliases=aliases)
        run.finish()


if __name__ == "__main__":
    main()