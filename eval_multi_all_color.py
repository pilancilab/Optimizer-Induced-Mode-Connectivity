import argparse
import json
import os
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

import matplotlib.pyplot as plt

from optimizer_lmc.merger import GPTMerger, GPTMergerWrapper

# Optional safetensors
try:
    from safetensors.torch import load_file as load_safetensors_file
    HAS_SFT = True
except Exception:
    HAS_SFT = False


# ---------- Helpers ----------

def load_state_dict_generic(model_dir: str) -> dict:
    """
    Load a state_dict from a HF checkpoint directory.
    Supports:
      - model.safetensors (preferred)
      - pytorch_model.bin
    """
    sft = os.path.join(model_dir, "model.safetensors")
    ptb = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(sft):
        if not HAS_SFT:
            raise RuntimeError("Found model.safetensors but safetensors is not installed.")
        return load_safetensors_file(sft)
    elif os.path.isfile(ptb):
        return torch.load(ptb, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_dir}")


def are_state_dicts_compatible(sd_a: dict, sd_b: dict) -> bool:
    """
    Return True if the two state dicts are shape-compatible for vanilla interpolation.
    We require that for any shared float tensor key, shapes match.
    """
    for k, v_a in sd_a.items():
        v_b = sd_b.get(k, None)
        if v_b is None:
            continue
        if torch.is_floating_point(v_a) and torch.is_floating_point(v_b):
            if v_a.shape != v_b.shape:
                print(f"[vanilla] Incompatible tensor for key '{k}': {v_a.shape} vs {v_b.shape}")
                return False
    return True


def build_datasets_from_splits(splits_dir, tokenizer, block_size):
    """
    Works for:
      (A) raw-text splits with column 'text'  -> tokenize + chunk
      (B) pre-tokenized splits with columns 'input_ids','attention_mask' -> just format torch
    """
    ds = load_from_disk(splits_dir)
    raw_train, raw_val, raw_test = ds["train"], ds["validation"], ds["test"]

    cols = set(raw_train.column_names)

    # Case B: already tokenized/chunked
    if "input_ids" in cols and "attention_mask" in cols and "text" not in cols:
        for split in (raw_train, raw_val, raw_test):
            split.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return raw_train, raw_val, raw_test

    # Case A: raw text -> tokenize + chunk
    if "text" not in cols:
        raise ValueError(f"Expected 'text' or ('input_ids','attention_mask') in dataset. Got columns: {raw_train.column_names}")

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

    def prep(one_split):
        t = one_split.map(tokenize_fn, batched=True, remove_columns=["text"])
        c = t.map(group_texts, batched=True)
        c.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return c

    train_ds = prep(raw_train)
    val_ds = prep(raw_val)
    test_ds = prep(raw_test)
    return train_ds, val_ds, test_ds


def evaluate_model(model, trainer: Trainer) -> float:
    """Run a single eval pass and return eval_loss."""
    model.eval()
    trainer.model = model
    metrics = trainer.evaluate()
    if "eval_loss" not in metrics:
        raise KeyError("Trainer returned metrics without 'eval_loss'.")
    return float(metrics["eval_loss"])


def interpolate_state_dict(sd_a: dict, sd_b: dict, lam: float) -> dict:
    """
    Vanilla interpolation between two state dicts:

        θ(λ) = λ θ_A + (1 − λ) θ_B

    Assumes keys match and shapes are compatible (checked beforehand).
    Non-floating tensors are taken from θ_A.
    """
    out = {}
    for k, v_a in sd_a.items():
        v_b = sd_b.get(k, None)
        if v_b is None:
            out[k] = v_a
            continue
        if torch.is_floating_point(v_a) and torch.is_floating_point(v_b):
            out[k] = (1.0 - lam) * v_a + lam * v_b
        else:
            out[k] = v_a
    return out


# ---------- Main eval + plot ----------

def run_eval(
    merger_wrapper: GPTMergerWrapper,
    vanilla_model: GPT2LMHeadModel,
    sd_a: dict,
    sd_b: dict,
    tokenizer,
    test_ds,
    permutations_only: bool,
    can_vanilla: bool,
    args,
):
    """
    1) Sweep with merger_wrapper BEFORE loading trained state dict  → "Weight matching"
    2) Load trained state dict into merger_wrapper and sweep again   → "Learned matching"
    3) Sweep vanilla simple interpolation model                      → "Vanilla averaging"
   
    Saves JSON + plot.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merger_wrapper = merger_wrapper.to(device)
    vanilla_model = vanilla_model.to(device)

    eval_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "eval_tmp"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_drop_last=False,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=vanilla_model,  # dummy; will be overwritten
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    coeffs = []
    c = args.coeff_start
    while c <= args.coeff_end + 1e-9:
        coeffs.append(float(round(c, 10)))
        c += args.coeff_step
    coeffs = np.array(coeffs, dtype=float)

    # --- 1) Weight matching sweep (merger before loading checkpoint) ---
    print("\n=== Sweep: Weight matching (pre-trained checkpoint) ===")
    coeff_losses_weight = {}
    for coeff in coeffs:
        if hasattr(merger_wrapper, "merger_model") and hasattr(merger_wrapper.merger_model, "set_sampler"):
            merger_wrapper.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        elif hasattr(merger_wrapper, "set_sampler"):
            merger_wrapper.set_sampler(sampler_type=None, fixed_coeff=float(coeff))

        key = f"{coeff:.6f}"
        loss = evaluate_model(merger_wrapper, trainer)
        coeff_losses_weight[key] = loss
        print(f"[Weight matching] λ={key} -> eval_loss={loss:.6f}")

    # --- 2) Load trained merger checkpoint and sweep again (learned matching) ---
    print("\n=== Loading trained merger state dict (learned matching) ===")
    sd_trained = load_state_dict_generic(args.merged_model_dir)
    missing, unexpected = merger_wrapper.load_state_dict(sd_trained, strict=False)
    if missing or unexpected:
        print(f"⚠️ Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    print("\n=== Sweep: Learned matching (post-training checkpoint) ===")
    coeff_losses_learned = {}
    for coeff in coeffs:
        if hasattr(merger_wrapper, "merger_model") and hasattr(merger_wrapper.merger_model, "set_sampler"):
            merger_wrapper.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        elif hasattr(merger_wrapper, "set_sampler"):
            merger_wrapper.set_sampler(sampler_type=None, fixed_coeff=float(coeff))

        key = f"{coeff:.6f}"
        loss = evaluate_model(merger_wrapper, trainer)
        coeff_losses_learned[key] = loss
        print(f"[Learned matching] λ={key} -> eval_loss={loss:.6f}")

    # --- 3) Vanilla interpolation sweep (if compatible) ---
    coeff_losses_vanilla = None
    if can_vanilla:
        print("\n=== Sweep: Vanilla interpolation (θ(λ) = λ θ_A + (1 − λ) θ_B) ===")
        coeff_losses_vanilla = {}
        for coeff in coeffs:
            key = f"{coeff:.6f}"
            sd_interp = interpolate_state_dict(sd_a, sd_b, lam=float(coeff))
            vanilla_model.load_state_dict(sd_interp, strict=False)
            loss = evaluate_model(vanilla_model, trainer)
            coeff_losses_vanilla[key] = loss
            print(f"[Vanilla] λ={key} -> eval_loss={loss:.6f}")
    else:
        print("\n⚠️ Skipping Vanilla interpolation: base model state dicts are not shape-compatible.")

    # --- Save JSON payload ---
    json_path = os.path.join(args.output_dir, args.results_name)
    payload = {
        "coeffs": coeffs.tolist(),
        "coeff_losses_weight_matching": coeff_losses_weight,
        "coeff_losses_learned_matching": coeff_losses_learned,
        "coeff_start": args.coeff_start,
        "coeff_end": args.coeff_end,
        "coeff_step": args.coeff_step,
        "permutations_only": permutations_only,
        "vanilla_supported": bool(can_vanilla),
    }
    if coeff_losses_vanilla is not None:
        payload["coeff_losses_vanilla"] = coeff_losses_vanilla

    with open(json_path, "w") as f:  
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\n✅ Wrote sweep losses to: {json_path}")

    # --- Prepare arrays for plotting ---
    keys_sorted = sorted(coeff_losses_learned.keys(), key=lambda k: float(k))
    xs = np.array([float(k) for k in keys_sorted])
    ys_weight = np.array([coeff_losses_weight[k] for k in keys_sorted])
    ys_learned = np.array([coeff_losses_learned[k] for k in keys_sorted])

    if can_vanilla and coeff_losses_vanilla is not None:
        ys_vanilla = np.array([coeff_losses_vanilla[k] for k in keys_sorted])
    else:
        ys_vanilla = None

    # ---------- Loss plot ----------
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Labels depending on permutations_only
    if permutations_only:
        label_weight = "Weight (perm)"
        label_learned = "Learned (perm)"
    else:
        label_weight = "Weight"
        label_learned = "Learned"

    labels = [label_learned]
    ys_list = [ys_learned]


    if ys_vanilla is not None:
        labels.append("Vanilla")
        ys_list.append(ys_vanilla)

    colours = ["tab:orange", plt.cm.get_cmap("GnBu_r", 5)(1), "tab:green"]
    markers = ["v", "o", "D"]

    for i, (y, lab) in enumerate(zip(ys_list, labels)):
        ax.plot(
            xs,
            y,
            color=colours[i],
            marker=markers[i],
            markersize=5,
            markeredgecolor="black",
            linewidth=1.5,
            label=lab,
        )

    y_band_src = ys_list[1]
    if np.isfinite(y_band_src).any():
        ymin = np.nanmin(y_band_src)
        ymax = np.nanmax(y_band_src)
        band = 0.01 * (ymax - ymin) if ymax > ymin else 0.0
        if band > 0:
            ax.fill_between(xs, y_band_src - band, y_band_src + band, alpha=0.2, color=colours[1])

    xticks = np.round(np.linspace(0.0, 1.0, 6), 1)  # 0.0, 0.2, ..., 1.0
    if can_vanilla:
        xticklabels = [f"{t:.1f}" for t in xticks]
    else:
        xticklabels = [f"{t:.1f}" for t in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.grid(True, linestyle="dotted", alpha=0.5)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        handlelength=2,
        handletextpad=0.5,
        borderpad=0.4,
    )

    fname_base = "loss_interp"

    loss_pdf = Path(args.output_dir) / f"{fname_base}.pdf"
    loss_png = Path(args.output_dir) / f"{fname_base}.png"
    fig.savefig(loss_pdf, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(loss_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Saved loss sweep plot to: {loss_pdf} and {loss_png}")

    return json_path, str(loss_png)

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a trained GPTMerger checkpoint via coefficient sweep "
            "using merge_meta.json (weight matching vs learned matching vs optional vanilla)."
        )
    )

    # Merged checkpoint + metadata (required)
    p.add_argument(
        "--merged_model_dirs",
        type=str,
        nargs="+",          # at least 1
        required=True,
        help="Explicit list of merged checkpoint dirs (each contains merge_meta.json + model weights).",
    )
    p.add_argument(
        "--error_type",
        type=str,
        default="sem",
        choices=["std", "sem"],
        help="Error bar type across runs: std or sem (std/sqrt(n)).",
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="merger_eval_results",
        help="Where to save JSON + plots.",
    )
    p.add_argument(
        "--results_name",
        type=str,
        default="merged_coeff_losses_weight_learned_vanilla.json",
        help="Filename for JSON results (inside output_dir).",
    )

    # Sweep / eval knobs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--coeff_start", type=float, default=0.0)
    p.add_argument("--coeff_end", type=float, default=1.0)
    p.add_argument("--coeff_step", type=float, default=0.1)
    p.add_argument("--title", type=float, default=0.1)
    p.add_argument("--color_scheme", type=str, default="adamw",
                   choices=["muon", "adamw"],
                   help="Color scheme: 'muon' (green) or 'adamw' (blue)")

    return p.parse_args()


def sweep_one_merged_dir(
    merged_dir: str,
    coeffs: np.ndarray,
    tokenizer,
    test_ds,
    args,
):
    """
    For one merged checkpoint dir:
      - build merger_wrapper from meta's model_dir_0/model_dir_1
      - load trained merger weights from merged_dir
      - sweep learned losses over coeffs
      - sweep vanilla losses over coeffs (if compatible)

    Returns:
      learned_losses: np.ndarray [n_coeff]
      vanilla_losses: np.ndarray [n_coeff] or None
      permutations_only: bool
      can_vanilla: bool
    """
    meta_path = os.path.join(merged_dir, "merge_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing merge_meta.json in {merged_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model_dir_0 = meta["model_dir_0"]
    model_dir_1 = meta["model_dir_1"]
    permutations_only = bool(meta.get("permutations_only", False))
    token_freqs_path = meta.get("token_freqs_path", None)

    # ---- NEW: read curve type from metadata (default "linear" for old checkpoints) ----
    curve_type = meta.get("curve", "linear")
    use_polychain = (curve_type == "polychain")
    if use_polychain:
        print(f"[{os.path.basename(merged_dir)}] Using polychain interpolation (curve={curve_type})")

    # Base state dicts for vanilla compatibility
    sd_a = load_state_dict_generic(model_dir_0)
    sd_b = load_state_dict_generic(model_dir_1)
    can_vanilla = are_state_dicts_compatible(sd_a, sd_b)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vanilla model (starts from model_dir_0)
    vanilla_model = GPT2LMHeadModel.from_pretrained(model_dir_0).to(device).eval()
    try:
        vanilla_model.config.attn_implementation = "eager"
        vanilla_model._attn_implementation = "eager"
    except Exception:
        pass

    # Base models for merger (kept on CPU as your code did)
    base0_cpu = GPT2LMHeadModel.from_pretrained(model_dir_0).eval()
    base1_cpu = GPT2LMHeadModel.from_pretrained(model_dir_1).eval()
    for m in (base0_cpu, base1_cpu):
        try:
            m.config.attn_implementation = "eager"
            m._attn_implementation = "eager"
        except Exception:
            pass

    token_freqs = None
    if token_freqs_path and os.path.isfile(token_freqs_path):
        token_freqs = torch.load(token_freqs_path, map_location="cpu")

    merger_model = GPTMerger(
        base0_cpu,
        base1_cpu,
        token_freqs=token_freqs,
        permutations_only=permutations_only,
        use_polychain=use_polychain,       # <-- NEW
    )
    merger_wrapper = GPTMergerWrapper(config=base0_cpu.config, merger_model=merger_model).to(device).eval()

    # Trainer
    eval_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "eval_tmp"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_drop_last=False,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=vanilla_model,  # placeholder; overwritten in evaluate_model
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    # ---- Load trained merger weights (this is what makes it "learned") ----
    sd_trained = load_state_dict_generic(merged_dir)
    missing, unexpected = merger_wrapper.load_state_dict(sd_trained, strict=False)
    if missing or unexpected:
        print(f"[{os.path.basename(merged_dir)}] ⚠️ missing={len(missing)} unexpected={len(unexpected)}")

    # ---- Learned sweep ----
    learned_losses = []
    for coeff in coeffs:
        if hasattr(merger_wrapper, "merger_model") and hasattr(merger_wrapper.merger_model, "set_sampler"):
            merger_wrapper.merger_model.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        elif hasattr(merger_wrapper, "set_sampler"):
            merger_wrapper.set_sampler(sampler_type=None, fixed_coeff=float(coeff))
        learned_losses.append(evaluate_model(merger_wrapper, trainer))
    learned_losses = np.array(learned_losses, dtype=float)

    # ---- Vanilla sweep ----
    vanilla_losses = None
    if can_vanilla:
        vanilla_losses = []
        for coeff in coeffs:
            sd_interp = interpolate_state_dict(sd_a, sd_b, lam=float(coeff))
            vanilla_model.load_state_dict(sd_interp, strict=False)
            vanilla_losses.append(evaluate_model(vanilla_model, trainer))
        vanilla_losses = np.array(vanilla_losses, dtype=float)

    # cleanup per-dir
    del trainer, merger_wrapper, merger_model, vanilla_model, base0_cpu, base1_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return learned_losses, vanilla_losses, permutations_only, can_vanilla



def main():

    # Color schemes matching sv_hist_compare2.py style
    COLOR_SCHEMES = {
        "muon": {
            "learned": "#2ca02c",       # darker green (tab:green)
            "vanilla": "#a8d5a2",       # muted sage green
            "faint_learned": "#2ca02c",
            "faint_vanilla": "#a8d5a2",
        },
        "adamw": {
            "learned": "#1f77b4",       # darker blue (tab:blue)
            "vanilla": "#9ecae1",       # lighter blue
            "faint_learned": "#1f77b4",
            "faint_vanilla": "#9ecae1",
        },
    }

    args = parse_args()

    scheme = COLOR_SCHEMES.get(args.color_scheme, COLOR_SCHEMES["adamw"])
    C_LEARNED = scheme["learned"]
    C_VANILLA = scheme["vanilla"]
    C_FAINT_LEARNED = scheme["faint_learned"]
    C_FAINT_VANILLA = scheme["faint_vanilla"]
    set_seed(args.seed)

    merged_dirs = args.merged_model_dirs
    if len(merged_dirs) == 0:
        raise ValueError("Provide at least one directory via --merged_model_dirs.")

    for d in merged_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Not a directory: {d}")
        if not os.path.isfile(os.path.join(d, "merge_meta.json")):
            raise FileNotFoundError(f"Missing merge_meta.json in: {d}")

    # ---- Load tokenizer + dataset ONCE from first meta ----
    first_meta_path = os.path.join(merged_dirs[0], "merge_meta.json")
    with open(first_meta_path, "r") as f:
        meta0 = json.load(f)

    tokenizer_dir = meta0.get("tokenizer_dir", None)
    splits_dir = meta0.get("splits_dir", None)
    if tokenizer_dir is None or splits_dir is None:
        raise ValueError("tokenizer_dir / splits_dir missing in merge_meta.json (first merged dir).")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token

    _, _, test_ds = build_datasets_from_splits(
        splits_dir=splits_dir,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )

    # ---- Coeff grid ----
    coeffs = []
    c = args.coeff_start
    while c <= args.coeff_end + 1e-9:
        coeffs.append(float(round(c, 10)))
        c += args.coeff_step
    coeffs = np.array(coeffs, dtype=float)

    # ---- Sweep all merged dirs ----
    all_learned = []
    all_vanilla = []
    vanilla_dirs = []  # track which dirs contributed vanilla
    permutations_only = None

    for d in merged_dirs:
        print(f"\n=== Sweeping: {d} ===")
        learned, vanilla, perm_only, can_vanilla = sweep_one_merged_dir(
            merged_dir=d,
            coeffs=coeffs,
            tokenizer=tokenizer,
            test_ds=test_ds,
            args=args,
        )
        all_learned.append(learned)
        permutations_only = perm_only if permutations_only is None else permutations_only

        if vanilla is not None:
            all_vanilla.append(vanilla)
            vanilla_dirs.append(d)
        else:
            print(f"[{os.path.basename(d)}] Vanilla skipped (incompatible shapes).")

    all_learned = np.stack(all_learned, axis=0)  # [n_runs, n_coeff]

    def err_from_std(std, n):
        if args.error_type == "std":
            return std
        # sem
        return std / np.sqrt(max(n, 1))

    learned_mean = all_learned.mean(axis=0)
    learned_std = all_learned.std(axis=0, ddof=1) if all_learned.shape[0] > 1 else np.zeros_like(learned_mean)
    learned_err = err_from_std(learned_std, all_learned.shape[0])

    if len(all_vanilla) > 0:
        all_vanilla = np.stack(all_vanilla, axis=0)  # [n_vruns, n_coeff]
        vanilla_mean = all_vanilla.mean(axis=0)
        vanilla_std = all_vanilla.std(axis=0, ddof=1) if all_vanilla.shape[0] > 1 else np.zeros_like(vanilla_mean)
        vanilla_err = err_from_std(vanilla_std, all_vanilla.shape[0])
    else:
        vanilla_mean = vanilla_err = None

    # ---- Plot mean ± error band ----
    os.makedirs(args.output_dir, exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))
    ax.tick_params(axis='both', which='major', labelsize=11)

    xs = coeffs
    n_runs = all_learned.shape[0]
    multi = n_runs > 1

    # ---- Learned: mean line + shaded error region ----
    ax.plot(
        xs, learned_mean,
        color=C_LEARNED,
        marker="o", markersize=5, markeredgecolor="black",
        linewidth=2.0,
    )
    if multi:
        ax.fill_between(
            xs,
            learned_mean - learned_err,
            learned_mean + learned_err,
            color=C_LEARNED, alpha=0.20,
        )

    # ---- Vanilla: mean line + shaded error region ----
    if vanilla_mean is not None:
        ax.plot(
            xs, vanilla_mean,
            color=C_VANILLA,
            marker="D", markersize=5, markeredgecolor="black",
            linewidth=2.0, linestyle="--",
        )
        if multi and vanilla_err is not None:
            ax.fill_between(
                xs,
                vanilla_mean - vanilla_err,
                vanilla_mean + vanilla_err,
                color=C_VANILLA, alpha=0.20,
            )

    # ---- your tick style ----
    xticks = np.round(np.linspace(0.0, 1.0, 6), 1)
    xticklabels = [f"{t:.1f}" for t in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # ax.invert_xaxis()

    ax.grid(True, linestyle="dotted", alpha=0.5)

    loss_pdf = Path(args.output_dir) / "loss_interp_all10.pdf"
    loss_png = Path(args.output_dir) / "loss_interp_all10.png"
    fig.savefig(loss_pdf, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(loss_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n📈 Saved: {loss_pdf} and {loss_png}")


    # ---- Save per-run curves + aggregated stats ----
    payload = {
        "merged_model_dirs": merged_dirs,
        "coeffs": coeffs.tolist(),
        "error_type": args.error_type,
        "learned_all": all_learned.tolist(),
        "learned_mean": learned_mean.tolist(),
        "learned_err": learned_err.tolist(),
        "vanilla_dirs_used": vanilla_dirs,
    }
    if vanilla_mean is not None:
        payload.update({
            "vanilla_all": all_vanilla.tolist(),
            "vanilla_mean": vanilla_mean.tolist(),
            "vanilla_err": vanilla_err.tolist(),
        })

    out_json = os.path.join(args.output_dir, "loss_interp_mean_err.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Saved JSON: {out_json}")


if __name__ == "__main__":
    main()