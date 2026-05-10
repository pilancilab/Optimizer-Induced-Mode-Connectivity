"""
plot_barrier_polychain_ood.py

Like plot_barrier_both.py but plots only a single group (--dirs3).

Usage:
    python plot_barrier_polychain_ood.py \
      --dirs3 ./enwik8_merge_ood_polychain/adamw_muon_seed0 ... \
      --json_name_id  merged_sampler_losses.json \
      --json_name_ood merged_sampler_losses_ood.json \
      --label3 "Mix (polychain)" \
      --color_scheme3 mix \
      --error_type std \
      --output_dir ./barrier_both_plots/enwik8_train_polychain
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


COLOR_SCHEMES = {
    "adamw": {"learned": "#1f77b4"},
    "muon":  {"learned": "#2ca02c"},
    "mix":   {"learned": "#b22222"},
}


def load_barriers(dirs, json_name):
    all_barriers = []
    keys_sorted = None
    for d in dirs:
        json_path = os.path.join(d, json_name)
        if not os.path.isfile(json_path):
            print(f"Warning: Missing {json_path}, skipping.")
            continue
        with open(json_path) as f:
            payload = json.load(f)
        coeff_barriers = payload["coeff_barriers"]
        keys_sorted = sorted(coeff_barriers.keys(), key=lambda k: float(k))
        all_barriers.append([coeff_barriers[k] for k in keys_sorted])

    if not all_barriers:
        raise RuntimeError(f"No valid JSON files found for {json_name} in provided dirs.")

    coeffs   = np.array([float(k) for k in keys_sorted])
    barriers = np.array(all_barriers, dtype=float)

    # Flip: coeff=0 → model_0 (AdamW), coeff=1 → model_1 (Muon)
    coeffs   = 1.0 - coeffs[::-1]
    barriers = barriers[:, ::-1]
    return coeffs, barriers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs3", type=str, nargs="+", required=True)
    p.add_argument("--json_name_id",  type=str, default="merged_sampler_losses.json")
    p.add_argument("--json_name_ood", type=str, default="merged_sampler_losses_ood.json")
    p.add_argument("--label3",        type=str, default="Mix")
    p.add_argument("--title_id",      type=str, default="In-distribution")
    p.add_argument("--title_ood",     type=str, default="OOD")
    p.add_argument("--color_scheme3", type=str, default="mix",
                   choices=list(COLOR_SCHEMES.keys()))
    p.add_argument("--error_type",    type=str, default="std", choices=["std", "sem"])
    p.add_argument("--ymax",          type=float, default=None)
    p.add_argument("--coeff_min",     type=float, default=None)
    p.add_argument("--output_dir",    type=str, default="barrier_both_plots")
    return p.parse_args()


def plot_panel(ax, dirs, label, cs_name, json_name, args, title=None):
    color  = COLOR_SCHEMES[cs_name]["learned"]
    coeffs, barriers = load_barriers(dirs, json_name)

    if args.coeff_min is not None:
        mask     = coeffs >= args.coeff_min
        coeffs   = coeffs[mask]
        barriers = barriers[:, mask]

    n_runs = barriers.shape[0]
    mean   = barriers.mean(axis=0)
    std    = barriers.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)
    err    = (std / np.sqrt(max(n_runs, 1))) if args.error_type == "sem" else std

    print(f"\n[{title}] {label}: {n_runs} runs")
    for c, m, s in zip(coeffs, mean, std):
        print(f"  coeff={c:.2f}  barrier={m:.4f} +/- {s:.4f}")

    ax.plot(coeffs, mean, color=color, marker="D", markersize=5,
            markeredgecolor="black", linewidth=2.0, label=label)
    if n_runs > 1:
        ax.fill_between(coeffs, mean - err, mean + err, color=color, alpha=0.20)

    xmin   = args.coeff_min if args.coeff_min is not None else 0.0
    xticks = np.round(np.linspace(xmin, 1.0, 6), 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.1f}" for t in xticks])
    ax.set_xlim(left=xmin, right=1.0)
    # ax.axhline(0, color="black", linewidth=1.2, linestyle="--")

    if args.ymax is not None:
        ax.set_ylim(top=args.ymax)

    ax.grid(True, linestyle="dotted", alpha=0.5)
    # ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    plt.rcParams["font.family"]      = "serif"
    plt.rcParams["font.serif"]       = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, (ax_id, ax_ood) = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    ax_id.tick_params(axis="both", which="major", labelsize=11)
    ax_ood.tick_params(axis="both", which="major", labelsize=11)

    plot_panel(ax_id,  args.dirs3, args.label3, args.color_scheme3,
               args.json_name_id,  args, title=args.title_id)
    plot_panel(ax_ood, args.dirs3, args.label3, args.color_scheme3,
               args.json_name_ood, args, title=args.title_ood)

    ax_id.set_ylabel("Loss barrier", fontsize=11)
    for ax in (ax_id, ax_ood):
        ax.set_xlabel("Interpolation coefficient", fontsize=11)

    fig.tight_layout()

    out_pdf = Path(args.output_dir) / "barrier_id_ood.pdf"
    out_png = Path(args.output_dir) / "barrier_id_ood.png"
    fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()