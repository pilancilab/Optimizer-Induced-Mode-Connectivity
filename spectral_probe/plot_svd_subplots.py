"""
Plot individual SVD histogram subplots replicating Figure 5 style.

Each subplot is saved as a separate PNG + PDF.  You get one file per
(merger_pair × coeff) cell — assemble them in your figure editor.

Layout of Figure 5 (7 columns × 3 rows):

  Row 0  AdamW-Muon  :  AdamW | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Muon
  Row 1  AdamW-AdamW :  AdamW | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | AdamW
  Row 2  Muon-Muon   :  Muon  | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Muon

Color gradient for each row:
  AdamW-Muon  : tab:blue → tab:green
  AdamW-AdamW : tab:blue → tab:blue   (stays blue)
  Muon-Muon   : tab:green → tab:green (stays green)

Usage
-----
# 1. Extract SVD data (once per pair)
python extract_svd_polychain.py \
    --merger_dir ./lm1b_large_merge_polychain/adamw_muon_seed_0 \
    --model_dir_0 ./adamw_train_large/adamw_seed0_lr0.001_wd0.1_separate \
    --model_dir_1 ./muon_train_large/muon_seed0_lr0.005_wd0.01_separateqkv_wsd \
    --output_path svd_polychain_adamw_muon.pt

python extract_svd_polychain.py \
    --merger_dir ./lm1b_large_merge_polychain/adamw_seed_0_1 \
    --model_dir_0 ./adamw_train_large/adamw_seed0_lr0.001_wd0.1_separate \
    --model_dir_1 ./adamw_train_large/adamw_seed1_lr0.001_wd0.1_separate \
    --output_path svd_polychain_adamw_adamw.pt

python extract_svd_polychain.py \
    --merger_dir ./lm1b_large_merge_polychain/muon_seed_0_1 \
    --model_dir_0 ./muon_train_large/muon_seed0_lr0.005_wd0.01_separateqkv_wsd \
    --model_dir_1 ./muon_train_large/muon_seed1_lr0.005_wd0.01_separateqkv_wsd \
    --output_path svd_polychain_muon_muon.pt

# 2. Plot
python plot_svd_fig5_subplots.py \
    --adamw_muon  svd_polychain_adamw_muon.pt \
    --adamw_adamw svd_polychain_adamw_adamw.pt \
    --muon_muon   svd_polychain_muon_muon.pt \
    --output_dir  fig5_subplots \
    --key         layer1.fc
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── colour helpers ────────────────────────────────────────────────────────────

def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])


def lerp_color(c0, c1, t):
    return (1.0 - t) * c0 + t * c1


FILL_ADAMW  = hex_to_rgb(mcolors.TABLEAU_COLORS["tab:blue"])
FILL_MUON   = hex_to_rgb(mcolors.TABLEAU_COLORS["tab:green"])
TITLE_ADAMW = hex_to_rgb("#1f77b4")
TITLE_MUON  = hex_to_rgb("#2ca02c")

HIST_ALPHA  = 0.35
LABEL_COLOR = "#333333"

# ── font setup ────────────────────────────────────────────────────────────────

def setup_fonts():
    plt.rcParams["font.family"]      = "serif"
    plt.rcParams["font.serif"]       = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"


# ── subplot helpers ───────────────────────────────────────────────────────────

def condition_number(sv: np.ndarray, percentile: float = 5.0) -> float:
    """σ_max / σ_p  where σ_p is the given percentile (default 5th).
    
    Using σ_min directly is unstable because a single near-zero singular
    value produces cond ~ 10⁷ and dominates any average.  The 5th-percentile
    gives a robust measure of how spread the spectrum is.
    """
    sv_pos = sv[sv > 0]
    if len(sv_pos) == 0:
        return float("nan")
    sigma_min = np.percentile(sv_pos, percentile)
    if sigma_min < 1e-12:
        return float("nan")
    return float(sv_pos.max() / sigma_min)


def avg_condition_all_keys(results_at_coeff: dict,
                           percentile: float = 5.0) -> float:
    """Median condition number over ALL weight keys (layers × weight types).
    
    Median is used instead of mean because even with a percentile-based
    condition number, a handful of outlier matrices can skew the mean.
    """
    conds = []
    for name, sv in results_at_coeff.items():
        sv = np.asarray(sv).ravel()
        c = condition_number(sv, percentile=percentile)
        if not np.isnan(c):
            conds.append(c)
    return float(np.median(conds)) if conds else float("nan")


def stable_rank(sv: np.ndarray) -> float:
    """Stable rank = ||W||_F^2 / ||W||_op^2 = sum(σ²) / σ_max².
    
    Measures spectral isotropy: equals min(m,n) for a perfectly isotropic
    matrix, and 1 for a rank-1 matrix.
    """
    sv_pos = sv[sv > 0]
    if len(sv_pos) == 0:
        return float("nan")
    return float((sv_pos ** 2).sum() / (sv_pos.max() ** 2))


def median_stable_rank_all_keys(results_at_coeff: dict) -> float:
    """Median stable rank over ALL weight keys (layers × weight types)."""
    sranks = []
    for name, sv in results_at_coeff.items():
        sv = np.asarray(sv).ravel()
        sr = stable_rank(sv)
        if not np.isnan(sr):
            sranks.append(sr)
    return float(np.median(sranks)) if sranks else float("nan")


def plot_subplot(sv, coeff_str, fill_rgb, title_rgb,
                 col_label, row_label,
                 avg_cond=None, avg_srank=None,
                 eval_loss=None,
                 bins=50, figw=3.2, figh=3.2, dpi=300):
    """Draw a single histogram subplot."""
    cond = condition_number(sv)
    sr   = stable_rank(sv)
    avg_str  = f"{avg_cond:.1f}" if avg_cond is not None and not np.isnan(avg_cond) else "N/A"
    sr_str   = f"{avg_srank:.1f}" if avg_srank is not None and not np.isnan(avg_srank) else "N/A"
    print(f"  cond={cond:.1f}  srank={sr:.1f}  |  median_cond={avg_str}  median_srank={sr_str}  [{col_label}  {row_label}]")

    fig, ax = plt.subplots(1, 1, figsize=(figw, figh))

    # ── title (optimizer label / coeff) + stable rank ────────────────────────
    fig.suptitle(col_label,
                 fontsize=13, fontweight="bold", color=title_rgb, y=0.98)
    # annotate median stable rank below the title
    display_srank = avg_srank if avg_srank is not None and not np.isnan(avg_srank) else sr
    if not np.isnan(display_srank):
        ax.set_title(f"srank={display_srank:.1f}", fontsize=8, color="#666666", pad=2)

    # ── histogram ────────────────────────────────────────────────────────────
    ax.hist(sv, bins=bins,
            color=fill_rgb,
            edgecolor="none",
            alpha=HIST_ALPHA,
            linewidth=0.5)

    ax.set_xlim(left=0)
    ax.set_xlabel("singular value", fontsize=7, color=LABEL_COLOR)
    ax.set_ylabel(f"{row_label}\ncount", fontsize=8, color=LABEL_COLOR)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, linestyle="dotted", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, ax


# ── per-row colour logic ──────────────────────────────────────────────────────

# def row_colors(pair_name: str, t: float):
#     """
#     Return (fill_rgb, title_rgb) for a given pair and interpolation coeff.

#     pair_name  : one of  'adamw_muon' | 'adamw_adamw' | 'muon_muon'
#     t          : float in [0, 1]
#     """
#     if pair_name == "adamw_muon":
#         fill  = lerp_color(FILL_ADAMW,  FILL_MUON,  t)
#         title = lerp_color(TITLE_ADAMW, TITLE_MUON, t)
#     elif pair_name == "adamw_adamw":
#         fill  = lerp_color(FILL_ADAMW,  FILL_ADAMW,  t)   # stays blue
#         title = lerp_color(TITLE_ADAMW, TITLE_ADAMW, t)
#     elif pair_name == "muon_muon":
#         fill  = lerp_color(FILL_MUON,   FILL_MUON,  t)    # stays green
#         title = lerp_color(TITLE_MUON,  TITLE_MUON, t)
#     else:
#         raise ValueError(f"Unknown pair: {pair_name}")
#     return fill, title


def row_colors(pair_name: str, t: float):
    """
    All three pairs use the same blue→green gradient so colour encodes
    only the interpolation position, not which pair is being shown.
    """
    if pair_name not in ("adamw_muon", "adamw_adamw", "muon_muon"):
        raise ValueError(f"Unknown pair: {pair_name}")
    fill  = lerp_color(FILL_ADAMW, FILL_MUON,  t)
    title = lerp_color(TITLE_ADAMW, TITLE_MUON, t)
    return fill, title


def col_label_for(pair_name: str, coeff_str: str) -> str:
    """Human-readable column header (matches Figure 5 paper labels)."""
    t = float(coeff_str)
    if t == 0.0:
        return {"adamw_muon":  "AdamW",
                "adamw_adamw": "AdamW",
                "muon_muon":   "Muon"}[pair_name]
    if t == 1.0:
        return {"adamw_muon":  "Muon",
                "adamw_adamw": "AdamW",
                "muon_muon":   "Muon"}[pair_name]
    return f"coeff={coeff_str}"


ROW_LABELS = {
    "adamw_muon":  "AdamW→Muon",
    "adamw_adamw": "AdamW→AdamW",
    "muon_muon":   "Muon→Muon",
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adamw_muon",  type=str, default=None,
                        help="Path to .pt from extract_svd_polychain for AdamW-Muon pair")
    parser.add_argument("--adamw_adamw", type=str, default=None,
                        help="Path to .pt from extract_svd_polychain for AdamW-AdamW pair")
    parser.add_argument("--muon_muon",   type=str, default=None,
                        help="Path to .pt from extract_svd_polychain for Muon-Muon pair")
    parser.add_argument("--key",         type=str, default="layer1.fc",
                        help="Weight key to plot, e.g. layer1.fc, layer0.Q")
    parser.add_argument("--output_dir",  type=str, default="fig5_subplots")
    parser.add_argument("--coeffs",      type=str,
                        default="0.0,0.1,0.3,0.5,0.7,0.9,1.0",
                        help="Comma-separated list of coefficients to plot (must match .pt keys)")
    parser.add_argument("--bins",        type=int, default=50)
    parser.add_argument("--figw",        type=float, default=3.2)
    parser.add_argument("--figh",        type=float, default=3.2)
    parser.add_argument("--dpi",         type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_fonts()

    # ── load available datasets ─────────────────────────────────────────────
    data = {}
    if args.adamw_muon:
        data["adamw_muon"] = torch.load(args.adamw_muon, weights_only=False)
    if args.adamw_adamw:
        data["adamw_adamw"] = torch.load(args.adamw_adamw, weights_only=False)
    if args.muon_muon:
        data["muon_muon"] = torch.load(args.muon_muon, weights_only=False)

    if not data:
        print("Error: provide at least one of --adamw_muon, --adamw_adamw, --muon_muon")
        return

    target_coeffs = [c.strip() for c in args.coeffs.split(",")]

    # ── print summary metrics over all layers/weights ──────────────────────
    print("\n" + "=" * 75)
    print("Median metrics (over all layers × weight types)")
    print("=" * 75)
    avg_conds = {}    # {pair_name: {coeff_str: float}}
    avg_sranks = {}   # {pair_name: {coeff_str: float}}
    for pair_name, results in data.items():
        avg_conds[pair_name] = {}
        avg_sranks[pair_name] = {}
        available = sorted(results.keys(), key=float)
        header_coeffs = []
        for coeff_str in target_coeffs:
            if coeff_str in results:
                actual_key = coeff_str
            else:
                variants = [coeff_str,
                            f"{float(coeff_str):.1f}",
                            f"{float(coeff_str):.2f}"]
                actual_key = next((v for v in variants if v in results), None)
                if actual_key is None:
                    actual_key = min(available,
                                     key=lambda k: abs(float(k) - float(coeff_str)))
            avg_conds[pair_name][actual_key] = avg_condition_all_keys(results[actual_key])
            avg_sranks[pair_name][actual_key] = median_stable_rank_all_keys(results[actual_key])
            header_coeffs.append(actual_key)

        row_label = ROW_LABELS.get(pair_name, pair_name)
        print(f"\n  {row_label}")
        cond_vals  = "  ".join(f"{c}: {avg_conds[pair_name][c]:.1f}"
                               for c in header_coeffs)
        srank_vals = "  ".join(f"{c}: {avg_sranks[pair_name][c]:.1f}"
                               for c in header_coeffs)
        print(f"    cond (σ_max/σ_5th):  {cond_vals}")
        print(f"    stable rank (Σσ²/σ²_max):  {srank_vals}")
    print("=" * 75 + "\n")

    # ── iterate: row (pair) × column (coeff) ─────────────────────────────────
    total = 0
    for pair_name, results in data.items():
        row_label = ROW_LABELS[pair_name]

        for coeff_str in target_coeffs:
            # find the closest matching key in the .pt file
            available = sorted(results.keys(), key=float)
            # exact match first, then nearest
            if coeff_str in results:
                actual_key = coeff_str
            else:
                # try zero-padded variants
                variants = [coeff_str,
                            f"{float(coeff_str):.1f}",
                            f"{float(coeff_str):.2f}"]
                actual_key = next((v for v in variants if v in results), None)
                if actual_key is None:
                    # nearest by value
                    actual_key = min(available,
                                     key=lambda k: abs(float(k) - float(coeff_str)))
                    print(f"  [warn] coeff {coeff_str} not found in {pair_name}; "
                          f"using {actual_key}")

            sv = results[actual_key].get(args.key, None)
            if sv is None:
                print(f"  [skip] key '{args.key}' missing for {pair_name} coeff={actual_key}")
                continue

            t           = float(actual_key)
            fill, title = row_colors(pair_name, t)
            col_lbl     = col_label_for(pair_name, actual_key)

            fig, ax = plot_subplot(
                sv         = sv,
                coeff_str  = actual_key,
                fill_rgb   = fill,
                title_rgb  = title,
                col_label  = col_lbl,
                row_label  = row_label,
                avg_cond   = avg_conds.get(pair_name, {}).get(actual_key, float("nan")),
                avg_srank  = avg_sranks.get(pair_name, {}).get(actual_key, float("nan")),
                bins       = args.bins,
                figw       = args.figw,
                figh       = args.figh,
                dpi        = args.dpi,
            )

            safe_coeff = actual_key.replace(".", "p")
            stem       = f"{pair_name}_{args.key.replace('.','_')}_coeff{safe_coeff}"
            png_path   = os.path.join(args.output_dir, stem + ".png")
            pdf_path   = os.path.join(args.output_dir, stem + ".pdf")
            fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
            fig.savefig(pdf_path, format="pdf", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            total += 1
            print(f"  saved {png_path}")

    print(f"\nDone — {total} subplot files written to {args.output_dir}/")
    print("Naming convention:  <pair>_<key>_coeff<t>.png")
    print("Example grid order for Fig 5 row 0 (AdamW→Muon):")
    print("  adamw_muon_layer1_fc_coeff0p0.png  [AdamW endpoint]")
    print("  adamw_muon_layer1_fc_coeff0p1.png")
    print("  adamw_muon_layer1_fc_coeff0p3.png")
    print("  adamw_muon_layer1_fc_coeff0p5.png")
    print("  adamw_muon_layer1_fc_coeff0p7.png")
    print("  adamw_muon_layer1_fc_coeff0p9.png")
    print("  adamw_muon_layer1_fc_coeff1p0.png  [Muon endpoint]")


if __name__ == "__main__":
    main()