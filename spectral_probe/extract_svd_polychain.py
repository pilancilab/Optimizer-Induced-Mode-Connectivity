"""
Extract singular value histograms of layer1.fc weights along the polychain
interpolation path of a trained GPTMerger (polychain mode).

Produces one .pt file per merger pair, structured identically to the output
of extract_svd_along_path.py so the same plotting code can be reused.

Usage (run once per merger dir):
    python extract_svd_polychain.py \
        --merger_dir  ./lm1b_large_merge_polychain/adamw_muon_seed_0 \
        --model_dir_0 ./adamw_train_large/adamw_seed0_lr0.001_wd0.1_separate \
        --model_dir_1 ./muon_train_large/muon_seed0_lr0.005_wd0.01_separateqkv_wsd \
        --output_path svd_polychain_adamw_muon.pt \
        --key layer1.fc

    python extract_svd_polychain.py \
        --merger_dir  ./lm1b_large_merge_polychain/adamw_seed_0_1 \
        --model_dir_0 ./adamw_train_large/adamw_seed0_lr0.001_wd0.1_separate \
        --model_dir_1 ./adamw_train_large/adamw_seed1_lr0.001_wd0.1_separate \
        --output_path svd_polychain_adamw_adamw.pt \
        --key layer1.fc

    python extract_svd_polychain.py \
        --merger_dir  ./lm1b_large_merge_polychain/muon_seed_0_1 \
        --model_dir_0 ./muon_train_large/muon_seed0_lr0.005_wd0.01_separateqkv_wsd \
        --model_dir_1 ./muon_train_large/muon_seed1_lr0.005_wd0.01_separateqkv_wsd \
        --output_path svd_polychain_muon_muon.pt \
        --key layer1.fc
"""

import argparse
import os
import sys
import torch
import numpy as np
from transformers import GPT2LMHeadModel

# ── make sure your project root is on the path ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def interpolate_polychain(W0, W1, bend, t):
    """Piecewise-linear path  W0 → bend → W1."""
    if t <= 0.5:
        s = 2.0 * t
        return (1.0 - s) * W0 + s * bend
    else:
        s = 2.0 * (t - 0.5)
        return (1.0 - s) * bend + s * W1


def rebuild_merger_and_load(merger_dir, model_dir_0, model_dir_1,
                            token_freqs_path=None, permutations_only=False):
    from merger import GPTMerger

    model0 = GPT2LMHeadModel.from_pretrained(model_dir_0)
    model1 = GPT2LMHeadModel.from_pretrained(model_dir_1)
    model0.eval(); model1.eval()

    for m in (model0, model1):
        try:
            m.config.attn_implementation = "eager"
            m._attn_implementation = "eager"
        except Exception:
            pass

    token_freqs = None
    if token_freqs_path and os.path.isfile(token_freqs_path):
        token_freqs = torch.load(token_freqs_path, map_location="cpu")

    print("Building polychain merger …")
    merger = GPTMerger(model0, model1,
                       token_freqs=token_freqs,
                       permutations_only=permutations_only,
                       use_polychain=True)

    # ── load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(merger_dir, "pytorch_model.bin")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
    else:
        from safetensors.torch import load_file
        ckpt_path = os.path.join(merger_dir, "model.safetensors")
        state_dict = load_file(ckpt_path, device="cpu")

    try:
        merger.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint (direct key match)")
    except RuntimeError:
        stripped = {k.replace("merger_model.", "", 1) if k.startswith("merger_model.") else k: v
                    for k, v in state_dict.items()}
        merger.load_state_dict(stripped, strict=True)
        print("Loaded checkpoint (stripped merger_model. prefix)")

    merger.eval()
    return merger


# ── weight extractors ────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_fc_weight(merger, layer_idx, coeff):
    """Return the interpolated c_fc weight for one layer at one coeff."""
    cfc = merger.model.transformer.h[layer_idx].mlp.c_fc

    # aligned model-1 weight  (same as Conv1DMerger.forward logic)
    aligned_w1 = cfc.P_in @ cfc.conv1d_1_weight @ cfc.P_out

    if cfc.use_polychain and cfc.bend_weight is not None:
        w = interpolate_polychain(cfc.conv1d_0_weight, aligned_w1,
                                  cfc.bend_weight.data, coeff)
    else:
        w = (1.0 - coeff) * cfc.conv1d_0_weight + coeff * aligned_w1

    return w.cpu().float()


@torch.no_grad()
def extract_layer_weights(merger, coeff):
    """
    Project alignment matrices then extract interpolated Q/K/V/O/fc/proj
    weights for every layer.  Returns dict: {layerN.X: Tensor (2-D)}.
    """
    merger._project(coeff)
    weights = {}
    n_layers = len(merger.model.transformer.h)

    for i in range(n_layers):
        block  = merger.model.transformer.h[i]

        # ── c_attn  (QKV) ────────────────────────────────────────────────────
        cattn = block.attn.c_attn
        cattn_1_w, _ = cattn._permute_heads(
            cattn.conv1d_1_weight, cattn.conv1d_1_bias, cattn.P_out)

        if cattn.use_polychain and cattn.bend_weight is not None:
            cattn_w = interpolate_polychain(
                cattn.conv1d_0_weight, cattn_1_w, cattn.bend_weight.data, coeff)
        else:
            cattn_w = (1 - coeff) * cattn.conv1d_0_weight + coeff * cattn_1_w

        q_w, k_w, v_w = cattn_w.chunk(3, dim=1)
        weights[f"layer{i}.Q"] = q_w.cpu().float()
        weights[f"layer{i}.K"] = k_w.cpu().float()
        weights[f"layer{i}.V"] = v_w.cpu().float()

        # ── c_proj  (O) ──────────────────────────────────────────────────────
        cproj = block.attn.c_proj
        cproj_1_w, _ = cproj._permute_heads(
            cproj.conv1d_1_weight @ cproj.P_out,
            cproj.conv1d_1_bias @ cproj.P_out,
            cproj.P_in)

        if cproj.use_polychain and cproj.bend_weight is not None:
            cproj_w = interpolate_polychain(
                cproj.conv1d_0_weight, cproj_1_w, cproj.bend_weight.data, coeff)
        else:
            cproj_w = (1 - coeff) * cproj.conv1d_0_weight + coeff * cproj_1_w

        weights[f"layer{i}.O"] = cproj_w.cpu().float()

        # ── mlp.c_fc ─────────────────────────────────────────────────────────
        cfc = block.mlp.c_fc
        aligned_fc1 = cfc.P_in @ cfc.conv1d_1_weight @ cfc.P_out

        if cfc.use_polychain and cfc.bend_weight is not None:
            fc_w = interpolate_polychain(
                cfc.conv1d_0_weight, aligned_fc1, cfc.bend_weight.data, coeff)
        else:
            fc_w = (1 - coeff) * cfc.conv1d_0_weight + coeff * aligned_fc1

        weights[f"layer{i}.fc"] = fc_w.cpu().float()

        # ── mlp.c_proj ───────────────────────────────────────────────────────
        mproj = block.mlp.c_proj
        aligned_mp1 = mproj.P_in @ mproj.conv1d_1_weight @ mproj.P_out

        if mproj.use_polychain and mproj.bend_weight is not None:
            mp_w = interpolate_polychain(
                mproj.conv1d_0_weight, aligned_mp1, mproj.bend_weight.data, coeff)
        else:
            mp_w = (1 - coeff) * mproj.conv1d_0_weight + coeff * aligned_mp1

        weights[f"layer{i}.proj"] = mp_w.cpu().float()

    return weights


@torch.no_grad()
def compute_singular_values(weights):
    return {name: torch.linalg.svdvals(w).numpy() for name, w in weights.items()}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merger_dir",  required=True)
    parser.add_argument("--model_dir_0", required=True)
    parser.add_argument("--model_dir_1", required=True)
    parser.add_argument("--token_freqs_path", default=None)
    parser.add_argument("--permutations_only", action="store_true")
    parser.add_argument("--coeffs", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--output_path", default="svd_polychain.pt")
    args = parser.parse_args()

    coeffs = [float(c) for c in args.coeffs.split(",")]

    merger = rebuild_merger_and_load(
        args.merger_dir, args.model_dir_0, args.model_dir_1,
        token_freqs_path=args.token_freqs_path,
        permutations_only=args.permutations_only,
    )

    results = {}
    for coeff in coeffs:
        print(f"\n--- coeff = {coeff:.2f} ---")
        weights = extract_layer_weights(merger, coeff)
        sv      = compute_singular_values(weights)
        results[f"{coeff:.2f}"] = sv
        for name, vals in list(sv.items())[:4]:        # print a few
            print(f"  {name:20s}  top-5: {vals[:5]}")

    torch.save(results, args.output_path)
    print(f"\nSaved to {args.output_path}")
    print("Access: results['0.50']['layer1.fc']  -> np.array of singular values")


if __name__ == "__main__":
    main()