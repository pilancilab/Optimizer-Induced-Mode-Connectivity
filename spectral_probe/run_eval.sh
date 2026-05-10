#!/bin/bash
# Full pipeline: extract SVD along polychain path → svd subplots
# Edit the paths below to match your directory layout.

set -e

# cd to project root so Python finds merger.py, utils.py, etc.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL0_ADAMW="./lm1b_train/adamw_seed0"
MODEL1_MUON="./lm1b_train/muon_seed0"
MODEL1_ADAMW="./lm1b_train/adamw_seed1"
MODEL1_MUON1="./lm1b_train/muon_seed1"

MERGE_ADAMW_MUON="./lm1b_merge_polychain/adamw_muon_seed_0"
MERGE_ADAMW_ADAMW="./lm1b_merge_polychain/adamw_seed_0_1"
MERGE_MUON_MUON_bendmuon="./lm1b_merge_polychain_muonbend/muon_seed_0_1"

KEY="layer1.fc"   # change to layer0.Q etc. as needed
COEFFS="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

echo "=== Step 1: Extract SVD (AdamW → Muon) ==="
python spectral_probe/extract_svd_polychain.py \
    --merger_dir  ${MERGE_ADAMW_MUON} \
    --model_dir_0 ${MODEL0_ADAMW} \
    --model_dir_1 ${MODEL1_MUON} \
    --coeffs      ${COEFFS} \
    --output_path svd_polychain_adamw_muon.pt

echo "=== Step 2: Extract SVD (AdamW → AdamW) ==="
python spectral_probe/extract_svd_polychain.py \
    --merger_dir  ${MERGE_ADAMW_ADAMW} \
    --model_dir_0 ${MODEL0_ADAMW} \
    --model_dir_1 ${MODEL1_ADAMW} \
    --coeffs      ${COEFFS} \
    --output_path svd_polychain_adamw_adamw.pt

echo "=== Step 3: Extract SVD (Muon → Muon) ==="
python spectral_probe/extract_svd_polychain.py \
    --merger_dir  ${MERGE_MUON_MUON_bendmuon} \
    --model_dir_0 ${MODEL1_MUON} \
    --model_dir_1 ${MODEL1_MUON1} \
    --coeffs      ${COEFFS} \
    --output_path svd_polychain_muon_muon.pt

echo "=== Step 4: Plot Fig 5 subplots ==="
python spectral_probe/plot_svd_subplots.py \
    --adamw_muon  svd_polychain_adamw_muon.pt \
    --adamw_adamw svd_polychain_adamw_adamw.pt \
    --muon_muon svd_polychain_muon_muon.pt \
    --key         ${KEY} \
    --coeffs      ${COEFFS} \
    --output_dir  svd_subplots

echo "Done!  Subplots are in ./svd_subplots/"