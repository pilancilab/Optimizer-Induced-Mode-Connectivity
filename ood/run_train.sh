#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(0 1 2 3 4)
NPROC=8
MAX_STEPS=15000
EVAL_STEPS=16000

# ── enwik8 ──────────────────────────────────────────────
DATASET="enwik8"
SPLITS_DIR="splits_enwik8_chunked256"

for SEED in "${SEEDS[@]}"; do
  torchrun --standalone --nproc_per_node=$NPROC train_adamw_lm1b_separate.py \
    --seed $SEED --max_steps $MAX_STEPS \
    --adamw_lr 0.01 --weight_decay 0.1 \
    --splits_dir $SPLITS_DIR \
    --output_dir ${DATASET}_train/adamw_seed${SEED} \
    --eval_steps $EVAL_STEPS
done

for SEED in "${SEEDS[@]}"; do
  torchrun --standalone --nproc_per_node=$NPROC train_muon_lm1b_separateqkv_wsd.py \
    --seed $SEED --max_steps $MAX_STEPS \
    --muon_lr 0.03 --weight_decay 0.1 \
    --splits_dir $SPLITS_DIR \
    --output_dir ${DATASET}_train/muon_seed${SEED} \
    --eval_steps $EVAL_STEPS
done

# ── stories ─────────────────────────────────────────────
DATASET="stories"
SPLITS_DIR="splits_stories_contig_chunked256"

for SEED in "${SEEDS[@]}"; do
  torchrun --standalone --nproc_per_node=$NPROC train_adamw_lm1b_separate.py \
    --seed $SEED --max_steps $MAX_STEPS \
    --adamw_lr 0.004 --weight_decay 0.1 \
    --splits_dir $SPLITS_DIR \
    --output_dir ./${DATASET}_train/adamw_seed${SEED} \
    --eval_steps $EVAL_STEPS
done

for SEED in "${SEEDS[@]}"; do
  torchrun --standalone --nproc_per_node=$NPROC train_muon_lm1b_separateqkv_wsd.py \
    --seed $SEED --max_steps $MAX_STEPS \
    --muon_lr 0.004 --weight_decay 0.1 \
    --splits_dir $SPLITS_DIR \
    --output_dir ./${DATASET}_train/muon_seed${SEED} \
    --eval_steps $EVAL_STEPS
done