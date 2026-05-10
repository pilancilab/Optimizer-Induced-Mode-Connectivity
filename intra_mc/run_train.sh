#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"


NPROC=8
MAX_STEPS=15000
EVAL_STEPS=16000
SEEDS=(0 1 2 3 4 5 6 7 8 9)

#         dataset      splits_dir                            adamw_lr  muon_lr
CONFIGS=(
  "enwik8     splits_enwik8_chunked256              0.01   0.03"
  "stories    splits_stories_contig_chunked256      0.004  0.004"
  "bookcorpus splits_bookcorpus_contig_chunked256   0.004  0.005"
  "lm1b       splits_lm1b_contig_chunked256         0.004  0.003"
)

for CFG in "${CONFIGS[@]}"; do
  read -r DS SPLITS ADAMW_LR MUON_LR <<< "$CFG"

  for SEED in "${SEEDS[@]}"; do
    torchrun --standalone --nproc_per_node=$NPROC train_adamw_lm1b_separate.py \
      --seed $SEED --max_steps $MAX_STEPS --eval_steps $EVAL_STEPS \
      --adamw_lr $ADAMW_LR --weight_decay 0.1 \
      --splits_dir $SPLITS \
      --output_dir ${DS}_train/adamw_seed${SEED}
  done

  for SEED in "${SEEDS[@]}"; do
    torchrun --standalone --nproc_per_node=$NPROC train_muon_lm1b_separateqkv_wsd.py \
      --seed $SEED --max_steps $MAX_STEPS --eval_steps $EVAL_STEPS \
      --muon_lr $MUON_LR --weight_decay 0.1 \
      --splits_dir $SPLITS \
      --output_dir ${DS}_train/muon_seed${SEED}
  done
done