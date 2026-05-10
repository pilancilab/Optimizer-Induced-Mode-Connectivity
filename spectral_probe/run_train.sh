#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for SEED in $(seq 0 1); do
  torchrun --standalone --nproc_per_node=8 train_adamw_lm1b_separate.py \
    --seed ${SEED} --max_steps 15000 \
    --adamw_lr 0.001 --weight_decay 0.1 \
    --splits_dir splits_lm1b_contig_chunked256 \
    --output_dir lm1b_train/adamw_seed${SEED} \
    --n_layer 8 --n_embd 1024 --n_inner 4096 --eval_steps 16000
done


for SEED in $(seq 0 1); do
  torchrun --standalone --nproc_per_node=8 train_muon_lm1b_separateqkv_wsd.py \
    --seed ${SEED} --max_steps 15000 \
    --muon_lr 0.005 --weight_decay 0.01 \
    --splits_dir splits_lm1b_contig_chunked256 \
    --output_dir lm1b_train/muon_seed${SEED} \
    --n_layer 8 --n_embd 1024 --n_inner 4096 --eval_steps 16000
done