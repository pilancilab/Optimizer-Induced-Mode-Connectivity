#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"


MODEL0="./lm1b_train/adamw_seed0"
MODEL1="./lm1b_train/adamw_seed1"
torchrun --standalone --nproc_per_node=8 train_merger.py \
  --model_dir_0 ${MODEL0} \
  --model_dir_1 ${MODEL1} \
  --tokenizer_dir ./gpt2_tokenizer \
  --output_dir ./lm1b_merge_polychain/adamw_seed_0_1 \
  --epochs 0.5 \
  --batch_size 32 \
  --fp16 \
  --splits_dir splits_lm1b_contig_chunked256 \
  --early_stop \
  --sampler NARROW_UNIFORM \
  --eval_steps 1000 \
  --curve polychain


MODEL0="./lm1b_train/adamw_seed0"
MODEL1="./lm1b_train/muon_seed0"
torchrun --standalone --nproc_per_node=8 train_merger.py \
  --model_dir_0 ${MODEL0} \
  --model_dir_1 ${MODEL1} \
  --tokenizer_dir ./gpt2_tokenizer \
  --output_dir ./lm1b_merge_polychain/adamw_muon_seed_0 \
  --epochs 0.5 \
  --batch_size 32 \
  --fp16 \
  --splits_dir splits_lm1b_contig_chunked256 \
  --early_stop \
  --sampler NARROW_UNIFORM \
  --eval_steps 1000 \
  --curve polychain



MODEL0="./lm1b_train_large/adamw_seed0"
MODEL1="./lm1b_train_large/muon_seed0"

torchrun --standalone --nproc_per_node=8 train_merger_muon.py \
  --model_dir_0 ${MODEL0} \
  --model_dir_1 ${MODEL1} \
  --tokenizer_dir ./gpt2_tokenizer \
  --output_dir ./lm1b_merge_polychain_muonbend/adamw_muon_seed_0 \
  --epochs 0.5 \
  --batch_size 32 \
  --fp16 \
  --splits_dir splits_lm1b_contig_chunked256 \
  --early_stop \
  --sampler NARROW_UNIFORM \
  --eval_steps 1000 \
  --curve polychain \
  --use_muon_for_bend \
  --muon_lr 0.001


MODEL0="./lm1b_train_large/muon_seed0"
MODEL1="./lm1b_train_large/muon_seed1"

torchrun --standalone --nproc_per_node=8 train_merger_muon.py \
  --model_dir_0 ${MODEL0} \
  --model_dir_1 ${MODEL1} \
  --tokenizer_dir ./gpt2_tokenizer \
  --output_dir ./lm1b_merge_polychain_muonbend/muon_seed_0_1 \
  --epochs 0.5 \
  --batch_size 32 \
  --fp16 \
  --splits_dir splits_lm1b_contig_chunked256 \
  --early_stop \
  --sampler NARROW_UNIFORM \
  --eval_steps 1000 \
  --curve polychain \
  --use_muon_for_bend \
  --muon_lr 0.001