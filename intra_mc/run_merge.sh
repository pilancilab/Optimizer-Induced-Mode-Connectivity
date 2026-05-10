#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NPROC=8
TOKENIZER="./gpt2_tokenizer"
COMMON="--batch_size 32 --fp16 --early_stop --sampler NARROW_UNIFORM --eval_steps 100 --curve polychain"

# Helper: run one merge
merge() {
  local m0=$1 m1=$2 out=$3 splits=$4 epochs=$5
  torchrun --standalone --nproc_per_node=$NPROC train_merger.py \
    --model_dir_0 "$m0" --model_dir_1 "$m1" \
    --tokenizer_dir $TOKENIZER --output_dir "$out" \
    --splits_dir "$splits" --epochs "$epochs" $COMMON
}

#       dataset      splits_dir                          epochs
CONFIGS=(
  "enwik8      splits_enwik8_chunked256              10"
  "wikitext103  splits_wikitext103_chunked256          10"
  "stories     splits_stories_contig_chunked256       10"
  "bookcorpus  splits_bookcorpus_contig_chunked256    1.0"
  "lm1b        splits_lm1b_contig_chunked256          1.0"
)

SAME_PAIRS=("0 1" "2 3" "4 5" "6 7" "8 9")
CROSS_SEEDS=(0 1 2 3 4)

for CFG in "${CONFIGS[@]}"; do
  read -r DS SPLITS EPOCHS <<< "$CFG"

  # Same-optimizer: AdamW–AdamW, Muon–Muon
  for OPT in adamw muon; do
    for PAIR in "${SAME_PAIRS[@]}"; do
      read -r S0 S1 <<< "$PAIR"
      merge "./${DS}_train/${OPT}_seed${S0}" \
            "./${DS}_train/${OPT}_seed${S1}" \
            "./${DS}_merge/${OPT}_seed_${S0}_${S1}" \
            "$SPLITS" "$EPOCHS"
    done
  done

  # Cross-optimizer: AdamW–Muon
  for SEED in "${CROSS_SEEDS[@]}"; do
    merge "./${DS}_train/adamw_seed${SEED}" \
          "./${DS}_train/muon_seed${SEED}" \
          "./${DS}_merge/adamw_muon_seed_${SEED}" \
          "$SPLITS" "$EPOCHS"
  done
done