set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"


TRAIN_SCRIPT="ood/train_merger_polychain_ood.py"
TOKENIZER="./gpt2_tokenizer"

DATASETS=("enwik8" "stories")
SPLITS_ID=("splits_enwik8_chunked256" "splits_stories_contig_chunked256")
SPLITS_OOD=("splits_stories_contig_chunked256" "splits_wikitext103_chunked256")

for i in "${!DATASETS[@]}"; do
  DS="${DATASETS[$i]}"
  for SEED in 0 1 2 3 4; do
    torchrun --standalone --nproc_per_node=8 ${TRAIN_SCRIPT} \
      --model_dir_0 ./${DS}_train/adamw_seed${SEED} \
      --model_dir_1 ./${DS}_train/muon_seed${SEED} \
      --output_dir  ./${DS}_merge_ood_polychain/adamw_muon_seed${SEED} \
      --tokenizer_dir "${TOKENIZER}" \
      --splits_dir "${SPLITS_ID[$i]}" \
      --ood_splits_dir "${SPLITS_OOD[$i]}" \
      --epochs 10 --batch_size 32 --fp16 --early_stop \
      --sampler UNIFORM --eval_steps 100 --curve polychain
  done
done