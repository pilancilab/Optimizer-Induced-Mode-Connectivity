#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PAIRS=("0_1" "2_3" "4_5" "6_7" "8_9")
DATASETS=(enwik8 wikitext103 stories bookcorpus lm1b)

for DS in "${DATASETS[@]}"; do
  for OPT in adamw muon; do
    DIRS=()
    for P in "${PAIRS[@]}"; do
      DIRS+=("./${DS}_merge/${OPT}_seed_${P}")
    done

    python eval_multi_all_color.py \
      --merged_model_dirs "${DIRS[@]}" \
      --coeff_start 0.0 --coeff_end 1.0 --coeff_step 0.1 \
      --error_type std --color_scheme $OPT \
      --output_dir merger_eval_results_${OPT}_${DS}
  done
done