set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_PC="./enwik8_merge_ood_polychain"   


python ood/plot_barrier_polychain_ood.py \
  --dirs3 \
    ${BASE_PC}/adamw_muon_seed0 \
    ${BASE_PC}/adamw_muon_seed1 \
    ${BASE_PC}/adamw_muon_seed2 \
    ${BASE_PC}/adamw_muon_seed3 \
    ${BASE_PC}/adamw_muon_seed4 \
  --json_name_id  merged_sampler_losses.json \
  --json_name_ood merged_sampler_losses_ood.json \
  --label3 "Mix (polychain)" \
  --color_scheme3 muon \
  --error_type std \
  --output_dir ./ood_barrier_plots/enwik8

BASE_PC="./stories_merge_ood_polychain"   


python ood/plot_barrier_polychain_ood.py \
  --dirs3 \
    ${BASE_PC}/adamw_muon_seed0 \
    ${BASE_PC}/adamw_muon_seed1 \
    ${BASE_PC}/adamw_muon_seed2 \
    ${BASE_PC}/adamw_muon_seed3 \
    ${BASE_PC}/adamw_muon_seed4 \
  --json_name_id  merged_sampler_losses.json \
  --json_name_ood merged_sampler_losses_ood.json \
  --label3 "Mix (polychain)" \
  --color_scheme3 adamw \
  --error_type std \
  --output_dir ./ood_barrier_plots/stories