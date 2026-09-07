#!/bin/bash
# DELTA/WLTA baselines on 50Salads -- the numbers our Y* is compared against.
#
#   sbatch scripts/slurm_delta_baselines.sh          # both model types, split 1
#   sbatch --array=1-5 scripts/slurm_delta_baselines.sh   # all 5 splits
#
# Needs the group's WLTA code at third_party/delta_wlta (gitignored) and its own
# conda env -- python 3.9 / torch 1.11 / cu113 -- NOT the repo's .venv.
# See docs/delta-code.md for what the two model types do.
#
#SBATCH --job-name=delta-baselines
#SBATCH --array=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/delta_baseline_%A_%a.out

set -euo pipefail
mkdir -p logs

module load cuda || true
source activate clot          # py3.9 / torch1.11 / cu113 -- see scripts/hal_setup.md

SPLIT="${SLURM_ARRAY_TASK_ID:-1}"
cd third_party/delta_wlta

for MODEL in atba wclot; do
  echo "=== ${MODEL}, split ${SPLIT} ==="
  python train.py \
      --dataset 50salads \
      --split "${SPLIT}" \
      --model_type "${MODEL}" \
      --exp_name "baseline_${MODEL}_s${SPLIT}" \
      2>&1 | tee "../../logs/wlta_${MODEL}_s${SPLIT}.log"
done

# Both write dense pseudo-labels + LTA metrics. To score Y* with our own
# harness, dump preds/gts to an .npz and run delta.align.segmentation_report --
# recipe in scripts/hal_setup.md section 5.
