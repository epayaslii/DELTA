#!/usr/bin/env bash
#SBATCH --job-name=delta50s-base
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# DELTA baseline (--ta_source delta: ATBA boundary detector + drop-DP), 50Salads, 5 splits.
#   sbatch --array=1-5 scripts/slurm/submit_50salads_baseline.sh      # one split per task (recommended)
#   sbatch scripts/slurm/submit_50salads_baseline.sh                  # all splits in one job
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/common.sh"
for s in $SPLITS; do run_split baseline "$s" --ta_source delta; done
