#!/bin/bash
#SBATCH --job-name=delta-feat-50s
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/extract_50s_%A_%a.out

set -euo pipefail
module load cuda || true
source .venv/bin/activate

# 8-way shard; --overwrite off so re-runs resume
python -m delta.features.extract \
    --config configs/50salads.yaml \
    --shard "${SLURM_ARRAY_TASK_ID}/8" \
    --device cuda --dtype bf16 --batch-size 64

# after all shards finish, merge manifests:
#   cat data/50salads/features_vl3siglip/manifest.shard*of8.jsonl > data/50salads/features_vl3siglip/manifest.jsonl
