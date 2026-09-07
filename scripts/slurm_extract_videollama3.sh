#!/bin/bash
# VideoLLaMA3 feature extraction for 50Salads (M-A4).
#
# FIRST, once, on a login/interactive node:
#   python scripts/verify_backbone_alignment.py --backbone videollama3 --device cuda
# Do not launch this array until that prints PASS -- see the script's docstring
# for why (we already lost one run to unprojected features).
#
#   sbatch scripts/slurm_extract_videollama3.sh
#
#SBATCH --job-name=vl3-feat-50s
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vl3_extract_%A_%a.out

set -euo pipefail
mkdir -p logs

module load cuda || true
source .venv/bin/activate

# VL3-SigLIP-NaViT's remote code needs transformers 4.x (VideoInput was dropped in v5)
python - <<'PY'
import transformers, sys
major = int(transformers.__version__.split(".")[0])
if major >= 5:
    sys.exit(f"transformers {transformers.__version__} is too new for VL3-SigLIP-NaViT; "
             "pip install 'transformers<5' in this env")
print(f"transformers {transformers.__version__} OK")
PY

# 4-way shard; no --overwrite so re-runs resume where they stopped
python -m delta.features.extract \
    --config configs/50salads.yaml \
    --backbone videollama3 \
    --out-dir data/50salads/features_videollama3 \
    --shard "${SLURM_ARRAY_TASK_ID}/4" \
    --device cuda --dtype bf16 --batch-size 32

# After all shards finish:
#   cat data/50salads/features_videollama3/manifest.shard*of4.jsonl \
#       > data/50salads/features_videollama3/manifest.jsonl
#
# Then score Stage A + Stage B1 against the baselines:
#   python -m delta.align.evaluate --config configs/50salads.yaml --provider naive \
#       --split 1 --ignore-startend
#   python -m delta.align.evaluate --config configs/50salads.yaml --provider asot+refine \
#       --split 1 --ignore-startend --rho 0.5 --alpha 0.1 --radius 90 --window 40 \
#       --frame-dir data/50salads/features_videollama3 \
#       --class-emb data/50salads/features_videollama3/action_name_embeddings.npy
