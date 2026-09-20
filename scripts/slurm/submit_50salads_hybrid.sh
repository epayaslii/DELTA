#!/usr/bin/env bash
#SBATCH --job-name=delta50s-hybrid
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Same run, same losses and eval, but Y* comes from hybrid_ta (--ta_source hybrid).
# Needs per-video similarity files first (offline, once):
#   python scripts/make_hybrid_sim.py --feat-dir <vlm feats> --text-emb <C x D .npy> --out-dir $SIM_DIR --keep-feat
# Ablate the refinement with HYBRID_BRANCHES=none|a|b|ab and HYBRID_METHOD=asot|dp:
#   sbatch --array=1-5 --export=ALL,SIM_DIR=/path/sim,HYBRID_BRANCHES=a scripts/slurm/submit_50salads_hybrid.sh
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/common.sh"
SIM_DIR="${SIM_DIR:?set SIM_DIR to the directory of <video>.npz similarity files}"
HYBRID_METHOD="${HYBRID_METHOD:-asot}"
HYBRID_BRANCHES="${HYBRID_BRANCHES:-ab}"
[ -d "$SIM_DIR" ] || { echo "SIM_DIR not found: $SIM_DIR"; exit 2; }
for s in $SPLITS; do
    run_split "hybrid_${HYBRID_METHOD}_${HYBRID_BRANCHES}" "$s" \
        --ta_source hybrid --hybrid_sim_dir "$SIM_DIR" --hybrid_method "$HYBRID_METHOD" --hybrid_branches "$HYBRID_BRANCHES"
done
