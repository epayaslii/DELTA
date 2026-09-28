#!/usr/bin/env bash
# Shared by submit_50salads_{baseline,hybrid}.sh -- sourced, not submitted.
# Everything site-specific is an env var with a default; override at submit time:
#   sbatch --export=ALL,CONDA_ENV=myenv,FS_ROOT=/data/50Salads_I3D scripts/slurm/submit_50salads_baseline.sh
#
# Submit from the repo root (Slurm copies the batch script, so paths hang off
# $SLURM_SUBMIT_DIR). One split per array task:  sbatch --array=1-5 ...
# Without --array the job loops over SPLITS sequentially (5 x 80 epochs will likely
# not fit in 12 h -- prefer the array).
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
WLTA="${WLTA:-$REPO/third_party/delta_wlta}"
FS_ROOT="${FS_ROOT:-/home/datasets/50Salads_I3D}"
CONDA_ENV="${CONDA_ENV:-delta}"
SPLITS="${SPLITS:-1 2 3 4 5}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-5e-4}"              # supplementary Table 2 (the group's launcher uses 5e-3)
WD="${WD:-3e-4}"
DRY_RUN="${DRY_RUN:-0}"     # 1 = print the training commands only (used by tests/test_pipeline_dryrun.py)
WANDB="${WANDB:-0}"           # 1 = pass --wandb (compute nodes often have no internet)
OUT_ROOT="${OUT_ROOT:-$REPO/results/slurm}"

echo "[$(date '+%F %T')] host=$(hostname) job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-none}"

# ---- environment -----------------------------------------------------------
if [ "$DRY_RUN" != 1 ]; then
if [ -n "${MODULES:-}" ]; then module load $MODULES; fi        # e.g. MODULES="cuda/12.1 anaconda"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
elif [ -n "${VENV:-}" ]; then
    source "$VENV/bin/activate"
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
[ "${OFFLINE_HF:-1}" = 1 ] && export TRANSFORMERS_OFFLINE=1     # pre-fetch distilbert on the login node
[ "$WANDB" = 1 ] || export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# ---- the patched group code must be in place --------------------------------
[ -f "$WLTA/src/wlta_fixes.py" ] || bash "$REPO/patches/wlta_repro/apply.sh" "$WLTA"
grep -q -- "--ta_source" "$WLTA/src/train_window_tokenizer.py" || { echo "train_window_tokenizer.py is not patched"; exit 2; }
pip install -q -e "$REPO" >/dev/null 2>&1 || true              # makes `hybrid_ta` / `delta.align` importable
fi

# ---- one split -------------------------------------------------------------
# COMMON flags = supplementary Table 2 (50Salads): batch 4, 80 epochs, 8-layer/4-head encoder with
# 512-d input, 4-layer decoder (hidden 256, 4 heads, 20 queries), dropout 0.5, CRF weight 1.0,
# DistilBERT text grounding, gamma = (0.6, 0.01, 1.0), optimizer reset at epoch 30.
run_split () {
    local tag="$1" split="$2"; shift 2
    local out="$OUT_ROOT/${tag}/split${split}"
    local cmd=(python3 src/train_window_tokenizer.py
        -d FS -ac all -c 19 -mpos 19000 -ne "$EPOCHS" -g 0 --seed "$SEED" -f 256
        -lat 0.2 --rho 0.1 -r 0.02 -vf 0 -lr "$LR" -wd "$WD" -ua
        --fs_root "$FS_ROOT" --split "$split" --group "$tag" --path "$out" --mode binary
        --model_type atba --ABLAT_tsm --nofprojections 3 --nseg 0 -bs 4 --dropout 0.5
        --atba_enc_layers 8 --atba_encIn_dim 512
        --LTA_dec_hidden_dim 256 --LTA_dec_n_head 4 --LTA_dec_layers 3 --LTA_dec_n_query 20
        --crf_weight 1.0 --gamma1 0.6 --gamma2 0.01 --gamma3 1.0
        --use_text --text_encoder distilbert)
    [ "$WANDB" = 1 ] && cmd+=(--wandb)
    cmd+=("$@")
    if [ "$DRY_RUN" = 1 ]; then printf 'CMD: '; printf '%q ' "${cmd[@]}"; printf '\n'; return 0; fi
    mkdir -p "$out"
    echo "== ${tag} split ${split} -> $out"
    ( cd "$WLTA" && "${cmd[@]}" ) 2>&1 | tee "$out/train.log"
    # the two numbers we compare across runs (masked MoC; see patches/wlta_repro/README.md)
    grep -E "test_(mean|top1)_moc_obs[0-9.]+_pred[0-9.]+" "$out/train.log" | tail -n 16 > "$out/moc_lines.txt" || true
}

# array task -> its split; otherwise all of them
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SPLITS="$SLURM_ARRAY_TASK_ID"; fi
