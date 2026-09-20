# WLTA 50Salads reproduction patch

Changes to `train_window_tokenizer.py` (+ `atba_loss.py`) (the group's code, gitignored here, so
kept as patches). `apply.sh <delta_wlta dir>` copies `wlta_fixes.py` in and applies
`train_window_tokenizer.patch` and `atba_loss.patch`. Tests: `tests/test_wlta_fixes.py` (helpers) and
`tests/test_pipeline_dryrun.py` (the real model + Trainer on CPU, mock data).

| # | change | flag / default |
|---|---|---|
| 1 | class names read from the same `mapping.txt` the loader uses (was a Breakfast file); `_`→space for DistilBERT | always on |
| 2 | MoC over core classes only; boundary tokens found **by name** | on; `--moc_all_classes` to disable |
| 3 | anticipation decoder depth 4, encoder input 512 | `--LTA_dec_layers 4`, `--atba_encIn_dim 512` (new defaults) |
| 4 | γ exposed; AdamW state + hyper-parameters reset at epoch 30 | `--gamma1 0.6 --gamma2 0.01 --gamma3 1.0 --gamma1_lta 0.8`; `--no_opt_reset` to disable |
| 5 | `--ta_source delta\|hybrid` (`--hybrid_sim_dir`, `--hybrid_method`, `--hybrid_branches`): Y* from `src/hybrid_ta` fed into the same losses | default `delta` = original behaviour |
| 6 | `--fs_root` replaces the hard-coded `/home/datasets/50Salads_I3D` | default = old path |

Notes
- `--num_decoder_layers` is a no-op for `--model_type atba`; the anticipation decoder depth is `--LTA_dec_layers`.
- `--gamma1_lta 0.8` keeps the original code's value; set 0.6 if γ1 should stay 0.6 after stage 2.
- There is no LR scheduler in the code, so "reset the scheduler" has nothing to act on; a warning is printed if one is ever attached.
- Reference command (supplementary Table 2 values; the group's launcher uses `-lr 5e-3`):
  `python3 src/train_window_tokenizer.py -d FS -c 19 -ne 80 -bs 4 -lr 5e-4 -wd 3e-4 --dropout 0.5 --model_type atba --atba_enc_layers 8 --atba_encIn_dim 512 --LTA_dec_hidden_dim 256 --LTA_dec_n_head 4 --LTA_dec_layers 4 --LTA_dec_n_query 20 --crf_weight 1.0 --use_text --text_encoder distilbert` + the launcher's remaining flags.

## Local CPU dry-run (no GPU)

```bash
uv pip install --python .venv/bin/python pytorch-lightning scipy scikit-learn wandb pot rotary-embedding-torch einops transformers
patches/wlta_repro/apply.sh third_party/delta_wlta      # once
.venv/bin/python -m pytest tests/test_pipeline_dryrun.py -v -s     # ~30 s; downloads distilbert once
```

It runs the real `VideoSSL` (8-layer encoder, DistilBERT grounding, 4-layer decoder, CRF, ATBA loss) through a real
`pl.Trainer` for 32 epochs on random [B, 64, 2048] features, so the epoch-30 reset and the stage switch actually fire.
Covers: 50Salads names reach DistilBERT; 2048 -> 512 -> 256 -> 4-layer decoder; Adam step counter 30 -> 1 at epoch 30
(and 31 with `--no_opt_reset`); 17-class masked MoC; `hybrid_ta` shapes/order/recovery; `--ta_source hybrid` training;
and that both Slurm scripts emit commands the training script's own argparse accepts.

## Cluster (Slurm)

```bash
mkdir -p logs                                   # Slurm will not create the --output directory
python -c "from transformers import DistilBertModel, DistilBertTokenizer as T; T.from_pretrained('distilbert-base-uncased'); DistilBertModel.from_pretrained('distilbert-base-uncased')"   # login node: cache the weights
sbatch --array=1-5 scripts/slurm/submit_50salads_baseline.sh
sbatch --array=1-5 --export=ALL,SIM_DIR=/path/sim,HYBRID_BRANCHES=a scripts/slurm/submit_50salads_hybrid.sh
```
Site knobs are env vars (`CONDA_ENV`, `FS_ROOT`, `MODULES`, `LR`, `WANDB`, ...; see `scripts/slurm/common.sh`).
Untested on a real cluster -- only `DRY_RUN=1` command generation is exercised.

## About the boundary ids

The 17-class mask finds `action_start` / `action_end` **by name** in `mapping.txt`. In our 50Salads bundle those are ids
17 and 18 (id 0 is `cut_tomato`); in the MS-TCN alphabetical mapping they are 0 and 1. Masking by position "0 and 18"
would drop a real class here. The test checks both layouts.
