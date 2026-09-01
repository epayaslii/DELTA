# The DELTA / "WLTA" code — what it is and how it runs

The professor's group shared the DELTA implementation (internally **WLTA** =
Weakly-supervised Long-Term Anticipation). Local copy:
`third_party/delta_wlta/` (gitignored). Snapshot dated **Oct 2025**,
research-grade.

## What it is

DELTA is built **on top of the CLOT / ASOT codebase** (both from Dimiccoli's
group at IRI — CLOT = Bueno-Benito & Dimiccoli ICCV'25, ASOT = Xu & Gould
CVPR'24). That inheritance explains a lot of the code:

| file | role |
|---|---|
| `src/train.py` (97 KB) | main training entrypoint (LightningModule). Base model + `--rope`. |
| `src/train_window_tokenizer.py` (84 KB) | variant: adds a window tokenizer + `--use_text` (DistilBERT crossmodal grounding) |
| `src/atba_loss.py` | **the TA module** — `BoundaryDetector` (7×7 class-agnostic kernel, JS-divergence `pair_sim`, NMS candidate selection, drop-allowed `dp()`), `LossFn` (token/video-level BCE + frame-CE on pseudo-labels + global-local InfoNCE, `warm_epc=10`). Matches `docs/temporal-alignment.md` §2 exactly. |
| `src/asot.py`, `src/gsw.py` | **ASOT** optimal-transport alignment + Gromov-Wasserstein structure term — the *alternative* alignment (`--model_type wclot`) |
| `src/cross_att.py` | crossmodal cross-attention (locally-masked grounding) |
| `src/wlta/wlta.py`, `src/wlta/crf.py` | LTA parallel decoder + linear-chain CRF |
| `src/ctc_loss.py` | `transcript_ctc_loss` (sequence-order consistency) |
| `src/tsm_loss.py` | `TruncatedMeanSquareLoss` — temporal-smoothness regulariser (ASOT/MS-TCN style) |
| `src/transformer_atba.py`, `src/transformer_futr.py` | the two encoder/decoder backbones |
| `src/video_dataset.py` | `VideoDataset_splits` — data loading |
| `src/metrics.py` | MoF, mIoU, F1, **JSD** (segment-duration histogram JS divergence), edit, F1@{10,25,50}, `calculate_moc` |
| `src/test_tas_lta.py` | `predict_long` — the LTA anticipation eval (Obs%/pred% grid) |
| `run_*.sh` | per-dataset driver scripts (see below) |

**Two alignment modes** (`--model_type`): `wclot` (ASOT optimal transport) and
`atba` (ATBA boundary detector). **The 50Salads run scripts use `--model_type
'atba'`.**

## How the tests / experiments run

`run_50S_allmetrics*.sh <exp_name>`:
1. loops `splits = 1..5`, `seed 0`
2. per split, runs the training command (below), teeing stdout to
   `results/50S/model_anticipation_50S-<exp>/tmp_logs/split_<k>.log`
3. `parse_split_metrics` greps the log for `test_mean_moc_obsX_predY` /
   `test_top1_moc_obsX_predY` lines (wandb output) → per-split JSON
4. accumulates segmentation numerators/denominators into
   `50S_frames_accumulated.json` (the python writes these via `--path`)
5. after all splits: computes `MOF / F1 / mIOU / JSD` (%) into
   `50S_final_metrics.json`, and averages the MoC grid into
   `50S_anticipation_metrics.json`

**50Salads training command** (from `run_50S_allmetrics.sh`, mid-level):
```
python3 src/train.py -d FS -ac all -mpos 19000 -c 19 -ne 100 -g <gpu> --seed 0 \
  -f 256 -lat 0.2 --rho 0.1 -r 0.02 -vf 0 -lr 5e-4 -wd 1e-4 -ua \
  --num_decoder_layers 4 --nofprojections 3 --nseg 0 --dropout 0.1 \
  --split <k> --mode binary --model_type atba --rope --rope_variant standard \
  --LTA_dec_hidden_dim 256 --LTA_dec_n_head 4 --LTA_dec_layers 3 --LTA_dec_n_query 20 \
  --crf_weight 1.0 -bs 4 --wandb --group <name> --path <accum.json>
```
Text/tokenizer variant (`run_50S_allmetrics_tokenizer_window_best.sh`) instead:
`src/train_window_tokenizer.py ... --ABLAT_tsm --atba_enc_layers 8
--atba_encIn_dim 512 --use_text --text_encoder distilbert`.

Flag glossary (inherited from ASOT — names are misleading):
- `-ua` / `--ub-actions` = **unbalanced OT on actions**, not "use alignment"
- `-lat/-lae`, `--eps-*`, `-r` (radius-gw), `--rho`, `--n-ot-*`,
  `--nofprojections`, `--nseg` = **ASOT / Gromov-Wasserstein solver params**
- `-f 256` = **sample 256 frames per video** for train/val (aggressive
  downsample; 50S videos are ~11.5k frames). `-mpos 19000` = pos-embedding table size.
- `-c 19` (mid) / `-c 12` (`-d FSeval`, eval granularity)
- `-vf 0` = no periodic validation

## Data format

`VideoDataset_splits` expects (Kukleva `unsup_temp_embed` HOWTO layout):
```
data/FS/
  features/<vid>.npy         # (D, T), loaded as .T -> (T, D); .txt also accepted
  groundTruth/<vid>.txt      # one action-name per line   (note: loader also
                             #   accepts no-extension names)
  mapping.txt                # "<id> <name>"   (mapping/mapping.txt for BF/YTI;
  mappingeval.txt            #   mappingeval.txt for -d FSeval)
```
`-d FS` == 50Salads ("Fifty Salads"), `-d FSeval` == 50Salads eval granularity.
Per-video standardisation (`--std-feats`) + `/= sqrt(D)`.

**Our `data/50salads/` (from `dinggd/50salads`) is compatible** — 2048-d I3D,
`(D,T)` npy, `groundTruth/*.txt`, `mapping.txt`, 19 classes. Only need to
arrange it as `data/FS/` (or point `--base-path`/`--root`) and add
`mappingeval.txt` if running `FSeval`.

## The "situation" — gotchas

1. **`README.md` is CLOT's, not DELTA's** (wrong doc; conda env named `clot`).
2. **The run scripts call `src/train_edit_elena9sept.py`** — a personal
   working file **not in the snapshot**. Only `train.py` and
   `train_window_tokenizer.py` exist. The scripts' flags map cleanly onto
   `train.py`; use that.
3. Hardcoded absolute paths (`/home/ebueno/datasets/breakfast_action`,
   `/home/datasets/clot_transcripts/...`). Need `--base-path` / `--root`.
4. Requires **Linux + CUDA + conda + a wandb account** (`--wandb` in every run
   script; metrics are parsed out of wandb log lines). Not Mac-runnable as-is.
   `requirements_env.txt` is a full conda export (`conda create --name clot
   --file requirements_env.txt`).
5. Two granularities for 50S: `FS` (`-c 19`, mid) and `FSeval` (`-c 12`). The
   paper's Table 1 numbers are the eval protocol.
6. Metrics come out as ratio pairs `[num, den]` accumulated across splits by the
   bash script, not printed directly by python.

## What this changes for our project

- **DELTA already has both alignment mechanisms** — ATBA boundary detector
  *and* ASOT optimal transport. Our "monotonic/OT alignment" is partly already
  here (`asot.py`, `gsw.py`, the `dp()` in `atba_loss.py`).
- So the **VLM-direct contribution narrows cleanly**: replace the *frame
  classifier posteriors* (`fr_logit` → `prob` in `BoundaryDetector.forward`,
  and the cost matrix fed to ASOT) with a **frozen-VLM transcript×frame
  similarity**. Everything downstream (CTC, CRF, TSM, LTA decoder, the whole
  eval harness) is reused unchanged.
- We can now **run the real baseline** on the cluster and get true 50Salads
  numbers under both `--model_type atba` and `wclot`, instead of
  reconstructing them.
- Our `src/delta/align/{ta,similarity,evaluate}.py` stays useful as the clean,
  tested, CPU-side prototype of the VLM-direct aligner; the integration target
  is `third_party/delta_wlta/src/atba_loss.py` + `asot.py`.

## Next

1. Get raw 50Salads video (still the blocker for VLM features).
2. On the cluster: conda env, arrange `data/FS/`, reproduce
   `run_50S_allmetrics.sh` split 1 → a baseline MoC.
3. Locate where `prob` / the OT cost matrix is built; swap in VLM similarity.
