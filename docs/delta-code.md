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

## The training loop, in detail (`src/train.py`, `--model_type atba`)

`VideoSSL(pl.LightningModule)`. Stage boundaries are **hardcoded**:
`stage1to2 = 10`, `stage2to3 = 30` (`train.py:288`). With `-ne 100` for 50S:
epochs 0–10 = stage 1, 11–30 = stage 2, 31–100 = stage 3.

### `__init__` — the `atba` branch
- `in_proj`: `Conv1d(2048 → atba_encIn_dim)` (512 in the run script).
- `enc_cls_token`: `Embedding(n_clusters=19, 512)` — the |C| class tokens.
- `pos_embedding`: `Embedding(max_pos_len, 512)`, or **RoPE** (`--rope`).
- `TAS_encoder`: `atba_enc_layers` (8) transformer layers with **pyramid local
  attention** (`local_r = 2**i` at layer i) — class tokens attend globally,
  frames locally.
- `tas_classifier`: `Linear(512 → 19)`.
- `atba_loss = LossFn(...)` (the ATBA module).
- `lta_model = LTAModel(...)` (the anticipation decoder + CRF).

### `training_step` flow
```
features_raw (B,T,2048)  →  in_proj  →  [cls_tokens ‖ features]  →  TAS_encoder
  →  tas_logits = tas_classifier(features)
       tok_logit = diagonal(tas_logits[:, :19])      # video-level, per class
       fr_logit  = tas_logits[:, 19:]                # (B,T,19) frame logits
  →  atba_out = atba_loss(epoch, tok_logit, fr_logit, mask, transcripts,
                          vid_multi_hot, features, ...)
       atba_tot         = tok + fr + glc  (ATBA's own losses; internal warm-up 10 epc)
       tas_pseudolabels = atba_out['pseudolabels']   # (B,T)  ← ATBA DP on softmax(fr_logit)
  →  features_refined = features[:, 19:, :]           # drop class-token rows
  ── split at obs_len = int(T * perc_obs[b]) ─────────────────────────────
       past_features, past_masks
       future_pseudolabs = tas_pseudolabels[:, obs_len:]
       future_transcripts = compress( future_pseudolabs ) + [NONE], pad to LTA_dec_n_query=20
       future_durations   = _extract_durations( future_pseudolabs[:, obs_len:] )
  →  lta_output = lta_model(clot_output=past_features, input_padding=past_masks,
                            future_transcript=future_transcripts, mode='train')
```

**The future transcript `T*` is `compress(ATBA pseudo-labels on the future
frames)` — NOT the GT transcript.** The alignment defines everything downstream.

### Loss assembly (`atba`)
| stage | condition | `loss` = |
|---|---|---|
| 1 | `epoch ≤ 10` | `atba_tot` |
| 2 | `10 < epoch < 30` | `(atba_tot + 0.5·tsm_loss) · 0.6` |
| 3 | `epoch ≥ 30` | `0.8·(atba_tot + 0.5·tsm_loss)  +  LTA_loss` |

- `tsm_loss` — `TruncatedMeanSquareLoss` on `fr_logit` (log-softmax temporal
  smoothness, clip 4). Skipped if `--ABLAT_tsm`.
- `LTA_loss = 0.2·loss_crf + loss_bacr_next + loss_bacr_prev + loss_duration
  + 0.01·ctc_overview_loss` (`+ loss_future_matrix` for EGTEA only):
  - `loss_crf` — `−CRF.log_likelihood(emissions, T*)` over the 20 query slots.
  - `loss_bacr_next/prev` — KL between the decoder's `anticipate_next` /
    `anticipate_prev` heads and the query sequence shifted by ±1 (a
    self-consistency regulariser on the query chain).
  - `loss_duration` — `MSE(normalize_duration(pred), future_durations)`.
  - `ctc_overview_loss` — `transcript_ctc_loss([obs fr_logit ‖ lta_action], GT
    transcript)` — reconstruct the *full* transcript from observed-frame +
    predicted-future logits. Skipped if `--ABLAT_ctc`.

### The LTA decoder (`src/wlta/wlta.py::LTAModel`)
- project `past_features` → `memory`; `LTA_dec_n_query=20` learnable queries
  (+ query pos-emb) → `TransformerDecoder` (cross-attends `memory`) → `tgt`
  `(B,20,D)`.
- `LTA_classifier` → `future_action` logits `(B,20,19+1)` (+1 = EOS).
- **CRF** (`src/wlta/crf.py`, standard linear-chain, `num_tags=20`,
  `transition_weight=crf_weight=1.0`): train = log-likelihood of `T*`;
  **inference = `crf.decode(top_k)` → the top-k Viterbi future sequences** (this
  is the paper's stochastic / Top-1 protocol).
- `anticipate_next`, `anticipate_prev` heads (BACR).
- **duration head**: `Linear(cat(tgt, softmax(future_action)) → 1)` per query.

### Where the transcript enters (train only)
1. `BoundaryDetector` inside `atba_loss` → `tas_pseudolabels` (via ATBA DP on `softmax(fr_logit)`).
2. `vid_multi_hot` — per-class presence, for `tok_loss`.
3. `future_transcripts` / `future_durations` — derived from `tas_pseudolabels`.
4. `transcript_ctc_loss(..., transcripts)` — CTC against the GT transcript.

At inference (`test_step` / `test_tas_lta.py`): forward pass → `fr_logit` argmax
for the observed part; LTA decoder + `crf.decode` for the future. **No
`atba_loss`, no `BoundaryDetector`, no transcript.**

## `--model_type wclot` (the ASOT path)

`training_step` `wclot` branch: `mlp` (2048→64→128→40) → FUTR-style
`transformer` with `segments_embed` queries → `features_refined` +
`attn_weights` (via `FeatureFusion`). Then **ASOT**:
```
cost_matrix = (1 − normalize(features) @ masked_clusters.T)  +  temporal_prior_weak(transcript)
opt_codes,_ = asot.segment_asot(cost_matrix, monotonic_mask, eps, alpha, radius_gw, ...)
tas_pseudolabels = opt_codes_refined.argmax(dim=2)
```
- `masked_clusters` = learnable `(19,40)` prototype codebook, restricted to the
  transcript's classes.
- `temporal_prior_weak(mask, K, transcripts, rho, mode)` (`asot.py`) — a soft
  per-action positional cost: action *i* is cheap near `linspace` position *i*,
  expensive elsewhere. **This is the only place the transcript order enters the
  cost.**
- `create_monotonic_mask(transcripts, T)` — a max-of-Gaussians soft mask over
  frames, keeping the OT roughly diagonal.
- `segment_asot` — fused-Gromov-Wasserstein OT via mirror descent
  (`n_ot ≈ [25,1]`, `eps`, `alpha`, `radius_gw`, unbalanced on actions `-ua`).
  Returns a soft frame×class plan `T`; `argmax` → pseudo-labels.
- `wclot` losses: `loss_ce`/`loss_ce_refined` (CE between `opt_codes` and the
  softmax cluster codes), `loss_weak` (video-level), `attn_entropy`, `ctc_loss`,
  `tsm_loss`, `loss_fr_lvl`.

## Crossmodal grounding (`src/cross_att.py`) — only in `train_window_tokenizer.py`

- `CrossModalBlock`: `V + σ(gate(V)) ⊙ MultiheadAttention(query=V, key=T, value=T)`
  — video features query the text tokens, gated residual. (`train.py` does
  **not** use this; the base 50S run has no semantic grounding.)
- `build_text_tokens_per_action_distilbert(transcripts, id2name, tokenizer, txt_model)`
  → **one DistilBERT `[CLS]` embedding per transcript action name**, `(B,S,H)`.
- `build_windows_from_pseudolabels(tas_pseudolabels, transcripts, radius=8)`
  → per-action temporal windows `(B,S,T)` = the paper's local mask `M`.

**DELTA already builds "one text vector per transcript step" and cross-attends
video to it, masked by the pseudo-label windows** — with DistilBERT, and only as
a late-fusion feature refinement, not as the alignment driver.

## Where the VLM similarity plugs in — precisely

Let `s ∈ ℝ^{N×T}` = `cos( g_text(action_n) , f_vis(frame_t) )` from a frozen VLM
(N = transcript length). Two clean insertion points, one per alignment mode:

**`--model_type wclot` — one line.** Replace the feature-vs-prototype term:
```
cost_matrix = (1 − s.transpose)  +  temporal_prior_weak(transcript)   # instead of 1 − normalize(features) @ masked_clusters.T
```
Keep `segment_asot`, `monotonic_mask`, `temporal_prior_weak`, and every
downstream loss. Optionally drop the learnable `clusters` codebook entirely
(the text embeddings replace it). `train.py:548, 555`.

**`--model_type atba` — inside `BoundaryDetector` (`src/atba_loss.py`).**
`get_cls_score` reads `norm_prob[:, [first_seg, second_seg]]` — i.e. exactly
`s[first_seg]` and `s[second_seg]`. `get_bdy_score` needs a per-frame
distribution — feed `softmax_n(s / τ)` (column-wise) or the VLM frame
embeddings' self-similarity. The DP (`dp()`) and the pseudo-label construction
are unchanged.

**Grounding (`train_window_tokenizer.py`):** swap
`build_text_tokens_per_action_distilbert` → a VLM text tower (`delta.features.text_encoder`).

Everything after `tas_pseudolabels` — the stage schedule, TSM, CTC, the LTA
decoder, CRF, duration head, `test_tas_lta.py`, `metrics.py` — is untouched.

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
2. On the cluster: conda env (`requirements_env.txt`), arrange
   `data/FS/{features,groundTruth,mapping.txt,splits}` from our `data/50salads/`
   bundle, run `python3 src/train.py -d FS -c 19 ... --model_type atba` (map the
   run-script flags — the script's `train_edit_elena9sept.py` doesn't exist) for
   split 1 → baseline MoC under both `atba` and `wclot`.
3. Insertion point is now known exactly (see "Where the VLM similarity plugs
   in"): `wclot` = swap the cost-matrix feature term at `train.py:548/555`;
   `atba` = feed `s` into `BoundaryDetector.get_cls_score` / `get_bdy_score`.
4. Instrument `training_step` to dump `tas_pseudolabels` per video → score with
   `delta.align` (MoC / edit / F1@k / boundary offset) vs our naive floor.
