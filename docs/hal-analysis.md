# HAL analysis — can it be a stronger TA baseline than ATBA for DELTA?

**HAL** = *Hierarchical Action Learning for Weakly-Supervised Action Segmentation*,
Huang et al., **CVPR 2026** (`arXiv:2602.24275`). Official code:
`github.com/DMIRLAB-Group/HAL` (inspected directly).

Tags: **[PAPER]** · **[CODE]** (official repo) · **[INFER]** · **[UNKNOWN]**.

---

## Bottom line

**HAL is not a different way of doing temporal alignment — HAL *is* ATBA plus a
small variational auxiliary regulariser.**

- `models/model.py` header: *"This file was adapted from … CVPR24_ATBA"*. **[CODE]**
- `models/loss.py::BoundaryDetector` is **byte-for-byte identical to ATBA** (and to
  DELTA's `third_party/delta_wlta/src/atba_loss.py`): same 7×7 kernel, JS-divergence
  `pair_sim`, NMS candidate selection, `cs_kernel=31` transition score, drop-allowed
  `dp()`. **[CODE]**
- `options.py` carries every ATBA hyper-parameter under its ATBA name
  (`--bdy-kernel` = "wb", `--bdy-scale` = "mu", `--cs-kernel` = "wa",
  `--candidate-mul` = "lambda", `--cts-temp` = "tau", `--warm-epc 40`). **[CODE]**
- HAL adds exactly 5 knobs: `--z_layer, --layer_nums, --diff_weight, --kl_weight,
  --rec_weight`. **[CODE]**

So "replace DELTA's ATBA-based TA with HAL" = **"keep DELTA's ATBA TA and add
HAL's dual-latent VAE branch + 3 auxiliary losses"** — an additive ablation, not a
module swap.

**Recommendation:** proceed with HAL, but as a **cheap one-run ablation** on top of
the ATBA/DELTA baseline — not a research thrust, not before the ATBA baseline, and
measured by **pseudo-label MoC + boundary offset**, not MoF. If the gain is within
seed noise (±1), drop it and go to the VLM work.

---

## What HAL adds to ATBA

**Encoder** (`models/model.py`): ATBA's encoder (Conv1d in_proj, learnable pos-emb,
|C| class tokens prepended, pyramid local attention `local_r = 2**i`), **unchanged**.
At two adjacent encoder layers `z_layer`, `z_layer+1` (default 1, 2) a parallel VAE
branch (`transformer_var` + MLP `decoder`) computes `mu, logvar`, reparametrises to a
latent `z`, and reconstructs the layer input. Yields `z_2` (fast / visual, shallower
layer) and `z_1` (slow / action, deeper layer). **The classifier still reads the
final encoder tokens; `z_1, z_2` feed only the aux losses.** At inference the VAE
branch is skipped entirely.

**Losses** (`models/loss.py::LossFn`):
`L = α·tok + β·fr + γ·glc + rec_weight·recon + kl_weight·kl + diff_weight·diff`
- `tok, fr, glc` — ATBA's (video-level BCE; frame-CE on ATBA pseudo-labels; global-
  local InfoNCE). α=1, β=1, γ=0.1.
- **`recon`** — MSE(layer input, VAE reconstruction). weight **0.1**.
- **`kl`** — standard VAE KL for `z_1, z_2`. weight **1e-3**.
- **`diff`** (the "sparse transition constraint") — `ReLU(slow_change − fast_change)
  + σ·slow_change` on L2-normalised frame-to-frame differences of `z_1` vs `z_2`;
  penalises the slow latent changing faster than the fast one. weight **1e-3**, σ=0.1.

Two-stage: `warm_epc=40` (vs ATBA/DELTA's ~10) — before that only `tok + kl + recon`;
after, add `fr + glc + diff`. 400 epochs, AdamW lr 5e-4, cosine decay.

---

## Inputs / outputs / supervision

- **Supervision:** transcripts only (`transcripts/<name>.txt` loaded from disk). No
  frame labels. Same weak setting as ATBA and DELTA's TA. **[PAPER + CODE]**
- **Features:** frozen pre-extracted **2048-d I3D** (3200 for CrossTask), `(D,T)` npy.
  Encoder trained, extractor frozen. **[CODE]**
- **Output — training:** `fr_cls (T,C)`, `tok_cls`, `z_1, z_2`, and internally ATBA's
  `pseudo_labels (T,)`. **Output — inference:** `argmax(interpolate(fr_cls))` — one
  label per raw frame. **No boundary list, no alignment matrix, no pseudo-label
  export at test.** **[CODE]**
- No explicit boundary head; over-segmentation is fought by `diff_loss`. Ambiguous
  transitions handled exactly as ATBA. **[CODE]**

## Datasets & metrics

- **Datasets:** Breakfast, Hollywood Extended, CrossTask, GTEA. **50Salads is NOT
  evaluated, and `datasets.py` asserts `dataset_name in
  ('breakfast','hollywood','crosstask','gtea')`** — instantiating with `'50salads'`
  raises (there are vestigial 50salads branches in `model.py`/`loss.py`, but no
  config/splits/results). **[PAPER + CODE]**
- **Metrics:** MoF, MoF-Bg, IoU, IoD (ISBA + TASL conventions). **No edit score, no
  F1@k, no boundary-localisation metric.** **[CODE]**
- **Results vs ATBA** **[PAPER]**: Breakfast MoF 53.9→56.3 (+2.4), IoU 41.1→42.6;
  CrossTask +3.4 MoF; Hollywood +3.3; GTEA +3.0 (within ±4–5 std). Modest.
- HAL optimises **MoF**; DELTA's headline is **MoC** (per-class). A MoF gain need
  not be a MoC gain — MoF is dominated by long frequent actions. **[INFER]**

## Official codebase

```
main.py options.py datasets.py test.py train.py utils.py get_transcript.py
models/{model.py, loss.py, transformer.py}
evaluation/{isba_code.py, tasl_code.py}   # SAME eval files as DELTA/WLTA
```
Python 3.9.23, PyTorch 1.11.0+cu113, single GPU, TensorBoardX (not wandb). **No
pretrained weights released.** Data: Breakfast/Hollywood/CrossTask features from
ATBA's Google Drive; GTEA processed via `get_transcript.py`. Layout
`data/<ds>/{features,groundTruth,transcripts,splits,mapping.txt}` — **needs
`transcripts/` on disk** (ATBA/DELTA derive them). Minor code issues:
`Model.__init__` reads undefined `args.lags`; dead scheduler code; double import.
**[CODE]**

Training: `python main.py --dataset breakfast --split 1 --sample-rate 10 --seed 0
--epoch 400 --cs-kernel 31 --rec_weight 0.1 --diff_weight 1e-3 --kl_weight 1e-3
--n-encoder 5`. Eval: add `--test --ckpt <name> --save`.

---

## ATBA vs HAL — what actually differs

| aspect | ATBA | HAL |
|---|---|---|
| feature type, encoder, boundary/transition scoring, NMS, DP, pseudo-labels, inference, metrics | — | **identical** |
| supervision losses | tok + fr + glc | **+ recon (0.1) + kl (1e-3) + diff (1e-3)** |
| latent structure | none explicit | dual-scale VAE latents at 2 encoder depths (train only) |
| warm-up | ~10 epc | 40 epc |
| datasets | BF / Hollywood / CrossTask | BF / Hollywood / CrossTask / GTEA (never 50S) |

In one line: HAL trains ATBA's encoder to also emit a slowly-varying latent (VAE +
a slow-≤-fast change-rate penalty), hoping for cleaner `fr_logit` → slightly better
ATBA pseudo-labels.

---

## Can HAL replace DELTA's TA? — the six axes

| axis | HAL vs ATBA | relevance to DELTA's TA |
|---|---|---|
| 1. WSAS MoF | +2–3 | weak proxy — DELTA wants MoC + future-transcript quality |
| 2. alignment mechanism | **identical** (same DP on same posteriors) | HAL changes the *inputs* to alignment, not the alignment |
| 3. pseudo-label quality | slightly better *if* the encoder regularisation transfers | the one plausible win, and it's small |
| 4. boundary localisation | same DP → same failure modes; **does not fix the 50Salads weak-`bdy_score` problem** (I3D 1.11× at boundaries) | no help where DELTA needs it |
| 5. DELTA compatibility | **very high** — shared ATBA lineage, shared eval code, same feature format, same schedule shape | integration = "add losses", ~70 lines |
| 6. downstream DLTA | **[UNKNOWN]** — never fed an anticipation decoder by anyone | no evidence either way |

**Verdict:** integrable and a legitimate stronger *segmentation* baseline, but **not
a TA replacement** in any architectural sense, and it **does not advance the VLM
direction** — HAL still aligns *through* the frame classifier.

## Adapter (do NOT build yet)

No HAL→DELTA adapter — integration means editing DELTA's encoder + `atba_loss.py`:
1. add the two VAE taps to the encoder (~40 lines, mirror `HAL/models/model.py::compute_latent`), return `z_1, z_2` during training only;
2. add `recon + kl + diff` to the loss (~30 lines from `HAL/models/loss.py`);
3. nothing downstream changes (`Y*`, `T*`, `d*`, mask, CTC, CRF, LTA decoder, eval);
4. new hyper-params `z_layer, rec_weight, kl_weight, diff_weight, sigma`; likely raise DELTA's `warm_epc` to ~40.
Standalone-HAL on 50Salads additionally needs: widen the loader assert, build
`data/50salads/{transcripts,splits,mapping}` (transcripts via `delta.data.to_transcript`).

## How HAL informs the eventual VLM method

- **Slow/fast disentanglement is the right prior — a frozen VLM gives the slow
  variable for free** (`g_text(action_n)` = one semantic vector per transcript step;
  no VAE, no identifiability theory needed). HAL is empirical evidence the axis matters.
- **The `diff_loss` change-rate penalty is reusable** to regularise a VLM aligner's
  soft-assignment sequence against over-segmentation.
- The video-level (`tok_loss`) and contrastive (`glc_loss`) terms carry over — the
  VLM version replaces the *learned* class prototype with the *frozen text embedding*.
- Do **not** claim a multimodal lineage — HAL has no text encoder; "transcript" is
  class indices only.

## Recommended first experiment

**Target: Breakfast** (the only dataset where reproduction is verifiable against a
published HAL *and* ATBA number; both run out of the box; fast iteration).
50Salads has no HAL reference or config — flying blind.

1. Reproduce **ATBA** on Breakfast split 1 → ≈53.9 MoF (pipeline works).
2. Reproduce **HAL** on Breakfast split 1 → ≈56 MoF (HAL works & reproduces).
3. Instrument both loops to dump per-video ATBA `pseudo_labels` + boundaries.
4. Score `Y*` with `delta.align`: **MoC**, edit, F1@{10,25,50}, **median per-transition
   boundary offset** — ATBA-`Y*` vs HAL-`Y*` vs naive-uniform floor.
5. **Gate:** if HAL-`Y*` MoC/offset beats ATBA-`Y*` beyond ±1 seed noise → worth a
   DELTA integration ablation. Else → skip, cite the numbers, go to VLM.

Never start with a full 5-split DELTA training with HAL swapped in.
