# Baselines & references: HAL, CVA, MASRA

Legend: **[PAPER]** · **[CODE]** (official repo) · **[INFER]** · **[UNKNOWN]**.

## ⚠️ Scope update — 2026-09-02: HAL → MASRA

Supervisor's direction: **the TA must use a VLM; no segmentation-based approach.**
Consequences:
- **HAL is dropped as a workstream.** It is ATBA + a variational regulariser —
  still "alignment *through* a trained frame classifier". Kept only as a
  *baseline number* (its paper's Breakfast MoF; it never ran 50Salads).
- **MASRA workstream** (this doc, `masra` section). CVA is the parallel workstream.
  Both are **VLM video-text alignment** (from grounding), adapted to
  transcript-supervised 50Salads.
- Method: frozen **InternVideo2** transcript×frame similarity → **MASRA-style
  LRCA/ESTA regularisation** → order-preserving alignment (**ASOT**, already in
  DELTA as `--model_type wclot`) → `Y*` → DELTA decoder.
- The "Phase T gate → Phase D" structure and the TA-metrics→DLTA-metrics chain
  still hold; just the method inside changed (VLM+MASRA instead of HAL).

The HAL analysis below is retained as the record of *why* it was dropped.

---

## The two axes

```
                  transcript → dense frame labels        NL query → one span
                  (= our task's TA stage)                (adjacent task, supervised)
 classifier-based ┌─────────────────────────┐
 alignment        │  ATBA (CVPR'24) → HAL   │            (n/a)
                  └─────────────────────────┘
 VLM / semantic   ┌─────────────────────────┐            ┌──────────┐
 alignment        │   ← OUR CONTRIBUTION →   │ ◄──ideas───│   CVA    │
                  └─────────────────────────┘            └──────────┘
```

- **HAL** — same task, same supervision as us. Baseline for the pseudo-label /
  segmentation stage.
- **CVA** — different task (video temporal grounding), fully supervised, but it
  *is* "align through a VLM, done well and SOTA". Methodological reference +
  transferable components, not a number we beat directly.

Papers: `~/Desktop/HAL_CVPR.pdf`, `~/Desktop/cva_cvpr.pdf`.

---

## HAL — same-task baseline

Full analysis: [`hal-analysis.md`](hal-analysis.md). CVPR PDF confirms the arXiv
version — nothing new.

- **HAL = ATBA + a variational regulariser.** `L_total = L_y^ATBA − α·ELBO + β·L_s`.
  The boundary detector, DP, pseudo-labels, and `L_y` (tok + fr + glc) are
  ATBA's, unchanged. **[PAPER + CODE]**
- Additions: a two-scale VAE tap (`z_1` slow "action" latent, `z_2` fast
  "visual" latent) at adjacent encoder layers; `ELBO` = MSE reconstruction + KL;
  **smoothness transition constraint**
  `L_s = ReLU(Σ w_c·ΔC − Σ w_v·ΔV) + δ·Σ w_c·ΔC`
  on L2-normalised latents (penalises the slow latent changing faster than the
  fast one; `w = SoftMax(Δ)`). Plus block-wise identifiability theory. **[PAPER]**
- **Datasets:** Breakfast, CrossTask, Hollywood, GTEA — **no 50Salads**;
  `datasets.py` asserts against it. **[CODE]**
- **Results vs ATBA (Breakfast):** MoF 53.9→**56.3**, MoF-Bg 54.4→57.2,
  IoU 41.1→42.6, IoD 61.7→62.4. CrossTask +3.4 MoF, Hollywood +3.3, GTEA +3.0
  (within ±4–5 std). **No edit, no F1@k, no boundary metric.** **[PAPER]**
- Code: `github.com/DMIRLAB-Group/HAL`, Python 3.9 / PyTorch 1.11, TensorBoardX,
  single GPU, no pretrained weights. Needs `transcripts/` on disk. **[CODE]**

**Role for us:** baseline #2 for `Y*` quality (ATBA = #1, naive-uniform = floor).
Integration into DELTA = add `ELBO + L_s` to the `--model_type atba` path
(~70 lines, additive; nothing downstream changes). It does **not** advance the
VLM direction — still aligns *through* the frame classifier. Its slow/fast prior
is an **argument for** the VLM method: a frozen text embedding per transcript
step *is* the slow action variable, obtained directly, no VAE.

---

## CVA — the "VLM alignment done right" reference

**Task:** Video Temporal Grounding (VTG) — given a video `V` and a natural-
language query `Q`, predict the moment span `m̂ = (ĉ, σ̂)` + per-clip saliency.
Joint Moment Retrieval + Highlight Detection. **Fully supervised** (GT spans).
Not transcripts, not dense labels, not our benchmarks. **[PAPER]**

**Features:** frozen **SlowFast + CLIP** vision, **CLIP text**. DETR-style
decoder with learnable span queries + bipartite matching. **[PAPER]**

**Datasets:** QVHighlights, Charades-STA, TACoS. **Results:** SOTA, e.g.
QVHighlights R1@0.7 **55.32** (prev best TD-DETR 50.37, +4.95); TACoS mIoU
41.07; Charades-STA R1@0.5 62.61. **[PAPER]**

**Code:** `github.com/byeol3325/CVA_cvpr` (linked from the project page;
population status **[UNKNOWN]**).

### The three components — and what's transferable

1. **QCD — Query-aware Context Diversification** (data aug). Mix background
   (non-moment) clips from other videos, but only clips whose **CLIP similarity
   to the query** falls in `[Percentile_α(non-GT), Percentile_β(GT)]` — avoids
   "false negatives" from mixing in semantically-related clips. Preserves a
   `p`-clip window around the GT boundary. `α=10, β=60, ratio 0.3, p=1`.
   *Attacks: "models over-associate the query with the static background."*
   → **For us:** on 50Salads this is "don't align `add_oil` to the whole
   cluttered board". A background-suppression term or transcript-aware
   augmentation.

2. **CTE — Context-enhanced Transformer Encoder** (architecture). `N_b` blocks;
   each: **windowed self-attention** on video (`L/W` non-overlapping windows,
   local) + **global self-attention** on learnable queries + **bidirectional
   cross-attention** (video↔queries) + FFN + residual. Concatenate the video
   output of every block → multi-scale feature; learnable weighted sum with the
   raw features: `F_CTE = ω·F_v + (1−ω)·Norm(MLP(F_b))`. Then a multimodal
   encoder (cross-attn / self-attn / 1D-conv / self-attn) fuses `F_CTE` with the
   text features.
   → **For us:** a better encoder for frozen VLM features before alignment than
   DELTA's pyramid-local-attention — local windows + learnable global queries +
   bidirectional cross-attention.

3. **CBD — Context-invariant Boundary Discrimination loss** (the key idea).
   Given two QCD augmentations of the same video, for each GT-span boundary
   index `b ∈ {min(G), max(G)}`: anchor `z_b = MLP(f'_{m,b})`, positive
   `z⁺_b = MLP(f''_{m,b})` (same index, other augmentation), negatives from
   (a) **temporally adjacent background** `|j − b| ≤ N_adj` and
   (b) the **`N_hard` most cosine-similar background clips** elsewhere.
   InfoNCE: `L_CBD = −(1/|B|) Σ_b log[ exp(s_p,b/τ) / (exp(s_p,b/τ) + Σ exp(s_n,b/τ)) ]`.
   Weight `λ_CBD = 0.005`.
   → **For us:** the most directly transferable piece. Our aligner produces
   boundary frames; a CBD-style loss says "the transition frame's representation
   is invariant to what surrounds it, and distinct from adjacent + look-alike
   non-transition frames." Fights over-segmentation and the fixed-camera
   ambiguity at once. No GT spans needed — the aligned pseudo-boundaries + our
   own augmentations suffice.

### Ablation (QVHighlights, R1@0.7): baseline 46.77 → +QCD 51.98 → +CTE 52.63 → +CBD 53.02 → all 54.84.

**Role for us:** the aligner reference (parallel workstream). CBD + CTE are the
transferable ingredients.

---

## MASRA — Eliz's focus  (`~/Desktop/MASRA_CVPR.pdf`, arXiv:2605.03398, 2026)

**MLLM-Assisted Semantic-Relational Consistent Alignment for Video Temporal
Grounding.** Zhang/Ran et al. **ACM-track** (© 2026 ACM; likely ACM MM) — *not*
CVPR, despite the local filename. **[PAPER]**

- **Task:** video temporal grounding (NL query → one span), **fully supervised**
  (GT spans). Features: SlowFast + CLIP. Datasets: QVHighlights, Charades-STA,
  TACoS, ActivityNet.
- **Contribution = a training-time language regulariser** (not an aligner —
  structurally like HAL was): an **MLLM** generates event descriptions + clip
  captions at training; two alignment objectives use them:
  - **LRCA** — build a *textual relation matrix* from the clip captions and
    **align it with the video's temporal feature-similarity matrix**. ← the
    piece closest to "transcript×frame similarity alignment".
  - **ESTA** — align temporally-pooled context with event/action semantics.
  - Supporting: DAI (context-aware codebook to absorb query-irrelevant
    semantics), SGE, SORA (sharpens the similarity map).
  - **The MLLM is used only at training, discarded at inference** — matches
    DELTA's philosophy.

**Where it does NOT fit (all adaptable / relaxed):** grounding not TA (single
query, no ordered sequence, no monotonicity); fully supervised on GT spans; the
whole "don't align to the background" motivation is moot on 50Salads (no
background); SlowFast+CLIP not InternVideo2; it regularises a base grounding
model, doesn't produce the alignment.

**What transfers:** LRCA, ESTA, the "MLLM/language prior at train only" design,
SORA. Cite MASRA for *"language-derived similarity-matrix alignment, train-time
regulariser"*; pair with **MLLM4WTAL** (Zhang et al., **CVPR 2025**,
arXiv:2411.08466) for a CVPR-stamped "MLLM guides a weakly-supervised temporal
model, discarded at inference" paradigm.

---

## Genuine transcript/sequence → video aligners (the mechanism references)

Not 2025-26, but these are what "TA" actually means (ordered sequence → video):
- **StepFormer** (CVPR 2023, arXiv:2304.13265) — ordered step slots → video,
  order-aware alignment loss, self-supervised from subtitles. Closest mechanism.
- **TAN** (CVPR 2022 Oral, `TengdaHan/TemporalAlignNet`) — sentences → long
  video, weakly from noisy narration; beats CLIP/MIL-NCE; helps weakly-supervised
  Breakfast segmentation.
- **Drop-DTW** (NeurIPS 2021) — the differentiable sequence↔video DTW primitive.
- **ASOT** — already inside DELTA (`--model_type wclot`, `src/asot.py`).

Also-rans: Ali et al. (ICCV 2025, self-supervised video↔video alignment +
segmentation, OT — *not* VLM); REMAP/REALIGN (2025, partial-GW-OT procedure alignment).

### OVTAS — the training-free VLM-similarity front-end (arXiv 2602.21406, Feb 2026)

*Exploring VLMs for Open-Vocabulary Zero-Shot Action Segmentation.* Training-free,
segmentation-by-classification: **FAES** (frame↔action embedding cosine, 14 VLMs
compared) → **SMTS** (similarity-matrix temporal smoothing). Zero-shot, no
supervision. **Does not use action order** (open-vocab label-set matching).

Role for us (peer review 2026-09-04): the **semantic-evidence generator / V0
baseline**, not the whole Stage A. `delta.align.cost.fused_cost` is FAES with the
transcript position as the label; `delta.align.asot` replaces SMTS's order-agnostic
smoothing with an order-preserving OT decode. Its 14-VLM comparison is the guide
for picking the frame/clip backbone (M-A4).

---

## Workstream split

- **Eliz → MASRA.** Adapt LRCA/ESTA into a training-time regulariser on the
  InternVideo2 transcript×frame similarity for 50Salads; then order-preserving
  alignment (ASOT) → `Y*` → DELTA. Plan M1–M7 below.
- **CVA workstream.** Build the VLM aligner: CTE encoder + CBD boundary loss.
- **Synergy:** CVA produces the alignment; MASRA (as HAL did on the segmentation
  side) is an auxiliary training signal that improves it. Both use InternVideo2.
  Measured on TA metrics, then downstream MoC.

### Eliz's MASRA plan — 50Salads

| # | step | needs / output |
|---|---|---|
| M1 | get MASRA code + a VTG base model; reproduce on **TACoS** (cooking VTG — no 50S video yet); read LRCA/ESTA/DAI | understanding, no blocker |
| M2 | add an `internvideo2` `FrameBackbone` to `delta.features.backbones`; build `s (N×T)` for 50Salads | **raw video** |
| M3 | adapt **LRCA** — align a transcript relation-matrix (order structure / co-occurrence) with the video similarity matrix | language regulariser loss |
| M4 | adapt **ESTA** — align pooled temporal context with the 17 action-name embeddings | semantic alignment loss |
| M5 | order-preserving alignment (ASOT / `delta.align.ta`) reads `Y*` off the regularised `s` | `Y*` |
| M6 | score `Y*` vs ATBA-in-DELTA / ASOT-in-DELTA / naive floor (MoF/MoC/edit/F1@k/boundary offset) | **the TA-metrics gate** |
| M7 | feed best `Y*` into DELTA's decoder (CTC/CRF/duration unchanged) → Obs%/Pred% MoC | the downstream result |

Baselines: naive-uniform (MoC 0.34) · ATBA-in-DELTA (`--model_type atba`) ·
ASOT-in-DELTA (`--model_type wclot`) · HAL (paper number, Breakfast, reference only).

### [retained] The dropped HAL plan — record only

*Superseded by the MASRA plan above (supervisor: VLM required, no segmentation).
Kept for reference — the Phase-T-gate → Phase-D structure and the
TA-metrics→DLTA-metrics chain carried over to MASRA.*

Original decision (2026-09-02): **50Salads only**, **test HAL's TA part in
isolation first**; only integrate into DELTA if it beats ATBA's TA part.

"TA part" = transcript + video → dense pseudo-labels `Y*`. For ATBA *and* HAL
this is: train an encoder → run the boundary-detector + drop-allowed DP → `Y*`.
The boundary-detector + DP are **identical** between them; only the encoder
differs (HAL adds a two-scale VAE branch + `recon`/`kl`/`diff` losses). So the
comparison is: *pseudo-labels from HAL's encoder* vs *from ATBA's encoder*.

Everything runs on the **pre-extracted I3D features** (`dinggd/50salads` bundle).
**No raw video needed for any of this.**

#### Phase T — TA-part comparison (the gate)

| # | Step | Output |
|---|---|---|
| T1 | Add a **50Salads config** to the ATBA repo (`iSEE-Laboratory/CVPR24_ATBA`) and the HAL repo (`DMIRLAB-Group/HAL`) — both use the same folder-structure loader; HAL has vestigial 50S branches. Generate `transcripts/<vid>.txt` by collapsing the groundTruth (`delta.data.to_transcript`); add splits, widen the loader assert | 50S runnable in both repos |
| T2 | Train **ATBA** on 50S (≥3 seeds); extract `Y*_ATBA` on the held-out split | **TA metrics** — see below |
| T3 | Train **HAL** on 50S (same seeds); extract `Y*_HAL` | same |
| T4 | Compare `Y*_HAL` vs `Y*_ATBA`, and both vs the naive-uniform floor | **GATE: does HAL's TA beat ATBA's TA beyond seed noise?** |

**TA metrics** (`delta.align` — `Y*` vs held-out GT): MoF · **MoC** · Edit ·
F1@{10, 25, 50} · **median per-transition boundary offset (frames)**. (These are
"is the alignment good", distinct from the downstream DLTA metrics.)

**One-video view (understanding, not a metric):** run both trained models on a
single test video → `delta.viz.plot_segmentation({"GT":…, "ATBA":…, "HAL":…})`.
Shows *where* HAL changed a boundary; the number is the split average.

#### Phase D — integrate into DELTA (only if Phase T passes)

| # | Step | Output |
|---|---|---|
| D1 | Port HAL's VAE tap (2 adjacent encoder layers → `z_1` slow / `z_2` fast) + `recon`/`kl`/`diff` losses into DELTA's `--model_type atba` path (`train.py` encoder + `atba_loss.py::LossFn`, ~70 lines; insertion in `delta-code.md`). `warm_epc`→~40; flags `--z_layer --rec_weight --kl_weight --diff_weight` | `DELTA-atba+HAL` |
| D2 | Train `DELTA-atba+HAL` vs plain `DELTA-atba` on 50S → **Obs {20,30}% × Pred {10,20,30,50}% MoC** | does the improved TA improve DLTA anticipation? |
| D3 | Ablate: recon-only / kl-only / diff-only | which HAL term carries it |
| D4 | If `diff_loss` (smoothness) helps → hand the change-rate penalty to the VLM-aligner track | shared component |

Expected magnitude is small (HAL is +2–3 MoF on *segmentation*; the DLTA MoC
effect is unknown). Fine: (a) it's a baseline for the joint paper, (b) `diff_loss`
may transfer to the VLM aligner, (c) a clean negative ("hierarchy priors that
help segmentation don't transfer to transcript-only anticipation") is a real
finding. **If Phase T fails, stop after T4** — no DELTA integration.

### The VLM track — 50Salads only

`s(n,t) = cos( VLM_text(action_n), VLM_video(clip_t) )` with **InternVideo2**
(`OpenGVLab/InternVideo2`, video-native, shared video-text space — better than
CVA's SlowFast+frame-CLIP for fine-grained/temporal).

| # | Step | Needs |
|---|---|---|
| V1 | Add an `internvideo2` `FrameBackbone` to `delta.features.backbones` (~40 lines) | — |
| V2 | Extract InternVideo2 video + action-name text features for 50Salads | **raw video** |
| V3 | Build `s (N×T)`; diagnostics — zero-shot argmax confusion vs GT, boundary peakedness (vs I3D's 1.11×) | V2 |
| V4 | Swap `s` into DELTA's alignment — `wclot` cost matrix (`train.py:548`, ~1 line) or `atba` `BoundaryDetector`. Score `Y*` vs ATBA-in-DELTA / ASOT-in-DELTA / HAL / naive | V3 |
| V5 | + CVA's **CBD** boundary-contrastive loss on the aligned boundaries; + a **CTE**-style encoder on the InternVideo2 features | V4 |
| V6 | Best `Y*` → DELTA decoder → **Obs%/Pred% MoC** on 50Salads | V5 |

**Cheap de-risk:** with even 3–5 raw `.avi`, run V2–V3 on those and check the
diagnostics before committing to full extraction.

## Scoped research statement

**50Salads only, for now.** Bring **CVA-style VLM alignment** — semantic
video-text similarity, a boundary-contrastive objective, context-robust
hierarchical encoding — into the **transcript-supervised dense long-term
anticipation** setting, where **ATBA and HAL** win with classifier-based
alignment. VLM-direct transcript→frame alignment for DLTA is unaddressed.

**Baselines (all on 50Salads, via WLTA):** naive-uniform (floor, MoC 0.34) ·
**ATBA-in-DELTA** (`--model_type atba`) · **ASOT-in-DELTA** (`--model_type
wclot`) · **HAL-losses-in-DELTA** (H4–H6). Reference (not compared directly):
CVA on VTG.

**Ordering:** the **HAL track (Phase T → Phase D)** runs now — I3D features,
cluster, no raw video. Phase T is a **gate**: integrate into DELTA only if HAL's
TA beats ATBA's TA. The **VLM track (V1–V6)** starts when raw 50Salads video
arrives (V1 — adding the InternVideo2 backbone — can be done now). The two
converge on the DELTA VLM-direct method, regularised by `diff_loss` (from HAL,
if Phase T passes) and CBD/CTE (from CVA).
