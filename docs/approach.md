# Multimodal self-supervised temporal alignment for transcript-only DLTA

**Project:** *Learning to predict future actions by multimodal self-supervised learning* — UPC-IRI
**Status:** approach memo + literature review, v1 (2026-08-31)
**Scope of this phase:** temporal alignment on **50Salads** first, then Breakfast.

---

## 1. Where we start: DELTA and its 50Salads gap

DELTA (Maté & Dimiccoli) does **dense long-term action anticipation (DLTA)** —
predict a *frame-level* labelling of a future horizon — trained from **transcripts
only**: an ordered list of actions with **no timestamps and no durations**. One
shared transformer encoder feeds three branches:

| Branch | What it does | Supervision |
|---|---|---|
| **Temporal Alignment (TA)** | boundary detector + dynamic programming maps transcript → dense pseudo-labels `Y*` over the *full* video | transcript only (weak) |
| **TAS head** | segments the observed part | `Y*` (frame CE) + **CTC** (order-consistency, boundary-free) |
| **Crossmodal grounding** | DistilBERT action-name embeddings injected into visual features via **locally masked** cross-attention (mask from `Y*`) | `Y*` |
| **Parallel decoder + duration head + CRF** | predicts future action order + relative durations | pseudo future transcript `T* = B(Y*_pred)` |

Inputs are frozen **I3D** (2048-d) visual features and **DistilBERT** text.
Training is two-stage: stabilise `L_align + L_TAS`, *then* add the DLTA losses.

**Results.** On **Breakfast** DELTA already *matches or beats* fully supervised
methods (avg MoC ≈ 29–37 vs ActFusion 28.5). On **50Salads** it clearly trails:

| 50Salads (Top-1 MoC, avg over Obs 20/30 % × pred 10/20/30/50 %) | avg |
|---|---|
| FUTR (supervised) | 25.96 |
| ActFusion (supervised) | 28.39 |
| WS-DA (prev. weakly supervised, *uses frame labels for observed part*) | 21.30 (single cell) |
| **DELTA (deterministic)** | **20.92** |
| DELTA (stochastic, best-of-sample) | 28.51 |

The paper's own diagnosis: *"50S remains below fully supervised methods, reflecting
the greater difficulty of recovering reliable boundaries and durations from
transcripts in videos with frequent transitions and higher temporal variability."*

### Why 50Salads is hard for transcript-only alignment

- **~20 action instances per video, long videos (avg ~6 min), fine-grained
  transitions.** A single misplaced boundary in `Y*` propagates through both the
  segmentation target and the future transcript `T*`.
- **17 visually similar actions** (`cut_tomato`, `cut_cucumber`, `cut_cheese`, …)
  from a fixed top-down camera — I3D motion features under-separate them, so the
  boundary detector's frame posteriors are noisy exactly where precision matters.
- **High intra-class duration variance** → the duration head has almost no signal
  (paper: duration loss helps only +0.2 MoC on 50S vs +3.3 on BF).
- Breakfast, by contrast, has ~6 actions/video and stronger scene/appearance cues
  per step, so the DP alignment locks on quickly.

**Thesis of this project:** the bottleneck is the *quality of the transcript→frame
alignment* on 50Salads, and that is dominated by (a) how discriminative the frozen
visual features are and (b) how the alignment is inferred (greedy DP over noisy
posteriors). Both are addressable with **vision-language foundation models** and a
**self-supervised, structurally-constrained alignment objective**.

---

## 2. Literature review

### 2.1 Weakly-supervised temporal action segmentation from transcripts

The lineage DELTA sits in — recover frame labels from an ordered action list:

- **ECTC** (Huang et al., ECCV 2016) — CTC extended with a visual-similarity prior.
- **NN-Viterbi** (Richard et al., CVPR 2018), **D³TW** (Chang et al., CVPR 2019),
  **CDFL** (Li et al., ICCV 2019) — HMM / differentiable-DTW / energy-based
  transcript alignment.
- **MuCon** (Souri et al., TPAMI 2021) — mutual consistency between a temporal
  segmentation branch and a sequence (length + transcript) branch; fast, no
  Viterbi at train time.
- **ATBA** (Xu & Zheng, CVPR 2024) — **action-transition-aware boundary alignment**:
  a boundary detector filters noisy frames and aligns to transitions efficiently.
  *This is the module DELTA's TA is built on* — the natural place to intervene.
- **HAL** (Huang et al., CVPR 2026) — hierarchical causal process: slowly-varying
  latent actions govern fast visual dynamics. Strong recent transcript-level baseline.

Takeaway: every method here is a way to make a *noisy frame→label posterior* yield
a *monotone, transcript-consistent* segmentation. Better posteriors (features) and
a better structural prior (OT / monotonic attention) both move the needle.

### 2.2 Self-supervised temporal alignment of video (and video↔text)

- **TCC** (Dwibedi et al., CVPR 2019) — temporal cycle-consistency between two
  videos of the same activity; the standard *representation* objective. Eval:
  Kendall's τ, phase progression, phase classification.
- **LAV** (Haresh et al., CVPR 2021), **VAVA** (Liu et al., CVPR 2022) — add
  optimal-transport / DTW alignment robust to background frames and
  non-monotonic content.
- **GTCC** (Donahue & Kambhamettu, CVPR 2023) — multi-cycle, handles
  repeated/missing steps.
- **Temporally-Consistent (Unbalanced) Optimal Transport for Action
  Segmentation** (Xu & Gould, CVPR 2024 / ANU) — decode a temporally-consistent
  segmentation from a noisy frame↔class cost matrix via a Gromov-Wasserstein
  problem with a temporal-consistency prior; GPU-friendly (few mirror-descent
  iters). Directly relevant as an *alignment decoder* to replace greedy DP.
- **CLOT — Closed-Loop Optimal Transport** (Bueno-Benito & Dimiccoli, ICCV 2025)
  — *from the same lab*; unsupervised action segmentation via a closed OT loop.
- **Joint Self-Supervised Video Alignment and Action Segmentation** (Ali et al.,
  ICCV 2025) — one **fused Gromov-Wasserstein OT** model with a structural prior
  does alignment *and* segmentation together; beats VAVA-style pipelines on
  segmentation, comparable on alignment. Closest methodological neighbour to what
  we want; a strong baseline and a source of the OT formulation.

Takeaway: OT with a monotonicity / temporal-consistency prior is the current
best-practice replacement for DP/Viterbi alignment, and it is differentiable, so
it can sit *inside* training rather than as a fixed pre-pass.

### 2.3 Vision-language / video foundation models as features

Nobody has yet reported I3D → FM feature swaps for transcript-only DLTA; for
segmentation the comparisons that exist (e.g. on EgoExoLearn) show FM features
are competitive-to-better and, crucially, **language-aligned**.

- **VideoLLaMA 3** (Zhang et al., 2025) — vision tower is **SigLIP-so400m** adapted
  with **any-resolution tokenization (AVT)** + **2D-RoPE**; released standalone as
  `DAMO-NLP-SG/VL3-SigLIP-NaViT` (~0.4 B params, ~1152-d). Mid/late layers are
  strong for retrieval *without* retrieval fine-tuning → good frozen features.
- **InternVideo2** (Wang et al., 2024) — video-native, strong on temporal tasks.
- **SigLIP 2** (Tschannen et al., 2025) — image+text in one space; the text tower
  pairs naturally with the VL3 vision tower for crossmodal grounding.
- **DINOv2 / DINOv3** — pure-vision SSL; useful *ablation* (does language pairing
  matter, or just better visual features?).
- **Procedure-aware pretraining**: *Learning Procedure-aware Video Representation
  from Instructional Videos and their Narrations* (Zhong et al., CVPR 2023),
  **StepFormer** (Dvornik et al., CVPR 2023, unsupervised step localisation from
  narration), **VideoTaskformer**. These learn exactly the step-structure prior we
  want; candidates for the visual backbone or an auxiliary objective.

### 2.4 Language / LLMs for long-term anticipation (adjacent)

**PALM** (Kim et al., ECCV 2024), **plausible LVLM anticipation** (Mittal et al.,
CVPR 2024), **ObjectPrompt** (Zhang et al., WACV 2024). These predict *symbolic*
future actions with language priors — complementary to DELTA's *dense* forecast;
a possible late-stage add-on (LLM prior over `T*`), not this phase.

---

## 3. Proposed approach

**One sentence:** replace I3D+DistilBERT with a paired VL foundation model, and
replace DELTA's greedy DP alignment with a *self-supervised, OT-based,
monotonicity-constrained* transcript→frame alignment trained jointly with the
encoder — evaluated first by alignment/segmentation quality on 50Salads, then by
downstream DLTA.

### Phase 1 — Foundation-model features *(done — this repo)*

`delta.features.extract` emits `(D, T)` per video, `T = len(groundTruth)`, a
drop-in replacement for I3D. Backbones: `vl3-siglip` (default), `siglip2`,
`dinov2`, `i3d-compat` (parity). Text: SigLIP-2 tower (same space) or DistilBERT
(parity). Frame grid is nearest-neighbour resampled to the annotation fps.

**Deliverable:** `features_vl3siglip/` + `action_name_embeddings.npy` for 50S & BF.

### Phase 2 — Drop-in swap, measure alignment quality

Re-run DELTA's TA + TAS stage with each feature set, **hold everything else fixed**,
and report on the *observed* part against ground truth (we have GT — we just don't
train on it):

- pseudo-label `Y*` vs GT: **MoF, MoC, edit, F1@{10,25,50}** (`delta.align`)
- boundary localisation error (median frame offset per transition)
- τ-alignment of encoder features across same-activity videos (TCC metric)

Grid: `{I3D, VL3-SigLIP, SigLIP2, DINOv2} × {DistilBERT, SigLIP2-text}`. This
isolates *"do better features alone fix 50Salads alignment?"* before we add any
new objective. (Needs a DELTA re-implementation — no official code yet; ATBA +
FUTR/ActFusion are public starting points.)

### Phase 3 — Multimodal self-supervised alignment objective

Add to the TA stage, in order of expected payoff / risk:

1. **Contrastive transcript-step ↔ frame-window alignment.** Using paired
   VL features, pull each transcript step `T_n` toward its aligned frames and push
   from others (InfoNCE with the current `Y*` as soft assignment). This makes the
   crossmodal grounding a *real* similarity, not just an embedding lookup, and
   gives the boundary detector a language-anchored signal. Cheap, low-risk.
2. **OT alignment decoder** replacing / regularising the DP step: solve a
   transcript→frame transport with a **monotonicity (temporal-consistency) prior**
   (fused Gromov-Wasserstein, à la Ali et al. 2025 / Xu & Gould 2024), fully
   differentiable, re-solved each iteration. Removes the "one early error
   propagates" failure mode the paper calls out for 50Salads.
3. **Cross-video temporal cycle-consistency** among same-activity videos (TCC/GTCC)
   as an auxiliary encoder loss — enforces a shared, monotone phase representation,
   which is exactly what long-horizon anticipation needs.
4. **VLM pseudo-supervision** (higher risk / cost): caption or score frame windows
   with VideoLLaMA-3 against the transcript vocabulary → an extra soft label to
   regularise `Y*`. Zero-shot, no training, but slow to precompute.

Keep DELTA's CTC (order consistency) and CRF (future structure) untouched — they
are complementary to all of the above.

### Phase 4 — Downstream DLTA

Feed the improved alignment into DELTA's two-stage schedule; evaluate on the
standard grid (Obs 20/30 % → pred 10/20/30/50 %), 5 splits, MoC. Targets:
close the gap to ActFusion (28.4) on 50S; **no regression** on Breakfast.

### Ablations to report

FM vs I3D · paired-text vs DistilBERT · +contrastive · +OT · +TCC · OT-vs-DP ·
per-objective on 50S *and* BF · robustness to degraded alignment (repeat the
paper's `deg-TA` probe).

---

## 4. Milestones

| # | Milestone | Gate |
|---|---|---|
| M0 | Datasets staged, sanity checks pass, features extracted (50S) | transcripts + frame counts correct |
| M1 | DELTA TA+TAS reproduced with I3D on 50S | `Y*` MoF within ~2 pts of paper's implied alignment |
| M2 | Feature-swap grid (Phase 2) | which backbone best; ≥ +3 MoF on `Y*` vs I3D → go |
| M3 | Contrastive alignment objective (Phase 3.1) | improves `Y*` MoC + boundary error |
| M4 | OT alignment decoder (Phase 3.2) | ≥ M3, fewer propagated boundary errors |
| M5 | Full DLTA, 50S grid | beats DELTA 20.9; approach/beat 25 (FUTR) |
| M6 | Breakfast, ablations, writeup | no BF regression |

---

## 5. Risks & open questions

- **No official DELTA code.** Re-implementation is on the critical path. Ask the
  professor whether the authors will share it.
- **Kaggle mirrors.** Confirm 50Salads features are I3D (not IDT) and the label
  fps (15 vs 30) before trusting `configs/50salads.yaml`. Raw videos may need the
  official Dundee download.
- **VL3-SigLIP throughput.** ~0.5–1 M frames for 50S at 15 fps × 0.4 B params —
  fine on the cluster with the provided SLURM array; window mode multiplies cost.
- **Duration prediction stays ill-posed** from transcripts (acknowledged in the
  paper, and even for fully-supervised methods). Not a target of this phase.
- **Fairness of comparison.** Swapping features changes the input distribution vs
  published I3D numbers; always report an I3D re-run under our own code as the
  control, not just the paper's table.
- **Open:** do we also want the stochastic (best-of-sample) protocol, or
  deterministic only for now? (Notes say alignment first — deterministic.)

---

## References (short)

DELTA — Maté & Dimiccoli. · ATBA — Xu & Zheng, CVPR 2024. · HAL — Huang et al.,
CVPR 2026. · MuCon — Souri et al., TPAMI 2021. · NN-Viterbi — Richard et al.,
CVPR 2018. · TCC — Dwibedi et al., CVPR 2019. · VAVA — Liu et al., CVPR 2022. ·
GTCC — Donahue & Kambhamettu, CVPR 2023. · Temporally-Consistent Unbalanced OT —
Xu & Gould, CVPR 2024. · CLOT — Bueno-Benito & Dimiccoli, ICCV 2025. · Joint
SSL Video Alignment & Action Segmentation — Ali et al., ICCV 2025 (arXiv
2503.16832). · VideoLLaMA 3 — Zhang et al., 2025 (arXiv 2501.13106). ·
InternVideo2 — Wang et al., 2024. · SigLIP 2 — Tschannen et al., 2025. ·
Procedure-aware video repr. — Zhong et al., CVPR 2023. · StepFormer — Dvornik et
al., CVPR 2023. · PALM — Kim et al., ECCV 2024. · Plausible LVLM anticipation —
Mittal et al., CVPR 2024. · FUTR — Gong et al., CVPR 2022. · ActFusion — Gong,
Kwak & Cho, NeurIPS 2024.
