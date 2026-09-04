# Multimodal self-supervised temporal alignment for transcript-only DLTA

**Project:** *Learning to predict future actions by multimodal self-supervised learning* — UPC-IRI
**Status:** approach memo + literature review, v1 (2026-08-31)
**Scope of this phase:** temporal alignment on **50Salads** first, then Breakfast.

---

## 1. Where we start: DELTA and its 50Salads gap

DELTA does **dense long-term action anticipation (DLTA)** —
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

**Thesis of this project:** existing transcript-supervised methods (ATBA, and the
CVPR-2026 SOTA HAL) do **alignment *through* segmentation** — a trained frame
classifier produces posteriors `P`, and the alignment is read off `P`. On
50Salads that classifier is a near-worst case (fixed camera, near-duplicate
fine-grained classes; Stage 1 measured I3D consecutive-frame distance at only
**1.11×** the random level at true boundaries). We instead do **alignment
directly**: a frozen **vision-language model** gives a transcript×frame
similarity matrix, and a **monotonic / optimal-transport** alignment reads `Y*`
straight off it — no frame classifier in the alignment path. See
`docs/temporal-alignment.md` §5.

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
- **ATBA** (Xu & Zheng, CVPR 2024) — action-transition-aware boundary alignment:
  a class-agnostic boundary detector + drop-allowed DP over classifier
  posteriors. DELTA's TA follows its boundary detector. **Our baseline**, not
  our method. Reports Breakfast / Hollywood / CrossTask — **not 50Salads**.
- **HAL** (Huang et al., CVPR 2026, `arXiv:2602.24275`) — current
  transcript-supervised segmentation SOTA. Hierarchical causal process:
  slowly-varying latent actions govern fast visual dynamics; hierarchical
  pyramid transformer, sparse transition constraints. Purely visual + transcript,
  **no text embeddings**; also **does not report 50Salads**. Optional baseline.
- **2by2** (Xu et al., CVPR 2025, `arXiv:2412.12829`) — weakly-supervised
  *global* action segmentation (activity-level, not per-video transcript).

Takeaway: all of these do "alignment through a trained classifier" and mostly
**skip 50Salads** — the regime where that approach is weakest. That gap is the
opening.

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
- **CLOT — Closed-Loop Optimal Transport** (ICCV 2025)
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
- **Exploring VLMs for Open-Vocabulary Zero-Shot Action Segmentation**
  (`arXiv:2602.21406`) — CLIP-style frame↔text similarity on 50Salads / Breakfast
  / GTEA, **per-frame classification, no transcript, no alignment**. Reports that
  frame-CLIP struggles with subtle visual differences and temporal consistency →
  argues for **video-native** VLM towers and an explicit **alignment** step
  (i.e. our direction). The nearest published point to what we're doing, and a
  baseline for "VLM similarity without alignment".

### 2.4 Language / LLMs for long-term anticipation (adjacent)

**PALM** (Kim et al., ECCV 2024), **plausible LVLM anticipation** (Mittal et al.,
CVPR 2024), **ObjectPrompt** (Zhang et al., WACV 2024). These predict *symbolic*
future actions with language priors — complementary to DELTA's *dense* forecast;
a possible late-stage add-on (LLM prior over `T*`), not this phase.

---

## 3. Proposed approach

**One sentence:** do transcript→video temporal alignment **directly** from a
frozen VLM's transcript×frame similarity, via a monotonic / optimal-transport
alignment — no trained frame classifier in the alignment path — and feed the
resulting `Y*` into a DLTA decoder; established methods (ATBA, HAL) and the
naive-uniform floor are baselines. Evaluated first by `Y*` quality on 50Salads,
then by downstream DLTA MoC.

**Why this framing:** ATBA / HAL do "alignment through segmentation" and mostly
skip 50Salads — the regime where a trained classifier is weakest. Removing the
classifier from the alignment path is the structural change; 50Salads is where
it should matter most. (`docs/temporal-alignment.md` §5.)

### Phase 1 — Foundation-model features *(done — this repo)*

`delta.features.extract` emits `(D, T)` per video, `T = len(groundTruth)`, a
drop-in replacement for I3D. Backbones: `vl3-siglip` (default), `siglip2`,
`dinov2`, `i3d-compat` (parity). Text: SigLIP-2 tower (same space) or DistilBERT
(parity). Frame grid is nearest-neighbour resampled to the annotation fps.

**Deliverable:** `features_vl3siglip/` + `action_name_embeddings.npy` for 50S & BF.

### Phase 2 — Baselines + VLM similarity matrix

- **Baselines** on 50Salads I3D features, scored as `Y*` vs held-out GT
  (**MoF, MoC, edit, F1@{10,25,50}**, median per-transition boundary offset;
  `delta.align`):
  - naive-uniform alignment — **MoC 0.34** (Stage 1)
  - vendored **ATBA** (`iSEE-Laboratory/CVPR24_ATBA`) + a 50Salads config
  - *(optional)* HAL
  - supervised warm-up classifier → ATBA — the **ceiling** for classifier-based alignment
- **VLM similarity matrix** `s(n,t) = sim(g_text(action_n), f_vis(frame_t))` for
  `{VL3-SigLIP, SigLIP2, InternVideo2} × {bare label, generated description}`.
  Diagnostic: zero-shot argmax confusion vs GT, and how peaked `s` is at true
  boundaries — does the VLM separate `cut_tomato`/`cut_cheese` where I3D doesn't?

### Phase 3 — VLM-direct alignment

Read `Y*` straight off `s` with a **monotonic alignment** — no frame classifier:

1. **Order-preserving DP / DTW** on `s` respecting transcript order (drop-allowed
   for background). Simplest; the first real number.
2. **Order-preserving optimal transport** (soft-DTW / ASOT-style Gromov–
   Wasserstein with a monotonicity prior; Ali et al. ICCV'25, Xu & Gould CVPR'24)
   — differentiable, soft assignment.
3. **Boundary uncertainty** — from the soft assignment, emit `P(boundary_r = t)`;
   carry the *distribution* into `T*`, `d*` and the crossmodal mask.
4. *(later)* fine-tune the alignment cost (small adapter on `f_vis` / `g_text`)
   with CTC + cross-video temporal cycle-consistency — still no per-frame
   classification head.

### Phase 4 — Downstream DLTA

Feed the best `Y*` into a minimal FUTR-style decoder (+ CRF, CTC later). Evaluate
the standard grid (Obs 20/30 % → pred 10/20/30/50 %), 5 splits, MoC. Targets:
beat DELTA deterministic **20.9** on 50S; approach FUTR 25.96; no Breakfast
regression. Report the upstream `Y*` quality as the bound.

### Ablations to report

VLM vs I3D similarity · video-native vs frame-CLIP `f_vis` · label vs description
text · DP vs OT alignment · hard vs uncertainty-aware boundaries · vs ATBA/HAL ·
50S *and* BF · robustness to a degraded similarity matrix.

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

DELTA. · ATBA — Xu & Zheng, CVPR 2024. · HAL — Huang et al.,
CVPR 2026. · MuCon — Souri et al., TPAMI 2021. · NN-Viterbi — Richard et al.,
CVPR 2018. · TCC — Dwibedi et al., CVPR 2019. · VAVA — Liu et al., CVPR 2022. ·
GTCC — Donahue & Kambhamettu, CVPR 2023. · Temporally-Consistent Unbalanced OT —
Xu & Gould, CVPR 2024. · CLOT — ICCV 2025. · Joint
SSL Video Alignment & Action Segmentation — Ali et al., ICCV 2025 (arXiv
2503.16832). · VideoLLaMA 3 — Zhang et al., 2025 (arXiv 2501.13106). ·
InternVideo2 — Wang et al., 2024. · SigLIP 2 — Tschannen et al., 2025. ·
Procedure-aware video repr. — Zhong et al., CVPR 2023. · StepFormer — Dvornik et
al., CVPR 2023. · PALM — Kim et al., ECCV 2024. · Plausible LVLM anticipation —
Mittal et al., CVPR 2024. · FUTR — Gong et al., CVPR 2022. · ActFusion — Gong,
Kwak & Cho, NeurIPS 2024.
