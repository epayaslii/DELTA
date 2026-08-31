# Temporal Alignment (TA) — reference

The internship focuses on the **Temporal Alignment** component. This note
consolidates how TA works in DELTA, grounded in its source method
**ATBA** (Xu & Zheng, *Efficient and Effective Weakly-Supervised Action
Segmentation via Action-Transition-Aware Boundary Alignment*, CVPR 2024,
arXiv:2403.19225, code: `iSEE-Laboratory/CVPR24_ATBA`). DELTA states its TA
module *"follow[s] the boundary detector of (Xu and Zheng 2024)"*.

Legend: **[paper]** stated/derivable from a paper · **[interp]** our reading ·
**[hypothesis]** our research conjecture, not established.

---

## 1. What TA is for

A transcript gives the **ordered** actions in a video but **no timestamps**.
TA infers the hidden boundaries so each transcript action gets a temporal
interval, producing dense **pseudo-labels** `Y*` that stand in for the
frame-level annotation DELTA is not given. `Y*` is the *only* supervision that
reaches the segmentation head and the anticipation decoder.

```
transcript  A → B → C → D          (order known, boundaries unknown)
video       x_1 ............. x_T
TA output   |--A--|----B----|--C--|----D----|   → Y* = per-frame labels
```

**[paper]** ATBA's own pseudo-labels are only ~62–68 % frame-accurate on
Breakfast (ATBA supp. Tables 3–4, "P.L." column). The targets DELTA trains on
are noisy even on the easier dataset.

---

## 2. The mechanism (ATBA §3)

### 2.1 Front end — produces the signal TA consumes
- Video features `X ∈ ℝ^{T×2048}` (I3D) → Transformer encoder (pre-norm,
  learnable positional embeddings, **pyramid hierarchical local attention**:
  window radius `2^{l-1}` at layer `l`) → `X' ∈ ℝ^{T×d'}`.
- Shared **linear classifier + softmax** → frame posteriors `P ∈ ℝ^{T×|C|}`.
- `|C|` learnable **class tokens** run through the same encoder → predict
  per-class *occurrence* (used by the video-level loss; in DELTA they are
  elevated to persistent "action prototypes" kept at inference).

### 2.2 ATBA module: `(P, transcript 𝒜) → Y*`
Transcript has `M` segments, `M−1` transitions `ℛ = {(a_r, a_{r+1})}`.

1. **Class-agnostic boundary score** `v^b_t` — *parameter-free*. In a window
   `w^b = 7` around `t`, build `Γ^{(t)}_{i,j} = 1 − 2·JS(p_i, p_j)` (Jensen–
   Shannon divergence between frame class distributions), correlate with a
   **fixed** `7×7` template `Ω^b` (block-diagonal ≈ +1). High where `P` changes
   sharply.
2. **Candidate selection** — greedy + NMS **on `v^b` only**: repeatedly take the
   highest `v^b_t`, suppress a neighborhood of radius `μT/M` (μ ≈ 0.3), until
   `K = λ(M−1)` candidates (λ = 4).
3. **Action-transition score** `V^a_{k,r}` — *parameter-free*. A fixed `2×w^a`
   template (`w^a = 31`) over classes `{a_r, a_{r+1}}` × time around candidate
   `b_k`, read directly from `P`: rewards "`P[a_r]` high before `b_k`,
   `P[a_{r+1}]` high after".
4. **Combine** `V_{k,r} = V^a_{k,r} + v^b_{b_k}`.
5. **Alignment = drop-allowed DP** (CTC-style, *not* frame-level Viterbi):
   choose `M−1` monotone candidates minimizing `−Σ_r V_{k_r,r}`, dropping
   `K−M+1`. Complexity **`O(λM²)` — independent of video length `T`**. Hard
   argmin + backtracking → **not differentiable** w.r.t. the chosen boundaries.
6. **Pseudo-labels** — drop the `M` transcript labels into the `M` intervals → `Y*`.
7. **Recomputed every training iteration** from the current `P`.

### 2.3 The three transcript-derived objectives DELTA inherits from ATBA
- `L_cls` — frame cross-entropy `CE(P, Y*)`.
- `L_vid` — BCE: does class `c` appear in the video? (transcript says) — the
  *trusted, boundary-invariant* signal.
- `L_glc` — InfoNCE pulling frame-feature centroids of each pseudo-class toward
  the class-token embeddings (`τ = 0.2`).

DELTA adds on top: **CTC** (sequence-order consistency, marginalises over all
boundary placements), **locally-masked crossmodal grounding** (DistilBERT
action-name embeddings, mask derived from `Y*`), and the whole DLTA branch
(parallel decoder + duration head + linear-chain CRF).

### 2.4 Training schedule
Two-stage, because the decoder's targets come from `Y*`:
Stage 1 optimises alignment + segmentation until pseudo-labels stabilise;
Stage 2 adds the DLTA losses and down-weights segmentation. **[paper]** Joint
optimisation from scratch collapses (DELTA: Mean MoC ≈ 29 → 11) because early
pseudo-labels are near-arbitrary.

### 2.5 Inference
`ŷ_t = argmax_c P_{t,c}`. **The transcript, the full video, the TA module and
the pseudo-labels are all discarded.** TA exists only at training time.

---

## 3. TA → TAS → DLTA dependency

```
transcript ──TA──▶ Y* ──┬──▶ TAS head        (L_cls, + CTC)
                         ├──▶ crossmodal mask M
                         └──▶ future transcript T* = B(Y*_pred) + durations d*
                                     └──▶ DLTA decoder (L_crf, L_dur)
```

**[interp]** If `Y*` boundaries are wrong, both the segmentation target and the
anticipation target are wrong — DELTA calls this "a single early error would
propagate". **[paper]** But DELTA's robustness probe (`deg-TA`: keep only the
video-level loss) shows only a ~1.5 % Mean-MoC drop, i.e. gradual not
catastrophic — CTC and crossmodal grounding absorb moderate pseudo-label noise.

---

## 4. Weaknesses of the current TA

### Supported by the papers
- **[paper]** Pseudo-labels are noisy (~65 % on Breakfast) and the model can
  overfit that noise — hence the video-level regulariser.
- **[paper]** Class-agnostic boundary detection alone is far worse than adding
  the transition pattern (ATBA Table 3) — boundaries are ambiguous.
- **[paper]** NMS radius `μT/M` **suppresses short segments** (ATBA supp.).
- **[paper]** DELTA underperforms fully-supervised methods on **50Salads**
  specifically ("frequent transitions, higher temporal variability"); duration
  loss barely helps there (+0.2 MoC vs +3.3 on Breakfast).
- **[paper]** ATBA was **never evaluated on 50Salads** — its knobs (`μ, w^b,
  w^a, λ`) were tuned for Breakfast / Hollywood / CrossTask.

### Our hypotheses
- **[hypothesis]** On 50Salads (fixed overhead camera, near-continuous hand
  motion) `v^b` is weak → true transitions may never become candidates → the DP
  cannot recover them regardless of `V^a`.
- **[hypothesis]** `V^a` is only as discriminative as a linear classifier on
  I3D at separating `cut_tomato` / `cut_cucumber` / `cut_cheese` / `add_oil` /
  `add_vinegar` — i.e. poorly. So alignment on 50Salads rests on the weakest
  link.
- **[hypothesis]** Hard, single-hypothesis boundaries discard uncertainty the
  downstream could otherwise use.
- **[hypothesis]** The detector + DP being **parameter-free** means the only
  thing training can improve is `P`; the alignment procedure itself never
  learns from its mistakes.

---

## 5. Research directions (conceptual only)

Ordered by leverage, each tied to a specific mechanism above:

1. **Better `P` via vision-language similarity.** `V^a` wants a per-frame,
   per-*named*-class score — exactly `sim(text("cut tomato"), frame_t)`. A VLM
   supplies it informatively from step 0, removing the cold-start that forces
   the 40-epoch warm-up, and disambiguates fine-grained classes via
   objects/context. *Lowest risk.*
2. **Better boundary term.** Replace `v^b` (JS-divergence of *posteriors*) with
   discontinuity in a self-supervised frame embedding (DINOv2 / VideoMAE), or a
   learned boundary head — attacks the candidate-generation bottleneck.
3. **Differentiable / OT alignment.** Swap the hard DP for a Sinkhorn transcript
   →frame transport with a monotonicity prior (ASOT CVPR'24; Ali et al.
   ICCV'25), so gradients flow through the alignment cost.
4. **Soft / uncertainty-aware boundaries.** Emit `P(boundary = t)` per
   transition; propagate the distribution into `T*`, `d*`, and the crossmodal
   mask instead of a single hard label. *Cleanest "TA-specific" contribution;
   not done for transcript-only anticipation.*
5. **Tune the untuned knobs** (`μ, w^b, w^a, λ`) for 50Salads before concluding
   the method fails there.

**[note]** Directions 1–3 individually resemble StepFormer / VAVA / Drop-DTW /
Ali et al. The open space is: doing this **for dense long-term anticipation**
(alignment error → forecast error, unstudied), **uncertainty-aware**
pseudo-boundaries feeding a decoder, and the fixed-camera fine-grained regime
(50Salads) where `v^b` demonstrably underperforms. Using a VLM is not by itself
a contribution.

---

## 6. Reproduction notes

- **ATBA code is public** → a TA-only or TA+TAS reproduction is realistically
  "fork ATBA, add a 50Salads config, add DELTA's CTC loss".
- Reproducing DELTA's TA ≈ reproducing ATBA + aligning over the *full*
  (observed + future) video + CTC. The DELTA-specific unknowns are all
  *downstream* of TA (decoder, CRF, loss weights `γ₁γ₂γ₃`, query count `K`,
  mask width) and need the DELTA supplementary material.
- Pseudo-label accuracy (`Y*` vs held-out GT frame labels: MoF / MoC / edit /
  F1@k) is a directly measurable baseline — see `delta.align`.
- ATBA downsamples Breakfast 10×; decide our temporal resolution before
  extracting features. Budget for multi-seed runs (WSAS results fluctuate).

## 7. Open questions for the professor

See `docs/approach.md` §"Questions for our professor" and the expanded list in
the meeting notes (parameter-free detector vs learned module; align observed
only vs full video; pseudo-label accuracy as a deliverable; novelty bar for
OT-based alignment; access to DELTA code + supplementary).
