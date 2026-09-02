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

### 1.1 Supervision — a common point of confusion

**ATBA, HAL and DELTA all use the same weak signal: the transcript** (ordered
action list, no timestamps). "Weakly supervised" means *weaker than frame-level*,
not *unsupervised*. In practice the transcript is obtained by run-length-collapsing
the ground-truth frame labels — you keep the *order*, discard the *boundaries*;
the method never sees the boundaries. (In the wild it comes from narration /
metadata.) DELTA is **not** different from ATBA/HAL on the supervision axis — its
novelty is the *task* (dense long-term anticipation, not just segmentation).

**All three discard the transcript at inference.** It only ever feeds the training
loss:
- ATBA/HAL: transcript → `BoundaryDetector` → pseudo-labels → frame-CE. At test
  (`test.py`), the model does a forward pass and `argmax(fr_logit)`; the boundary
  detector and the `transcript` argument are never called. **[code]**
- DELTA: transcript → TA module → `Y*` → `T*`, `d*` → decoder/CRF targets. At test
  the model gets only `X_obs`; *"the transcript, the full video, the alignment
  module, and the pseudo-labels are discarded"* (paper). **[paper]**

The transcript's information lives in the trained weights (like pseudo-labelling /
distillation). A new test video genuinely has no transcript — that is the point of
the weak setting. GT is used at test only to *score* the output.

Older WSAS methods (NN-Viterbi, CDFL, TASL, D³TW) do **not** discard it — they run
Viterbi/DTW against candidate transcripts at inference. ATBA/HAL/DELTA are
"alignment-free at inference".

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

### The root issue
All of the above trace to one design choice: **alignment is derived from a
trained frame classifier** (`P`). The classifier is the bottleneck, and
50Salads is a near-worst case for it. §5 attacks this by removing the
classifier from the alignment path.

---

## 5. Our direction: align *directly*, not *through segmentation*

### 5.1 The structural observation

Every transcript-supervised segmenter — ATBA, and the current SOTA **HAL**
(Huang et al., CVPR 2026) — does **alignment through segmentation**:

```
frozen I3D → trained frame classifier → posteriors P → (boundary/transition scores) → align → Y*
```

The alignment can be no better than that trained classifier at separating the
vocabulary. On 50Salads the classifier is weak by construction (fixed overhead
camera, near-duplicate fine-grained classes) — Stage 1 measured the I3D
consecutive-frame distance at only **1.11×** the random-frame level at true
boundaries, and the naive-uniform floor is already **MoC 0.34**. Notably,
**neither ATBA nor HAL even reports 50Salads** — the transcript-supervised
literature mostly avoids it, precisely because "alignment through segmentation"
struggles there. DELTA is one of the few transcript-only methods that runs on
50Salads at all.

### 5.2 The alternative

Replace the trained classifier with a **frozen vision-language model** and align
on its similarity directly:

```
VLM:  s(n, t) = sim( g_text(action_n) , f_vis(frame_t) )        # transcript × frame
      → monotonic alignment (order-preserving DTW / OT / DP) on s → Y*
```

No frame classifier, no cold-start, no 40-epoch warm-up. The alignment is
anchored in language-grounded visual semantics from step 0, and the fine-grained
vocabulary (`cut_tomato` vs `cut_cheese`, `add_oil` vs `add_vinegar`) is
disambiguated by the *noun in the label*, not by a classifier that 50Salads
makes hard to train.

Components:
- **`f_vis`** — a video-native VLM tower (VideoLLaMA3 / InternVideo2), not
  frame-CLIP. CLIP-family frame encoders are known to be weak at fine-grained
  and temporal distinctions (arXiv 2602.21406 says exactly this); a video
  encoder with temporal context is the bet, and measuring the gap is a Stage 2
  experiment.
- **`g_text`** — the paired text tower; optionally VLM-generated *descriptions*
  ("pours dark vinegar from a bottle") rather than bare labels.
- **alignment** — order-preserving OT / soft-DTW with a monotonicity prior
  (ASOT CVPR'24 — *already in DELTA as `--model_type wclot`*, `src/asot.py`;
  Ali et al. ICCV'25), differentiable so the cost can later be fine-tuned.
- **boundary uncertainty** — emit `P(boundary_r = t)` from the soft alignment
  and pass the *distribution* into `T*`, `d*`, and the crossmodal mask, instead
  of one hard label.
- **boundary-contrastive loss** — a CBD-style objective (CVA, CVPR'26): the
  aligned boundary frames' representations should be invariant to surrounding
  context and distinct from adjacent + look-alike non-boundary frames. Fights
  over-segmentation and fixed-camera ambiguity; no GT spans needed. See
  [`baselines-hal-cva.md`](baselines-hal-cva.md).
- **encoder** — a CTE-style hierarchical encoder (windowed self-attn + learnable
  global queries + bidirectional cross-attn) on the frozen VLM features, instead
  of DELTA's pyramid-local-attention.

**CVA** (Context-aware Video-text Alignment, CVPR'26) is this approach done well
for the adjacent task of video temporal grounding (query→span, supervised): CLIP
video-text similarity + the CBD boundary-contrastive loss + the CTE encoder,
SOTA on QVHighlights/Charades/TACoS. It is the methodological reference; we adapt
its ingredients to the transcript-supervised DLTA setting.

### 5.3 What is and isn't novel

| | status |
|---|---|
| VLM similarity + monotonic alignment for **step localization / segmentation** | done — StepFormer, VAVA, Drop-DTW, Ali et al. |
| VLM **per-frame** zero-shot action segmentation (no transcript, no alignment) | done — arXiv 2602.21406 |
| VLM-direct transcript→frame alignment feeding a **dense long-term anticipation** decoder (alignment error → forecast error) | **open** |
| **uncertainty-aware** pseudo-boundaries into an anticipation decoder | **open** |
| the **fixed-camera fine-grained regime (50Salads)** that segmentation-based methods skip | **open** |

Lead the contribution with the last three. "We used a VLM" is not itself a
contribution.

### 5.4 ATBA / HAL become baselines; CVA is the reference

The plan is no longer "improve ATBA". It's: build VLM-direct alignment, and
benchmark `Y*` quality against (a) the naive-uniform floor (MoC 0.34),
(b) **ATBA-in-DELTA** (`--model_type atba`) and **ASOT-in-DELTA**
(`--model_type wclot`) on the same features, (c) optionally **HAL**
(= ATBA + a VAE regulariser; segmentation SOTA, but never 50Salads —
[`baselines-hal-cva.md`](baselines-hal-cva.md)), (d) a supervised warm-up
classifier as the ceiling. **CVA** (CVPR'26) is the methodological reference for
what VLM alignment achieves, and a source of transferable pieces (CBD loss, CTE
encoder). Then feed the best `Y*` into the DLTA decoder.

---

## 6. Reproduction / baseline notes

- **ATBA is a baseline now, not the method.** Vendor its public code
  (`iSEE-Laboratory/CVPR24_ATBA`), add a 50Salads config, drive it with the
  same frozen features as our VLM-direct aligner, and compare `Y*` quality.
- **HAL** (CVPR 2026, `arXiv:2602.24275`) — current transcript-supervised
  segmentation SOTA; purely visual + transcript, no text embeddings; does not
  report 50Salads. Optional second baseline.
- Reproducing DELTA's full pipeline ≈ our aligner over the *full* (observed +
  future) video → `T*` / `d*` → a FUTR-style decoder + CRF + CTC. The
  DELTA-specific unknowns are all *downstream* of alignment (decoder, CRF, loss
  weights `γ₁γ₂γ₃`, query count `K`, mask width) and need the DELTA supp.
- Pseudo-label accuracy (`Y*` vs held-out GT: MoF / MoC / edit / F1@k) is
  directly measurable — see `delta.align`. Floor: naive-uniform MoC 0.34.
- Budget for multi-seed runs (weakly-supervised results fluctuate).

## 7. Open questions for the professor

- Is **VLM-direct alignment** (no trained frame classifier in the alignment
  path) the intended contribution, or an improvement to DELTA's existing TA?
- Target metric: `Y*` alignment quality on 50Salads, or downstream DLTA MoC?
- Is establishing a **50Salads** transcript-only alignment benchmark (which ATBA
  and HAL skip) itself a contribution?
- Frozen VLM vs later fine-tuning the alignment cost — how far can we go?
- Do we need our own DELTA decoder reproduction, or is the professor's group
  sharing code / the DELTA supplementary?

Plus the earlier list in `docs/approach.md`.
