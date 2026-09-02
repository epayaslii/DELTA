# Baselines: HAL and CVA (CVPR 2026)

Two recent CVPR'26 papers, on **two different axes** of the problem. Neither is
our contribution; both scope it.

Legend: **[PAPER]** · **[CODE]** (official repo) · **[INFER]** · **[UNKNOWN]**.

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

**Role for us:** methodological upper reference + a menu of ingredients (CBD,
CTE, QCD). Not a benchmark we compete on.

---

## Scoped research statement

Bring **CVA-style VLM alignment** — semantic video-text similarity, a
boundary-contrastive objective, context-robust hierarchical encoding — into the
**transcript-supervised dense long-term anticipation** setting, where **ATBA and
HAL** currently win with classifier-based alignment. VLM-direct transcript→frame
alignment for DLTA is unaddressed.

Baselines: naive-uniform (floor) · ATBA-in-DELTA (`--model_type atba`) ·
ASOT-in-DELTA (`--model_type wclot`) · HAL (segmentation, Breakfast) ·
optionally HAL-losses-in-DELTA. Reference: CVA numbers on VTG (not compared
directly).

## How to proceed with the TA

| Phase | Work | Needs | Output |
|---|---|---|---|
| **0** | DELTA/WLTA `atba` + `wclot` on Breakfast + 50S; (opt) standalone HAL on Breakfast | cluster | real MoC baselines; pseudo-label dumps |
| **1** | frozen VLM frame feats + action-name text embeds for 50S; build `s (N×T)`; diagnostics | raw video | `s`; zero-shot confusion; boundary peakedness |
| **2** | plug `s` into DELTA's alignment — `wclot` cost matrix (`train.py:548`) or `atba` `BoundaryDetector` | phase 1 | `Y*` quality vs ATBA / HAL / naive |
| **3** | borrow from CVA: CBD-style boundary contrastive loss on aligned boundaries; CTE-style encoder on VLM feats; QCD-style background robustness | phase 2 | ablation: which CVA idea helps |
| **4** | best `Y*` → DELTA decoder; Obs%/Pred% MoC | phase 3 | the result vs ATBA / HAL / DELTA |

Phase 0 runs now (I3D, cluster); phases 1–4 need raw 50Salads video.
