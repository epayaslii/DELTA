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

## Workstream split

- **Eliz → HAL.** Integrate it into DELTA's TA **on 50Salads** and answer the
  open question: *does HAL's hierarchical-latent + smoothness regulariser help
  dense long-term anticipation (MoC), not just segmentation (MoF)?* Nobody has
  tested this. (50Salads-only for now → all via WLTA, no standalone HAL.)
- **Co-intern → CVA.** The VLM video-text-alignment angle (CBD, CTE, semantic
  similarity).
- **Synergy:** HAL contributes structure/regularisation (the `L_s` change-rate
  penalty, the slow/fast prior); CVA contributes the VLM semantic-matching
  alignment. The DELTA VLM-direct method can draw from both — e.g. a CVA-style
  VLM aligner regularised by a HAL-style smoothness term on the soft assignment.

### Eliz's HAL plan — **50Salads only, all via the WLTA code**

Decision (2026-09-02): **focus on 50Salads only for now.** HAL's standalone repo
asserts against 50Salads and there's no published HAL/ATBA number there, so we
skip the standalone-HAL / Breakfast detour and work entirely through
**DELTA/WLTA**, which runs 50Salads natively (`-d FS -c 19`) with both alignment
modes (`--model_type atba` = ATBA-in-DELTA, `--model_type wclot` = ASOT-in-DELTA).

| # | Step | Where | Output / gate |
|---|---|---|---|
| H1 | Cluster env (Py3.9 / torch1.11, conda); arrange `data/FS/` from the `dinggd/50salads` bundle (+ `mappingeval.txt` for `-d FSeval`) | cluster | data ready |
| H2 | Run WLTA `src/train.py -d FS -c 19 … --model_type atba` (and `--model_type wclot`) on 50S, ≥1 split. Map the run-script flags onto `train.py` (the scripts' `train_edit_elena9sept.py` is missing) | cluster | **baseline** — TAS MoF/MoC + Obs%/Pred% MoC grid |
| H3 | Instrument `training_step` to dump `tas_pseudolabels` per video → score `Y*` vs held-out GT with `delta.align` (MoC, edit, F1@{10,25,50}, median per-transition boundary offset) — vs the naive-uniform floor (MoC 0.34) | Mac | the pseudo-label number HAL must beat |
| H4 | **Port HAL into DELTA** `--model_type atba`: VAE tap on 2 adjacent encoder layers → `z_1`(slow)/`z_2`(fast); `recon`/`kl`/`diff` losses in `atba_loss.py::LossFn` (~70 lines; insertion documented in `delta-code.md`). Raise `warm_epc`→~40; add `--z_layer --rec_weight --kl_weight --diff_weight` | code | `DELTA-atba+HAL` |
| H5 | Run `DELTA-atba+HAL` on 50S → compare (a) `Y*` quality vs H3, (b) **Obs%/Pred% MoC** vs H2 | cluster | **does HAL's hierarchy/smoothness prior transfer to DLTA?** |
| H6 | Ablate: recon-only / kl-only / diff-only / `warm_epc` sweep | cluster | which term matters |
| H7 | If `diff_loss` (smoothness) helps: hand the change-rate penalty to the VLM-aligner track (regularise the soft assignment) | — | shared component |

Optional, off the critical path: one throwaway standalone-HAL run on Breakfast
split 1 (~1 day) just to confirm the HAL repo behaves as published.

Expected magnitude is small (HAL is +2–3 MoF on *segmentation*; the DLTA MoC
effect is unknown, possibly ~0). That is fine: (a) it's the baseline table for
the joint paper, (b) `diff_loss` may transfer to the VLM aligner, (c) a clean
negative ("hierarchy priors that help segmentation don't transfer to
transcript-only anticipation") is a real finding.

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

**Ordering:** the **HAL track (H1–H7)** runs now — I3D features, cluster, no raw
video. The **VLM track (V1–V6)** starts when raw 50Salads video arrives (V1 —
adding the InternVideo2 backbone — can be done now). The two converge on the
DELTA VLM-direct method, regularised by `diff_loss` (from HAL) and CBD/CTE
(from CVA).
