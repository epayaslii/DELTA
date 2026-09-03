# MASRA — analysis and adaptation to transcript-supervised VLM-direct alignment

Legend: **[PAPER]** (arXiv:2605.03398v1, `~/Desktop/MASRA_CVPR.pdf`) ·
**[INFER]** · **[UNKNOWN]**.

*MASRA: MLLM-Assisted Semantic-Relational Consistent Alignment for Video Temporal
Grounding* — Ran Ran et al., UESTC. **[PAPER]** ACM-track submission (the PDF uses
the `Conference'17` template and an anonymous running header); arXiv 2026-05-05.

## 1. What MASRA is

**[PAPER]** MASRA is **not a standalone model** — it is a DETR-style VTG backbone
(baseline numbers in Tab. 4/5 sit in CG-DETR / TR-DETR / RGTR territory) plus
**five additions**, three of which are training-time only:

| Module | Role | Train-only? | We port it? |
|---|---|---|---|
| **DAI** — Decoupled Alignment Interaction | context-aware codebook (K_B=1024, top-50) subtracts pooled query from pooled video → retrieves "background" tokens that soak up query-irrelevant attention. VQ codebook loss `L_cb` (Eq. 2). | no (runs at inference) | **no** — our backbone is ASOT/DP, not cross-modal attention |
| **ESTA** — Event Semantic Temporal Alignment | pull the mean-pooled temporal-context features of event span `[s_i,e_i]` toward the text embedding `o_i` of that event's description. `L_sem = mean_i (1 − cos(u_i, o_i))` (Eq. 3–4). | **yes** | **yes** — adapted |
| **SGE** — Semantic-Guided Enhancement | two-stage attention: `E1 = Attn(Q=H,K=H,V=I)`, `E2 = Attn(Q=E1,K=H,V=H)` — ESTA-cleaned context `H` decides *where* to aggregate the interaction feature `I`, then re-injects semantics. | no | **no** — backbone-specific |
| **LRCA** — Local Relational Consistency Alignment | make the clip×clip visual self-similarity `S = cos(E,E)` match a **text relation matrix** `R = cos(C,C)` built from per-clip caption embeddings. `L_rel = (1/T²) Σ_ij (s_ij − r_ij)²` (Eq. 7–8). | **yes** | **yes** — adapted, this is the main one |
| **SORA** — Second-Order Relational Attention | `S̃ = S + φ(S)` (conv + activation refine net), then `F = softmax(S̃)·MLP(E)`. Lightweight local denoise of the similarity map. | no | **maybe** — cheap, backbone-agnostic; port after LRCA works |

**[PAPER]** Overall loss (Eq. 11):
`L = L_vtg + λ_sal·L_sal + λ_sem·L_semantic + λ_rel·L_relation + λ_cb·L_cb`.
`L_vtg` = classification + L1 + GIoU on the span; `L_sal` = clip saliency.
**[UNKNOWN]** the four λ values — not in the paper text. **[INFER]** start
`λ_sem = λ_rel = 1`, `λ_cb ≈ 0.25`, tune.

**[PAPER]** Backbone features: **SlowFast + CLIP** (SF+C), CLIP text encoder;
VGG+GloVe variant for Charades. MLLM = **GPT-5**, used **offline** — captions and
event analyses are generated once, encoded with the text encoder, and the
**text features are cached and read during training**. The MLLM is never in the
training loop and never at inference.

**[PAPER]** Training: AdamW, lr 1e-4, wd 1e-4, batch 32, 400 epochs, 1× A100.
Benchmarks: QVHighlights, Charades-STA, **TACoS** (long cooking videos — the one
relevant to us). Gains: TACoS mIoU 37.44 → 38.84 vs RGTR; QVH avg mAP +~1.5.

**[PAPER]** Ablations worth knowing:
- Fig. 5 (`T-T` best): **both** priors should be text-derived, not visual. Fully
  visual priors (`V-V`) are the worst.
- Fig. 6: MLLM event spans > GT-only spans > feature-shift spans. GT spans "lack
  dense supervisory signal"; MLLM spans win because the *descriptions* carry
  extra semantics. **[INFER]** for us: an action *name* alone is a weak `o_i`; a
  short generated description per action should help.
- Tab. 5: DAI/SGE/SORA each add ~0.2–0.5 — small. ESTA + LRCA are the substance
  (Tab. 4: +1.2 mAP combined).

## 2. No code

**[UNKNOWN]** No official repository — anonymous submission, no link in the PDF.
Reproducing MASRA-as-published means re-implementing the five modules on top of a
public VTG DETR (**CG-DETR** `github.com/wjun0830/CGDETR` is the closest baseline
and has TACoS configs). **We do not need the full backbone** — see §3.

## 3. Why it half-fits, and the adaptation

MASRA's task is **query → one span**, fully supervised on GT spans. Our task is
**ordered transcript → dense frame labels `Y*`**, supervised only by the
transcript (no spans, no timestamps). What transfers is the **two training-time
regularizers**, moved from "regularize a DETR's temporal features" to
"regularize the frozen-VLM transcript×frame similarity `s` before ASOT".

| MASRA | Our VLM-direct alignment (`delta.align`) |
|---|---|
| query `Q` (one sentence) | the whole ordered transcript `{a_1…a_N}` |
| target moment `(t_s,t_e)` | full dense labeling `Y*` = N contiguous blocks |
| temporal feature `E ∈ R^{T×C}` | frozen **InternVideo2** frame features `F` (I3D as a stand-in until raw video) |
| MLLM event `(y_i,[s_i,e_i])` | transcript entry `i` as the "event"; **span = its block under the current alignment `Y*⁽ᵏ⁾`**; `o_i` = text embedding of the action name (later: a generated description) |
| MLLM clip caption `c_t` → `R = cos(c_t,c_{t'})` | **text relation matrix from the transcript**: `r_ij = cos(g(class of frame i), g(class of frame j))` where the class of a frame is the action its current block belongs to, `g` = class-name embedding. No MLLM, no video. (Optionally: real per-frame VLM/MLLM caption embeddings once we have video.) |
| `S = cos(E,E)` | `S = cos(F,F)` — frame×frame self-similarity of the VLM features |
| `L_sem` (ESTA) | same formula — pull each block's pooled `F` toward its action embedding |
| `L_rel` (LRCA) | same formula — pull `S` toward the transcript-derived `R` (block-diagonal-ish → sharper boundaries) |

**The adaptation is EM-style.** We have no GT spans, so the spans that ESTA/LRCA
need come from the alignment we are trying to improve:

```
s⁽⁰⁾  = InternVideo2 transcript×frame cosine            (delta.align.similarity)
Y*⁽ᵏ⁾ = align_dp(s⁽ᵏ⁾, transcript)                       (delta.align.ta)
        ├─ ESTA target: block-pooled F  ↔  action embeddings
        └─ LRCA target: R from Y*⁽ᵏ⁾ + class embeddings
s⁽ᵏ⁺¹⁾ = s⁽ᵏ⁾  updated by a gradient step on λ_sem·L_sem + λ_rel·L_rel
        (on the cluster, in torch, with F / the s-projection trainable)
```

At inference the regularizers and any MLLM text are gone — exactly MASRA's
"no inference overhead" property, and exactly DELTA's "transcript is training-only"
property. Clean fit.

**Companion citation** for the "MLLM guides a weakly-supervised model at training
only" paradigm with a CVPR stamp: **MLLM4WTAL** (Zhang et al., CVPR 2025,
arXiv:2411.08466, `~/Desktop/ArticleBest.pdf`).

## 4. What runs locally today (no GPU, no raw video)

`src/delta/align/masra.py` — numpy forward-pass of the adapted pieces:
- `pool_events`, `esta_alignment` — the ESTA term for a given `(F, Y*)`.
- `visual_relation_matrix`, `transcript_relation_matrix`, `lrca_residual` — the
  LRCA term; `R` in `block` (hard) or `class-sim` (soft, MASRA's `T` prior) mode.
- `masra_report` — both numbers for one video, plus a relation-derived boundary
  novelty score vs GT boundaries.

Use it now for a **Stage-1-style motivating measurement** on the I3D bundle:
*how far is `S = cos(F,F)` from the transcript block structure `R`?* — the residual
`L_rel` under the naive-uniform and oracle alignments bounds what LRCA can buy.
The differentiable torch versions come with M3/M4 on the cluster.

**[INFER] Measurement — 50Salads split-1 test (10 videos), bundled I3D features:**

| quantity | value | reading |
|---|---|---|
| `L_rel` (block `R`), **oracle** alignment `Y*=GT` | 0.465 | even with perfect blocks, `cos(F,F)` is far from block-diagonal |
| `L_rel` (block `R`), **naive-uniform** alignment | 0.483 | I3D relational structure barely separates a correct alignment from a wrong one (Δ = 0.018) |
| `1 − s_{t,t+1}` percentile at GT boundaries (0.5 = chance) | 0.666 | I3D self-similarity carries *some* boundary signal — weak but real (cf. the 1.11× consecutive-frame result) |

So on I3D the LRCA target is only weakly informative — which is the argument for
(a) a stronger video-native VLM (**InternVideo2**) as `F`, and (b) LRCA as an
*active push* toward block structure rather than a passive match. Re-run with the
InternVideo2 features at M3 to get the real ceiling.

## 5. Plan (M1–M7, mirrors `baselines-hal-cva.md`)

| # | step | blocker |
|---|---|---|
| M1 | ✅ read MASRA (this doc); pick CG-DETR as the reference backbone; note no code | — |
| M2 | add an `internvideo2` `FrameBackbone`; build `s (N×T)` for 50Salads | **raw video** |
| M3 | LRCA (torch): `R` from the transcript, align `S` to it | needs M2's `s` / features |
| M4 | ESTA (torch): pool blocks of `F`, pull to action embeddings (+ generated descriptions) | needs M2 |
| M5 | EM loop: `align_dp` → regularize → re-align → `Y*` | M3+M4 |
| M6 | **gate**: `Y*` vs ATBA-in-DELTA / ASOT-in-DELTA / naive floor (MoF/MoC/edit/F1@k/boundary offset) | M5 |
| M7 | feed best `Y*` into DELTA's decoder → Obs%/Pred% MoC | M6 passes |

M1 groundwork + the numpy diagnostics (§4) are done locally. M2 onward is
cluster + raw-50Salads-video gated.
