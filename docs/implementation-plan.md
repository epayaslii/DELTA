# Implementation plan — weak alignment → refine, VLM-driven

Combines the best-fitting piece of each analysed paper. All stages stay
**weakly supervised** (transcript only; GT is eval / oracle / warm-up only —
see `50salads-notes.md`). Companion: `method-weak-then-refine.md` (the design),
`masra-analysis.md` / `baselines-hal-cva.md` (the sources).

**Framing (peer review, 2026-09-04):** the contribution is *how to turn weak
transcript-order structure into accurate VLM-guided boundaries without ever
seeing a GT timestamp*. HiERO-StepG gives the weak structure; the **local
semantic boundary search** (Stage B1) is the novel core; the contrastive
refinement (Stage B2) is our own objective, **not** a reuse of CVA's CBD.

## Component sources

| stage | idea | from | how we adapt it |
|---|---|---|---|
| **0** semantic-guided sampling | language decides where to spend VLM compute | **LGTTP** (EMNLP'25) | encode densely near likely transitions, sparse inside stable segments |
| **A1** VLM semantic evidence | frozen VLM clip↔action similarity | **OVTAS** (2026, training-free FAES/SMTS) | clip↔*transcript-position* similarity — we keep the order OVTAS discards |
| **A2** video segments not frames | represent actions with temporal context | **HiERO** | short multi-scale clips, not single images (single-frame SigLIP2 failed) |
| **A3** ordered grounding + monotonic decode | strict temporal monotonicity over procedural steps | **HiERO-StepG** (2605.31227) | DELTA's real ordered transcript; DP monotonic decode |
| **A-alt** OT decoder | multimodal OT: visual structure + semantic cost + temporal reg | **TASOT** (2602.24138) | transcript actions as the semantic source, order imposed — *ablation vs A3, not default* |
| **B1** local boundary search | search candidate positions around each coarse transition | **OUR DESIGN** | ±r window; score "left = A, right = B" + visual-change term; pick best + confidence |
| **B2** pseudo-boundary contrastive refinement (PBCR) | boundaries deserve their own contrastive representation | **CVA CBD — inspiration only** | published CBD anchors on **GT spans**; we build anchors from *confident* pseudo-boundaries, negatives from confident action interiors — **our objective, new name** |
| **B** relational sharpening | LRCA / ESTA on local windows | **MASRA** (ACM'26) | `delta.align.masra_torch`, already built |
| **B3** hard-case reasoning | pseudo temporal grounding + consistency, no GT timestamps | **TOGA** (2506.09445) | call a chat-VLM only on low-confidence boundaries, locally |
| **C** frame↔segment feedback | closed-loop OT refinement | **CLOT / D-CLOT** | transcript-defined segment identities, not anonymous clusters — *V3 only* |
| features | video-native embeddings | **VideoLLaMA3** (supervisor's call, last meeting — vision tower for `s`, chat model for captions/reasoning). **V-JEPA2** (D-CLOT release) optional as a pure-visual structure term. NOT InternVideo2. | replace single-frame SigLIP2 |
| output | dense pseudo-labels `Y*` | **DELTA interface** | boundaries → per-frame labels, unchanged downstream |

### ⚠️ Correction on CVA CBD

Earlier notes said "CVA's CBD works on any boundary frames, no GT spans needed."
**Not true of published CBD** — it anchors on GT-span boundary indices, defines
negatives relative to the GT span, and the outer objective Hungarian-matches
predictions to GT moments. Our `delta.align.cbd` is *already* the adapted form
(takes predicted boundaries), but the adaptation is **ours** and has a real
failure mode: a wrong pseudo-boundary self-reinforces. Mitigation = confidence
weighting from Stage B1 + iterative realignment. Report it as **PBCR**, our
contribution, crediting CVA for the *principle* only.

## Modules (this repo)

```
delta.align.asot        DONE fused-GW OT: segment_asot / decode / temporal_prior /
                             monotonic_mask / align_asot   (Stage A3 + A-alt)
delta.align.cost        DONE fused cost = w_sem(1-cos(text,frame)) + w_cap(...) + rho*prior
delta.align.refine      NEW  Stage B1: local semantic boundary search
                             refine_boundaries(sim, coarse, entries, radius, w)
                               -> refined positions + per-boundary confidence
delta.align.cbd         DONE PBCR objective (torch) -- contrastive on *predicted*
                             boundaries; needs confidence weighting (Stage B2)
delta.align.masra_torch DONE ESTA / LRCA / MasraRegularizer
delta.align.ta          DONE align_dp / align_soft  (hard-DP baseline)
delta.features.extract  DONE + --every N ; add a videollama3 backbone (cluster;
                             VL3-SigLIP-NaViT needs transformers 4.x -> pin in the
                             cluster env). V-JEPA2 optional.
```

## Version ladder (peer review)

- **V0 — sanity**: OVTAS-style VLM similarity + ordered transcript + monotonic
  decode. *Question: does VLM semantics carry any temporal signal?* → measured
  2026-09-04 on coarse SigLIP2: no (ASOT 0.34 < naive 0.37). Redo with V-JEPA2.
- **V1 — the proposed method**: clip-level VLM evidence → HiERO-StepG monotonic
  alignment → coarse boundaries → **local semantic boundary search** → refined
  `Y*`. *Priority.*
- **V2 — boundary learning**: add pseudo-boundary uncertainty + PBCR (redesigned
  contrastive, no GT). Stronger novelty if V1 works.
- **V3 — iterative segment refinement**: CLOT/D-CLOT ideas, if V2 still struggles
  with short/ambiguous actions.
- **V4 — expensive reasoning**: TOGA-like / VideoLLaMA3, low-confidence boundaries only.

## Milestones

- **M-A1..3** ✅ ASOT + fused cost + measured (2026-09-04): recovers the hard-DP
  collapse (0.199 → 0.34), still < naive (0.37). **Bottleneck = features.**
- **M-B1** ✅ `delta.align.cbd` (PBCR objective, torch, tests)
- **M-B1b** ✅ `delta.align.refine` — local semantic boundary search (V1 core).
  On coarse SigLIP2: +0.01 MoC, but 95 % of boundaries low-confidence → the
  confidence signal correctly flags "features too weak"; feeds B3 routing.
- **M-A4** ← **next**: VideoLLaMA3 features (+ optional V-JEPA2 structure term)
  + TASOT-style VideoLLaMA3 captions — re-run M-A3 + M-B1b. *cluster*
- **M-B2** PBCR loop with confidence weighting. *cluster (training)*
- **M-B3** TOGA-style chat-VLM on low-confidence boundaries. *cluster*
- **M-C** best `Y*` → DELTA decoder (`wclot`/`atba` unchanged) → Obs%/Pred% MoC.

## Open questions for the supervisor

1. which "pass-to-VLM real-time-reasoning" paper (TOGA? something else)?
2. segment prompt = visual span marker or text timestamps + names?
3. Stage B on VLM embeddings + contrastive, or chat-VLM judgement, or both?
