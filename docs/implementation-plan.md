# Implementation plan — weak alignment → refine, VLM-driven

Combines the best-fitting piece of each analysed paper. All stages stay
**weakly supervised** (transcript only; GT is eval / oracle / warm-up only —
see `50salads-notes.md`). Companion: `method-weak-then-refine.md` (the design),
`masra-analysis.md` / `baselines-hal-cva.md` (the sources).

## Component sources

| stage | idea | from | why that paper |
|---|---|---|---|
| **0** semantic-guided sampling | language decides where to spend VLM compute | **LGTTP** (EMNLP'25) | model-agnostic query→token-density; "cost is a constraint" |
| **A** weak alignment | fused-GW **OT** with a transcript temporal prior + **VLM semantic cues in the cost** | **TASOT** (2602.24138) + **ASOT/CLOT** (group code) | TASOT = ASOT + VLM captions in the cost, annotation-free; ASOT already in `wclot` |
| **A′** monotonic decode | strict temporal monotonicity + fine/coarse agreement + noise post-proc | **HiERO-StepG** (2605.31227) | zero-shot ordered-steps→spans, exactly Stage A |
| **B** boundary refinement | **CBD** context-invariant boundary contrastive loss on *predicted* pseudo-boundaries | **CVA** (CVPR'26) | works on any boundary frames + augmentations, no GT spans |
| **B** relational sharpening | **LRCA / ESTA** on local windows | **MASRA** (ACM'26) | `delta.align.masra_torch`, already built |
| **B′** hard-case reasoning | prompt a chat-VLM with candidate `[s,e]` and the two action names | **TOGA** (2506.09445) | "segments in the VLM imports"; consistency check |
| features | video-native embeddings | **V-JEPA2** (from D-CLOT release) or InternVideo2 | single-frame SigLIP2 failed (`50salads-notes.md`) |

## Modules (this repo)

```
delta.align.asot        NEW  minimal self-contained fused-GW OT solver (torch):
                             segment_asot(cost, mask, ...) -> transport plan T
                             monotonic_mask(transcript, T) -> (T,) soft plausibility
                             temporal_prior(transcript, T, K) -> (T,K) cost
                             decode(T) -> Y*  (argmax + monotonic cleanup)
delta.align.cost        NEW  fused cost matrix (TASOT):
                             C = a*(1 - s_visual) + b*(1 - s_semantic) + rho*temporal_prior
                             s_visual   : cos(frame_i, frame_j)-driven GW handled by asot
                             s_semantic : cos(action_text_n, frame_t)  (siglip2 / VLM caption)
delta.align.cbd         NEW  CVA CBD loss (torch): cbd_loss(feat, boundaries, augment_fn, ...)
delta.align.masra_torch DONE ESTA / LRCA / MasraRegularizer
delta.align.ta          DONE align_dp / align_soft  (kept as a baseline / hard-DP option)
delta.features.extract  DONE + --every N ; add internvideo2 / vjepa2 backbones (cluster)
```

## Milestones

- **M-A1** ✅ minimal ASOT in `delta.align.asot` (+ tests, 2026-09-04)
- **M-A2** ✅ `delta.align.cost` fused cost; `asot` provider in `evaluate.py`
- **M-A3** ✅ measured on coarse siglip2, split-1: ASOT 0.342 recovers the
  hard-DP collapse (0.199) but doesn't clear naive (0.366). GW term hurts on
  near-constant features; VLM semantic cost negligible. **Bottleneck = features.**
- **M-A4** ← **next**: V-JEPA2 / InternVideo2 features + TASOT VLM captions
  (needs cluster). re-measure.
- **M-B1** ✅ `delta.align.cbd` — CVA CBD loss (+ tests, 2026-09-04)
- **M-B2** Stage B loop: local window around each Stage-A boundary → CBD + LRCA
  refine → updated `Y*`. measure boundary offset. *cluster (training)*
- **M-B3** TOGA-style hard-case reasoning: chat-VLM prompt with `[s,e]` + names,
  consistency check against the segment. *cluster*
- **M-C**  best `Y*` → DELTA decoder (`wclot`/`atba` unchanged) → Obs%/Pred% MoC.

## Open questions for the supervisor (unchanged)

1. which "pass-to-VLM real-time-reasoning" paper (TOGA? something else)?
2. segment prompt = visual span marker or text timestamps + names?
3. Stage B on VLM embeddings + CBD, or chat-VLM judgement, or both?
