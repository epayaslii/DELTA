# Method v2 — weak alignment, then refine

Source: supervisor meeting **2026-09-03**. Supersedes the "one-shot VLM-direct
alignment" framing in `temporal-alignment.md` §5 / `masra-analysis.md` §3 —
those regularizers still apply, but inside the two-stage structure below.

## Supervisor's direction (verbatim points, expanded)

| point | reading |
|---|---|
| "weak temporal alignment and then refine" | **two stages, coarse → fine.** Stage A gets an approximate transcript↔video alignment cheaply; Stage B fixes the boundaries. |
| "semantic should guide the sampling" + "cost is also important" | **do not encode every frame.** Use cheap text–image similarity to decide where to sample densely (near likely transitions) vs sparsely (inside stable segments). Compute budget is a first-class constraint. |
| "semantic is enough for words, the video for [timing]" | **modality division of labour:** text/semantic embeddings settle *which* action (the "what"); video settles *where the cut is* (the "when"). Don't ask the text side to localize. |
| "boundary contrastive losses" | Stage B uses a CVA-CBD-style loss — the transition frame's representation is invariant to surrounding context and distinct from adjacent + look-alike frames. |
| "video llama3" | the VLM is **VideoLLaMA3** — explicitly *instead of* InternVideo2 (restated at the 2026-09-03 meeting). SigLIP-based vision tower for embeddings; Chat model for captions / reasoning. One model family for Stage 0, A, B. |
| "you don't want to use the short term actions, top down strategies" | **top-down**, not bottom-up: localize from the whole transcript structure downward; do **not** build the alignment out of short-horizon / atomic action detections. |
| "segments should be included in the vlm imports" | candidate segment spans are part of what is fed to the VLM, not bare frames. |
| "recent paper: pass things on vlm, do reasoning in real time" | a reference for VLM-in-the-loop reasoning at inference — **[NEED: which paper]**. |
| "action segmentation" / "raw vids shared" / "50Salads shared" | task framing is action segmentation; we now have the 50Salads videos. |

## Pipeline

### Stage 0 — semantic-guided sparse sampling
- Coarse frame grid (e.g. 1 fps) → VideoLLaMA3 SigLIP tower → `s0 (N_actions × T_coarse)`
  cosine similarity to the transcript action-name embeddings.
- Adaptive density: where the per-frame argmax / similarity profile over transcript
  actions shifts, mark a "transition-likely" zone; elsewhere "stable".
- Re-encode only the transition-likely zones at higher rate.
- Output: a non-uniform frame set + embeddings, at a fraction of full-rate cost.
- Tooling: `delta.features.extract --every N` (added 2026-09-03) does the coarse
  pass; adaptive re-sampling is Stage-0 code to write.

### Stage A — weak temporal alignment
- Order-preserving monotonic alignment (ASOT / DP — `delta.align.ta.align_dp`,
  and `--model_type wclot` in the WLTA code) of transcript → sampled frames on `s0`.
- **Must include a strong temporal / length prior.** Measured 2026-09-03: hard DP
  on a single-frame SigLIP2 similarity scores **MoC 0.199 < naive-uniform 0.366**
  on 50Salads split-1 — a weak `s` + unpriored DP is worse than "split evenly".
  So Stage A = ASOT with a temporal prior (or interpolate from the naive prior),
  not argmax-DP on raw VLM similarity.
- Output: **coarse segment spans**, boundaries good to ± the sampling gap.
- No boundary precision expected here — that's Stage B's job.

### Stage B — boundary refinement
For each coarse boundary between action `A` (ending) and `B` (starting):
- Densely encode a local window (± a few seconds) with VideoLLaMA3.
- Feed the candidate segment pair into the VLM as context ("segments in the
  VLM imports") — as a visual span marker or a text prompt with the candidate
  timestamps + the two action names.
- **Boundary contrastive loss** (CVA CBD): anchor = candidate transition frame,
  positives = same frame under augmentation, negatives = adjacent background +
  the `N_hard` most cosine-similar non-transition frames.
- **MASRA LRCA** on the window: push the frame×frame similarity toward the
  two-block (A | B) structure; **ESTA**: pull each side's pooled frames toward
  its action embedding.
- Output: refined boundary position.
- All boundaries refined → dense `Y*`.

### Stage C — downstream
Feed `Y*` to DELTA's decoder (CTC / CRF / duration unchanged) → Obs%/Pred% MoC.

## Where the analyzed papers land

| paper | role |
|---|---|
| ASOT (in DELTA `wclot`) | Stage A weak alignment |
| **MASRA** LRCA / ESTA | Stage 0 semantic sim + Stage B semantic-relational refinement (Eliz) |
| **CVA** CBD / CTE | Stage B boundary contrastive loss + local encoder (parallel workstream) |
| HAL | baseline number only |

## Cost argument

Full-rate VideoLLaMA3 encode of 50Salads ≈ 630k frames. Semantic-guided coarse
grid (1 fps ≈ 20k frames) + local refinement windows (≈ 50 boundaries × 300
frames × 50 videos ≈ 40k after dedup) ≈ **~10 % of full-rate**. Directly answers
"cost is also important".

## Open questions for the supervisor

1. Which recent "pass things to a VLM, reason in real time" paper?
2. "Segments in the VLM imports" — visual prompt (span drawn on frames) or text
   prompt (candidate timestamps + action names to VideoLLaMA3-Chat)?
3. Stage B: VideoLLaMA3-**SigLIP** (embeddings + contrastive loss) or
   VideoLLaMA3-**Chat** (generate a boundary judgement)? Or both — Chat proposes,
   SigLIP + contrastive verifies?
