# Literature summaries — short version

One line each. Full paragraphs: `literature-summaries.md`. Split by what's
recent SOTA we adapt from vs. what we inherited as the starting infrastructure.

## SOTA papers we adapt from (2025–2026)

| paper | one-liner |
|---|---|
| **OVTAS** (Feb 2026) | Training-free VLM clip↔action similarity for zero-shot segmentation, open-vocabulary (no order) — we keep its scoring mechanism, add back the transcript's order. |
| **HiERO** (ICCV 2025) | Weakly-supervised hierarchical video features (action → thread → routine) from narrations, top-down, no manual annotation. |
| **HiERO-StepG** (2605.31227) | HiERO applied to zero-shot step grounding with strict temporal monotonicity — architecturally the closest match to our Stage A. |
| **CVA** (CVPR 2026) | SOTA supervised video-text grounding; its CBD loss (boundary frames should be context-invariant) is the *idea* behind our own PBCR objective — not reused directly, since CBD needs GT spans we don't have. |
| **MASRA** (ACM 2026) | MLLM-assisted training-time regularizer for grounding (LRCA/ESTA align a relational structure to a text-derived target) — we drop the MLLM, build the target from the transcript instead. |
| **TASOT** (Feb 2026) | ASOT + VLM-caption semantics fused into one OT cost, annotation-free, surgical domain — closest existing template to our whole method; used as a Stage-A ablation. |
| **TOGA** (Jun 2025) | Weakly-supervised VLM grounding via self-consistency, no GT timestamps — reserved for our hardest, lowest-confidence boundaries only. |
| **D-CLOT** (TPAMI sub.) | CLOT's successor; re-estimates prototypes around ambiguous transitions/short actions — exactly our failure mode, kept as a V3 fallback. |
| **VideoLLaMA3** (2025) | The frozen backbone for the whole pipeline — vision tower for similarity, chat model for captions/reasoning. |

## Parts we inherited (the starting infrastructure, not something we found)

| what | from | what it already gave us |
|---|---|---|
| **DELTA** (base paper + code) | the group | the whole task definition, the downstream decoder (CTC/CRF/duration head), and the ATBA-style TA it originally used — which we're replacing. |
| **ASOT** (CVPR 2024) | already in the group's code (`--model_type wclot`) | the optimal-transport solver we use as our Stage-A engine, unmodified in spirit — we swap its unsupervised cluster identities for our transcript entries and add a temporal prior. |
| **CLOT** (ICCV 2025) | the group's own prior work | the closed-loop frame↔segment OT refinement idea layered on top of ASOT; same lineage as `wclot`. |
| **ATBA** (CVPR 2024) | baseline reference | "alignment through a trained classifier" — the mechanism DELTA's original TA follows; now our baseline to beat, not our method. |
| **HAL** (CVPR 2026) | baseline reference | ATBA + a VAE regularizer; dropped as a workstream (still classifier-based, no VLM), kept only as a number. |

## What's genuinely ours, not inherited from anywhere

**Stage B1 — the local semantic boundary search** (`delta.align.refine`): search
a window around each coarse boundary, score "left = A not B, right = B not A"
+ a visual-change term, pick the best, attach a confidence from how prominent
that peak is. No paper source — this is the project's own contribution.
