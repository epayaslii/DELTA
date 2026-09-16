# Literature summaries

Full paragraph per paper — companion to the terse mapping in
`docs/articles-by-pipeline-part.xlsx`. Grouped by role in the pipeline. Links
in the xlsx; not repeated here.

## The base paper

**DELTA** — *Dense Long-Term Action Anticipation from Procedural Transcripts*
(UPC-IRI). Task: given an observed prefix of a video and *only* the ordered
list of actions in the recipe (no timestamps, ever), predict the dense
frame-level labelling of the unobserved future. Architecture: a shared
transformer encoder feeds three branches — a **Temporal Alignment (TA)**
module that turns the transcript into dense pseudo-labels `Y*` (following
ATBA's boundary-detector + dynamic-programming mechanism), a **TAS head**
supervised by `Y*` + CTC, and a **parallel anticipation decoder** (20
learnable queries, a linear-chain CRF, a duration head) that predicts the
future action sequence and its durations. The transcript, the alignment
module, and `Y*` are all discarded at inference — the decoder runs on the
visual stream alone. On 50Salads it trails fully-supervised methods (avg MoC
20.92 vs ~28.4), which the paper attributes to unreliable boundary/duration
recovery under frequent transitions — exactly the gap this project targets.

## Stage-by-stage method sources (our hybrid)

**OVTAS** — *Exploring Vision-Language Models for Open-Vocabulary Zero-Shot
Action Segmentation* (arXiv 2602.21406, Feb 2026). Training-free
segmentation-by-classification: **FAES** (Frame-Action Embedding Similarity)
scores every frame against every candidate action label with a frozen VLM;
**SMTS** (Similarity-Matrix Temporal Segmentation) smooths that into a
segmentation, with no assumption about action order (open-vocabulary — the
action set is unordered and not tied to a specific video's recipe). Surveys
14 VLMs for this. We take FAES's mechanism (frozen VLM clip↔action cosine) but
keep the transcript's order, which OVTAS itself discards — our version scores
similarity per *transcript position*, not per open-vocabulary label.

**HiERO** — *understanding the hierarchy of human behavior enhances reasoning
on egocentric videos* (ICCV 2025). Weakly-supervised representation learning:
aligns video clips with narrated descriptions to enrich clip features with the
video's *hierarchical* activity structure (actions → threads → routines),
learned from unscripted egocentric video with no manual annotation beyond
narrations. Zero-shot procedure-step localization by simple clustering in the
resulting feature space beats fully-supervised baselines by +12.5 F1 on
EgoProceL. We take the principle — represent actions with temporal *context*,
not isolated frames, and reason top-down from structure rather than building
up from atomic action detections. Domain gap: egocentric (head-mounted,
constant camera motion) vs. our fixed overhead camera.

**HiERO-StepG** — *HiERO-StepG @ Ego4D Step Grounding Challenge* (arXiv
2605.31227). Applies HiERO's features to the Ego4D Step Grounding task:
localize a procedural step's temporal boundaries given its free-form text
description, zero-shot, no procedure-specific annotation. Augments the base
HiERO clustering with **strict temporal monotonicity** on the grounded steps,
fine/coarse-level agreement between step assignments, and noise
post-processing. Ranked 2nd on the public leaderboard (56.27% R@1 @ IoU 0.3)
fully zero-shot. This is architecturally closest to our Stage A: "ordered step
descriptions → monotonic temporal grounding" is precisely the coarse-alignment
problem we solve, just on a different domain and without HiERO's own features.

**ASOT** (Xu & Gould, CVPR 2024) / **CLOT** (ICCV 2025) — the optimal-transport
alignment lineage the group's own `wclot` code already implements. ASOT casts
segmentation as fused Gromov-Wasserstein OT between frames and action
labels/prototypes, with no assumption about action ordering — a class-agnostic
structural prior (frames near each other in time should share a label) plus an
unbalanced marginal on the action side. CLOT adds a closed multi-level cyclic
refinement loop (frame embeddings ↔ segment embeddings ↔ refined pseudo-labels,
three chained OT problems) on top. We reuse ASOT's solver as our Stage A
engine, replacing its unsupervised cluster identities with our transcript
entries and adding a transcript-order temporal prior.

**Our design — local semantic boundary search (Stage B1).** Not from any
paper. For each coarse boundary the OT/monotonic decode produces, open a
±window and score every candidate crossover point by "does the left side look
like action A and not B, does the right side look like B and not A" (+ an
optional visual-change term), pick the best, and attach a confidence from how
*prominent* that peak is relative to the rest of the search window (a real
localized boundary stands out from noise; a flat/noisy profile doesn't). This
confidence feeds the refinement stage and, eventually, which boundaries get
escalated to expensive VLM reasoning.

**CVA** (CVPR 2026) — *Context-aware Video-text Alignment*. Task: video
temporal grounding (one NL query → one span), fully supervised on ground-truth
spans, SlowFast+CLIP features, SOTA on QVHighlights/Charades-STA/TACoS. Three
components: QCD (query-aware background-mixing augmentation), CTE (a
hierarchical encoder — windowed local self-attention + learnable global
queries + bidirectional cross-attention), and **CBD** (a context-invariant
boundary-discrimination contrastive loss — the boundary frame's representation
should be invariant to augmentation and distinct from adjacent/confusable
background frames). We take only the *principle* from CBD, not the loss
itself: published CBD anchors on GT-span boundary indices and defines
negatives relative to the GT span, neither of which we have. Our version
(PBCR, `delta.align.cbd`) rebuilds it on *confident* pseudo-boundaries from
Stage B1 instead — a genuine adaptation, not a reuse.

**MASRA** (arXiv 2605.03398, ACM track, 2026) — *MLLM-Assisted
Semantic-Relational Consistent Alignment*. Same task as CVA (grounding, fully
supervised), same feature backbone family, but its contribution is a
training-time regularizer rather than the aligner itself: an MLLM (GPT-5,
run offline, discarded at inference) generates event-level descriptions and
clip-level captions; **ESTA** aligns pooled temporal context with those event
descriptions, **LRCA** aligns a text-derived relation matrix (cosine
similarity between clip captions) with the model's own frame-similarity
matrix. We take LRCA/ESTA's mechanism — align a relational structure to a
text-derived target — dropping the MLLM entirely and building the target from
the transcript + class-name embeddings instead.

**VideoLLaMA3** (arXiv 2501.13106, 2025) — the frozen encoder for every stage
above. A SigLIP-based vision tower (for clip↔action similarity) paired with an
LLM decoder (for offline captioning and, later, hard-boundary reasoning) —
one model family for the whole pipeline. Chosen over InternVideo2 by explicit
instruction.

## Later-version / fallback sources

**TASOT** (arXiv 2602.24138, Feb 2026) — *Multimodal Optimal Transport for
Training-free Temporal Segmentation in Surgical Robotics*. Extends ASOT by
fusing a VLM-generated temporal-caption semantic cost into the same fused-GW
OT objective, annotation-free, on surgical phase segmentation. Closest
existing template to our whole method (OT alignment + VLM captions in the
cost, no labels) — different domain, and it doesn't have an ordered transcript
to exploit the way we do. Planned as a Stage-A **ablation**, not the default.

**TOGA** (arXiv 2506.09445, Jun 2025) — *Temporally Grounded Open-Ended Video
QA with Weak Supervision*. A VLM instruction-tuned to jointly generate an
open-ended answer *and* its temporal grounding, trained without any GT
grounding labels via self-generated pseudo-labels validated by a
consistency constraint (a grounding answer and its "what happens at
[start,end]?" counterpart must agree). Genuinely weakly supervised — the
closest paper on that axis to our own constraint. Not a segmentation method
(one query → one span); planned use is narrow: prompt it with a low-confidence
candidate boundary's `[start,end]` and the two action names, use its judgement
or the consistency check as a tie-breaker, only for the hardest cases.

**D-CLOT** (TPAMI submission, arXiv 2608.05877) — CLOT's successor. Re-estimates
action prototypes from the OT-refined frame geometry (rather than only via
loss gradients) through a graph-constrained regularizer, explicitly targeting
"representation–prototype inconsistency" around ambiguous transitions and
short/infrequent actions — precisely 50Salads's failure mode. Also releases
V-JEPA2 features for Breakfast/Assembly101 (not 50Salads). Planned as a V3
fallback if Stage B2 alone doesn't handle noisy boundaries well enough.

## Baselines

**ATBA** (Xu & Zheng, CVPR 2024) — *Action-Transition-Aware Boundary
Alignment*. "Alignment through segmentation": a class-agnostic boundary
detector (JS-divergence over a local kernel on the frame classifier's
posteriors) proposes candidates, NMS filters them, drop-allowed dynamic
programming selects the transitions best matching the transcript, inducing
`Y*`. DELTA's TA module follows this mechanism directly. Reports Breakfast /
Hollywood / CrossTask — never 50Salads.

**HAL** (arXiv 2602.24275, CVPR 2026) — *ATBA + a two-scale VAE regularizer.*
Adds a slow/fast latent split at two encoder layers with reconstruction + KL
+ a smoothness constraint (slow latent shouldn't change faster than fast).
The boundary detector, DP, and core losses are byte-identical to ATBA's. +2–3
MoF on Breakfast, some within noise. Never evaluated 50Salads. Dropped as a
workstream (still classifier-based, no VLM) — kept only as a baseline number.

## Companion citations

**MLLM4WTAL** (CVPR 2025, arXiv 2411.08466) — *Weakly Supervised Temporal
Action Localization via Dual-Prior Collaborative Learning Guided by MLLMs.*
The CVPR-stamped citation for the "MLLM guides a weakly-supervised model at
training time only, discarded at inference" paradigm — different task (WTAL,
video-level labels → localization) and not procedural, but the same
methodological family as MASRA.

**StepFormer** (CVPR 2023), **TAN** (CVPR 2022 Oral), **Drop-DTW** (NeurIPS
2021) — the genuine pre-VLM "ordered sequence → video" aligners: learnable
step slots with an order-aware alignment loss (StepFormer), sentence-to-video
alignment from noisy narration (TAN), a differentiable sequence-to-video DTW
that can drop outliers (Drop-DTW). Historical grounding for what "temporal
alignment" meant before VLM-based semantic matching.

**VAOT/VASOT** (arXiv 2503.16832) — *Joint Self-Supervised Video Alignment and
Action Segmentation.* A single fused-GW OT model for both video-to-video
alignment (not transcript-to-video) and segmentation, self-supervised, no VLM.
Background reference for the OT-with-structural-priors recipe.
