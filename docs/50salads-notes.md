# 50Salads — why it's hard for transcript-only temporal alignment

Findings from Stage 1 (`notebooks/stage1_dataset_analysis.ipynb`), on the
`dinggd/50salads` benchmark bundle: 50 videos, 19 classes (17 actions +
`action_start`/`action_end`), I3D 2048-d features + `groundTruth` at **30 fps**.
No raw video (official host down). Numbers are means over all 50 videos unless
noted.

## The shape of the data

| | 50Salads |
|---|---|
| videos | 50 (40/10 train/test per split, 5 splits) |
| frames / video | 7.6k – 18.1k (mean 11.5k) |
| length | mean 6.4 min |
| segments / video | 15 – 26 (**mean 20.0**) |
| `action_start`+`action_end` share | 14% of frames |

Matches the DELTA paper's "roughly 20 action instances each".

## Five reasons it's hard

### 1. Frame counts are dominated by a few long actions → use MoC, not MoF
`cut_*` / `place_*_into_bowl` / `mix_*` own most frames; the
`add_{salt,vinegar,oil,pepper}` steps are frame-rare but each occurs once per
recipe. A frame-weighted loss (fully-supervised methods) is driven by the long
actions; a transcript counts every action once. → **Mean-over-Classes is the
headline metric**, and this is exactly the "frequency bias vs procedural
structure" argument in the paper.

### 2. Duration is weakly determined by the transcript
Most variable classes sit at **CV ≈ 0.5–0.8** (`place_tomato_into_bowl`: mean
~8 s, CV 0.78 → ~2–20 s). One ordered occurrence carries little information
about extent → the duration head has weak signal (paper: duration loss gives
**+0.2 MoC on 50S** vs +3.3 on Breakfast).

### 3. Order ≠ timing
Mean next-action entropy over real classes: **~1.9 bits** (~3–4 plausible
successors per action). The transcript order alone does **not** pin down the
timeline — visual evidence has to carry the alignment.

### 4. …but the visual evidence is weak (the core problem)
ATBA's candidate boundaries come from a *class-agnostic* frame-change score.
Proxy: consecutive-frame cosine distance of the I3D features.

| I3D consecutive-frame cosine distance | value |
|---|---|
| at ground-truth boundaries | 0.0240 |
| at random frames | 0.0216 |
| **ratio** | **1.11×** |

Boundaries barely stand out. Fixed overhead camera, near-continuous hand
motion, and near-duplicate fine-grained classes
(`cut_tomato`/`cut_cheese`/`cut_lettuce`/`cut_cucumber`,
`add_oil`/`add_vinegar`/…) mean the class-agnostic score `v^b` is noisy → true
transitions may never enter the candidate set, and no downstream module can
recover them. This is the concrete form of the
[`temporal-alignment.md`](temporal-alignment.md) §4 hypothesis.

### 5. Long sequences → error propagation
~20 segments over ~11k frames. ATBA's DP chain is long; a single early
mislocalised boundary shifts everything after it, and in DELTA that error flows
straight into the future transcript `T*` and durations `d*`.

## The floor to beat

Naive uniform alignment (split each video into `len(transcript)` equal parts,
in transcript order — zero visual evidence):

| metric | naive uniform |
|---|---|
| MoF | 0.335 |
| MoC | 0.342 |
| edit | 100.0 *(order is exact by construction)* |
| F1@10 / @25 / @50 | 49.4 / 40.6 / 20.3 |

Any real temporal-alignment method must clear **MoC ≈ 0.34**. ATBA reports
pseudo-label accuracy ~65% on Breakfast; the §4 evidence above says 50Salads
will land lower, and closing that gap is the project.

## Implications for Stage 2

- The highest-leverage intervention is a **better boundary signal** (`v^b`) and
  **better posteriors** `P` feeding `V^a` — not the DP itself.
- Measure `Y*` quality with **MoC** primarily; report MoF, edit, F1@k, and
  median per-transition boundary offset alongside.
- Keep the naive-uniform and (later) supervised-warm-up numbers as the floor
  and ceiling around every ATBA result.

## Measurement — single-frame SigLIP2 as a direct aligner (2026-09-03)

Coarse **SigLIP2-so400m** frame embeddings (`get_image_features`, 1 fps, nearest-
filled), split-1 test (n=10). Transcript×frame cosine → hard DP alignment
(`delta.align.ta.align_dp`).

| | naive-uniform | SigLIP2 + DP |
|---|---|---|
| MoC | 0.366 | **0.199** |
| F1@50 | 26.2 | 6.7 |
| edit | 100 | 100 |

**A single-frame image-CLIP direct aligner is worse than the no-evidence floor.**
Why: fixed overhead camera → frame–frame cosine ≈ 0.94 (features barely vary);
text–image cosine ≈ 0.09 (aligned but weak); it separates coarse verbs
(cut vs add-dressing) but not *which* vegetable, and locks onto one wrong action
for long runs. A noisy `s` + hard DP underperforms the equal-length prior.

Diagnostics vs the I3D reference (oracle/naive):
- LRCA-block residual 0.52 (I3D 0.465 / 0.483) — **less** block structure than I3D
- ESTA `1−cos` 0.90 → semantic pull ≈ 0.10
- boundary score at GT 0.52 ≈ chance (I3D 0.67)

Conclusion: the "VLM-direct alignment" framing needs (a) a **video / temporal**
encoder, not single frames, and (b) **weak-align-then-refine** — let the
transcript prior carry the coarse structure (ASOT with a temporal prior, or the
naive prior), use the VLM only for local boundary refinement. See
`docs/method-weak-then-refine.md`.

### Stage A with fused-GW OT — same features (2026-09-04)

`delta.align.asot` (fused-GW OT + transcript temporal prior), same coarse
SigLIP2 features, split-1 test:

| method | MoC | F1@50 |
|---|---|---|
| naive-uniform | **0.366** | 26.2 |
| hard DP on VLM sim | 0.199 | 6.7 |
| **ASOT** (rho 0.5, alpha 0.1) | 0.342 | 22.9 |
| ASOT (rho 0.3, alpha 0.3) | 0.323 | 17.7 |

ASOT + the temporal prior **recovers the hard-DP collapse** (0.199 → 0.34) but
still can't clear the naive prior: the GW structure term *hurts* (alpha↑ → MoC↓,
the near-constant features blur), and `w_sem·(1−cos)` contributes almost nothing
because SigLIP2 cosines vary only ~0.05 across actions. The Stage-A machinery is
correct; **the features are the bottleneck** — next is VideoLLaMA3 features
(supervisor's call) + TASOT-style VideoLLaMA3 captions (M-A4, cluster).

### Stage B1 — local semantic boundary search (2026-09-04)

`delta.align.refine`: ±r window around each coarse boundary, score each position
by "left = A not B, right = B not A" + a visual-change term, pick the best,
attach a prominence confidence.

| | MoC | F1@50 |
|---|---|---|
| ASOT | 0.342 | 22.9 |
| ASOT + refine (r=90, w=40) | 0.352 | 23.4 |

+0.01 MoC. The **confidence signal is the useful output**: over 199 boundaries,
mean confidence 0.27, only **5 % above 0.5**. Mean \|shift\| 21.6 frames. The
search correctly reports it *cannot* confidently localise boundaries from these
features — which is (a) the right diagnostic, (b) exactly the input Stage B3
needs (route the low-confidence 95 % to a chat-VLM). Re-run once VideoLLaMA3
features exist.
