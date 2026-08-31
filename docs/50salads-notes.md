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
