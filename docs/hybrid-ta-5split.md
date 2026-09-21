# Hybrid TA on 50Salads — all 5 splits (local, coarse SigLIP2)

`scripts/eval_hybrid_5split.py`, raw numbers in `hybrid-ta-5split.json`. Frozen SigLIP2 single-frame features every
30 frames, class-name text embeddings, **no trained parameters** (so "5 splits" = every one of the 50 videos scored once,
grouped by the split whose test set holds it). Input = the ordered transcript only; frame labels are used for scoring.
This scores the alignment `Y*` against ground truth (the TA task), **not** anticipation MoC.

## Result (mean over 50 videos)

| method | 19-class MoF | 19-class MoC | F1@50 | 17-class MoC |
|---|---|---|---|---|
| naive-uniform (floor) | 0.335 | 0.342 | 20.2 | 0.286 |
| hard DP on similarity | 0.232 | 0.188 | 5.2 | 0.197 |
| ASOT (coarse only) | 0.425 | 0.362 | 19.3 | 0.342 |
| **ASOT + branch A** | **0.489** | **0.422** | **24.8** | **0.395** |
| ASOT + branch B | 0.378 | 0.335 | 18.0 | 0.311 |
| ASOT + A + B | 0.441 | 0.395 | 22.9 | 0.366 |

* The naive floor reproduces the earlier reference (0.335 / 0.342 / 20.3 at 19 classes; 0.283 / 0.286 at 17).
* **Branch A helps in every split** (19-class MoC, ASOT → ASOT+A: 0.327→0.413, 0.347→0.428, 0.400→0.449, 0.420→0.433,
  0.318→0.388). **Branch B hurts in four splits and ties in the fifth (split 3: 0.398 vs 0.400)**, and A+B is below A alone in all five. Hence the default is ASOT + A.
* ASOT+A beats the naive floor on 4 of 5 splits at 19 classes (split 1: 0.413 vs 0.422 — a tie/loss) and on all 5 at 17
  classes. Overall +8.0 MoC points (19-class) and +10.9 (17-class) over naive.
* On split 1 alone (17-class) the earlier ATBA-style probe scored 0.202 and naive 0.366; ASOT+A scores 0.393.
* `edit` is 100 for every order-enforcing method by construction — uninformative here.

## Caveats
* Hyper-parameters are defaults (`rho 0.15, alpha 0.3, A radius 30 / window 20` **in grid units** of 30 frames). The
  `rho 0.5 / alpha 0.1` setting picked by eye on split 1 earlier does *not* generalise to splits 2–5 (0.360 vs 0.362).
  A radius/window of 90/40 grid units (~2.7 k frames) is far too large (0.254): those rows are in the json for
  completeness, not as a fair comparison with the earlier full-resolution 90/40.
* Coarse, single-frame SigLIP2 (cosines vary ~0.05 across actions). Video-level features (VideoLLaMA3) are the untested
  upgrade. This is training-free alignment quality; the downstream effect on anticipation MoC needs the GPU runs.
* No ATBA / DELTA-TA number on this same protocol yet — that comes from the instrumented DELTA run.
