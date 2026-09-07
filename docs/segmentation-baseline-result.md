# Does alignment-through-segmentation beat VLM-direct on 50Salads?

Branch: `hal-segmentation-baseline`. Run 2026-09-04, split-1 test, I3D features.

## Why

Our VLM path keeps landing under the naive-uniform floor. Before committing to
it, test whether the classifier-based paradigm (ATBA, HAL) — which the project
moved away from — would have done better on this dataset.

## What was actually run

`delta.align.segbaseline` — ATBA's core loop, stripped:

```
pseudo-labels <- naive-uniform blocks
repeat 4x:
    train a temporal-conv frame classifier on the current pseudo-labels
    pseudo-labels <- order-preserving DP through its posteriors
```

`recon_weight > 0` adds an auxiliary reconstruction head as a light stand-in for
HAL's two-scale VAE regulariser.

**This is not HAL.** No VAE latents, no ELBO, no boundary detector, ~1 min of
training instead of 400 epochs. It tests the *paradigm*, not the paper. HAL's
published numbers are on Breakfast; it never ran 50Salads at all.

## Result

| method | MoF | MoC | edit | F1@50 |
|---|---|---|---|---|
| **naive-uniform floor** | 0.342 | **0.366** | 100.0 | 26.2 |
| VLM: ASOT + boundary refine | 0.323 | 0.352 | 100.0 | 23.4 |
| VLM: ASOT | — | 0.342 | 100.0 | 22.9 |
| **ATBA-style (classifier + DP)** | 0.241 | **0.202** | 100.0 | 14.0 |
| VLM: hard DP | 0.255 | 0.199 | 100.0 | 6.7 |
| **HAL-style (+ recon regulariser)** | 0.197 | **0.149** | 100.0 | 9.4 |

Segmentation scored *worse* than the VLM path, and the HAL-style regulariser
made it worse again.

## Why — read the training trace

```
round 0: loss 2.47, 73% of frames relabelled
round 1: loss 1.16, 21%
round 2: loss 0.68, 14%
round 3: loss 0.45,  8%
```

Loss collapses, relabelling stops. The classifier is **confidently fitting its
own round-0 mistakes** — textbook self-confirmation. Real ATBA carries a
class-agnostic boundary detector specifically to break this loop, and this
version has none. So the honest reading is *"the naive form of the paradigm
self-confirms on 50Salads"*, **not** *"HAL is bad"*.

The regulariser making things worse is consistent with the same story: it adds
a reconstruction pressure that further stabilises whatever labelling the model
has already locked onto.

## The finding that matters

**Every approach tried so far sits below the no-evidence floor** — VLM-direct,
classifier-based, with and without regularisers, hard DP and OT alike.

That points away from "which method" and toward the data: a fixed overhead
camera where consecutive frames are 0.94 cosine-similar, and where
`cut_tomato` / `cut_cucumber` / `cut_cheese` are near-indistinguishable from a
single frame (`50salads-notes.md`).

Implication for the plan: the open question is not *segmentation vs VLM* but
*what evidence could localise a boundary here at all*. That argues for temporal
context (video encoders, not frame encoders) and for the boundary-refinement
stage carrying the load — which is the current direction, so this run supports
continuing rather than pivoting.

## Caveats before quoting any of this

- Compact reimplementation, minutes of training, one split, one seed.
- No boundary detector — the component most responsible for ATBA's gains.
- Real HAL/ATBA on the cluster (`scripts/slurm_delta_baselines.sh`) remains the
  number to report. This is a directional probe, not a baseline result.
