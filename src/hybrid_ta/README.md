# hybrid_ta — Hybrid Temporal Alignment (pseudo-labels Y*)

```
frame features (T,D) ──┐
class-name embeddings ─┴─► similarity.py ──► sim (T,N)          # cosine, transcript columns
                                             │
                          coarse_align.py ◄──┘   monotone, every transcript entry once (asot | dp)
                                             │
                     ┌───────────────────────┴───────────────────────┐
        boundary_refinement.py A                        boundary_refinement.py B
        semantic transition sharpening                  relational block-structure (LRCA-style)
        (CVA principle; delta.align.refine)             + neighbour-clipped windows
                     └───────────────────────┬───────────────────────┘
                              pseudo_labels.py  (confidence-weighted fusion, ordered, min length)
                                             │
                                             ▼  Y* (B,T) long, -100 on padding
```

* Built on the unit-tested `delta.align` primitives (needs `pip install -e .`).
* `provider.py` is the training hook behind `--ta_source hybrid` (reads `scripts/make_hybrid_sim.py` output).
* Y* replaces ATBA's pseudo-labels *inside* `LossFn.forward` (`external_pse_la`), so the frame loss, contrastive loss,
  text grounding, CTC/CRF/decoder and the evaluation protocol are untouched.
* Weak supervision: only the ordered transcript is used; no frame labels or durations.
* Similarity is only meaningful in a text-aligned space (SigLIP2 / VideoLLaMA3). `FrameActionSimilarity` can project 2048-D
  I3D and text to a shared 256-d, but those projections are untrained.

## First numbers (10 local videos, coarse SigLIP2, T=256, frame accuracy vs GT — GT used for scoring only)

| coarse | none | A | B | A+B |
|---|---|---|---|---|
| dp   | 0.261 | 0.290 | 0.200 | 0.235 |
| asot | 0.366 | **0.423** | 0.289 | 0.390 |

ASOT ≫ DP; branch A helps; **branch B hurts and drags A+B below A alone.** So the default is **ASOT + A**; B stays
available (`--hybrid_branches b|ab`) as an ablation. This is one split's test set (10 videos), no tuning.
