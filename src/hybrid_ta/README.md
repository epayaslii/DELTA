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

## Numbers

All 50 videos / 5 splits, coarse SigLIP2, training-free (`docs/hybrid-ta-5split.md`):

| method | 19-class MoC | 17-class MoC | F1@50 |
|---|---|---|---|
| naive-uniform | 0.342 | 0.286 | 20.2 |
| ASOT | 0.362 | 0.342 | 19.3 |
| **ASOT + A (default)** | **0.422** | **0.395** | **24.8** |
| ASOT + B | 0.335 | 0.311 | 18.0 |
| ASOT + A + B | 0.395 | 0.366 | 22.9 |

A helps in every split; B hurts (four splits) or ties (one), so B is opt-in (`--hybrid_branches b|ab`).
