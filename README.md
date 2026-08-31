# DELTA — multimodal self-supervised temporal alignment

Follow-up to **DELTA: Dense Long-Term Action Anticipation from Procedural
Transcripts** (Maté & Dimiccoli, UPC-IRI). That work learns dense long-term
action anticipation (DLTA) from *transcripts only* — ordered action lists with
no timestamps — via a temporal-alignment (TA) module that turns transcripts into
dense pseudo-labels, plus CTC sequence-consistency and locally-masked crossmodal
grounding.

**This repo's line of work:** replace the frozen **I3D + DistilBERT** inputs with
**vision-language foundation-model features** and strengthen the transcript→video
**temporal alignment**, targeting **50Salads first** (where DELTA trails fully
supervised methods: avg MoC 20.9 vs ~28.4, because boundaries/durations are hard
to recover from transcripts under frequent transitions).

See [`docs/approach.md`](docs/approach.md) for the research plan and
[`docs/data-preparation.md`](docs/data-preparation.md) for dataset setup.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

## Feature extraction

Produces one `<video_id>.npy` of shape `(D, T)` per video, with `T` equal to the
length of `groundTruth/<video_id>.txt` — a drop-in replacement for the benchmark
I3D features.

```bash
# 50Salads, VideoLLaMA3 vision tower (SigLIP so400m), all videos
python -m delta.features.extract --config configs/50salads.yaml

# restrict to a split / shard for cluster array jobs
python -m delta.features.extract --config configs/50salads.yaml --shard 0/8

# action-name text embeddings only (same space as the visual backbone)
python -m delta.features.extract --config configs/50salads.yaml --text-only
```

Backbones (`--backbone`): `vl3-siglip` (default), `siglip2`, `dinov2`,
`i3d-compat` (parity passthrough). Add more in
[`src/delta/features/backbones.py`](src/delta/features/backbones.py).

## Alignment / segmentation metrics

```python
from delta.align import segmentation_report, kendall_tau_alignment
segmentation_report(pred_frame_labels, gt_frame_labels)   # MoF, MoC, edit, F1@{10,25,50}
```

## Layout

```
src/delta/
    data/        50Salads / Breakfast conventions, transcripts, splits
    features/    frame sampling, frozen backbones, text encoder, extraction CLI
    align/       alignment & segmentation evaluation metrics
configs/         per-dataset YAML
scripts/         SLURM array job for extraction
tests/           CPU-only unit tests  (pytest -q)
```

## Status

- [x] Dataset indexing + transcript derivation (50Salads, Breakfast)
- [x] Foundation-model feature extraction pipeline (frame-grid aligned to GT)
- [x] Alignment / segmentation metrics
- [ ] DELTA re-implementation (no official code yet) — baseline reproduction
- [ ] Swap I3D → FM features in the TA module; measure pseudo-label quality
- [ ] Multimodal self-supervised alignment objective
- [ ] Full two-stage DLTA training + Obs%/pred% evaluation
