# DELTA — multimodal self-supervised temporal alignment

Follow-up to **DELTA: Dense Long-Term Action Anticipation from Procedural
Transcripts** (UPC-IRI). That work learns dense long-term
action anticipation (DLTA) from *transcripts only* — ordered action lists with
no timestamps — via a temporal-alignment (TA) module that turns transcripts into
dense pseudo-labels, plus CTC sequence-consistency and locally-masked crossmodal
grounding.

**This repo's line of work:** replace the frozen **I3D + DistilBERT** inputs with
**vision-language foundation-model features** and strengthen the transcript→video
**temporal alignment**, targeting **50Salads first** (where DELTA trails fully
supervised methods: avg MoC 20.9 vs ~28.4, because boundaries/durations are hard
to recover from transcripts under frequent transitions).

## Docs

| Doc | What's in it |
|---|---|
| [`docs/temporal-alignment.md`](docs/temporal-alignment.md) | **TA reference** — how the alignment module works (ATBA-grounded), TA→TAS→DLTA dependency, weaknesses, the VLM-direct direction. Start here. |
| [`docs/delta-code.md`](docs/delta-code.md) | The DELTA implementation ("WLTA", `third_party/delta_wlta/`) — what it is, how the experiments run, the gotchas |
| [`docs/hal-analysis.md`](docs/hal-analysis.md) | HAL (CVPR'26) = ATBA + a VAE regulariser; can it be a stronger TA baseline? verdict + first experiment |
| [`docs/baselines-hal-cva.md`](docs/baselines-hal-cva.md) | HAL & CVA (CVPR'26) as baselines on the two axes; scoped research statement + phased TA plan |
| [`docs/approach.md`](docs/approach.md) | Research plan, literature review, milestones, questions for the professor |
| [`docs/50salads-notes.md`](docs/50salads-notes.md) | Why 50Salads is hard for transcript-only alignment, with numbers |
| [`docs/data-preparation.md`](docs/data-preparation.md) | 50Salads / Breakfast layout, download, sanity checks |

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

## Alignment / segmentation

```python
from delta.align import segmentation_report, similarity_matrix, align_dp

s = similarity_matrix(text_emb, frame_emb)          # (N, T) frozen-VLM transcript×frame
y_star = align_dp(s, transcript).y_star             # order-preserving alignment -> pseudo-labels
segmentation_report(y_star, gt_frame_labels)        # MoF, MoC, edit, F1@{10,25,50}
```

```bash
# baselines on 50Salads (no VLM features yet)
python -m delta.align.evaluate --config configs/50salads.yaml --provider naive  --split 1
python -m delta.align.evaluate --config configs/50salads.yaml --provider oracle --split 1   # aligner sanity
# once VLM features exist:
python -m delta.align.evaluate --config configs/50salads.yaml --provider vlm --split 1 \
    --frame-dir data/50salads/features_vl3siglip --class-emb .../action_name_embeddings.npy
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
- [x] **Stage 0** — env (`.venv`, py3.11) + 50Salads benchmark bundle downloaded & verified (`dinggd/50salads`, 30 fps)
- [x] **Stage 1** — 50Salads dataset analysis: `delta.data.stats`, `delta.viz.timeline`, [`notebooks/stage1_dataset_analysis.ipynb`](notebooks/stage1_dataset_analysis.ipynb), [`docs/50salads-notes.md`](docs/50salads-notes.md) *(video-watching skipped — raw `.avi` host is down)*
- [~] **Stage 2** — VLM-direct alignment (`docs/temporal-alignment.md` §5). Aligner done: `delta.align.{similarity,ta,evaluate}` (order-preserving DP + soft forward-backward), validated on real 50Salads transcripts; ATBA vendored as baseline. **Blocked on raw video** for the VLM similarity matrix.
- [ ] **Stage 3** — minimal FUTR-style DLTA decoder on the VLM-direct `Y*`

