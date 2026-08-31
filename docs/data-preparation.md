# Data preparation

## Expected layout

```
data/50salads/
    videos/            rgb-01-1.avi ...            (raw video, 30 fps, top view)
    groundTruth/       rgb-01-1.txt ...            (one action name per frame @ 15 fps)
    mapping.txt        "<idx> <class_name>"
    splits/            train.split{1..5}.bundle  test.split{1..5}.bundle
    features/          rgb-01-1.npy ...            (existing I3D, optional, for parity)

data/breakfast/
    videos/            P03/cam01/P03_cereals.avi ...   (15 fps)
    groundTruth/       P03_cam01_P03_cereals.txt ...
    mapping.txt
    splits/            train.split{1..4}.bundle  test.split{1..4}.bundle
    features/          P03_cam01_P03_cereals.npy ...
```

The pipeline is driven by `groundTruth/` + `mapping.txt` + `splits/` (the
standard MS-TCN/FUTR/ActFusion/DELTA bundle). Raw videos are needed only for
feature extraction.

## Sources

### Benchmark bundle (features + labels + splits) — what we use

**50Salads:** `dinggd/50salads` on HuggingFace — one `50salads.zip` (~4.0 GB)
containing `50salads/{features/*.npy, groundTruth/*.txt, splits/*.bundle,
mapping.txt}` in exactly the layout above. This is the standard MS-TCN / FUTR /
ActFusion bundle.

```bash
curl -L -C - -o data/_dl/50salads.zip \
  https://huggingface.co/datasets/dinggd/50salads/resolve/main/50salads.zip
unzip -q data/_dl/50salads.zip -d data/        # -> data/50salads/
```

Verified facts (from the download):
- **30 fps** — `features/<id>.npy` is `(2048, T)` with `T == len(groundTruth)`,
  T ≈ 7.6k–18.1k. `configs/50salads.yaml` and `DATASET_DEFAULTS` set
  `source_fps = label_fps = 30`. (FUTR/ActFusion additionally downsample ×2 at
  load time — a model choice, not the stored grid.)
- 19 classes: 17 actions + `action_start` + `action_end` (mid-level granularity).
- No `background` label; `action_start`/`action_end` wrap every video (~14% of
  frames) and are treated as background by `delta.data.stats`.

**Breakfast:** `dinggd/breakfast` on HuggingFace (same structure, larger).

### Raw videos

Needed only for frame-level visualisation. The official host
`cvip.computing.dundee.ac.uk` is **currently down (NXDOMAIN)** and there is no
known public mirror of the `.avi` files. Ask the professor / lab for a copy.
Drop them at `data/50salads/videos/rgb-XX-Y.avi` and set `video_dir` in the
config; the pipeline picks them up automatically.

## Sanity checks before extracting

```bash
python - <<'PY'
from delta.data import ActionSegDataset
ds = ActionSegDataset("50salads", "data/50salads", video_dir="data/50salads/videos")
print(ds)
rec = ds.record(ds.all_ids()[0])
print(rec.video_id, "labels:", rec.num_label_frames, "transcript:", rec.transcript)
print("video found:", rec.has_video())
PY
```

Confirm: (1) every `groundTruth` id resolves to a video file, (2) `num_label_frames`
is ~ `video_frames * label_fps / source_fps`, (3) transcripts look right
(≈20 actions for 50Salads, ≈6 for Breakfast).
