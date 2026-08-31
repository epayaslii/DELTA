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

- **50Salads**: raw videos + annotations from the official page
  (<https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/>).
  Kaggle mirror `fedecvg/50salads-idt` provides pre-extracted features and the
  benchmark bundle; **check whether its `features/` are I3D or IDT** and whether
  the label grid is 15 or 30 fps, then set `features.label_fps` / `source_fps`
  in `configs/50salads.yaml` accordingly.
- **Breakfast**: `mohamedadlyi/breakfast-activity-recognition-dataset` on Kaggle,
  or the official coarse/fine annotations from
  <https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/>.

```bash
pip install kaggle
kaggle datasets download -d fedecvg/50salads-idt -p data/_dl --unzip
kaggle datasets download -d mohamedadlyi/breakfast-activity-recognition-dataset -p data/_dl --unzip
# then arrange into the layout above (a helper is intentionally not committed
# until we see the real archive structure)
```

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
