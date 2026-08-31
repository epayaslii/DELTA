"""Dataset conventions for 50Salads and Breakfast.

Both datasets are distributed (and consumed by MS-TCN / FUTR / ActFusion / DELTA)
in the same layout::

    <root>/
        features/       <video_id>.npy        # (D, T) or (T, D)  -- pre-extracted I3D
        groundTruth/    <video_id>.txt         # one action-name per frame, T lines
        splits/         train.split{k}.bundle  # list of "<video_id>.txt"
                        test.split{k}.bundle
        mapping.txt     "<class_idx> <class_name>" per line

The *temporal resolution* of ``groundTruth`` defines the canonical frame grid.
Our foundation-model feature extractor must emit exactly ``len(groundTruth)``
vectors per video so the new features are a drop-in replacement for I3D.

Frame-rate notes (verify against your actual download -- the Kaggle mirrors may
differ slightly):
  * 50Salads : source video 30 fps, ground truth annotated at 30 fps but the
    standard benchmark features/labels are *downsampled to 15 fps* (stride 2).
  * Breakfast: source video 15 fps, ground truth at 15 fps (stride 1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# per-dataset defaults
# --------------------------------------------------------------------------------------

DATASET_DEFAULTS = {
    "50salads": dict(
        num_splits=5,
        source_fps=30.0,
        label_fps=30.0,          # benchmark grid: features & groundTruth are per-frame @ 30 fps
                                 # (dinggd/50salads bundle: feature T == len(groundTruth), ~11.7k-17.8k frames).
                                 # Many models (FUTR/ActFusion) additionally downsample by 2 at load time.
        num_classes=19,          # 17 actions + action_start + action_end  (mapping.txt is authoritative)
        video_glob=["*.avi"],
        # 50Salads ships one "rgb-XX-Y.avi" per recording; ground-truth id == video stem
        video_id_from_stem=lambda s: s,
    ),
    "breakfast": dict(
        num_splits=4,
        source_fps=15.0,
        label_fps=15.0,
        num_classes=48,
        video_glob=["*.avi", "*.mp4"],
        # Breakfast videos live in P##/<cam>/... ; the benchmark id joins them,
        # e.g. file "P03/cam01/P03_cereals.avi"  ->  id "P03_cam01_P03_cereals"
        video_id_from_stem=lambda s: s,
    ),
}


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------


@dataclass
class VideoRecord:
    video_id: str
    video_path: Path | None
    gt_path: Path | None
    feature_path: Path | None
    num_label_frames: int | None = None            # len(groundTruth)
    frame_labels: np.ndarray | None = field(default=None, repr=False)  # (T,) int
    transcript: list[int] | None = None            # collapsed consecutive labels

    def has_video(self) -> bool:
        return self.video_path is not None and self.video_path.exists()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def load_mapping(mapping_file: str | os.PathLike) -> tuple[dict[str, int], list[str]]:
    """Return (name->idx, idx->name) from a 'mapping.txt' ('<idx> <name>' per line)."""
    name_to_idx: dict[str, int] = {}
    pairs: list[tuple[int, str]] = []
    with open(mapping_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split(maxsplit=1)
            idx = int(idx_str)
            name_to_idx[name] = idx
            pairs.append((idx, name))
    pairs.sort()
    idx_to_name = [name for _, name in pairs]
    return name_to_idx, idx_to_name


def read_groundtruth(gt_path: str | os.PathLike, name_to_idx: dict[str, int]) -> np.ndarray:
    with open(gt_path) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    try:
        return np.array([name_to_idx[n] for n in names], dtype=np.int64)
    except KeyError as e:  # pragma: no cover - config error
        raise KeyError(f"label {e} in {gt_path} missing from mapping.txt") from e


def to_transcript(frame_labels: np.ndarray) -> list[int]:
    """Collapse consecutive equal labels -> ordered action list (the transcript)."""
    if len(frame_labels) == 0:
        return []
    out = [int(frame_labels[0])]
    for v in frame_labels[1:]:
        if int(v) != out[-1]:
            out.append(int(v))
    return out


def split_video_ids(split_file: str | os.PathLike) -> list[str]:
    """Read a '*.bundle' split file -> list of video ids (strip dir + '.txt')."""
    ids: list[str] = []
    with open(split_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ids.append(Path(line).stem)
    return ids


def _index_videos(video_dir: Path, globs: list[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for pattern in globs:
        for path in video_dir.rglob(pattern):
            index.setdefault(path.stem, path)
    return index


# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------


class ActionSegDataset:
    """Light index over a 50Salads / Breakfast root. Not a torch Dataset -- it just
    resolves paths and (optionally) loads ground truth + transcripts. The feature
    extractor iterates over ``.records()``.
    """

    def __init__(
        self,
        name: str,
        root: str | os.PathLike,
        video_dir: str | os.PathLike | None = None,
        feature_dir: str | os.PathLike | None = None,
        load_labels: bool = True,
    ):
        name = name.lower()
        if name not in DATASET_DEFAULTS:
            raise ValueError(f"unknown dataset {name!r}; expected one of {list(DATASET_DEFAULTS)}")
        self.name = name
        self.cfg = DATASET_DEFAULTS[name]
        self.root = Path(root)
        self.gt_dir = self.root / "groundTruth"
        self.mapping_file = self.root / "mapping.txt"
        self.splits_dir = self.root / "splits"
        self.video_dir = Path(video_dir) if video_dir else self.root / "videos"
        self.feature_dir = Path(feature_dir) if feature_dir else self.root / "features"

        self.name_to_idx, self.idx_to_name = (
            load_mapping(self.mapping_file) if self.mapping_file.exists() else ({}, [])
        )
        self._video_index = (
            _index_videos(self.video_dir, self.cfg["video_glob"])
            if self.video_dir.exists()
            else {}
        )
        self.load_labels = load_labels and self.mapping_file.exists()

    # ---- ids --------------------------------------------------------------------------

    def all_ids(self) -> list[str]:
        if self.gt_dir.exists():
            return sorted(p.stem for p in self.gt_dir.glob("*.txt"))
        return sorted(self._video_index)

    def split(self, k: int, subset: str = "test") -> list[str]:
        f = self.splits_dir / f"{subset}.split{k}.bundle"
        return split_video_ids(f)

    # ---- records ---------------------------------------------------------------------

    def record(self, video_id: str) -> VideoRecord:
        gt_path = self.gt_dir / f"{video_id}.txt"
        feat_path = self.feature_dir / f"{video_id}.npy"
        rec = VideoRecord(
            video_id=video_id,
            video_path=self._video_index.get(video_id),
            gt_path=gt_path if gt_path.exists() else None,
            feature_path=feat_path if feat_path.exists() else None,
        )
        if self.load_labels and rec.gt_path is not None:
            rec.frame_labels = read_groundtruth(rec.gt_path, self.name_to_idx)
            rec.num_label_frames = int(len(rec.frame_labels))
            rec.transcript = to_transcript(rec.frame_labels)
        return rec

    def records(self, ids: list[str] | None = None):
        for vid in (ids or self.all_ids()):
            yield self.record(vid)

    # ---- misc ----------------------------------------------------------------------

    def action_names(self) -> list[str]:
        return list(self.idx_to_name)

    def __len__(self) -> int:
        return len(self.all_ids())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ActionSegDataset(name={self.name!r}, root={str(self.root)!r}, "
            f"n_videos={len(self)}, n_classes={len(self.idx_to_name)})"
        )
