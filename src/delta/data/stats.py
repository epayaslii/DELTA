"""Dataset statistics for 50Salads / Breakfast.

Pure functions over :class:`delta.data.ActionSegDataset`. Everything works from
``groundTruth/`` + ``mapping.txt`` alone (no video, no features needed), so this
runs on the benchmark bundle before any raw video is available.

The point of each statistic is to make concrete *why* 50Salads is hard for
transcript-only temporal alignment; see ``notebooks/stage1_dataset_analysis``
and ``docs/temporal-alignment.md`` §4.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from delta.align import segments
from delta.data.datasets import ActionSegDataset, DATASET_DEFAULTS

# names that denote "no action" across the datasets we use
BACKGROUND_NAMES = {"background", "sil", "SIL", "action_start", "action_end", "none"}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def label_fps(ds: ActionSegDataset) -> float:
    return float(DATASET_DEFAULTS[ds.name]["label_fps"])


def background_ids(ds: ActionSegDataset, names: set[str] | None = None) -> set[int]:
    names = {n.lower() for n in (names or BACKGROUND_NAMES)}
    return {i for i, n in enumerate(ds.idx_to_name) if n.lower() in names}


def _records(ds: ActionSegDataset, ids: list[str] | None):
    for vid in (ids or ds.all_ids()):
        rec = ds.record(vid)
        if rec.frame_labels is not None:
            yield rec


# --------------------------------------------------------------------------------------
# per-segment / per-video tables
# --------------------------------------------------------------------------------------


def segment_rows(ds: ActionSegDataset, ids: list[str] | None = None) -> list[dict]:
    """One row per action segment across the given videos."""
    fps = label_fps(ds)
    out: list[dict] = []
    for rec in _records(ds, ids):
        segs = segments(rec.frame_labels)
        n = len(segs)
        for k, (cls, s, e) in enumerate(segs):
            out.append(
                dict(
                    video_id=rec.video_id,
                    seg_index=k,
                    n_segments=n,
                    rel_position=k / max(n - 1, 1),
                    class_idx=cls,
                    class_name=ds.idx_to_name[cls] if cls < len(ds.idx_to_name) else str(cls),
                    start=s,
                    end=e,
                    dur_frames=e - s,
                    dur_sec=(e - s) / fps,
                )
            )
    return out


def video_rows(ds: ActionSegDataset, ids: list[str] | None = None) -> list[dict]:
    """One row per video."""
    fps = label_fps(ds)
    bg = background_ids(ds)
    out: list[dict] = []
    for rec in _records(ds, ids):
        y = rec.frame_labels
        segs = segments(y)
        bg_frames = int(np.isin(y, list(bg)).sum()) if bg else 0
        out.append(
            dict(
                video_id=rec.video_id,
                n_frames=len(y),
                dur_sec=len(y) / fps,
                n_segments=len(segs),
                transcript_len=len(rec.transcript or []),
                n_unique_classes=int(len(np.unique(y))),
                bg_frac=bg_frames / max(len(y), 1),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# class-level aggregates
# --------------------------------------------------------------------------------------


def frame_counts(ds: ActionSegDataset, ids: list[str] | None = None) -> dict[int, int]:
    c: Counter = Counter()
    for rec in _records(ds, ids):
        vals, cnts = np.unique(rec.frame_labels, return_counts=True)
        for v, n in zip(vals, cnts):
            c[int(v)] += int(n)
    return dict(c)


def segment_counts(ds: ActionSegDataset, ids: list[str] | None = None) -> dict[int, int]:
    c: Counter = Counter()
    for rec in _records(ds, ids):
        for cls, _, _ in segments(rec.frame_labels):
            c[cls] += 1
    return dict(c)


def class_duration_summary(ds: ActionSegDataset, ids: list[str] | None = None) -> list[dict]:
    """Per class: how many instances, and how variable their length is.

    High ``cv`` (coefficient of variation = std/mean) means a single ordered
    transcript occurrence carries almost no information about extent -- the
    duration head has nothing to learn (paper: +0.2 MoC on 50S vs +3.3 on BF).
    """
    fps = label_fps(ds)
    by_class: dict[int, list[int]] = {}
    for row in segment_rows(ds, ids):
        by_class.setdefault(row["class_idx"], []).append(row["dur_frames"])
    out = []
    for cls, durs in sorted(by_class.items()):
        a = np.asarray(durs, dtype=float)
        mean = float(a.mean())
        out.append(
            dict(
                class_idx=cls,
                class_name=ds.idx_to_name[cls] if cls < len(ds.idx_to_name) else str(cls),
                n_instances=len(a),
                mean_sec=mean / fps,
                std_sec=float(a.std()) / fps,
                cv=float(a.std() / mean) if mean > 0 else float("nan"),
                min_sec=float(a.min()) / fps,
                max_sec=float(a.max()) / fps,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# transition structure
# --------------------------------------------------------------------------------------


def transition_matrix(
    ds: ActionSegDataset, ids: list[str] | None = None, normalize: bool = True
) -> np.ndarray:
    """(C, C) matrix from consecutive transcript actions. Row c = P(next | current=c)
    if ``normalize``. Tells you how much the transcript *order alone* pins down the
    next action -- i.e. how much work the visual transition score V^a has to do.
    """
    C = len(ds.idx_to_name)
    M = np.zeros((C, C), dtype=float)
    for rec in _records(ds, ids):
        t = rec.transcript or []
        for a, b in zip(t[:-1], t[1:]):
            if a < C and b < C:
                M[a, b] += 1
    if normalize:
        row = M.sum(axis=1, keepdims=True)
        M = np.divide(M, row, out=np.zeros_like(M), where=row > 0)
    return M


def transition_entropy(ds: ActionSegDataset, ids: list[str] | None = None) -> dict[int, float]:
    """Per-class Shannon entropy (bits) of the next-action distribution.
    0 = deterministic successor; higher = the order is ambiguous.
    """
    P = transition_matrix(ds, ids, normalize=True)
    out = {}
    for c in range(P.shape[0]):
        p = P[c][P[c] > 0]
        out[c] = float(-(p * np.log2(p)).sum()) if len(p) else float("nan")
    return out


# --------------------------------------------------------------------------------------
# the naive floor for Stage 1c
# --------------------------------------------------------------------------------------


def naive_uniform_labeling(transcript: list[int], n_frames: int) -> np.ndarray:
    """Split the video into ``len(transcript)`` equal parts, one per transcript
    action, in order. This is the alignment you get with *zero* visual evidence --
    the floor any real temporal-alignment method must beat.
    """
    n = len(transcript)
    if n == 0 or n_frames <= 0:
        return np.zeros(max(n_frames, 0), dtype=np.int64)
    bounds = np.linspace(0, n_frames, n + 1).round().astype(int)
    y = np.empty(n_frames, dtype=np.int64)
    for k, cls in enumerate(transcript):
        y[bounds[k] : bounds[k + 1]] = cls
    return y


# --------------------------------------------------------------------------------------
# one-line cross-dataset comparison
# --------------------------------------------------------------------------------------


def dataset_summary(ds: ActionSegDataset, ids: list[str] | None = None) -> dict:
    """Scalar summary for a 50Salads-vs-Breakfast table."""
    vrows = video_rows(ds, ids)
    srows = segment_rows(ds, ids)
    ent = transition_entropy(ds, ids)
    ent_vals = [v for v in ent.values() if not math.isnan(v)]
    seg_durs = np.array([r["dur_sec"] for r in srows], dtype=float)
    return dict(
        dataset=ds.name,
        n_videos=len(vrows),
        n_classes=len(ds.idx_to_name),
        mean_video_sec=float(np.mean([r["dur_sec"] for r in vrows])),
        mean_segments_per_video=float(np.mean([r["n_segments"] for r in vrows])),
        mean_transcript_len=float(np.mean([r["transcript_len"] for r in vrows])),
        median_segment_sec=float(np.median(seg_durs)),
        mean_bg_frac=float(np.mean([r["bg_frac"] for r in vrows])),
        mean_next_action_entropy_bits=float(np.mean(ent_vals)) if ent_vals else float("nan"),
    )


# --------------------------------------------------------------------------------------
# optional pandas convenience
# --------------------------------------------------------------------------------------


def _df(rows: list[dict]):
    import pandas as pd  # lazy: stats work without pandas, notebooks want it

    return pd.DataFrame(rows)


def segment_df(ds: ActionSegDataset, ids: list[str] | None = None):
    return _df(segment_rows(ds, ids))


def video_df(ds: ActionSegDataset, ids: list[str] | None = None):
    return _df(video_rows(ds, ids))


def class_duration_df(ds: ActionSegDataset, ids: list[str] | None = None):
    return _df(class_duration_summary(ds, ids))
