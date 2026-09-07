"""Stage 0 -- semantic-guided adaptive sampling.

Encoding every frame with a video LLM is wasteful: most of a 50Salads video is
the middle of a long action where nothing changes. The supervisor's framing was
"semantic should guide the sampling" and "cost is also important".

So: encode a cheap coarse grid first, use the resulting transcript x frame
similarity to guess *where the transitions probably are*, then spend the
remaining budget re-encoding only those neighbourhoods densely.

    coarse pass (e.g. 1 fps)
        -> transition_score(s)        per-coarse-frame "a boundary is near here"
        -> select_dense_zones(...)    the top-k neighbourhoods
        -> adaptive_frame_plan(...)   final frame indices to encode

Everything here is numpy and testable without a GPU; the actual encoding is
`delta.features.extract`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def transition_score(s: np.ndarray, smooth: int = 3) -> np.ndarray:
    """(N, T) transcript x frame similarity -> (T,) "a boundary is near here".

    Combines two cues, both normalised to [0, 1] and averaged:

    * **argmax churn** -- the best-matching transcript entry changes here
    * **low margin** -- top-1 and top-2 entries score almost the same, i.e. the
      model is undecided, which is what a transition looks like

    A confident frame in the middle of an action scores ~0; an ambiguous frame
    between two actions scores high.
    """
    if s.ndim != 2 or s.shape[0] < 2:
        return np.zeros(s.shape[-1])
    order = np.argsort(-s, axis=0)
    top1, top2 = order[0], order[1]
    t = np.arange(s.shape[1])

    churn = np.zeros(s.shape[1])
    switch = np.flatnonzero(np.diff(top1) != 0) + 1
    churn[switch] = 1.0

    margin = s[top1, t] - s[top2, t]
    rng = margin.max() - margin.min()
    if rng > 1e-9:
        undecided = 1.0 - (margin - margin.min()) / rng
    else:
        # constant margin -> no relative information anywhere, so no frame is
        # more transition-like than another (NOT "all maximally undecided")
        undecided = np.zeros_like(margin)

    score = 0.5 * churn + 0.5 * undecided
    if smooth > 1:                                   # spread each cue over neighbours
        k = np.ones(smooth) / smooth
        score = np.convolve(score, k, mode="same")
    m = score.max()
    return score / m if m > 1e-9 else score


@dataclass
class DensePlan:
    dense_frames: np.ndarray        # full-resolution frame indices to encode densely
    zones: list[tuple[int, int]]    # (start, end) in full-resolution frame indices
    n_encoded: int                  # coarse + dense, after dedup
    n_full_rate: int                # what encoding every frame would have cost
    saving: float                   # 1 - n_encoded / n_full_rate


def select_dense_zones(
    score: np.ndarray,              # (T_coarse,) from transition_score
    coarse_idx: np.ndarray,         # (T_coarse,) their full-resolution frame indices
    n_zones: int,                   # how many neighbourhoods to refine
    radius: int,                    # +- this many full-resolution frames per zone
    min_gap: int = 1,               # min separation between picked peaks, in coarse steps
    n_frames: int | None = None,    # clamp to the video length
) -> list[tuple[int, int]]:
    """Greedily take the ``n_zones`` highest-scoring peaks, keeping them at least
    ``min_gap`` coarse steps apart, and expand each by ``radius``."""
    picked: list[int] = []
    for c in np.argsort(-score):
        if len(picked) >= n_zones:
            break
        if all(abs(int(c) - p) >= min_gap for p in picked):
            picked.append(int(c))

    hi = (n_frames - 1) if n_frames is not None else int(coarse_idx[-1])
    zones = []
    for c in sorted(picked):
        centre = int(coarse_idx[c])
        zones.append((max(0, centre - radius), min(hi, centre + radius)))

    merged: list[tuple[int, int]] = []          # merge overlaps so we don't double-encode
    for a, b in zones:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def adaptive_frame_plan(
    s_coarse: np.ndarray,           # (N, T_coarse) similarity from the coarse pass
    coarse_idx: np.ndarray,         # (T_coarse,) full-resolution indices of those frames
    n_frames: int,                  # full-resolution length
    n_transcript: int,              # how many actions -> how many boundaries to expect
    dense_every: int = 2,           # step inside a refined zone
    radius: int = 90,               # +- frames around each predicted transition (3 s @ 30 fps)
    zones_per_boundary: float = 1.5,  # search a few more zones than there are boundaries
) -> DensePlan:
    """Decide which extra frames to encode after the coarse pass.

    Budget scales with the number of *transitions*, not video length -- a longer
    video with the same recipe costs barely more.
    """
    score = transition_score(s_coarse)
    n_zones = max(1, int(round((n_transcript - 1) * zones_per_boundary)))
    zones = select_dense_zones(score, coarse_idx, n_zones=n_zones, radius=radius,
                               n_frames=n_frames)

    dense = np.unique(np.concatenate(
        [np.arange(a, b + 1, dense_every) for a, b in zones]
    )) if zones else np.array([], dtype=int)

    encoded = np.union1d(np.asarray(coarse_idx, int), dense.astype(int))
    return DensePlan(
        dense_frames=dense.astype(int),
        zones=zones,
        n_encoded=int(encoded.size),
        n_full_rate=int(n_frames),
        saving=1.0 - encoded.size / max(n_frames, 1),
    )
