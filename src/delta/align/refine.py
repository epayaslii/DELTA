"""Stage B1 -- local semantic boundary search.

The novel core of the method (peer review, 2026-09-04). Stage A
(`delta.align.asot`) gives an ordered but *coarse* alignment. Instead of
trusting its boundary positions, open a local window around each coarse
transition and score every candidate position by *semantic transition
evidence*: the left side should look like action A and not B, the right side
like B and not A. Optionally add a visual-change term. Pick the best position
and attach a confidence (peaky score profile -> confident).

This is:
  * VLM-based        -- scores come from the transcript x frame similarity
  * segment-based    -- uses the candidate (A ending, B starting) pair
  * transcript-only  -- no GT boundary anywhere
  * cheap            -- only ``2*radius+1`` positions per boundary, local windows

The confidence feeds Stage B2 (PBCR contrastive weighting) and Stage B3
(which boundaries to hand to an expensive chat-VLM).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .similarity import l2norm


@dataclass
class BoundarySearchResult:
    entry_of_frame: np.ndarray      # (T,) refined monotone entry index
    boundaries: list[int]           # refined transition frame indices
    confidences: list[float]        # per-boundary, in [0, 1]
    coarse_boundaries: list[int]
    shifts: list[int]               # refined - coarse, per boundary


def _transition_score(
    sim: np.ndarray,                # (N, T) transcript-entry x frame similarity
    b: int, e_left: int, e_right: int, window: int,
    frame_emb: np.ndarray | None,   # (T, D) optional, for the visual-change term
    w_visual: float,
) -> float:
    T = sim.shape[1]
    l0, l1 = max(0, b - window), b
    r0, r1 = b, min(T, b + window)
    if l1 <= l0 or r1 <= r0:
        return -np.inf
    left, right = sim[:, l0:l1], sim[:, r0:r1]
    s = (left[e_left].mean() - left[e_right].mean()          # left = A, not B
         + right[e_right].mean() - right[e_left].mean())     # right = B, not A
    if frame_emb is not None and w_visual:
        fl = l2norm(frame_emb[l0:l1].mean(axis=0))
        fr = l2norm(frame_emb[r0:r1].mean(axis=0))
        s += w_visual * (1.0 - float(fl @ fr))               # reward a real visual change
    return float(s)


def _confidence(profile: np.ndarray, sep: int = 8) -> float:
    """Peak *prominence*: how much the best candidate beats the best candidate at
    least ``sep`` frames away, as a fraction of the profile's range. A real
    localised boundary has one clear spot (high); a noise profile has many
    equally-good spots (~0). Feeds Stage B2 weighting / B3 hard-case routing."""
    n = len(profile)
    if n < 2 * sep + 1:
        return 0.0
    b = int(profile.argmax())
    far = np.concatenate([profile[: max(0, b - sep)], profile[b + sep + 1:]])
    if len(far) == 0:
        return 0.0
    rng = profile.max() - profile.min() + 1e-9
    return float(np.clip((profile[b] - far.max()) / rng, 0.0, 1.0))


def search_one(
    sim: np.ndarray, b0: int, e_left: int, e_right: int,
    radius: int, window: int,
    frame_emb: np.ndarray | None = None, w_visual: float = 0.5,
) -> tuple[int, float]:
    """Refine one coarse boundary ``b0`` between entries ``e_left`` and
    ``e_right``. Returns ``(refined_b, confidence)``."""
    T = sim.shape[1]
    lo, hi = max(1, b0 - radius), min(T - 1, b0 + radius + 1)
    cand = np.arange(lo, hi)
    if len(cand) == 0:
        return b0, 0.0
    prof = np.array([_transition_score(sim, int(b), e_left, e_right, window,
                                       frame_emb, w_visual) for b in cand])
    best = int(cand[prof.argmax()])
    return best, _confidence(prof)


def refine_boundaries(
    sim: np.ndarray,                # (N, T) transcript-entry x frame similarity
    entry_of_frame: np.ndarray,     # (T,) coarse monotone entry index (Stage A)
    radius: int = 30,
    window: int = 20,
    frame_emb: np.ndarray | None = None,
    w_visual: float = 0.5,
) -> BoundarySearchResult:
    """Refine every coarse transition with a local semantic search. Refined
    boundaries are clamped to stay ordered (each >= the previous)."""
    entry = np.asarray(entry_of_frame, int).copy()
    T = len(entry)
    coarse = [int(i) for i in np.flatnonzero(np.diff(entry) != 0) + 1]
    refined, confs, shifts = [], [], []
    prev = 0
    for k, b0 in enumerate(coarse):
        e_left, e_right = int(entry[b0 - 1]), int(entry[b0])
        nb, c = search_one(sim, b0, e_left, e_right, radius, window, frame_emb, w_visual)
        nb = max(prev + 1, min(nb, T - 1))
        refined.append(nb); confs.append(c); shifts.append(nb - b0)
        prev = nb

    out = np.empty(T, int)
    edges = [0, *refined, T]
    entries_in_order = [int(entry[0])] + [int(entry[b]) for b in coarse]
    for e, (a, z) in zip(entries_in_order, zip(edges[:-1], edges[1:])):
        out[a:z] = e
    return BoundarySearchResult(entry_of_frame=out, boundaries=refined,
                                confidences=confs, coarse_boundaries=coarse,
                                shifts=shifts)
