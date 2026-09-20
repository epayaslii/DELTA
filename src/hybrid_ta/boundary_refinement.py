"""(c) Two parallel refinement branches over the coarse boundaries.

Both branches look at each coarse boundary b0 (between transcript entries e_l, e_r)
and independently propose a shifted position with a confidence in [0, 1]. They are
fused in `pseudo_labels.fuse_boundaries`.

Branch A -- semantic transition sharpening (CVA principle: a boundary frame should
  be discriminable from both neighbours). Local search maximising "left window looks
  like e_l and not e_r; right window looks like e_r and not e_l" (+ visual change).
  This is `delta.align.refine`; the search itself is our own design, CVA supplies only
  the principle (its CBD loss needs GT spans, which we do not have).

Branch B -- relational structure + temporal consistency (MASRA / LRCA principle: align
  the model's frame-frame relation matrix to a target derived from the text/transcript).
  Locally, the transcript predicts a block-diagonal relation matrix: frames on the same
  side of the boundary are related (1), frames across it are not (0). We pick the shift
  whose target best matches the observed cosine self-similarity (min LRCA residual).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from delta.align.refine import refine_boundaries, _confidence
from delta.align.similarity import l2norm


@dataclass
class BranchResult:
    boundaries: list[int]            # refined internal boundaries, one per coarse boundary
    confidences: list[float]         # in [0, 1]
    coarse: list[int] = field(default_factory=list)


def coarse_boundaries(entry_of_frame: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.diff(np.asarray(entry_of_frame)) != 0) + 1]


def branch_a_semantic(sim_nt: np.ndarray, entry_of_frame: np.ndarray, radius: int = 30,
                      window: int = 20, frame_emb: np.ndarray | None = None,
                      w_visual: float = 0.5) -> BranchResult:
    """sim_nt (N, T): transcript-entry x frame similarity."""
    r = refine_boundaries(sim_nt, entry_of_frame, radius=radius, window=window,
                          frame_emb=frame_emb, w_visual=w_visual)
    return BranchResult(list(r.boundaries), list(r.confidences), list(r.coarse_boundaries))


def _lrca_profile(S_loc: np.ndarray, offset: int, cands: np.ndarray, window: int,
                  left_lim: int, right_lim: int) -> np.ndarray:
    """-(mean squared residual between S and the block-diagonal target) per candidate.
    S_loc is the local self-similarity; `offset` = global index of S_loc[0]. The window
    around a candidate c is symmetric and clipped to (left_lim, right_lim) -- the
    neighbouring boundaries -- so only the two entries that meet at this boundary are
    compared (a third entry inside the window would fake a block)."""
    out = np.empty(len(cands))
    for k, c in enumerate(cands):
        m = int(min(window, c - left_lim, right_lim - c))
        idx = np.arange(c - m, c + m) - offset
        side = (idx >= c - offset).astype(np.int8)
        target = (side[:, None] == side[None, :]).astype(np.float64)
        out[k] = -np.mean((S_loc[np.ix_(idx, idx)] - target) ** 2)
    return out


def branch_b_relational(frame_feat: np.ndarray, entry_of_frame: np.ndarray, radius: int = 30,
                        window: int = 20) -> BranchResult:
    """frame_feat (T, D): any per-frame embedding (VLM features, or the similarity
    profile itself). Only a local relation matrix is built per boundary -- never the
    full T x T."""
    entry = np.asarray(entry_of_frame, int)
    T = len(entry)
    f = l2norm(np.asarray(frame_feat, dtype=np.float64))
    coarse = coarse_boundaries(entry)
    refined, confs, prev = [], [], 0
    for k, b0 in enumerate(coarse):
        nxt = coarse[k + 1] if k + 1 < len(coarse) else T
        lo, hi = max(prev + 2, b0 - radius), min(nxt - 1, b0 + radius + 1)
        cands = np.arange(lo, hi)
        if len(cands) == 0:
            refined.append(b0); confs.append(0.0); prev = b0
            continue
        a, z = max(0, int(cands[0]) - window), min(T, int(cands[-1]) + window)
        loc = f[a:z]
        prof = _lrca_profile(loc @ loc.T, a, cands, window, prev, nxt)
        nb = int(cands[prof.argmax()])
        refined.append(nb); confs.append(_confidence(prof)); prev = nb
    return BranchResult(refined, confs, coarse)
