"""(b) Monotonic coarse alignment, constrained by the ordered transcript.

Every transcript entry must appear, in order, exactly once (entry i occupies a
contiguous block; blocks are ordered). Two solvers, both from `delta.align`:
  * "dp"   : order-preserving Viterbi over the similarity (HiERO-StepG-style).
  * "asot" : unbalanced fused Gromov-Wasserstein OT (ASOT/CLOT lineage) with the
             transcript-order temporal prior, then the same monotone decode.
"""
from __future__ import annotations

import numpy as np

from delta.align.ta import align_dp
from delta.align.asot import align_asot


def coarse_align(sim_tn: np.ndarray, method: str = "dp", transition_penalty: float = 0.0,
                 rho: float = 0.15, **asot_kw) -> np.ndarray:
    """sim_tn (T, N) similarity of every frame to each transcript entry, in
    transcript order. Returns (T,) monotone non-decreasing entry index that covers
    all N entries."""
    sim_tn = np.asarray(sim_tn, dtype=np.float64)
    T, N = sim_tn.shape
    if N == 0 or T < N:
        raise ValueError(f"need T >= N >= 1 (got T={T}, N={N})")
    if N == 1:
        return np.zeros(T, dtype=np.int64)
    if method == "dp":
        return align_dp(sim_tn.T, list(range(N)), transition_penalty=transition_penalty).entry_of_frame
    if method == "asot":
        cost = 1.0 - sim_tn
        res = align_asot(cost, list(range(N)), rho=rho, **asot_kw)
        return np.asarray(res.y_star, dtype=np.int64)  # identity transcript -> labels == entry index
    raise ValueError(f"unknown method {method!r} (dp | asot)")
