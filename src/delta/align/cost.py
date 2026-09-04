"""Stage-A fused cost matrix (TASOT-style).

The (T, N) cost fed to `delta.align.asot.align_asot`:

    C[t, n] = w_sem * (1 - cos(text_n, frame_t))          # frame <-> transcript entry
            + w_cap * (1 - cos(caption_t, text_n))        # VLM per-frame caption (optional)
            + rho   * |t/T - n/N|                         # transcript-order temporal prior

``text_n`` is the VLM text embedding of transcript entry ``n`` (bare action name,
a generated description, or -- TASOT -- an MLLM event description). ``caption_t``
is an optional per-frame VLM caption embedding in the same space. The
frame<->frame structure term (GW smoothness) is handled inside the solver.
"""

from __future__ import annotations

import numpy as np

from .similarity import l2norm
from .asot import temporal_prior


def fused_cost(
    frame_emb: np.ndarray,          # (T, D)
    transcript: list[int],
    class_emb: np.ndarray,          # (C, D) -- same space as frame_emb
    caption_emb: np.ndarray | None = None,   # (T, D) per-frame VLM caption, same space
    w_sem: float = 1.0,
    w_cap: float = 0.0,
    rho: float = 0.15,
) -> np.ndarray:
    """(T, N) fused cost, lower = better match."""
    text = l2norm(class_emb[np.asarray(transcript, int)])            # (N, D)
    frm = l2norm(frame_emb)                                          # (T, D)
    C = w_sem * (1.0 - frm @ text.T)                                 # (T, N)
    if caption_emb is not None and w_cap:
        cap = l2norm(caption_emb)                                    # (T, D)
        C = C + w_cap * (1.0 - (cap @ text.T))
    C = C + temporal_prior(frame_emb.shape[0], len(transcript), rho=rho)
    return C
