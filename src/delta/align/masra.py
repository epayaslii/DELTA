"""MASRA's two training-time regularizers, adapted to transcript-supervised
VLM-direct alignment.

MASRA (arXiv:2605.03398) improves a video-temporal-grounding DETR with two
alignments driven by cached MLLM text features:

  * ESTA  -- pull the mean-pooled temporal features of an event span toward that
             event's text embedding:   L_sem = mean_i (1 - cos(u_i, o_i))
  * LRCA  -- match the clip x clip visual self-similarity S = cos(E, E) to a text
             relation matrix R:         L_rel = mean_ij (s_ij - r_ij)^2

We keep the formulas and move the target from "a DETR's temporal features + GT
spans + MLLM captions" to "frozen-VLM frame features + the block structure of the
*current* transcript alignment + class-name embeddings". No MLLM, no GT spans.
See ``docs/masra-analysis.md``.

This module is numpy forward-pass only -- a CPU-side prototype and a diagnostic on
the I3D bundle (how far is cos(F, F) from the transcript block structure?). The
differentiable torch versions live on the cluster (plan M3/M4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .similarity import l2norm


def _subsample(T: int, max_frames: int) -> np.ndarray:
    """Indices that thin ``T`` frames to at most ``max_frames`` (T x T matrices
    blow up otherwise -- 50Salads has ~11k frames)."""
    if T <= max_frames:
        return np.arange(T)
    stride = int(np.ceil(T / max_frames))
    return np.arange(0, T, stride)


def entry_classes(y_star: np.ndarray, transcript: list[int]) -> np.ndarray:
    """Per-frame action class = the class of the transcript entry the frame is
    assigned to."""
    tr = np.asarray(transcript, dtype=int)
    return tr[np.asarray(y_star, dtype=int)]


# --------------------------------------------------------------------------------------
# ESTA -- event semantic temporal alignment
# --------------------------------------------------------------------------------------


def pool_events(frame_feat: np.ndarray, y_star: np.ndarray, n_entries: int):
    """Mean-pool ``frame_feat`` (T, D) within each transcript entry's block.
    Returns ``(U (N, D), filled (N,) bool)`` -- ``filled[i]`` is False when entry
    ``i`` got no frames (empty block)."""
    y = np.asarray(y_star, dtype=int)
    D = frame_feat.shape[1]
    U = np.zeros((n_entries, D), dtype=float)
    filled = np.zeros(n_entries, dtype=bool)
    for i in range(n_entries):
        m = y == i
        if m.any():
            U[i] = frame_feat[m].mean(axis=0)
            filled[i] = True
    return U, filled


@dataclass
class EstaResult:
    loss: float                 # mean_i (1 - cos(u_i, o_i)) over filled entries
    per_entry: np.ndarray       # (N,) cosine, nan where the block is empty
    n_empty: int


def esta_alignment(
    frame_feat: np.ndarray,     # (T, D)
    y_star: np.ndarray,         # (T,) transcript-entry index per frame
    transcript: list[int],      # (N,) class id per entry, in order
    class_emb: np.ndarray,      # (C, D_txt) -- must share D with frame_feat (or be projected)
) -> EstaResult:
    """MASRA Eq. 3-4 with the event span = the entry's block under ``y_star`` and
    ``o_i`` = the action-name embedding. Frames currently assigned to action ``i``
    should look like action ``i``."""
    N = len(transcript)
    U, filled = pool_events(frame_feat, y_star, N)
    O = class_emb[np.asarray(transcript, dtype=int)]
    if O.shape[1] != U.shape[1]:
        raise ValueError(
            f"class_emb dim {O.shape[1]} != frame_feat dim {U.shape[1]}; "
            "project into a shared space first"
        )
    cos = np.full(N, np.nan)
    uu, oo = l2norm(U), l2norm(O)
    cos[filled] = np.sum(uu[filled] * oo[filled], axis=1)
    loss = float(np.mean(1.0 - cos[filled])) if filled.any() else float("nan")
    return EstaResult(loss=loss, per_entry=cos, n_empty=int((~filled).sum()))


# --------------------------------------------------------------------------------------
# LRCA -- local relational consistency alignment
# --------------------------------------------------------------------------------------


def visual_relation_matrix(frame_feat: np.ndarray, max_frames: int = 1000):
    """``S = cos(F, F)`` (Eq. 7), subsampled. Returns ``(S (k, k), idx (k,))``."""
    idx = _subsample(frame_feat.shape[0], max_frames)
    f = l2norm(frame_feat[idx])
    return f @ f.T, idx


def transcript_relation_matrix(
    y_star: np.ndarray,
    transcript: list[int],
    class_emb: np.ndarray | None = None,
    mode: str = "class-sim",           # "class-sim" (soft, MASRA's T prior) | "block" (hard)
    idx: np.ndarray | None = None,
    max_frames: int = 1000,
) -> np.ndarray:
    """Text-side relation matrix ``R`` (Eq. 8 target).

    * ``block``     -- ``r_ij = 1`` iff frames i, j are in the same transcript
                       entry, else 0. The hard block-diagonal structure.
    * ``class-sim`` -- ``r_ij = cos(g(class_i), g(class_j))`` where ``class`` is
                       the action of each frame's current block and ``g`` the
                       class-name embedding. MASRA's caption-similarity prior,
                       built from the transcript instead of an MLLM.
    """
    y = np.asarray(y_star, dtype=int)
    if idx is None:
        idx = _subsample(y.shape[0], max_frames)
    y = y[idx]
    if mode == "block":
        return (y[:, None] == y[None, :]).astype(float)
    if mode == "class-sim":
        if class_emb is None:
            raise ValueError("mode='class-sim' needs class_emb")
        cls = entry_classes(y, transcript)
        g = l2norm(class_emb[cls])
        return g @ g.T
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class LrcaResult:
    loss: float                 # mean_ij (s_ij - r_ij)^2   (Eq. 8)
    per_frame: np.ndarray       # (k,) row-mean residual -- high = this frame's
                                #      neighbourhood disagrees with the transcript
    idx: np.ndarray             # frame indices S/R were computed on


def lrca_residual(S: np.ndarray, R: np.ndarray, idx: np.ndarray | None = None) -> LrcaResult:
    if S.shape != R.shape:
        raise ValueError(f"S {S.shape} vs R {R.shape}")
    d = (S - R) ** 2
    return LrcaResult(
        loss=float(d.mean()),
        per_frame=d.mean(axis=1),
        idx=np.arange(S.shape[0]) if idx is None else idx,
    )


def relation_boundary_score(S: np.ndarray) -> np.ndarray:
    """Cheap boundary detector from the self-similarity matrix: ``1 - s_{t,t+1}``.
    A diagnostic -- does the VLM feature relation carry boundary information
    (cf. the I3D 1.11x result in ``docs/50salads-notes.md``)?"""
    off = np.diag(S, k=1)
    return np.concatenate([1.0 - off, [0.0]])


# --------------------------------------------------------------------------------------
# one-video report
# --------------------------------------------------------------------------------------


def masra_report(
    frame_feat: np.ndarray,          # (T, D)
    y_star: np.ndarray,              # (T,)
    transcript: list[int],           # (N,)
    class_emb: np.ndarray | None = None,   # (C, D) -- ESTA + class-sim R need it
    gt_boundaries: list[int] | None = None,
    max_frames: int = 1000,
) -> dict:
    """ESTA + LRCA numbers for one video under a given alignment. ``class_emb``
    must already share the feature dimension of ``frame_feat``."""
    out: dict = {"T": int(frame_feat.shape[0]), "N": len(transcript)}

    S, idx = visual_relation_matrix(frame_feat, max_frames=max_frames)
    R_block = transcript_relation_matrix(y_star, transcript, mode="block", idx=idx)
    out["lrca_block"] = lrca_residual(S, R_block).loss
    if class_emb is not None:
        R_cls = transcript_relation_matrix(
            y_star, transcript, class_emb=class_emb, mode="class-sim", idx=idx
        )
        out["lrca_classsim"] = lrca_residual(S, R_cls).loss
        out["esta_loss"] = esta_alignment(frame_feat, y_star, transcript, class_emb).loss

    if gt_boundaries:
        bs = relation_boundary_score(S)
        gb = np.unique(np.clip(np.searchsorted(idx, gt_boundaries), 0, len(idx) - 1))
        rank = bs.argsort().argsort() / max(len(bs) - 1, 1)   # 0..1, 1 = most boundary-like
        out["bdy_score_at_gt"] = float(np.mean(rank[gb]))     # >0.5 = better than chance
    return out
