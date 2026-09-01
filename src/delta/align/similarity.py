"""Transcript x frame similarity matrix — the input to VLM-direct alignment.

`s[n, t]` = how well transcript action `n` matches frame `t`, from a **frozen**
vision-language model (no trained frame classifier). Downstream, `delta.align.ta`
reads `Y*` off `s` with an order-preserving alignment.

This module is model-agnostic: it takes already-extracted embeddings. Frame
embeddings come from `delta.features` (a VLM vision tower); text embeddings from
`delta.features.text_encoder` (the paired text tower) or `encode_action_names`.
"""

from __future__ import annotations

import numpy as np


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def similarity_matrix(
    text_emb: np.ndarray,       # (N, D) — one row per transcript entry, in order
    frame_emb: np.ndarray,      # (T, D)
    temperature: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """(N, T) cosine similarity. With ``temperature`` set, returns a softmax over
    transcript entries per frame (a column-stochastic soft assignment)."""
    a = l2norm(text_emb) if normalize else text_emb
    b = l2norm(frame_emb) if normalize else frame_emb
    s = a @ b.T                                        # (N, T), cosine in [-1, 1]
    if temperature is not None:
        z = s / float(temperature)
        z = z - z.max(axis=0, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=0, keepdims=True)
    return s


def transcript_text_embeddings(transcript: list[int], class_emb: np.ndarray) -> np.ndarray:
    """Expand per-class embeddings (C, D) to per-transcript-entry (N, D) following
    the transcript order (repeats included)."""
    return class_emb[np.asarray(transcript, dtype=int)]


def boundary_peakedness(s: np.ndarray, gt_boundaries: list[int], window: int = 1) -> dict:
    """Diagnostic: does the row-argmax of ``s`` actually switch near GT boundaries?
    Returns the fraction of GT boundaries within ``window`` frames of an argmax
    switch, and the mean gap. A blunt check of whether the VLM similarity carries
    boundary information at all (cf. the I3D 1.11x result in docs/50salads-notes)."""
    switch = np.flatnonzero(np.diff(s.argmax(axis=0)) != 0) + 1
    if len(switch) == 0 or len(gt_boundaries) == 0:
        return {"hit_rate": 0.0, "mean_gap": float("nan"), "n_switches": int(len(switch))}
    gaps = [int(np.min(np.abs(switch - b))) for b in gt_boundaries]
    return {
        "hit_rate": float(np.mean([g <= window for g in gaps])),
        "mean_gap": float(np.mean(gaps)),
        "n_switches": int(len(switch)),
    }
