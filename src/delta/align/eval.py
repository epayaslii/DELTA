"""Metrics for temporal alignment / segmentation quality.

Two groups:

1. Segmentation quality (how good are the pseudo-labels / TAS output):
   - mean_over_frames (MoF)         frame accuracy
   - mean_over_classes (MoC)        per-class frame accuracy, averaged  <- 50S/BF headline
   - edit_score                     segmental edit distance (Lea et al.)
   - f1_at_k                        segmental F1 @ IoU {0.10, 0.25, 0.50}

2. Alignment representation quality (for the self-supervised alignment encoder):
   - kendall_tau_alignment          temporal ordering consistency between two videos
   - pseudo_label_agreement         frame agreement of alignment pseudo-labels vs GT

All functions take integer numpy arrays of per-frame class ids.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------------------


def mean_over_frames(pred: np.ndarray, gt: np.ndarray, ignore: set[int] | None = None) -> float:
    pred, gt = np.asarray(pred), np.asarray(gt)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    mask = np.ones(n, dtype=bool)
    if ignore:
        mask &= ~np.isin(gt, list(ignore))
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == gt[mask]).mean())


def mean_over_classes(pred: np.ndarray, gt: np.ndarray, ignore: set[int] | None = None) -> float:
    pred, gt = np.asarray(pred), np.asarray(gt)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    accs = []
    for c in np.unique(gt):
        if ignore and c in ignore:
            continue
        m = gt == c
        accs.append((pred[m] == c).mean())
    return float(np.mean(accs)) if accs else float("nan")


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encode a per-frame labeling -> list of (label, start, end_exclusive)."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    bounds = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


_segments = segments  # backwards-compatible alias


def edit_score(pred: np.ndarray, gt: np.ndarray, norm: bool = True) -> float:
    p = [s[0] for s in _segments(pred)]
    g = [s[0] for s in _segments(gt)]
    m, n = len(p), len(g)
    d = np.zeros((m + 1, n + 1))
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p[i - 1] == g[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    dist = d[m, n]
    return float((1.0 - dist / max(m, n, 1)) * 100.0) if norm else float(dist)


def f1_at_k(pred: np.ndarray, gt: np.ndarray, overlaps=(0.1, 0.25, 0.5)) -> dict[float, float]:
    p_seg = _segments(pred)
    g_seg = _segments(gt)
    out: dict[float, float] = {}
    for k in overlaps:
        tp = fp = 0
        used = np.zeros(len(g_seg), dtype=bool)
        for pl, ps, pe in p_seg:
            best_iou, best_j = 0.0, -1
            for j, (gl, gs, ge) in enumerate(g_seg):
                if gl != pl or used[j]:
                    continue
                inter = max(0, min(pe, ge) - max(ps, gs))
                union = max(pe, ge) - min(ps, gs)
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= k and best_j >= 0:
                tp += 1
                used[best_j] = True
            else:
                fp += 1
        fn = int((~used).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[k] = float(2 * prec * rec / (prec + rec) * 100.0) if prec + rec else 0.0
    return out


def segmentation_report(pred: np.ndarray, gt: np.ndarray, ignore: set[int] | None = None) -> dict:
    r = {
        "MoF": mean_over_frames(pred, gt, ignore),
        "MoC": mean_over_classes(pred, gt, ignore),
        "edit": edit_score(pred, gt),
    }
    r.update({f"F1@{int(k*100)}": v for k, v in f1_at_k(pred, gt).items()})
    return r


# --------------------------------------------------------------------------------------
# alignment representation
# --------------------------------------------------------------------------------------


def pseudo_label_agreement(pseudo: np.ndarray, gt: np.ndarray) -> float:
    """Frame agreement between alignment pseudo-labels and ground truth (== MoF)."""
    return mean_over_frames(pseudo, gt)


def kendall_tau_alignment(embed_a: np.ndarray, embed_b: np.ndarray) -> float:
    """Kendall's tau over the nearest-neighbour frame correspondence between two
    videos' embedding sequences (Dwibedi et al., TCC). +1 = perfectly monotone.

    embed_a: (Ta, D), embed_b: (Tb, D)  (L2-normalised recommended)
    """
    a = np.asarray(embed_a, dtype=np.float64)
    b = np.asarray(embed_b, dtype=np.float64)
    # for each frame in a, nn index in b
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    nn = d.argmin(axis=1)
    n = len(nn)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(nn[j] - nn[i])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else float("nan")
