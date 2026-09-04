"""Stage-B2: Pseudo-Boundary Contrastive Refinement (PBCR).

Inspired by CVA's CBD loss (CVPR'26), but **not** a reuse of it. Published CBD is
fully supervised: it anchors the contrastive loss on **GT-span boundary indices**,
defines negatives relative to the GT span, and the outer VTG objective
Hungarian-matches predicted moments to GT moments. None of that is available to us.

PBCR keeps only the *principle* -- "a transition frame deserves its own
contrastive representation, invariant to context and distinct from action
interiors" -- and rebuilds the pieces from things we legitimately have:
  * anchors  = *confident* pseudo-boundaries from Stage A / the local search
               (`delta.align.refine`), NOT every predicted boundary
  * positive = the same index under a second augmentation
  * negatives = confident action interiors (left / right) + other confusable
               transitions

**Known failure mode:** if the pseudo-boundary is wrong, the loss reinforces the
wrong location. Mitigation lives outside this function -- confidence weighting
from the local search, local candidate windows, iterative realignment (Stage B2
loop). Only feed boundaries above a confidence threshold.

torch; not re-exported from ``delta.align`` (keeps the package importable without
torch). Import directly: ``from delta.align.cbd import cbd_loss, BoundaryHead``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    """Small projection applied to frame features before the contrastive loss."""

    def __init__(self, dim: int, proj: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, proj), nn.GELU(), nn.Linear(proj, proj))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _negatives(z_anchor_view: torch.Tensor, b: int, T: int, n_adj: int, n_hard: int,
               anchor_vec: torch.Tensor) -> torch.Tensor:
    """Indices: temporally adjacent (|j-b|<=n_adj, j!=b) + the n_hard most
    cosine-similar frames outside that band."""
    adj = [j for j in range(max(0, b - n_adj), min(T, b + n_adj + 1)) if j != b]
    banned = set(adj) | {b}
    far = torch.tensor([j for j in range(T) if j not in banned], device=z_anchor_view.device)
    if len(far) and n_hard:
        sims = z_anchor_view[far] @ anchor_vec
        hard = far[sims.topk(min(n_hard, len(far))).indices]
        idx = torch.cat([torch.tensor(adj, device=far.device), hard])
    else:
        idx = torch.tensor(adj, device=z_anchor_view.device)
    return idx


def cbd_loss(
    feat_view1: torch.Tensor,       # (T, D) projected features, augmentation 1
    feat_view2: torch.Tensor,       # (T, D) projected features, augmentation 2 (same indexing)
    boundaries: list[int],          # *confident* pseudo-boundary frame indices
    weights: list[float] | None = None,   # per-boundary confidence in [0, 1]
    n_adj: int = 2,
    n_hard: int = 10,
    tau: float = 0.07,
) -> torch.Tensor:
    """Confidence-weighted mean InfoNCE over pseudo-boundary frames. anchor =
    view1[b], positive = view2[b], negatives = adjacent + hard-mined from view1.
    Pass ``weights`` (e.g. the local-search confidence) so uncertain boundaries
    contribute less -- a wrong anchor otherwise self-reinforces."""
    T = feat_view1.shape[0]
    pairs = [(int(b), 1.0 if weights is None else float(w))
             for b, w in zip(boundaries, weights or [1.0] * len(boundaries))
             if 0 <= int(b) < T]
    if not pairs:
        return feat_view1.sum() * 0.0
    losses, ws = [], []
    for b, w in pairs:
        a = feat_view1[b]                                   # anchor
        pos = feat_view2[b]                                 # positive
        neg_idx = _negatives(feat_view1, b, T, n_adj, n_hard, a)
        if len(neg_idx) == 0:
            continue
        negs = feat_view1[neg_idx]                          # (M, D)
        s_pos = (a @ pos) / tau
        s_neg = (negs @ a) / tau
        logits = torch.cat([s_pos[None], s_neg])
        losses.append(F.cross_entropy(logits[None], torch.zeros(1, dtype=torch.long,
                                                                device=logits.device)))
        ws.append(w)
    if not losses:
        return feat_view1.sum() * 0.0
    w_t = torch.tensor(ws, device=feat_view1.device)
    return (torch.stack(losses) * w_t).sum() / w_t.sum().clamp_min(1e-6)


def jitter_views(feat: torch.Tensor, drop: float = 0.1, noise: float = 0.05
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Two cheap context perturbations of a (T, D) feature sequence for CBD:
    feature dropout + additive Gaussian noise. **The timeline is not touched** --
    frame ``b`` stays the transition in both views (a temporal shift would move
    the boundary and break the loss). A fuller version would mix in background
    clips from other videos (CVA's QCD)."""
    def one():
        x = F.dropout(feat, p=drop, training=True)
        return x + noise * torch.randn_like(x)
    return one(), one()
