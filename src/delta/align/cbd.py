"""Stage-B boundary refinement: context-invariant boundary discrimination (CBD).

CVA's CBD loss (CVPR'26), adapted to the weak setting. CVA computes it on
**ground-truth** span boundaries; we compute it on the **predicted** boundaries
of the Stage-A alignment (`delta.align.asot`). No GT spans -- so it is usable
here even though CVA-the-paper is fully supervised.

Idea: a true transition frame's representation should be
  * invariant to surrounding context   -> positive = the same frame index in a
    second augmentation of the clip
  * distinct from non-transition frames -> negatives = temporally adjacent frames
    and the hardest (most cosine-similar) frames elsewhere

InfoNCE over those, weight ~0.005 against the backbone loss.

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
    boundaries: list[int],          # predicted transition frame indices (Stage A)
    n_adj: int = 2,
    n_hard: int = 10,
    tau: float = 0.07,
) -> torch.Tensor:
    """Mean InfoNCE over boundary frames. anchor = view1[b], positive = view2[b],
    negatives = adjacent + hard-mined from view1."""
    T = feat_view1.shape[0]
    bs = [int(b) for b in boundaries if 0 <= int(b) < T]
    if not bs:
        return feat_view1.sum() * 0.0
    losses = []
    for b in bs:
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
    if not losses:
        return feat_view1.sum() * 0.0
    return torch.stack(losses).mean()


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
