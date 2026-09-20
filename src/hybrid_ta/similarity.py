"""(a) Frame x Action similarity.

`cosine_similarity` is the whole mechanism: L2-normalise both sides, dot product.
Meaningful only when frames and text live in one embedding space (a VLM such as
SigLIP2 / VideoLLaMA3). Raw 2048-D I3D features are NOT in a text-aligned space;
`FrameActionSimilarity` can project both sides to a shared dim (default 256) so the
shapes work, but those projections are untrained here -- use them for shape
plumbing / dry-runs, or train them yourself; do not expect zero-shot semantics.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def cosine_similarity(vis: torch.Tensor, txt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """vis (B,T,D) x txt (B,N,D) or (N,D) -> (B,T,N) cosine similarity."""
    if txt.dim() == 2:
        txt = txt.unsqueeze(0).expand(vis.shape[0], -1, -1)
    if vis.shape[-1] != txt.shape[-1]:
        raise ValueError(f"dim mismatch {vis.shape[-1]} vs {txt.shape[-1]}: project to a shared dim first")
    v = F.normalize(vis, dim=-1, eps=eps)
    t = F.normalize(txt, dim=-1, eps=eps)
    return torch.einsum("btd,bnd->btn", v, t)


class FrameActionSimilarity(nn.Module):
    def __init__(self, vis_dim: int, txt_dim: int, proj_dim: int | None = 256):
        super().__init__()
        same = vis_dim == txt_dim and proj_dim is None
        self.vis_proj = nn.Identity() if same else nn.Linear(vis_dim, proj_dim or txt_dim, bias=False)
        self.txt_proj = nn.Identity() if same else nn.Linear(txt_dim, proj_dim or txt_dim, bias=False)

    def forward(self, vis: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        if txt.dim() == 2:
            txt = txt.unsqueeze(0).expand(vis.shape[0], -1, -1)
        return cosine_similarity(self.vis_proj(vis), self.txt_proj(txt))
