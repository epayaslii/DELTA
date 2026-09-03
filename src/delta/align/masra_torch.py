"""Differentiable MASRA regularizers (ESTA + LRCA), torch.

The numpy module ``delta.align.masra`` is the CPU-side prototype / diagnostic;
this is the version that goes into a training loop -- CG-DETR for the M1 TACoS
reproduction, and later our 50Salads pipeline (plan M3/M4). ``torch`` is imported
at module load, so this file is *not* re-exported from ``delta.align.__init__``
(keep the package importable without torch). Import it directly:

    from delta.align.masra_torch import esta_loss, lrca_loss, MasraRegularizer

Formulas (MASRA arXiv:2605.03398, Eq. 3-4 and 7-8), adapted per
``docs/masra-analysis.md``: the "event span" is the block of the transcript
entry under the *current* alignment, and the LRCA text relation matrix is built
from the transcript + class-name embeddings instead of MLLM clip captions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# ESTA -- event semantic temporal alignment  (MASRA Eq. 3-4)
# --------------------------------------------------------------------------------------


def pool_spans(feat: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``feat`` (T, C) over each ``[s, e)`` row of ``spans`` (M, 2).
    Empty / degenerate spans pool to a zero row. Returns (M, C)."""
    T, C = feat.shape
    out = feat.new_zeros(len(spans), C)
    for i, (s, e) in enumerate(spans.tolist()):
        s, e = int(s), int(e)
        if e > s:
            out[i] = feat[s:e].mean(dim=0)
    return out


def spans_from_assignment(entry_of_frame: torch.Tensor, n_entries: int) -> torch.Tensor:
    """Contiguous ``[s, e)`` per transcript entry from a per-frame entry index
    (monotone non-decreasing, e.g. from ``delta.align.align_dp``). Missing
    entries get ``[0, 0)``."""
    e = entry_of_frame
    spans = torch.zeros(n_entries, 2, dtype=torch.long, device=e.device)
    for i in range(n_entries):
        idx = (e == i).nonzero(as_tuple=False).flatten()
        if len(idx):
            spans[i, 0] = idx[0]
            spans[i, 1] = idx[-1] + 1
    return spans


def esta_loss(
    temporal_ctx: torch.Tensor,        # (T, C) -- the context branch H
    spans: torch.Tensor,               # (M, 2) long -- event [s, e)
    event_emb: torch.Tensor,           # (M, C) -- text embedding per event (action name / description)
    proj: nn.Module | None = None,     # optional Linear(C_ctx -> C_txt) on the pooled video side
    reduction: str = "mean",
) -> torch.Tensor:
    """``mean_i (1 - cos(u_i, o_i))`` over non-empty spans (Eq. 4)."""
    u = pool_spans(temporal_ctx, spans)                 # (M, C)
    if proj is not None:
        u = proj(u)
    filled = (spans[:, 1] > spans[:, 0])
    if filled.sum() == 0:
        return temporal_ctx.sum() * 0.0
    cos = F.cosine_similarity(u[filled], event_emb[filled], dim=-1)
    loss = 1.0 - cos
    return loss.mean() if reduction == "mean" else loss.sum()


# --------------------------------------------------------------------------------------
# LRCA -- local relational consistency alignment  (MASRA Eq. 7-8)
# --------------------------------------------------------------------------------------


def self_similarity(feat: torch.Tensor) -> torch.Tensor:
    """Cosine ``S = F F^T`` for L2-normalised rows. ``feat`` (T, C) -> (T, T)."""
    f = F.normalize(feat, p=2, dim=-1)
    return f @ f.t()


def transcript_relation_target(
    entry_of_frame: torch.Tensor,      # (T,) long -- entry index per frame
    transcript: torch.Tensor,          # (N,) long -- class id per entry
    class_emb: torch.Tensor | None = None,   # (C_cls, D)
    mode: str = "class-sim",           # "class-sim" (soft) | "block" (hard)
) -> torch.Tensor:
    """Text-side relation matrix ``R`` (T, T), the Eq. 8 target. No grad w.r.t.
    the model -- this is a fixed supervision signal for the current alignment."""
    with torch.no_grad():
        if mode == "block":
            e = entry_of_frame
            return (e[:, None] == e[None, :]).float()
        if mode == "class-sim":
            if class_emb is None:
                raise ValueError("mode='class-sim' needs class_emb")
            cls_per_frame = transcript[entry_of_frame]              # (T,)
            g = F.normalize(class_emb[cls_per_frame], p=2, dim=-1)  # (T, D)
            return g @ g.t()
        raise ValueError(f"unknown mode {mode!r}")


def lrca_loss(
    feat: torch.Tensor,                # (T, C) -- temporal feature E
    relation_target: torch.Tensor,     # (T, T) -- R
    frame_mask: torch.Tensor | None = None,   # (T,) bool -- valid frames
) -> torch.Tensor:
    """``mean_ij (s_ij - r_ij)^2`` (Eq. 8)."""
    S = self_similarity(feat)
    d = (S - relation_target) ** 2
    if frame_mask is not None:
        m = frame_mask.float()
        w = m[:, None] * m[None, :]
        return (d * w).sum() / w.sum().clamp_min(1.0)
    return d.mean()


# --------------------------------------------------------------------------------------
# SORA -- second-order relational attention  (MASRA Eq. 9-10; optional refinement)
# --------------------------------------------------------------------------------------


class SoraRefine(nn.Module):
    """``S~ = S + phi(S)``; ``F = softmax(S~) @ MLP(E)``. A light conv denoise of
    the similarity map fed back into the features. Backbone-agnostic, cheap --
    port after LRCA is working (Tab. 5: ~+0.3)."""

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.phi = nn.Sequential(
            nn.Conv2d(1, 8, kernel, padding=pad), nn.GELU(),
            nn.Conv2d(8, 1, kernel, padding=pad),
        )
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.GELU(),
                                 nn.Linear(channels, channels))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        S = self_similarity(feat)                       # (T, T)
        S = S + self.phi(S[None, None])[0, 0]
        return torch.softmax(S, dim=-1) @ self.mlp(feat)


# --------------------------------------------------------------------------------------
# combined
# --------------------------------------------------------------------------------------


class MasraRegularizer(nn.Module):
    """ESTA + LRCA as one auxiliary term. Add its output to the backbone loss:

        reg = MasraRegularizer(lam_sem=1.0, lam_rel=1.0)
        loss = loss_vtg + reg(temporal_ctx=H, temporal_feat=E,
                              entry_of_frame=y, transcript=tr,
                              event_emb=O, class_emb=G)
    """

    def __init__(self, lam_sem: float = 1.0, lam_rel: float = 1.0,
                 relation_mode: str = "class-sim"):
        super().__init__()
        self.lam_sem = lam_sem
        self.lam_rel = lam_rel
        self.relation_mode = relation_mode

    def forward(self, *, temporal_ctx, temporal_feat, entry_of_frame, transcript,
                event_emb, class_emb=None, esta_proj=None, frame_mask=None) -> dict:
        n_entries = len(transcript)
        spans = spans_from_assignment(entry_of_frame, n_entries)
        l_sem = esta_loss(temporal_ctx, spans, event_emb, proj=esta_proj)
        R = transcript_relation_target(entry_of_frame, transcript, class_emb,
                                       mode=self.relation_mode)
        l_rel = lrca_loss(temporal_feat, R, frame_mask=frame_mask)
        total = self.lam_sem * l_sem + self.lam_rel * l_rel
        return {"loss": total, "loss_esta": l_sem.detach(), "loss_lrca": l_rel.detach()}
