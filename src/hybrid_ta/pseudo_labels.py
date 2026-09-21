"""(d) Fuse both branches into the frame-level pseudo-label tensor Y*.

Per boundary: each branch proposes (position, confidence). If both are confident and
agree (within `agree_tol` frames) -> confidence-weighted mean; if they disagree -> the
more confident one; if neither clears `min_conf` -> keep the coarse boundary. Then
enforce temporal consistency: strictly ordered, at least `min_len` frames per entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .coarse_align import coarse_align
from .boundary_refinement import BranchResult, branch_a_semantic, branch_b_relational, coarse_boundaries


@dataclass
class HybridConfig:
    method: str = "asot"          # coarse solver: asot | dp  (asot: +10 pts frame-acc over dp on local SigLIP2 videos)
    branches: str = "a"           # refinement: a | b | ab | none. Default a: on local SigLIP2 videos A helps, B hurts (see README)
    radius: int = 30
    window: int = 20
    w_visual: float = 0.5
    min_conf: float = 0.05
    agree_tol: int = 10
    min_len: int = 1
    transition_penalty: float = 0.0
    rho: float = 0.15             # asot: transcript-order temporal prior weight
    alpha: float = 0.3            # asot: Gromov-Wasserstein structure weight (0 = pure OT)


@dataclass
class HybridResult:
    y_star: np.ndarray            # (T,) class id per frame
    entry_of_frame: np.ndarray    # (T,) transcript-entry index per frame
    boundaries: list[int]
    coarse: list[int]
    conf: list[float]


def fuse_boundaries(coarse: list[int], a: BranchResult | None, b: BranchResult | None,
                    T: int, cfg: HybridConfig) -> tuple[list[int], list[float]]:
    out, conf = [], []
    for k, b0 in enumerate(coarse):
        props = []
        for br in (a, b):
            if br is not None and br.confidences[k] >= cfg.min_conf:
                props.append((br.boundaries[k], br.confidences[k]))
        if not props:
            out.append(b0); conf.append(0.0)
        elif len(props) == 2 and abs(props[0][0] - props[1][0]) <= cfg.agree_tol:
            (pa, ca), (pb, cb) = props
            out.append(int(round((pa * ca + pb * cb) / (ca + cb)))); conf.append(max(ca, cb))
        else:
            p, c = max(props, key=lambda x: x[1])
            out.append(int(p)); conf.append(float(c))
    # temporal consistency: ordered, each entry gets >= min_len frames
    n = len(out)
    for k in range(n):
        lo = (out[k - 1] if k else 0) + cfg.min_len
        hi = T - (n - k) * cfg.min_len
        out[k] = int(min(max(out[k], lo), hi))
    return out, conf


def generate_pseudo_labels(sim_tn: np.ndarray, transcript: list[int],
                           frame_feat: np.ndarray | None = None,
                           cfg: HybridConfig | None = None) -> HybridResult:
    """sim_tn (T, N): frame x transcript-entry similarity (column i <-> transcript[i]).
    frame_feat (T, D): per-frame embedding for branch B / branch A's visual term;
    defaults to the similarity profile itself."""
    cfg = cfg or HybridConfig()
    sim_tn = np.asarray(sim_tn, dtype=np.float64)
    T, N = sim_tn.shape
    assert N == len(transcript), (N, len(transcript))
    entry = coarse_align(sim_tn, cfg.method, cfg.transition_penalty, rho=cfg.rho, alpha=cfg.alpha)
    coarse = coarse_boundaries(entry)
    if cfg.branches == "none" or not coarse:
        bounds, conf = coarse, [0.0] * len(coarse)
    else:
        emb = sim_tn if frame_feat is None else frame_feat
        a = branch_a_semantic(sim_tn.T, entry, cfg.radius, cfg.window, emb, cfg.w_visual) if "a" in cfg.branches else None
        b = branch_b_relational(emb, entry, cfg.radius, cfg.window) if "b" in cfg.branches else None
        bounds, conf = fuse_boundaries(coarse, a, b, T, cfg)
    ent = np.empty(T, dtype=np.int64)
    edges = [0, *bounds, T]
    for e, (s, z) in enumerate(zip(edges[:-1], edges[1:])):
        ent[s:z] = e
    y = np.asarray(transcript, dtype=np.int64)[ent]
    return HybridResult(y, ent, list(bounds), list(coarse), list(conf))


def generate_batch(sim_btc: torch.Tensor, transcripts: list[list[int]], mask: torch.Tensor,
                   frame_feat: torch.Tensor | None = None, cfg: HybridConfig | None = None,
                   ignore_index: int = -100) -> torch.Tensor:
    """sim_btc (B, T, C): CLASS-level frame x action similarity. For each video the
    columns of its transcript are gathered (repeated actions -> repeated columns).
    mask (B, T) bool marks valid frames. Returns Y* (B, T) long, `ignore_index` on
    padding -- the same convention as ATBA's own pseudo-labels."""
    B, T, _ = sim_btc.shape
    out = torch.full((B, T), ignore_index, dtype=torch.long)
    for i in range(B):
        tv = int(mask[i].sum())
        cols = torch.as_tensor(transcripts[i], dtype=torch.long)
        s = sim_btc[i, :tv][:, cols].detach().cpu().numpy()
        f = None if frame_feat is None else frame_feat[i, :tv].detach().cpu().numpy()
        res = generate_pseudo_labels(s, list(map(int, transcripts[i])), f, cfg)
        out[i, :tv] = torch.from_numpy(res.y_star)
    return out
