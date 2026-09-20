"""Training-time hook: precomputed per-video similarity -> Y* for a batch.

Offline step (VLM features, GPU/cluster): for each video write
`<sim_dir>/<video_id>.npz` with `sim` (T_full, C) = cosine(frame, class text) from
`hybrid_ta.similarity`, and optionally `feat` (T_full, D). No ground-truth frame
labels are read here -- only the transcript (ordered action list).

Resampling: the WLTA loader keeps a prefix of the video (fraction obs_p + pred_p) and
draws `T` frames from it (one random frame per equal segment). We map batch frame k to
`(k + .5) / T * cutoff` in the full-resolution similarity -- exact up to the loader's
within-segment jitter (~ cutoff / T frames).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .pseudo_labels import HybridConfig, generate_batch


class HybridPseudoLabeler:
    def __init__(self, sim_dir: str, cfg: HybridConfig | None = None):
        self.sim_dir, self.cfg, self._cache = sim_dir, cfg or HybridConfig(), {}

    def _load(self, vid: str):
        if vid not in self._cache:
            path = os.path.join(self.sim_dir, os.path.splitext(vid)[0] + ".npz")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"no similarity file for {vid}: {path}")
            z = np.load(path)
            self._cache[vid] = (z["sim"].astype(np.float32), z["feat"].astype(np.float32) if "feat" in z else None)
        return self._cache[vid]

    def __call__(self, fnames, transcripts, mask: torch.Tensor, frac: torch.Tensor | float = 1.0) -> torch.Tensor:
        B, T = mask.shape
        frac = torch.as_tensor(frac, dtype=torch.float32).reshape(-1).expand(B)
        sims, feats, has_feat = [], [], True
        for i in range(B):
            sim, feat = self._load(fnames[i])
            cutoff = max(1, int(len(sim) * min(1.0, float(frac[i]))))
            idx = np.minimum(((np.arange(T) + 0.5) / T * cutoff).astype(int), cutoff - 1)
            sims.append(sim[idx])
            if feat is None:
                has_feat = False
            else:
                feats.append(feat[idx])
        sim_b = torch.from_numpy(np.stack(sims))
        feat_b = torch.from_numpy(np.stack(feats)) if has_feat else None
        return generate_batch(sim_b, transcripts, mask.cpu(), feat_b, self.cfg)
