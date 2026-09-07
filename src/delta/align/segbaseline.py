"""Alignment *through segmentation* -- a compact ATBA/HAL-style baseline.

The question this answers: on 50Salads, does the classifier-based paradigm
(ATBA, HAL) beat our VLM-direct alignment? Both are weakly supervised -- only
the transcript, never a timestamp.

The loop, which is ATBA's core stripped of its boundary-detector refinements:

    pseudo-labels  <-- naive-uniform alignment of the transcript
    repeat:
        train a frame classifier on the current pseudo-labels
        posteriors = classifier(features)
        pseudo-labels <-- order-preserving DP through those posteriors
    Y* = final pseudo-labels

HAL adds a two-scale VAE regulariser on top of exactly this; ``recon_weight``
enables a light stand-in for it (an auxiliary reconstruction head), so we can
see whether that *kind* of regularisation helps here at all.

This is a faithful-in-spirit reimplementation, NOT the published HAL -- it has
no VAE latents, no ELBO, no boundary detector, and trains far shorter. Treat it
as evidence about the paradigm, not as a reproduction of HAL's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegBaselineConfig:
    hidden: int = 256
    sample_rate: int = 10          # keep every N-th frame while training
    epochs_per_round: int = 8
    rounds: int = 4                # pseudo-label refinement rounds
    lr: float = 1e-3
    recon_weight: float = 0.0      # >0 = HAL-style auxiliary reconstruction
    seed: int = 0
    device: str = "cpu"


@dataclass
class SegBaselineResult:
    y_star: dict[str, np.ndarray] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


def _naive_entry_index(n_entries: int, T: int) -> np.ndarray:
    """Equal-length blocks in transcript order -- the round-0 pseudo-labels."""
    edges = np.linspace(0, T, n_entries + 1).astype(int)
    y = np.zeros(T, dtype=int)
    for i in range(n_entries):
        y[edges[i]:edges[i + 1]] = i
    return y


class _Net:
    """Frame classifier: 1-D temporal conv stack -> per-frame class logits.
    Optional decoder head for the HAL-style reconstruction term."""

    def __init__(self, dim: int, n_classes: int, cfg: SegBaselineConfig):
        import torch
        import torch.nn as nn

        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        h = cfg.hidden
        self.encoder = nn.Sequential(
            nn.Conv1d(dim, h, 5, padding=2), nn.ReLU(),
            nn.Conv1d(h, h, 5, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.ReLU(),
        ).to(cfg.device)
        self.head = nn.Conv1d(h, n_classes, 1).to(cfg.device)
        self.decoder = (nn.Conv1d(h, dim, 1).to(cfg.device)
                        if cfg.recon_weight > 0 else None)

    def parameters(self):
        ps = list(self.encoder.parameters()) + list(self.head.parameters())
        if self.decoder is not None:
            ps += list(self.decoder.parameters())
        return ps

    def __call__(self, x):                       # x: (1, D, T)
        z = self.encoder(x)
        return self.head(z), (self.decoder(z) if self.decoder is not None else None)


def run(ds, train_ids, test_ids, feature_dir: str,
        cfg: SegBaselineConfig | None = None) -> SegBaselineResult:
    """Train the loop on ``train_ids``, then emit ``Y*`` for ``test_ids``."""
    import torch
    import torch.nn.functional as F

    from .ta import align_dp

    cfg = cfg or SegBaselineConfig()
    dev = cfg.device
    rng = np.random.default_rng(cfg.seed)
    n_classes = len(ds.action_names())

    def load(vid):
        f = np.load(f"{feature_dir}/{vid}.npy")
        rec = ds.record(vid)
        if f.shape[0] != rec.num_label_frames and f.shape[1] == rec.num_label_frames:
            pass                                  # already (D, T)
        elif f.shape[0] == rec.num_label_frames:
            f = f.T
        return f[:, ::cfg.sample_rate], rec        # (D, T_sub)

    train = {v: load(v) for v in train_ids}
    dim = next(iter(train.values()))[0].shape[0]

    # round 0 pseudo-labels: naive-uniform blocks
    pseudo = {}
    for v, (f, rec) in train.items():
        pseudo[v] = _naive_entry_index(len(rec.transcript), f.shape[1])

    net = _Net(dim, n_classes, cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    result = SegBaselineResult()

    for rnd in range(cfg.rounds):
        # ---- fit the classifier to the current pseudo-labels ----------------
        losses = []
        for _ in range(cfg.epochs_per_round):
            for v in rng.permutation(list(train)):
                f, rec = train[v]
                x = torch.from_numpy(f[None]).float().to(dev)
                cls = np.asarray(rec.transcript, int)[pseudo[v]]     # entry -> class id
                y = torch.from_numpy(cls).long().to(dev)[None]
                logits, recon = net(x)
                loss = F.cross_entropy(logits, y)
                if recon is not None:
                    loss = loss + cfg.recon_weight * F.mse_loss(recon, x)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.detach()))

        # ---- realign: DP through the classifier's posteriors ---------------
        shifted = 0
        with torch.no_grad():
            for v, (f, rec) in train.items():
                logp = torch.log_softmax(
                    net(torch.from_numpy(f[None]).float().to(dev))[0], dim=1
                )[0].cpu().numpy()                                   # (C, T_sub)
                s = logp[np.asarray(rec.transcript, int)]            # (N, T_sub)
                new = align_dp(s, rec.transcript).entry_of_frame
                shifted += int((new != pseudo[v]).mean() * 100)
                pseudo[v] = new
        result.history.append({"round": rnd, "loss": float(np.mean(losses)),
                               "pct_frames_relabelled": shifted / max(len(train), 1)})

    # ---- inference on held-out videos ------------------------------------
    with torch.no_grad():
        for v in test_ids:
            f, rec = load(v)
            logp = torch.log_softmax(
                net(torch.from_numpy(f[None]).float().to(dev))[0], dim=1
            )[0].cpu().numpy()
            s = logp[np.asarray(rec.transcript, int)]
            y_sub = align_dp(s, rec.transcript).y_star               # subsampled grid
            # upsample back to the full label grid
            idx = np.clip(np.arange(rec.num_label_frames) // cfg.sample_rate,
                          0, len(y_sub) - 1)
            result.y_star[v] = y_sub[idx]
    return result
