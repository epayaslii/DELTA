"""Stage-A weak alignment: fused Gromov-Wasserstein optimal transport.

A minimal self-contained numpy port of ASOT (Xu & Gould, CVPR'24) / CLOT
(ICCV'25) -- the solver the group's WLTA code uses as
``--model_type wclot`` (``third_party/delta_wlta/src/asot.py``). We keep it in
numpy for parity with the rest of ``delta.align`` and so Stage A runs without a
GPU.

The alignment problem: assign every frame ``t`` to a transcript entry ``n`` so
that (a) the assignment cost ``C[t, n]`` is low, (b) visually similar frames get
similar assignments (the Gromov-Wasserstein term), (c) the assignment respects
transcript order in time (the temporal prior + monotonic mask). No trained
classifier; ``C`` comes from a frozen VLM (`delta.align.cost`).

Unlike hard DP (`delta.align.ta.align_dp`), the OT plan is soft and the temporal
prior keeps it from collapsing when the visual signal is weak -- which is exactly
where hard DP lost to the naive prior on 50Salads (`docs/50salads-notes.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eval import segments


# --------------------------------------------------------------------------------------
# transcript-derived priors
# --------------------------------------------------------------------------------------


def temporal_prior(n_frames: int, n_entries: int, rho: float = 0.15) -> np.ndarray:
    """(T, N) cost that grows with |t/T - n/N| -- a soft diagonal band pinning
    transcript order to time. ``rho`` scales its weight against the feature cost."""
    t = np.arange(n_frames)[:, None] / max(n_frames, 1)
    n = np.arange(n_entries)[None, :] / max(n_entries, 1)
    return rho * np.abs(t - n)


def monotonic_mask(n_frames: int, n_entries: int, sigma_scale: float = 0.35) -> np.ndarray:
    """(T,) soft per-frame plausibility -- max over per-entry Gaussians centred at
    ``(n+0.5)/N``. Multiplies the transport plan; keeps mass near the order-diagonal
    without a hard cutoff."""
    if n_entries == 0:
        return np.ones(n_frames)
    xs = np.arange(n_frames)[None, :]
    centers = ((np.arange(n_entries) + 0.5) * n_frames / n_entries)[:, None]
    sigma = max(1.0, sigma_scale * n_frames / n_entries)
    g = np.exp(-((xs - centers) ** 2) / (2 * sigma ** 2)).max(axis=0)
    g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    return np.clip(g, 1e-4, 1.0)


# --------------------------------------------------------------------------------------
# GW term -- O(TN) via a 1-D convolution over the time axis
# --------------------------------------------------------------------------------------


def _gw_grad(T: np.ndarray, radius: float) -> np.ndarray:
    """d/dT of the Gromov-Wasserstein structure term, approximated by ASOT's
    band filter: frames within ``radius*T`` should share an assignment."""
    n_frames = T.shape[0]
    r = max(1, int(n_frames * radius))
    k = np.ones(2 * r + 1) / max(radius, 1e-6)
    k[r] = 0.0
    row_tot = T.sum(axis=1, keepdims=True) - T                      # (T, N)
    out = np.empty_like(T)
    for j in range(T.shape[1]):
        out[:, j] = np.convolve(row_tot[:, j], k, mode="same")
    return out


# --------------------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------------------


@dataclass
class AsotResult:
    y_star: np.ndarray          # (T,) transcript-entry index per frame
    plan: np.ndarray            # (T, N) soft transport plan
    boundaries: list[int]
    obj_trace: list[float]


def segment_asot(
    cost: np.ndarray,           # (T, N) assignment cost, lower = better
    mask: np.ndarray | None = None,     # (T,) soft frame plausibility
    eps: float = 0.07,          # entropic regularisation
    alpha: float = 0.3,         # GW structure weight (0 = pure OT)
    radius: float = 0.04,       # GW band, fraction of T
    lambda_actions: float = 0.05,       # unbalanced-KL weight on the action marginal
    n_iters: int = 25,
    step_size: float | None = None,
) -> np.ndarray:
    """Mirror-descent fused-GW OT. Frames balanced, actions unbalanced (ASOT
    default). Returns the (T, N) plan."""
    n_frames, n_entries = cost.shape
    m = np.ones(n_frames) if mask is None else np.asarray(mask, float)
    nnz = max(m.sum(), 1e-8)
    dx = (m / nnz)[:, None]                                          # frame marginal
    dy = np.full((1, n_entries), 1.0 / n_entries)                    # action marginal

    plan = dx * dy
    trace: list[float] = []
    for it in range(n_iters):
        gw = _gw_grad(plan, radius)
        grad = alpha * gw + (1.0 - alpha) * cost
        grad = grad + eps * np.log(plan + 1e-12)                     # entropy
        marg_a = plan.sum(axis=0, keepdims=True)
        grad = grad + lambda_actions * (np.log(marg_a / dy + 1e-12) + 1.0)
        if it == 0 and step_size is None:
            step_size = 4.0 / (np.abs(grad).max() + 1e-12)
        plan = plan * np.exp(-step_size * grad)
        plan = plan / (plan.sum(axis=1, keepdims=True) + 1e-12) * dx   # renorm frames
        plan = plan * m[:, None]
        trace.append(float((grad * plan).sum()))
    return plan, trace


def decode(plan: np.ndarray, n_entries: int) -> np.ndarray:
    """(T, N) plan -> (T,) monotone non-decreasing entry index, every entry
    covered. Order-preserving DP over the plan (treated as a similarity):
    reuses the tested Viterbi in `delta.align.ta`."""
    from .ta import align_dp

    # align_dp wants (N, T) similarity and a transcript; use identity transcript
    return align_dp(plan.T, list(range(n_entries))).entry_of_frame


def align_asot(
    cost: np.ndarray,           # (T, N)
    transcript: list[int],
    mask: np.ndarray | None = None,
    rho: float | None = None,   # if set, add temporal_prior(rho) to `cost` here
    **kw,
) -> AsotResult:
    """Full Stage-A: solve + decode. Pass ``rho`` to fold in the transcript-order
    temporal prior, or pre-add it (see `delta.align.cost.fused_cost`)."""
    n_entries = len(transcript)
    C = cost if rho is None else cost + temporal_prior(cost.shape[0], n_entries, rho=rho)
    if mask is None:
        mask = monotonic_mask(C.shape[0], n_entries)
    plan, trace = segment_asot(C, mask=mask, **kw)
    y = decode(plan, n_entries)
    entry = np.asarray(y, int)
    labels = np.asarray(transcript, int)[entry]
    bounds = [int(i) for i in np.flatnonzero(np.diff(entry) != 0) + 1]
    return AsotResult(y_star=labels, plan=plan, boundaries=bounds, obj_trace=trace)
