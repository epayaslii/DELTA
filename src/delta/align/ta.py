"""VLM-direct temporal alignment.

Given a transcript x frame similarity matrix ``s`` (N x T) and the transcript
order, assign every frame to a transcript entry such that the assignment is
**monotone non-decreasing in time** (each action occupies one contiguous block,
order preserved) and total matched similarity is maximised.

No trained frame classifier is involved — this is the whole method; ``s`` comes
straight from a frozen VLM (`delta.align.similarity`).

Two solvers:
  * ``align_dp``   — hard order-preserving DP (Viterbi over the N x T lattice).
  * ``align_soft`` — entropic / soft version: a column-stochastic assignment from
                     forward-backward, giving per-boundary distributions.

Both return an :class:`AlignResult` with the dense pseudo-labels ``Y*``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignResult:
    y_star: np.ndarray            # (T,) int  -- dense pseudo-labels (class ids)
    entry_of_frame: np.ndarray    # (T,) int  -- transcript-entry index per frame
    boundaries: list[int]         # N-1 frame indices where the entry switches
    boundary_dist: np.ndarray | None = None  # (N-1, T) soft P(boundary_r = t), if soft
    score: float = 0.0            # total matched similarity


def _to_labels(entry_of_frame: np.ndarray, transcript: list[int]) -> np.ndarray:
    t = np.asarray(transcript, dtype=np.int64)
    return t[entry_of_frame]


def _boundaries(entry_of_frame: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.diff(entry_of_frame) != 0) + 1]


def align_dp(
    s: np.ndarray,
    transcript: list[int],
    transition_penalty: float = 0.0,
    allow_skip: bool = False,
) -> AlignResult:
    """Hard order-preserving alignment.

    s                 : (N, T) similarity, N == len(transcript)
    transition_penalty: subtracted each time the entry advances (discourages
                        over-segmentation); in the same units as ``s``
    allow_skip        : if True, an entry may be skipped (advance by >1) at an
                        extra ``transition_penalty`` each. Off by default —
                        every transcript action must appear.
    """
    s = np.asarray(s, dtype=np.float64)
    N, T = s.shape
    if N != len(transcript):
        raise ValueError(f"s has {N} rows but transcript has {len(transcript)} entries")
    if N > T:
        raise ValueError(f"transcript ({N}) longer than video ({T} frames)")

    NEG = -1e18
    D = np.full((N, T), NEG)          # best cumulative score, entry n ends at frame t
    back = np.zeros((N, T), dtype=np.int8)  # 0 = stay, 1 = came from n-1, 2 = from n-2

    D[0, :] = np.cumsum(s[0, :])      # entry 0 must cover a prefix [0..t]
    for n in range(1, N):
        # entry n must start at frame >= n (each earlier entry >=1 frame)
        for t in range(n, T):
            stay = D[n, t - 1] if t > 0 else NEG
            adv1 = (D[n - 1, t - 1] if t > 0 else NEG) - transition_penalty
            best, arg = (stay, 0) if stay >= adv1 else (adv1, 1)
            if allow_skip and n >= 2:
                adv2 = (D[n - 2, t - 1] if t > 0 else NEG) - 2 * transition_penalty
                if adv2 > best:
                    best, arg = adv2, 2
            D[n, t] = s[n, t] + best
            back[n, t] = arg

    # backtrace from (N-1, T-1)
    entry = np.empty(T, dtype=np.int64)
    n, t = N - 1, T - 1
    while t >= 0:
        entry[t] = n
        step = back[n, t]
        n = n - (1 if step == 1 else 2 if step == 2 else 0)
        t -= 1
        if n < 0:
            n = 0
    result_entry = entry
    y = _to_labels(result_entry, transcript)
    return AlignResult(
        y_star=y,
        entry_of_frame=result_entry,
        boundaries=_boundaries(result_entry),
        score=float(D[N - 1, T - 1]),
    )


def align_soft(
    s: np.ndarray,
    transcript: list[int],
    temperature: float = 0.1,
) -> AlignResult:
    """Soft order-preserving alignment via entropic forward-backward over the
    monotone lattice. Returns per-frame entry posteriors and, from them,
    ``P(boundary_r = t)`` for each of the N-1 transitions.

    The MAP labeling (argmax posterior, then enforce monotonicity by a cheap
    isotonic pass) is stored in ``y_star``; the distributions are the point.
    """
    s = np.asarray(s, dtype=np.float64)
    N, T = s.shape
    e = s / float(temperature)
    e = e - e.max(axis=0, keepdims=True)
    emit = np.exp(e)                                   # (N, T) unnormalised emissions

    # forward: alpha[n,t] ~ sum over monotone paths ending with entry n at frame t
    alpha = np.zeros((N, T))
    alpha[0, 0] = emit[0, 0]
    for t in range(1, T):
        alpha[0, t] = alpha[0, t - 1] * emit[0, t]
        for n in range(1, N):
            alpha[n, t] = (alpha[n, t - 1] + alpha[n - 1, t - 1]) * emit[n, t]
        c = alpha[:, t].sum()
        if c > 0:
            alpha[:, t] /= c

    # backward
    beta = np.zeros((N, T))
    beta[N - 1, T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[N - 1, t] = beta[N - 1, t + 1] * emit[N - 1, t + 1]
        for n in range(N - 2, -1, -1):
            beta[n, t] = beta[n, t + 1] * emit[n, t + 1] + beta[n + 1, t + 1] * emit[n + 1, t + 1]
        c = beta[:, t].sum()
        if c > 0:
            beta[:, t] /= c

    post = alpha * beta
    post /= post.sum(axis=0, keepdims=True) + 1e-12    # (N, T) P(entry=n | frame t)

    # boundary_r distribution: P(entry switches from r to r+1 at frame t)
    bdist = np.zeros((max(N - 1, 1), T))
    for r in range(N - 1):
        bdist[r, 1:] = post[r, :-1] * post[r + 1, 1:]
        z = bdist[r].sum()
        if z > 0:
            bdist[r] /= z

    # MAP entry, then make monotone (non-decreasing) by a forward max pass
    raw = post.argmax(axis=0)
    entry = np.maximum.accumulate(raw)
    entry = np.clip(entry, 0, N - 1)
    y = _to_labels(entry, transcript)
    return AlignResult(
        y_star=y,
        entry_of_frame=entry,
        boundaries=_boundaries(entry),
        boundary_dist=bdist,
        score=float((post * s).sum()),
    )
