"""Unit tests for Stage-A weak alignment (delta.align.asot, delta.align.cost)."""

import numpy as np
import pytest

from delta.align import segments
from delta.align.asot import (temporal_prior, monotonic_mask, segment_asot, decode,
                              align_asot)
from delta.align.cost import fused_cost


def test_temporal_prior_is_diagonal_band():
    P = temporal_prior(100, 4, rho=1.0)
    assert P.shape == (100, 4)
    # cheapest column for an early frame is entry 0, for a late frame the last
    assert P[0].argmin() == 0
    assert P[-1].argmin() == 3


def test_monotonic_mask_peaks_are_ordered():
    m = monotonic_mask(300, 3)
    assert m.shape == (300,)
    assert (m > 0).all()
    # mass is spread but the early third should favour entry-0 region: mask high near 50,150,250
    assert m[50] > m[0] and m[150] > m[110]


def _blocky_cost(transcript, block_lens, noise=0.0, seed=0):
    """(T, N) cost that is ~0 on each entry's true block, ~1 elsewhere."""
    rng = np.random.default_rng(seed)
    N, T = len(transcript), sum(block_lens)
    C = np.ones((T, N)) + (rng.normal(0, noise, (T, N)) if noise else 0.0)
    t = 0
    for n, L in enumerate(block_lens):
        C[t:t + L, n] = 0.0
        t += L
    return C


def test_segment_asot_plan_shapes_and_normalisation():
    C = _blocky_cost([0, 1, 2], [20, 20, 20])
    plan, trace = segment_asot(C, n_iters=15)
    assert plan.shape == (60, 3)
    assert np.all(plan >= 0)
    assert len(trace) == 15


def test_align_asot_recovers_clean_blocks():
    transcript = [5, 2, 7, 2]
    lens = [30, 20, 25, 15]
    C = _blocky_cost(transcript, lens)
    r = align_asot(C, transcript, alpha=0.2)
    assert [c for c, _, _ in segments(r.y_star)] == transcript
    # boundaries within a few frames of the truth
    truth = np.cumsum(lens)[:-1]
    assert all(min(abs(np.array(r.boundaries) - t)) <= 4 for t in truth)


def test_decode_is_monotone_and_covers_every_entry():
    rng = np.random.default_rng(1)
    plan = rng.random((200, 6))
    y = decode(plan, 6)
    assert np.all(np.diff(y) >= 0)
    assert set(y.tolist()) == set(range(6))


def test_align_asot_stays_ordered_under_heavy_noise():
    """The temporal prior keeps every action present and in order even when the
    feature cost is almost pure noise -- the failure mode where hard DP lost to
    the naive prior on 50Salads."""
    transcript = [0, 1, 2, 3]
    C = _blocky_cost(transcript, [25, 25, 25, 25], noise=1.5, seed=3)
    y = align_asot(C, transcript, alpha=0.3, rho=0.3).y_star
    assert [c for c, _, _ in segments(y)] == transcript


def test_fused_cost_shape_and_temporal_term():
    rng = np.random.default_rng(0)
    D = 16
    class_emb = rng.normal(size=(5, D))
    transcript = [3, 1, 4]
    frame_emb = np.concatenate([class_emb[c][None].repeat(20, 0) for c in transcript])
    C = fused_cost(frame_emb, transcript, class_emb, w_sem=1.0, rho=0.1)
    assert C.shape == (60, 3)
    # frames that ARE their class embedding -> near-zero semantic cost on the diagonal block
    assert C[0, 0] < C[0, 2]
    assert C[59, 2] < C[59, 0]


def test_fused_cost_caption_term_optional():
    rng = np.random.default_rng(2)
    D = 8
    ce = rng.normal(size=(4, D))
    tr = [0, 1]
    fe = rng.normal(size=(10, D))
    cap = rng.normal(size=(10, D))
    a = fused_cost(fe, tr, ce, w_sem=1.0, rho=0.1)
    b = fused_cost(fe, tr, ce, caption_emb=cap, w_sem=1.0, w_cap=0.5, rho=0.1)
    assert not np.allclose(a, b)
