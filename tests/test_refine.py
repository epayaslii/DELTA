"""Unit tests for the Stage-B1 local semantic boundary search (delta.align.refine)."""

import numpy as np
import pytest

from delta.align import segments
from delta.align.refine import search_one, refine_boundaries, _confidence


def _sim_with_boundary(T=120, N=3, true_bounds=(40, 80), sharp=1.0, seed=0):
    """(N, T) similarity where entry n is clearly best inside its true block."""
    rng = np.random.default_rng(seed)
    sim = rng.normal(0, 0.1, (N, T))
    edges = [0, *true_bounds, T]
    for n, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sim[n, a:b] += sharp
    return sim


def test_search_one_moves_a_wrong_coarse_boundary_toward_the_truth():
    sim = _sim_with_boundary(true_bounds=(40, 80))
    # coarse boundary between entry 0 and 1 is off by +12
    nb, conf = search_one(sim, b0=52, e_left=0, e_right=1, radius=30, window=15)
    assert abs(nb - 40) < abs(52 - 40)
    assert 0.0 <= conf <= 1.0


def test_search_one_confidence_higher_for_a_sharp_boundary():
    # average over seeds: a real boundary is consistently more prominent than noise
    sharp = np.mean([search_one(_sim_with_boundary(sharp=3.0, seed=s), 40, 0, 1, 25, 12)[1]
                     for s in range(8)])
    weak = np.mean([search_one(_sim_with_boundary(sharp=0.02, seed=s), 40, 0, 1, 25, 12)[1]
                    for s in range(8)])
    assert sharp > weak


def test_refine_boundaries_keeps_order_and_count():
    sim = _sim_with_boundary(true_bounds=(40, 80))
    # coarse entry assignment with boundaries at 30 and 70
    e = np.concatenate([[0] * 30, [1] * 40, [2] * 50])
    res = refine_boundaries(sim, e, radius=25, window=15)
    assert len(res.boundaries) == 2
    assert res.boundaries[0] < res.boundaries[1]
    assert np.all(np.diff(res.entry_of_frame) >= 0)
    assert set(res.entry_of_frame.tolist()) == {0, 1, 2}


def test_refine_boundaries_improves_boundary_offset():
    true_b = (40, 80)
    sim = _sim_with_boundary(true_bounds=true_b)
    e = np.concatenate([[0] * 25, [1] * 40, [2] * 55])          # coarse: 25, 65
    res = refine_boundaries(sim, e, radius=30, window=15)
    coarse_err = abs(25 - 40) + abs(65 - 80)
    refined_err = abs(res.boundaries[0] - 40) + abs(res.boundaries[1] - 80)
    assert refined_err < coarse_err


def test_visual_change_term_rewards_a_real_feature_jump():
    sim = _sim_with_boundary(true_bounds=(40, 80))
    D = 12
    frame_emb = np.zeros((120, D))
    frame_emb[:40] = np.array([1.0] + [0] * (D - 1))
    frame_emb[40:80] = np.array([0, 1.0] + [0] * (D - 2))
    frame_emb[80:] = np.array([0, 0, 1.0] + [0] * (D - 3))
    nb, _ = search_one(sim, 50, 0, 1, radius=25, window=12,
                       frame_emb=frame_emb, w_visual=2.0)
    assert abs(nb - 40) <= 3


def test_confidence_zero_for_flat_profile():
    assert _confidence(np.ones(20)) == pytest.approx(0.0, abs=1e-6)
