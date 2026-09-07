"""Unit tests for Stage-0 semantic-guided sampling (delta.features.sampling)."""

import numpy as np
import pytest

from delta.features.sampling import (
    transition_score,
    select_dense_zones,
    adaptive_frame_plan,
)


def _blocky_sim(n_entries=4, per_block=25, sharp=1.0, seed=0):
    """(N, T) similarity where entry n clearly wins inside its own block."""
    rng = np.random.default_rng(seed)
    T = n_entries * per_block
    s = rng.normal(0, 0.05, (n_entries, T))
    for n in range(n_entries):
        s[n, n * per_block:(n + 1) * per_block] += sharp
    return s


def test_transition_score_peaks_near_real_boundaries():
    s = _blocky_sim(per_block=25)
    score = transition_score(s)
    assert score.shape == (100,)
    assert 0.0 <= score.min() and score.max() <= 1.0
    truth = [25, 50, 75]
    for b in truth:                       # a local peak within a few frames of each cut
        assert score[b - 3:b + 4].max() > score[b - 15:b - 8].max()


def test_transition_score_flat_when_nothing_changes():
    s = np.repeat(np.array([[1.0], [0.2], [0.1]]), 60, axis=1)
    score = transition_score(s)
    assert score.max() - score.min() < 1e-6 or score.max() < 1e-6


def test_transition_score_handles_degenerate_input():
    assert transition_score(np.zeros((1, 20))).shape == (20,)


def test_select_dense_zones_respects_count_and_spacing():
    score = np.zeros(100)
    for c in (10, 11, 50, 80):
        score[c] = 1.0
    coarse_idx = np.arange(100) * 10
    zones = select_dense_zones(score, coarse_idx, n_zones=3, radius=20,
                               min_gap=5, n_frames=1000)
    assert len(zones) <= 3
    for a, b in zones:                     # inside the video, ordered
        assert 0 <= a < b <= 999
    assert all(zones[i][1] <= zones[i + 1][1] for i in range(len(zones) - 1))


def test_select_dense_zones_merges_overlaps():
    score = np.zeros(50)
    score[20] = 1.0
    score[21] = 0.9                        # adjacent peaks -> overlapping zones
    coarse_idx = np.arange(50) * 10
    zones = select_dense_zones(score, coarse_idx, n_zones=2, radius=40,
                               min_gap=1, n_frames=500)
    assert len(zones) == 1                 # merged into one


def test_adaptive_frame_plan_saves_budget_and_covers_boundaries():
    s = _blocky_sim(n_entries=4, per_block=25)
    coarse_idx = np.arange(0, 3000, 30)    # 1 fps over a 100 s video
    s_coarse = np.repeat(s, 1, axis=1)[:, : len(coarse_idx)]
    plan = adaptive_frame_plan(s_coarse, coarse_idx, n_frames=3000,
                               n_transcript=4, dense_every=2, radius=90)
    assert plan.n_full_rate == 3000
    assert plan.n_encoded < plan.n_full_rate
    assert 0.0 < plan.saving < 1.0
    assert plan.dense_frames.max() < 3000
    assert np.all(np.diff(plan.dense_frames) > 0)   # sorted, deduped


def test_adaptive_frame_plan_budget_scales_with_transcript_not_length():
    s = _blocky_sim(n_entries=4, per_block=25)
    short = adaptive_frame_plan(s, np.arange(0, 1000, 10), n_frames=1000,
                                n_transcript=4, radius=50)
    long = adaptive_frame_plan(s, np.arange(0, 4000, 40), n_frames=4000,
                               n_transcript=4, radius=50)
    # 4x the video, same recipe -> dense budget stays in the same ballpark
    assert long.dense_frames.size < 2 * short.dense_frames.size
    assert long.saving > short.saving                # relatively cheaper on longer video
