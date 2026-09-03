"""Unit tests for the adapted MASRA regularizers (delta.align.masra)."""

import numpy as np
import pytest

from delta.align import segments
from delta.align.masra import (
    entry_classes,
    pool_events,
    esta_alignment,
    visual_relation_matrix,
    transcript_relation_matrix,
    lrca_residual,
    masra_report,
)


def _clean_case(seed=0):
    """3 actions, 3 blocks of 20 frames; frame features == the block's class
    embedding + a little noise. ESTA loss ~ 0, LRCA-block residual small."""
    rng = np.random.default_rng(seed)
    D = 8
    class_emb = rng.normal(size=(5, D))
    transcript = [3, 1, 4]
    lens = [20, 20, 20]
    y_star = np.concatenate([[i] * L for i, L in enumerate(lens)])
    feat = np.concatenate(
        [class_emb[c][None].repeat(L, 0) for c, L in zip(transcript, lens)]
    ) + rng.normal(scale=1e-3, size=(60, D))
    return feat, y_star, transcript, class_emb


def test_entry_classes_maps_frames_through_transcript():
    y = np.array([0, 0, 1, 2, 2])
    assert entry_classes(y, [7, 3, 9]).tolist() == [7, 7, 3, 9, 9]


def test_pool_events_flags_empty_blocks():
    feat = np.ones((10, 4))
    y = np.array([0] * 5 + [2] * 5)          # entry 1 gets nothing
    U, filled = pool_events(feat, y, 3)
    assert filled.tolist() == [True, False, True]
    assert np.allclose(U[1], 0.0)


def test_esta_loss_near_zero_when_blocks_match_their_class():
    feat, y_star, transcript, class_emb = _clean_case()
    r = esta_alignment(feat, y_star, transcript, class_emb)
    assert r.n_empty == 0
    assert r.loss < 1e-2
    assert np.all(r.per_entry > 0.99)


def test_esta_loss_large_when_alignment_is_shifted():
    feat, y_star, transcript, class_emb = _clean_case()
    shifted = np.roll(y_star, 10)            # blocks no longer line up with content
    good = esta_alignment(feat, y_star, transcript, class_emb).loss
    bad = esta_alignment(feat, shifted, transcript, class_emb).loss
    assert bad > good + 0.1


def test_esta_dim_mismatch_raises():
    feat, y_star, transcript, _ = _clean_case()
    with pytest.raises(ValueError):
        esta_alignment(feat, y_star, transcript, np.zeros((5, 3)))


def test_relation_matrices_shape_and_block_structure():
    feat, y_star, transcript, class_emb = _clean_case()
    S, idx = visual_relation_matrix(feat, max_frames=1000)
    R = transcript_relation_matrix(y_star, transcript, mode="block", idx=idx)
    assert S.shape == R.shape == (60, 60)
    # block R is 1 inside a segment, 0 across
    assert R[0, 19] == 1.0 and R[0, 20] == 0.0
    # clean features -> visual similarity is high within a block too
    assert S[0, 19] > 0.9


def test_lrca_residual_smaller_for_correct_alignment():
    feat, y_star, transcript, class_emb = _clean_case()
    S, idx = visual_relation_matrix(feat)
    R_good = transcript_relation_matrix(y_star, transcript, mode="block", idx=idx)
    R_bad = transcript_relation_matrix(np.roll(y_star, 10), transcript, mode="block", idx=idx)
    assert lrca_residual(S, R_good).loss < lrca_residual(S, R_bad).loss


def test_classsim_mode_needs_class_emb():
    feat, y_star, transcript, class_emb = _clean_case()
    with pytest.raises(ValueError):
        transcript_relation_matrix(y_star, transcript, mode="class-sim")


def test_subsampling_keeps_matrices_bounded():
    rng = np.random.default_rng(1)
    feat = rng.normal(size=(5000, 6))
    y_star = np.sort(rng.integers(0, 10, size=5000))
    S, idx = visual_relation_matrix(feat, max_frames=800)
    assert S.shape[0] <= 800 and len(idx) == S.shape[0]
    R = transcript_relation_matrix(y_star, list(range(10)), mode="block",
                                   idx=idx, max_frames=800)
    assert R.shape == S.shape


def test_masra_report_runs_and_reports_both_terms():
    feat, y_star, transcript, class_emb = _clean_case()
    gt_b = [b for _, b, _ in segments(entry_classes(y_star, transcript))][:-1]
    rep = masra_report(feat, y_star, transcript, class_emb=class_emb, gt_boundaries=gt_b)
    assert set(rep) >= {"T", "N", "lrca_block", "lrca_classsim", "esta_loss", "bdy_score_at_gt"}
    assert rep["esta_loss"] < 1e-2
