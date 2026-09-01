"""Unit tests for VLM-direct alignment (delta.align.ta, delta.align.similarity)."""

import numpy as np
import pytest

from delta.align import segments
from delta.align.similarity import similarity_matrix, transcript_text_embeddings
from delta.align.ta import align_dp, align_soft


def _blocky_similarity(transcript, block_lens, noise=0.0, seed=0):
    """Build an (N, T) similarity that is high on the true block for each entry."""
    rng = np.random.default_rng(seed)
    N = len(transcript)
    T = sum(block_lens)
    s = rng.normal(0, noise, size=(N, T)) if noise else np.zeros((N, T))
    t = 0
    for n, L in enumerate(block_lens):
        s[n, t : t + L] += 1.0
        t += L
    return s


def test_align_dp_recovers_clean_blocks():
    transcript = [5, 2, 7, 2]                      # note: class 2 repeats
    lens = [30, 20, 25, 15]
    s = _blocky_similarity(transcript, lens)
    r = align_dp(s, transcript)
    # exact recovery
    assert r.boundaries == [30, 50, 75]
    assert r.y_star.tolist() == sum(([c] * L for c, L in zip(transcript, lens)), [])
    # transcript order preserved
    assert [c for c, _, _ in segments(r.y_star)] == transcript


def test_align_dp_is_monotone_and_covers_all_entries():
    transcript = [0, 1, 2, 3, 4]
    s = _blocky_similarity(transcript, [10, 10, 10, 10, 10], noise=0.4, seed=1)
    r = align_dp(s, transcript)
    assert np.all(np.diff(r.entry_of_frame) >= 0)         # monotone
    assert set(r.entry_of_frame.tolist()) == {0, 1, 2, 3, 4}  # every entry present
    assert len(r.boundaries) == 4


def test_align_dp_transition_penalty_reduces_oversegmentation():
    transcript = [0, 1, 0, 1]
    s = _blocky_similarity(transcript, [15, 15, 15, 15], noise=0.9, seed=2)
    n_lo = len(segments(align_dp(s, transcript, transition_penalty=0.0).y_star))
    n_hi = len(segments(align_dp(s, transcript, transition_penalty=5.0).y_star))
    assert n_hi <= n_lo


def test_align_dp_rejects_transcript_longer_than_video():
    with pytest.raises(ValueError):
        align_dp(np.zeros((5, 3)), [0, 1, 2, 3, 4])


def test_align_soft_boundary_dist_peaks_near_truth():
    transcript = [0, 1, 2]
    lens = [40, 40, 40]
    s = _blocky_similarity(transcript, lens, noise=0.2, seed=3)
    r = align_soft(s, transcript, temperature=0.1)
    assert r.boundary_dist.shape == (2, 120)
    # peak of P(boundary_0) should be near frame 40, boundary_1 near 80
    assert abs(int(r.boundary_dist[0].argmax()) - 40) <= 5
    assert abs(int(r.boundary_dist[1].argmax()) - 80) <= 5
    assert np.all(np.diff(r.entry_of_frame) >= 0)


def test_similarity_matrix_shapes_and_softmax():
    rng = np.random.default_rng(0)
    txt = rng.normal(size=(4, 16))
    frm = rng.normal(size=(50, 16))
    s = similarity_matrix(txt, frm)
    assert s.shape == (4, 50) and s.max() <= 1.0 + 1e-6 and s.min() >= -1.0 - 1e-6
    p = similarity_matrix(txt, frm, temperature=0.1)
    assert np.allclose(p.sum(axis=0), 1.0)


def test_transcript_text_embeddings_expands_by_order():
    class_emb = np.arange(3 * 2).reshape(3, 2).astype(float)
    out = transcript_text_embeddings([2, 0, 2], class_emb)
    assert out.tolist() == [[4, 5], [0, 1], [4, 5]]


def test_end_to_end_vlm_style():
    # synthetic "VLM": each class has a random unit direction; frames on the true
    # block point that way (+ noise). Align should recover the blocks.
    rng = np.random.default_rng(7)
    D = 32
    class_dirs = rng.normal(size=(5, D))
    transcript = [0, 1, 2, 3, 4]
    lens = [20, 25, 15, 30, 10]
    frames = []
    for c, L in zip(transcript, lens):
        frames.append(class_dirs[c] + 0.3 * rng.normal(size=(L, D)))
    frame_emb = np.concatenate(frames, 0)
    txt = transcript_text_embeddings(transcript, class_dirs)
    s = similarity_matrix(txt, frame_emb)
    r = align_dp(s, transcript, transition_penalty=0.5)
    from delta.align import mean_over_frames
    gt = np.array(sum(([c] * L for c, L in zip(transcript, lens)), []))
    assert mean_over_frames(r.y_star, gt) > 0.9
