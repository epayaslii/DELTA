"""Unit tests for the parts that don't need a GPU or video files."""

import numpy as np

from delta.data.datasets import to_transcript, load_mapping, read_groundtruth
from delta.features.video_io import sample_frame_indices, FrameSampler
from delta.align import (
    mean_over_frames,
    mean_over_classes,
    edit_score,
    f1_at_k,
    kendall_tau_alignment,
)


def test_transcript_collapse():
    labels = np.array([0, 0, 0, 3, 3, 1, 1, 1, 3])
    assert to_transcript(labels) == [0, 3, 1, 3]
    assert to_transcript(np.array([])) == []


def test_mapping_and_gt(tmp_path):
    (tmp_path / "mapping.txt").write_text("0 background\n1 add_oil\n2 add_salt\n")
    name_to_idx, idx_to_name = load_mapping(tmp_path / "mapping.txt")
    assert name_to_idx == {"background": 0, "add_oil": 1, "add_salt": 2}
    assert idx_to_name == ["background", "add_oil", "add_salt"]
    (tmp_path / "gt.txt").write_text("background\nadd_oil\nadd_oil\nadd_salt\n")
    gt = read_groundtruth(tmp_path / "gt.txt", name_to_idx)
    assert gt.tolist() == [0, 1, 1, 2]


def test_frame_index_sampling():
    # 30 fps source, 15 fps labels, 100 label frames -> ~200 source frames
    idx = sample_frame_indices(n_src=200, n_target=100, source_fps=30.0, label_fps=15.0)
    assert idx.shape == (100,)
    assert idx[0] == 1 and idx[-1] <= 199
    assert np.all(np.diff(idx) >= 0)
    # clamps when video is short
    idx2 = sample_frame_indices(n_src=50, n_target=100, source_fps=30.0, label_fps=15.0)
    assert idx2.max() == 49


def test_frame_sampler_window():
    s = FrameSampler(source_fps=30.0, label_fps=15.0, window=3, window_stride=2)
    plan = s.plan(n_src=200, n_target=10)
    assert plan.shape == (10, 3)
    assert plan.min() >= 0 and plan.max() <= 199


def test_segmentation_metrics():
    gt = np.array([0, 0, 1, 1, 1, 2, 2])
    assert mean_over_frames(gt, gt) == 1.0
    assert mean_over_classes(gt, gt) == 1.0
    pred = np.array([0, 0, 1, 1, 2, 2, 2])
    assert 0.0 < mean_over_frames(pred, gt) < 1.0
    assert edit_score(gt, gt) == 100.0
    f1 = f1_at_k(gt, gt)
    assert f1[0.5] == 100.0


def test_kendall_tau_monotone():
    a = np.linspace(0, 1, 20)[:, None] * np.ones((1, 4))
    b = a.copy()
    assert kendall_tau_alignment(a, b) > 0.99
    assert kendall_tau_alignment(a, b[::-1]) < -0.99
