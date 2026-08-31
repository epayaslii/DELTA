"""Unit tests for delta.data.stats on a synthetic 50salads-shaped dataset."""

import numpy as np
import pytest

from delta.data import ActionSegDataset
from delta.data import stats


@pytest.fixture
def toy_ds(tmp_path):
    root = tmp_path / "50salads"
    (root / "groundTruth").mkdir(parents=True)
    (root / "splits").mkdir()
    # mid-level-ish vocab: 0 action_start, 1 action_end, 2..4 real actions
    (root / "mapping.txt").write_text(
        "0 action_start\n1 action_end\n2 add_oil\n3 cut_tomato\n4 mix\n"
    )
    # video A: start(10) oil(20) tomato(30) mix(40) end(10)  -> 110 frames
    seqA = ["action_start"] * 10 + ["add_oil"] * 20 + ["cut_tomato"] * 30 + ["mix"] * 40 + ["action_end"] * 10
    # video B: start(5) tomato(10) oil(5) mix(20) end(10)     -> 50 frames, oil is SHORT here
    seqB = ["action_start"] * 5 + ["cut_tomato"] * 10 + ["add_oil"] * 5 + ["mix"] * 20 + ["action_end"] * 10
    (root / "groundTruth" / "rgb-01-1.txt").write_text("\n".join(seqA) + "\n")
    (root / "groundTruth" / "rgb-01-2.txt").write_text("\n".join(seqB) + "\n")
    (root / "splits" / "train.split1.bundle").write_text("rgb-01-1.txt\n")
    (root / "splits" / "test.split1.bundle").write_text("rgb-01-2.txt\n")
    return ActionSegDataset("50salads", root)


def test_frame_and_segment_counts(toy_ds):
    fc = stats.frame_counts(toy_ds)
    assert fc[2] == 25  # add_oil: 20 + 5
    assert fc[4] == 60  # mix: 40 + 20
    sc = stats.segment_counts(toy_ds)
    assert sc[2] == 2 and sc[4] == 2


def test_video_rows(toy_ds):
    rows = {r["video_id"]: r for r in stats.video_rows(toy_ds)}
    assert rows["rgb-01-1"]["n_frames"] == 110
    assert rows["rgb-01-1"]["n_segments"] == 5
    assert rows["rgb-01-1"]["transcript_len"] == 5
    # action_start/end are background -> 20/110 for video A
    assert rows["rgb-01-1"]["bg_frac"] == pytest.approx(20 / 110)


def test_class_duration_cv_flags_variable_classes(toy_ds):
    summ = {r["class_name"]: r for r in stats.class_duration_summary(toy_ds)}
    # add_oil is 20f then 5f -> high CV; mix is 40f then 20f -> lower relative spread
    assert summ["add_oil"]["cv"] > summ["mix"]["cv"]
    assert summ["add_oil"]["n_instances"] == 2


def test_transition_matrix_rows_sum_to_one(toy_ds):
    P = stats.transition_matrix(toy_ds, normalize=True)
    row_sums = P.sum(axis=1)
    for c in range(P.shape[0]):
        assert row_sums[c] == pytest.approx(1.0) or row_sums[c] == pytest.approx(0.0)
    # from add_oil(2): A goes oil->tomato, B goes oil->mix  => 0.5 each
    assert P[2, 3] == pytest.approx(0.5)
    assert P[2, 4] == pytest.approx(0.5)


def test_transition_entropy(toy_ds):
    ent = stats.transition_entropy(toy_ds)
    assert ent[2] == pytest.approx(1.0)  # add_oil -> {tomato, mix} equally = 1 bit
    # action_start always -> add_oil in A, cut_tomato in B => also 1 bit here
    assert ent[0] == pytest.approx(1.0)


def test_naive_uniform_labeling():
    y = stats.naive_uniform_labeling([2, 3, 4], 90)
    assert y.tolist() == [2] * 30 + [3] * 30 + [4] * 30
    assert stats.naive_uniform_labeling([], 10).tolist() == [0] * 10


def test_dataset_summary_keys(toy_ds):
    s = stats.dataset_summary(toy_ds)
    for k in ("n_videos", "n_classes", "mean_segments_per_video", "mean_next_action_entropy_bits"):
        assert k in s
    assert s["n_videos"] == 2 and s["n_classes"] == 5
