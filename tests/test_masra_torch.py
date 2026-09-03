"""Tests for the differentiable MASRA regularizers. Skipped without torch
(local .venv has no torch; these run on the cluster)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from delta.align.masra_torch import (
    pool_spans,
    spans_from_assignment,
    esta_loss,
    self_similarity,
    transcript_relation_target,
    lrca_loss,
    MasraRegularizer,
)


def _clean(seed=0):
    g = torch.Generator().manual_seed(seed)
    D = 8
    class_emb = torch.randn(5, D, generator=g)
    transcript = torch.tensor([3, 1, 4])
    lens = [20, 20, 20]
    y = torch.cat([torch.full((L,), i) for i, L in enumerate(lens)])
    feat = torch.cat([class_emb[c].expand(L, D) for c, L in zip(transcript.tolist(), lens)])
    feat = feat + 1e-3 * torch.randn(60, D, generator=g)
    return feat, y, transcript, class_emb


def test_spans_from_assignment_contiguous():
    y = torch.tensor([0, 0, 0, 1, 1, 2])
    s = spans_from_assignment(y, 3)
    assert s.tolist() == [[0, 3], [3, 5], [5, 6]]


def test_spans_from_assignment_marks_missing_entry_empty():
    y = torch.tensor([0, 0, 2, 2])
    s = spans_from_assignment(y, 3)
    assert s[1].tolist() == [0, 0]          # entry 1 absent -> empty span


def test_pool_spans_matches_manual_mean():
    feat = torch.arange(12, dtype=torch.float).reshape(6, 2)
    out = pool_spans(feat, torch.tensor([[0, 3], [3, 6]]))
    assert torch.allclose(out[0], feat[0:3].mean(0))
    assert torch.allclose(out[1], feat[3:6].mean(0))


def test_esta_loss_low_for_aligned_blocks_high_for_shifted():
    feat, y, tr, ce = _clean()
    O = ce[tr]
    good = esta_loss(feat, spans_from_assignment(y, 3), O)
    bad = esta_loss(feat, spans_from_assignment(torch.roll(y, 10), 3), O)
    assert good.item() < 0.05
    assert bad.item() > good.item() + 0.1


def test_esta_loss_is_differentiable():
    feat, y, tr, ce = _clean()
    feat.requires_grad_(True)
    esta_loss(feat, spans_from_assignment(y, 3), ce[tr]).backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


def test_relation_target_block_is_0_1_and_symmetric():
    y = torch.tensor([0, 0, 1, 1, 2])
    R = transcript_relation_target(y, torch.tensor([7, 3, 9]), mode="block")
    assert set(R.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(R, R.t())
    assert R[0, 1] == 1.0 and R[0, 2] == 0.0


def test_relation_target_classsim_needs_class_emb():
    with pytest.raises(ValueError):
        transcript_relation_target(torch.zeros(4, dtype=torch.long),
                                   torch.zeros(2, dtype=torch.long), mode="class-sim")


def test_lrca_loss_smaller_for_correct_alignment_and_differentiable():
    feat, y, tr, ce = _clean()
    feat.requires_grad_(True)
    R_good = transcript_relation_target(y, tr, mode="block")
    R_bad = transcript_relation_target(torch.roll(y, 10), tr, mode="block")
    good = lrca_loss(feat, R_good)
    bad = lrca_loss(feat, R_bad)
    assert good.item() < bad.item()
    good.backward()
    assert torch.isfinite(feat.grad).all()


def test_lrca_frame_mask_ignores_padding():
    feat, y, tr, ce = _clean()
    R = transcript_relation_target(y, tr, mode="block")
    m = torch.ones(60, dtype=torch.bool)
    m[50:] = False
    full = lrca_loss(feat, R)
    masked = lrca_loss(feat, R, frame_mask=m)
    assert masked.item() != full.item()


def test_regularizer_bundles_both_terms():
    feat, y, tr, ce = _clean()
    feat.requires_grad_(True)
    reg = MasraRegularizer(lam_sem=1.0, lam_rel=0.5, relation_mode="class-sim")
    out = reg(temporal_ctx=feat, temporal_feat=feat, entry_of_frame=y,
              transcript=tr, event_emb=ce[tr], class_emb=ce)
    assert {"loss", "loss_esta", "loss_lrca"} <= set(out)
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert torch.isfinite(feat.grad).all()
