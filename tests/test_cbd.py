"""Tests for the CBD boundary-contrastive loss. Skipped without torch."""

import pytest

torch = pytest.importorskip("torch")

from delta.align.cbd import BoundaryHead, cbd_loss, jitter_views


def _seq(T=60, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, D, generator=g)
    # inject a clean transition at t=30
    x[30:] += 3.0
    return torch.nn.functional.normalize(x, dim=-1)


def test_boundary_head_normalises():
    h = BoundaryHead(16, proj=32)
    out = h(torch.randn(5, 16))
    assert out.shape == (5, 32)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-4)


def test_cbd_loss_runs_and_is_differentiable():
    x = _seq()
    x.requires_grad_(True)
    v1, v2 = jitter_views(x)
    loss = cbd_loss(v1, v2, boundaries=[30])
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_cbd_loss_trains_down():
    """Training the projection head on CBD makes the boundary frame more
    discriminable -> loss trends down over training."""
    torch.manual_seed(0)
    x = _seq()
    v1, v2 = jitter_views(x, drop=0.0, noise=0.01)      # fixed views
    head = BoundaryHead(16, proj=32)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    hist = []
    for _ in range(60):
        opt.zero_grad()
        l = cbd_loss(head(v1), head(v2), boundaries=[30], n_hard=0)
        l.backward(); opt.step()
        hist.append(l.item())
    assert sum(hist[-10:]) / 10 < sum(hist[:10]) / 10


def test_cbd_loss_empty_boundaries_is_zero():
    x = _seq()
    v1, v2 = jitter_views(x)
    assert cbd_loss(v1, v2, boundaries=[]).item() == 0.0


def test_jitter_views_preserve_length():
    x = _seq(T=40, D=8)
    a, b = jitter_views(x)
    assert a.shape == b.shape == (40, 8)
