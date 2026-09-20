import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "wlta_fixes", Path(__file__).resolve().parents[1] / "patches" / "wlta_repro" / "wlta_fixes.py"
)
wf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wf)

# the mapping shipped with our 50Salads bundle (mid-level, 19 classes)
MAPPING_50S = """0 cut_tomato
1 place_tomato_into_bowl
2 cut_cheese
3 place_cheese_into_bowl
4 cut_lettuce
5 place_lettuce_into_bowl
6 add_salt
7 add_vinegar
8 add_oil
9 add_pepper
10 mix_dressing
11 peel_cucumber
12 cut_cucumber
13 place_cucumber_into_bowl
14 add_dressing
15 mix_ingredients
16 serve_salad_onto_plate
17 action_start
18 action_end
"""
# MS-TCN style alphabetical ordering: boundary tokens sit at 0 and 1
MAPPING_ALPHA = """0 action_end
1 action_start
2 add_dressing
3 add_oil
"""


def _write(tmp_path, text):
    p = tmp_path / "mapping.txt"
    p.write_text(text)
    return str(p)


# ---- 1. class names --------------------------------------------------------
def test_mapping_is_one_to_one_with_indices(tmp_path):
    id2name = wf.load_id2name_from_mapping(_write(tmp_path, MAPPING_50S))
    assert len(id2name) == 19
    assert id2name[0] == "cut_tomato" and id2name[16] == "serve_salad_onto_plate"
    assert len(set(id2name.values())) == 19  # no two ids share a name


def test_mapping_rejects_gaps_and_duplicates(tmp_path):
    with pytest.raises(ValueError):
        wf.load_id2name_from_mapping(_write(tmp_path, "0 a\n2 b\n"))
    with pytest.raises(ValueError):
        wf.load_id2name_from_mapping(_write(tmp_path, "0 a\n0 b\n"))
    with pytest.raises(FileNotFoundError):
        wf.load_id2name_from_mapping(str(tmp_path / "nope.txt"))


def test_humanize_keeps_ids():
    out = wf.humanize_id2name({0: "cut_tomato", 1: "add_oil"})
    assert out == {0: "cut tomato", 1: "add oil"}


# ---- 2. masking ------------------------------------------------------------
def test_boundary_ids_found_by_name_not_position(tmp_path):
    ours = wf.load_id2name_from_mapping(_write(tmp_path, MAPPING_50S))
    alpha = wf.load_id2name_from_mapping(_write(tmp_path, MAPPING_ALPHA))
    assert wf.boundary_class_ids(ours) == (17, 18)
    assert wf.boundary_class_ids(alpha) == (0, 1)
    # in our bundle id 0 is a real action and must stay in the core set
    assert 0 in wf.core_class_ids(19, wf.boundary_class_ids(ours))


def test_core_ids_are_the_17_actions(tmp_path):
    ours = wf.load_id2name_from_mapping(_write(tmp_path, MAPPING_50S))
    core = wf.core_class_ids(19, wf.boundary_class_ids(ours))
    assert list(core) == list(range(17))


def test_masked_moc_ignores_boundary_tokens():
    T = np.zeros(19)
    F = np.zeros(19)
    T[0], F[0] = 8, 2          # 0.8
    T[3], F[3] = 1, 1          # 0.5
    F[18] = 50                 # action_end, always wrong
    core = wf.core_class_ids(19, (17, 18))
    assert wf.masked_moc(T, F, None) == pytest.approx((0.8 + 0.5 + 0.0) / 3)
    assert wf.masked_moc(T, F, core) == pytest.approx((0.8 + 0.5) / 2)


def test_masked_moc_equals_calculate_moc_when_boundaries_absent():
    """Boundary tokens never occur in the anticipated window -> masking is a no-op
    (the reason the mask cannot by itself explain a several-point MoC gap)."""
    rng = np.random.default_rng(0)
    T = rng.integers(0, 20, 19).astype(float)
    F = rng.integers(0, 20, 19).astype(float)
    T[17:] = 0
    F[17:] = 0

    def calculate_moc(T_actions, F_actions):  # verbatim from third_party/.../metrics.py
        acc, n = 0, 0
        for c in range(len(T_actions)):
            total = T_actions[c] + F_actions[c]
            if total > 0:
                acc += T_actions[c] / total
                n += 1
        return acc / n if n > 0 else 0

    assert wf.masked_moc(T, F, wf.core_class_ids(19, (17, 18))) == pytest.approx(calculate_moc(T, F))


def test_masked_moc_empty_is_zero():
    assert wf.masked_moc(np.zeros(19), np.zeros(19), np.arange(17)) == 0.0


# ---- 3. optimizer reset ----------------------------------------------------
def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def _step(model, opt, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 4, generator=g)
    opt.zero_grad()
    model(x).pow(2).mean().backward()
    opt.step()


def test_reset_matches_reinstantiated_adamw():
    base_lr, wd = 5e-3, 3e-4
    a = _model()
    opt_a = torch.optim.AdamW(a.parameters(), lr=base_lr, weight_decay=wd)
    for s in range(5):
        _step(a, opt_a, s)
    assert len(opt_a.state) > 0

    b = copy.deepcopy(a)  # same weights at the switch point
    opt_b = torch.optim.AdamW(b.parameters(), lr=base_lr, weight_decay=wd)  # re-instantiated

    wf.reset_optimizer_state(opt_a)
    assert len(opt_a.state) == 0
    for s in range(10, 14):
        _step(a, opt_a, s)
        _step(b, opt_b, s)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=0, rtol=0)


def test_without_reset_trajectories_differ():
    """Guards the test above: stale Adam moments really change the update."""
    a = _model()
    opt_a = torch.optim.AdamW(a.parameters(), lr=5e-3, weight_decay=3e-4)
    for s in range(5):
        _step(a, opt_a, s)
    b = copy.deepcopy(a)
    opt_b = torch.optim.AdamW(b.parameters(), lr=5e-3, weight_decay=3e-4)
    for s in range(10, 12):
        _step(a, opt_a, s)
        _step(b, opt_b, s)
    assert any(not torch.allclose(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def test_reset_restores_base_hyperparameters():
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3, weight_decay=3e-4)
    _step(m, opt, 0)
    opt.param_groups[0]["lr"] = 1e-9  # e.g. left decayed by a scheduler
    opt.param_groups[0]["initial_lr"] = 1e-9
    wf.reset_optimizer_state(opt)
    g = opt.param_groups[0]
    assert g["lr"] == 5e-3 and g["weight_decay"] == 3e-4 and "initial_lr" not in g
