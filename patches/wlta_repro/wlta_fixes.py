"""Small, dependency-light helpers for reproducing the WLTA 50Salads baseline.

Copied next to `train_window_tokenizer.py` (see `apply.sh`). Only numpy/torch,
so each piece is unit-testable without Lightning or a GPU (tests/test_wlta_fixes.py).

  1. load_id2name_from_mapping / humanize_id2name -- class names read from the
     *same* mapping.txt the dataset loader uses, so id -> name is 1-to-1 with the
     action indices by construction (instead of a hard-coded Breakfast file).
  2. boundary_class_ids / core_class_ids / masked_moc -- MoC over the core
     action classes only, with the boundary tokens found by *name*.
  3. reset_optimizer_state -- in-place equivalent of re-instantiating the
     optimizer at the stage-1 -> stage-2 switch.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

BOUNDARY_NAMES = ("action_start", "action_end")


# ----------------------------------------------------------------------------
# 1. class names
# ----------------------------------------------------------------------------
def load_id2name_from_mapping(mapping_file: str) -> Dict[int, str]:
    """Parse `<id> <name>` lines. Raw names (underscores kept) so boundary tokens
    can still be found by name. Ids must be exactly 0..n-1."""
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"class mapping not found: {mapping_file}")
    id2name: Dict[int, str] = {}
    with open(mapping_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(" ", 1)
            if int(idx) in id2name:
                raise ValueError(f"duplicate class id {idx} in {mapping_file}")
            id2name[int(idx)] = name.strip()
    if sorted(id2name) != list(range(len(id2name))):
        raise ValueError(f"class ids in {mapping_file} are not contiguous 0..{len(id2name) - 1}")
    return id2name


def humanize_id2name(id2name: Dict[int, str]) -> Dict[int, str]:
    """`cut_tomato` -> `cut tomato`, so DistilBERT sees words rather than a
    snake_case token. Our choice, not something the paper specifies."""
    return {i: n.replace("_", " ") for i, n in id2name.items()}


# ----------------------------------------------------------------------------
# 2. core-class MoC
# ----------------------------------------------------------------------------
def boundary_class_ids(id2name: Dict[int, str], names: Sequence[str] = BOUNDARY_NAMES) -> Tuple[int, ...]:
    """Ids of the boundary tokens, found by name (never by position)."""
    return tuple(sorted(i for i, n in id2name.items() if n in names))


def core_class_ids(n_classes: int, boundary_ids: Iterable[int] = ()) -> np.ndarray:
    drop = set(int(i) for i in boundary_ids)
    return np.array([c for c in range(n_classes) if c not in drop], dtype=np.int64)


def masked_moc(T_actions, F_actions, core_ids=None) -> float:
    """Mean over classes of T/(T+F), restricted to `core_ids`; classes absent from
    the ground truth (T+F == 0) are skipped, exactly like metrics.calculate_moc.
    `core_ids=None` reproduces calculate_moc over every class."""
    T = np.asarray(T_actions, dtype=np.float64)
    F = np.asarray(F_actions, dtype=np.float64)
    ids = np.arange(len(T)) if core_ids is None else np.asarray(core_ids, dtype=np.int64)
    total = T[ids] + F[ids]
    seen = total > 0
    if not seen.any():
        return 0.0
    return float((T[ids][seen] / total[seen]).mean())


# ----------------------------------------------------------------------------
# 3. optimizer reset
# ----------------------------------------------------------------------------
def reset_optimizer_state(optimizer) -> None:
    """Make `optimizer` behave like a freshly constructed one, in place.

    Drops all per-parameter state (Adam's exp_avg / exp_avg_sq / step, so momentum
    and variance are zero and bias-correction restarts) and restores every
    hyper-parameter from `optimizer.defaults` (i.e. the base LR / weight decay the
    optimizer was built with). In place because Lightning holds references to the
    optimizer object; swapping in a new one would also mean re-wiring the
    strategy / precision plugin. Equivalence with a re-instantiated AdamW is
    checked in tests/test_wlta_fixes.py.

    Restores *all* groups to the constructor defaults -- fine for the single-group
    AdamW used here, would flatten per-group settings otherwise.
    """
    optimizer.state.clear()
    for group in optimizer.param_groups:
        for key, value in optimizer.defaults.items():
            group[key] = value
        group.pop("initial_lr", None)  # left over from any LR scheduler
