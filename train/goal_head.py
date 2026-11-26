"""Sub-goal prediction head: dataset label, loss, and adding a fresh head to weights
that lack one (2026-11-26, user request; no ADR covers it, built on the task message).

The head (agent/model.py `goal_head`, on when ModelConfig.goal_head) maps the GRU
state to configs.goal_modes.N_GOAL_MODES logits. It is an AUXILIARY prediction: its
output never feeds the policy heads, so every action still comes only from them.

Dataset label: `goal_mode` (T,) integer per episode npz, ids from configs/goal_modes.py.
  * a NEGATIVE value (GOAL_UNLABELLED = -1) marks a step without a label: masked out
    of the loss and the accuracy;
  * an episode file without the array counts as unlabelled at every step;
  * a value >= N_GOAL_MODES is refused at load.
A coefficient > 0 refuses a data dir in which NO episode carries the array.

Head addition (used by `--init-from-bc` in train/bc.py and `--init-from` in
train/init_from.py): allowed ONLY when the source model config equals the new one
with `goal_head` False instead of True. The source state must hold none of the
head's keys; exactly those keys are filled from the freshly built (seeded) new
model, then the caller loads strictly. Every other weight is copied unchanged.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agent.model import GOAL_HEAD_NAME
from configs.goal_modes import N_GOAL_MODES
from configs.model_config import ModelConfig

from .bc_spec import DatasetMismatch

GOAL_LABEL_KEY = "goal_mode"
GOAL_UNLABELLED = -1
GOAL_KEY_PREFIX = GOAL_HEAD_NAME + "."


class GoalLabelError(DatasetMismatch):
    """A goal_mode array is malformed, or required labels are missing."""


# ---------------------------------------------------------------- dataset label
def validate_goal(arr: np.ndarray, length: int, where: str) -> np.ndarray:
    """(T,) integer array -> int64 copy; refuses wrong shape, non-integer dtype or ids
    >= N_GOAL_MODES. Negative values are kept (they mean "no label")."""
    if arr.shape != (length,):
        raise GoalLabelError(f"{where}: {GOAL_LABEL_KEY} shape {arr.shape}, expected ({length},)")
    if not np.issubdtype(arr.dtype, np.integer):
        raise GoalLabelError(f"{where}: {GOAL_LABEL_KEY} dtype {arr.dtype}, expected an integer type")
    out = arr.astype(np.int64)
    if out.size and int(out.max()) >= N_GOAL_MODES:
        raise GoalLabelError(f"{where}: {GOAL_LABEL_KEY} id {int(out.max())} >= N_GOAL_MODES={N_GOAL_MODES}")
    return out


def require_goal_labels(episodes: list[Any], where: str) -> None:
    """Refuse (GoalLabelError) when no episode in `episodes` carries a goal_mode array."""
    if not any(getattr(e, "goal", None) is not None for e in episodes):
        raise GoalLabelError(f"{where}: no episode carries a {GOAL_LABEL_KEY!r} array, but the goal-head "
                             f"coefficient is > 0 (record the label, or use --goal-head-coef 0)")


def labelled_fraction(episodes: list[Any]) -> float:
    """Fraction of all steps with a goal label (>= 0)."""
    total = sum(int(e.length) for e in episodes)
    lab = sum(int((e.goal >= 0).sum()) for e in episodes if getattr(e, "goal", None) is not None)
    return lab / max(total, 1)


# ---------------------------------------------------------------- loss
@dataclass
class GoalLoss:
    loss: torch.Tensor     # scalar mean CE over labelled valid steps (0 when none)
    loss_sum: float        # CE summed over those steps
    correct: int
    n_labelled: int


def goal_loss(goal_logits: torch.Tensor, goal: torch.Tensor, valid: torch.Tensor) -> GoalLoss:
    """goal_logits (T, B, N), goal (T, B) long (negative = unlabelled), valid (T, B)
    bool. Only valid AND labelled steps count, in the value, gradient and metrics."""
    t, b = valid.shape
    assert goal_logits.shape == (t, b, N_GOAL_MODES), \
        f"goal_logits expected (T={t}, B={b}, {N_GOAL_MODES}), got {tuple(goal_logits.shape)}"
    assert goal.shape == (t, b), f"goal labels expected (T={t}, B={b}), got {tuple(goal.shape)}"
    use = valid & (goal >= 0)
    n = int(use.sum())
    lv, yv = goal_logits[use], goal[use]
    ce = F.cross_entropy(lv, yv, reduction="sum") if n else goal_logits.sum() * 0.0
    return GoalLoss(ce / max(n, 1), float(ce.detach()), int((lv.detach().argmax(-1) == yv).sum()) if n else 0, n)


# ---------------------------------------------------------------- head addition
def goal_head_added(src_cfg: ModelConfig, dst_cfg: ModelConfig) -> bool:
    """True when `dst_cfg` is `src_cfg` with the goal head switched on (the only
    config difference a head addition allows); False when the configs are equal;
    ValueError for any other difference (removing the head included)."""
    if src_cfg == dst_cfg:
        return False
    if (not src_cfg.goal_head) and dst_cfg.goal_head and dataclasses.replace(src_cfg, goal_head=True) == dst_cfg:
        return True
    raise ValueError("model config differs (the only difference accepted is adding the goal head: "
                     f"source goal_head={src_cfg.goal_head}, new goal_head={dst_cfg.goal_head})")


def add_goal_head_state(state: dict[str, torch.Tensor], fresh: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy of `state` plus the goal head's keys taken from `fresh` (the newly built,
    seeded model with the head). Refuses a source state that already holds any."""
    have = [k for k in state if k.startswith(GOAL_KEY_PREFIX)]
    if have:
        raise KeyError(f"goal-head addition: the source state already has {have}")
    fresh_sd = fresh.state_dict()
    keys = [k for k in fresh_sd if k.startswith(GOAL_KEY_PREFIX)]
    if not keys:
        raise KeyError("goal-head addition: the new model has no goal head")
    out = dict(state)
    for k in keys:
        out[k] = fresh_sd[k].detach().clone().cpu()
    return out


def addition_record() -> dict[str, Any]:
    return {"kind": "goal_head_added", "key_prefix": GOAL_KEY_PREFIX, "n_goal_modes": N_GOAL_MODES,
            "init": "fresh (orthogonal, bias 0) from the new model's seeded build"}
