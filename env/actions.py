"""Action encoding (ADR 0002 as amended by ADR 0006): six independent categorical
heads -> ViZDoom buttons, applied unchanged for every tic of the decision.

Uniform plumbing only: no masking, no observation- or history-dependent logic.
USE is plain button state (ADR 0006 ruling 2): head 1 = held down for all tics,
head 0 = up. A new press happens only on a 0 -> 1 change of the head.
"""
from __future__ import annotations

import numpy as np
import vizdoom as vzd

B = vzd.Button
BUTTONS = [
    B.MOVE_FORWARD, B.MOVE_BACKWARD, B.MOVE_LEFT, B.MOVE_RIGHT,
    B.TURN_LEFT_RIGHT_DELTA, B.ATTACK, B.USE, B.SPEED,
]
IDX_FWD, IDX_BACK, IDX_LEFT, IDX_RIGHT, IDX_TURN, IDX_ATTACK, IDX_USE, IDX_SPEED = range(8)

HEAD_NAMES = ("move", "strafe", "turn", "attack", "use", "speed")


def head_sizes(n_turn_bins: int) -> tuple[int, ...]:
    return (3, 3, n_turn_bins, 2, 2, 2)


def encode(action, turn_bins) -> list[float]:
    """Button vector held for all tics of one decision."""
    move, strafe, turn, attack, use, speed = (int(a) for a in action)
    v = [0.0] * len(BUTTONS)
    v[IDX_FWD] = 1.0 if move == 1 else 0.0
    v[IDX_BACK] = 1.0 if move == 2 else 0.0
    v[IDX_LEFT] = 1.0 if strafe == 1 else 0.0
    v[IDX_RIGHT] = 1.0 if strafe == 2 else 0.0
    v[IDX_TURN] = float(turn_bins[turn])
    v[IDX_ATTACK] = float(attack)
    v[IDX_USE] = float(use)
    v[IDX_SPEED] = float(speed)
    return v


def as_index_array(action) -> np.ndarray:
    return np.asarray(action, dtype=np.int64).reshape(len(HEAD_NAMES))
