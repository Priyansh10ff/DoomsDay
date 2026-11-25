"""Door-progress term: paid once per door per episode, the first time the player gets
through that door. TRAINING ONLY (rule 4.2: privileged pose + map geometry for reward).

Inputs are `info` alone (pos_x, pos_y, map); no action, no observation, and nothing
computed here is written into an observation.

Definition ("door passed"), per door slab [a0, a1] along its travel axis and
[lo, hi] across it (maps/doors.py):
  * the player is NEAR the door when lo <= lateral <= hi and
    a0 - margin <= along <= a1 + margin;
  * while near, the side it was last seen on is remembered: -1 (along < a0) or
    +1 (along > a1); a sample inside the slab keeps the remembered side;
  * the door is PASSED when, without leaving the near region, the player is seen
    on the side opposite the remembered one. Leaving the near region forgets the side.
  * each door pays at most once per episode, in either direction.
Why it cannot be farmed or earned through a closed door: a closed door has no gap,
so the far side can only be reached by leaving the near region (around a loop),
which forgets the side; stepping back and forth pays nothing after the first pass.
`margin` exceeds one decision's travel, so a step that jumps the 32-unit slab is
still seen on both sides.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from maps.doors import DoorSpec, DoorTable

from .terms import MissingInfoField

F_POS_X, F_POS_Y, F_MAP = "pos_x", "pos_y", "map"
NO_SIDE, NEAR_SIDE, FAR_SIDE = 0, -1, 1


def side_of(d: DoorSpec, x: float, y: float, margin: float) -> int | None:
    """None outside the near region; -1 / +1 before / behind the slab; 0 inside it."""
    along, lat = d.local(x, y)
    if not (d.lo <= lat <= d.hi and d.a0 - margin <= along <= d.a1 + margin):
        return None
    if along < d.a0:
        return NEAR_SIDE
    if along > d.a1:
        return FAR_SIDE
    return NO_SIDE


class DoorProgress:
    def __init__(self, table: DoorTable, weight: float, margin: float) -> None:
        if not margin > 0.0:
            raise ValueError(f"door_near_margin must be positive, got {margin}")
        self.table = dict(table)
        self.weight, self.margin = float(weight), float(margin)
        self.doors: tuple[DoorSpec, ...] = ()
        self._side: list[int] = []
        self._passed: list[bool] = []
        self._ready = False

    @staticmethod
    def _need(info: Mapping[str, Any], key: str) -> Any:
        if key not in info:
            raise MissingInfoField(f"door term is on but info has no {key!r}")
        return info[key]

    @property
    def n_doors(self) -> int:
        return len(self.doors)

    @property
    def passed(self) -> int:
        return sum(self._passed)

    @property
    def passed_flags(self) -> tuple[bool, ...]:
        """Per door (same order as `doors`): passed yet this episode. Diagnostics only."""
        return tuple(self._passed)

    @property
    def episode_max(self) -> float:
        """Largest door pay of this episode: weight x doors in the map."""
        return self.weight * self.n_doors

    def reset(self, info: Mapping[str, Any]) -> None:
        lump = self._need(info, F_MAP)
        if lump not in self.table:
            raise KeyError(f"map {lump!r} has no door geometry; the door table covers "
                           f"{len(self.table)} maps of the training bundle")
        self.doors = self.table[lump]
        self._side = [NO_SIDE] * len(self.doors)
        self._passed = [False] * len(self.doors)
        self._ready = True
        self._observe(info)

    def _observe(self, info: Mapping[str, Any]) -> int:
        x, y = float(self._need(info, F_POS_X)), float(self._need(info, F_POS_Y))
        new = 0
        for k, d in enumerate(self.doors):
            s = side_of(d, x, y, self.margin)
            if s is None:
                self._side[k] = NO_SIDE
            elif s != NO_SIDE:
                if self._side[k] not in (NO_SIDE, s) and not self._passed[k]:
                    self._passed[k] = True
                    new += 1
                self._side[k] = s
        return new

    def step(self, info: Mapping[str, Any]) -> float:
        """Raw value of this step: the number of doors passed for the first time."""
        if not self._ready:
            raise RuntimeError("DoorProgress.step called before reset")
        new = self._observe(info)
        assert self.passed <= self.n_doors, "door paid more often than the map has doors"
        return float(new)
