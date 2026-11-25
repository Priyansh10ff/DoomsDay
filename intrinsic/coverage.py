"""E1: episodic coverage bonus (ADR 0005, ADR 0010).

A fixed grid over the map's x/y plane. The first time the player is in a cell in
this episode pays 1 / budget_cells, for at most budget_cells cells; the cell set is
emptied at every reset. The raw bonus of an episode is therefore in [0, 1] by
construction, with no running statistics (a per-process running std would differ
between worker processes and restart on resume).

Position comes from the privileged `info` and is used for the reward only.

Guards:
  * heading is not part of the cell, so turning on the spot pays nothing;
  * the start cell is marked visited without pay, so standing still pays nothing;
  * a cell pays once per episode, so stepping back and forth across a cell edge
    pays for each of the two cells once and never again;
  * the per-episode budget bounds the total on any map, however large.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from configs.reward_config import CoverageConfig

F_X, F_Y = "pos_x", "pos_y"


class EpisodicCoverage:
    def __init__(self, cfg: CoverageConfig) -> None:
        self.cfg = cfg
        self._visited: set[tuple[int, int]] = set()
        self._paid_cells = 0
        self._ready = False

    def _cell(self, info: Mapping[str, Any]) -> tuple[int, int]:
        if F_X not in info or F_Y not in info:
            raise KeyError(f"coverage bonus is enabled but info has no {F_X!r}/{F_Y!r}; "
                           f"disable it (CoverageConfig.enabled=False) for envs without a pose")
        s = self.cfg.cell_size
        return math.floor(float(info[F_X]) / s), math.floor(float(info[F_Y]) / s)

    @property
    def cells_visited(self) -> int:
        """Distinct cells this episode, including the start cell."""
        return len(self._visited)

    def reset(self, info: Mapping[str, Any]) -> None:
        self._visited = {self._cell(info)}
        self._paid_cells = 0
        self._ready = True

    def step(self, info: Mapping[str, Any]) -> float:
        """Raw bonus for this decision: 1 / budget_cells on a first visit, else 0."""
        if not self._ready:
            raise RuntimeError("EpisodicCoverage.step called before reset")
        cell = self._cell(info)
        if cell in self._visited:
            return 0.0
        self._visited.add(cell)
        if self._paid_cells >= self.cfg.budget_cells:
            return 0.0
        self._paid_cells += 1
        return 1.0 / self.cfg.budget_cells
