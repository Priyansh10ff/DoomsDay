"""Engine-free helpers for the curriculum-training tests (importable by spawned workers)."""
from __future__ import annotations

from typing import Any

import numpy as np

from a0_helpers import FakeDoomEnv

TINY_CURRICULUM = {
    "stages": ["S0", "S1"],
    "promotion": {"threshold": {"S0": 0.5}, "window": 2, "count_replay_episodes": False},
    "replay": {"fraction": 0.0, "stage_weights": "uniform"},
    "seed": 7,
}
TINY_POOL = {"S0": [("DD0000", "S0-a"), ("DD0001", "S0-b")], "S1": [("DD0002", "S1-a"), ("DD0003", "S1-b")]}


class FakeMapEnv(FakeDoomEnv):
    """FakeDoomEnv that records the map it was reset to (info["map"], like env.DoomEnv)
    and ends every episode at `time_limit`: by level exit on maps in `exit_maps`, by
    time-out elsewhere. Observations do not depend on the map."""

    def __init__(self, time_limit: int = 3, exit_maps: tuple[str, ...] = ("DD0000", "DD0001")) -> None:
        super().__init__(time_limit=time_limit)
        self.exit_maps = exit_maps
        self.map: str | None = None
        self.resets: list[Any] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.resets.append(options)
        self.map = (options or {}).get("map", "MAP01")
        info["map"] = self.map
        return obs, info

    def step(self, action: np.ndarray):
        obs, r, term, trunc, info = super().step(action)
        info["map"] = self.map
        if trunc and self.map in self.exit_maps:
            term, trunc = True, False
            info.update(level_exit=True, timeout=False, dead=False)
        return obs, r, term, trunc, info


def fake_map_env() -> FakeMapEnv:
    return FakeMapEnv()
