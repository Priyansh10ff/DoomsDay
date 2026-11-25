"""Helpers for the reward / intrinsic tests (ADR 0010). No torch, no ViZDoom:
a scripted env replays synthetic `info` dictionaries shaped like env.DoomEnv's."""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.reward_config import FULL_LEVEL_COVERAGE, FULL_LEVEL_REWARD  # noqa: E402
from rewards.wrapper import INFO_EPISODE, INFO_STEP, ShapedRewardEnv  # noqa: E402

NO_KEYS = {"blue": False, "yellow": False, "red": False}
# Full-level terms with the two-signal exit rule off, for tests that do not study it.
FULL = dataclasses.replace(FULL_LEVEL_REWARD, exit_confirm_engine_reward=None)
COV = FULL_LEVEL_COVERAGE
CELL = COV.cell_size


def mk(x: float = 0.0, y: float = 0.0, angle: float = 0.0, kills: float = 0.0, keys: dict | None = None,
       health: float = 100.0, armor: float = 0.0, ammo: tuple = (50.0, 0.0), damage: float = 0.0,
       **extra: Any) -> dict[str, Any]:
    """One per-step `info` with every field the full reward reads."""
    d = {"pos_x": x, "pos_y": y, "pos_z": 0.0, "angle": angle, "killcount": kills,
         "keys_held": dict(keys or NO_KEYS), "health": health, "armor": armor, "ammo": ammo,
         "damage_taken_total": damage, "final_obs_fresh": True}
    d.update(extra)
    return d


def end(info: dict, how: str, engine_reward: float | None = None, last_damage: float = 0.0,
        stale: bool | None = None) -> dict[str, Any]:
    """Turn `info` into an ending-step info: how in {"exit", "dead", "timeout"}."""
    d = dict(info, dead=how == "dead", level_exit=how == "exit", timeout=how == "timeout",
             last_step_damage_taken=last_damage)
    d["final_obs_fresh"] = (how == "timeout") if stale is None else not stale
    if engine_reward is not None:
        d["engine_reward"] = engine_reward
    return d


class ScriptEnv(gym.Env):
    """Replays `infos[0]` at reset and `infos[1:]` as steps. An info with any of
    dead / level_exit / timeout set ends the episode; `engine` gives the engine
    reward per step (default 0). After the script ends it restarts on reset."""

    def __init__(self, infos: list[dict], engine: list[float] | None = None) -> None:
        self.infos, self.engine = infos, engine or [0.0] * (len(infos) - 1)
        self.observation_space = spaces.Dict({"self": spaces.Box(-1, 1, (3,), np.float32)})
        self.action_space = spaces.MultiDiscrete([2, 2])
        self._t = 0
        self.actions: list[Any] = []

    def _obs(self) -> dict[str, np.ndarray]:
        return {"self": np.full(3, self._t, np.float32)}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), dict(self.infos[0])

    def step(self, action: Any):
        self._t += 1
        self.actions.append(action)
        info = dict(self.infos[self._t])
        terminated = bool(info.get("dead") or info.get("level_exit"))
        truncated = bool(info.get("timeout")) and not terminated
        return self._obs(), float(self.engine[self._t - 1]), terminated, truncated, info


def play(infos: list[dict], reward=FULL, coverage=COV, engine: list[float] | None = None,
         start_decisions: int = 0, actions: list | None = None) -> tuple[list[float], list[dict]]:
    """Run the whole script through ShapedRewardEnv. Returns per-step rewards and infos."""
    env = ShapedRewardEnv(ScriptEnv(infos, engine), reward, coverage, start_decisions)
    env.reset(seed=0)
    rewards, out = [], []
    for i in range(len(infos) - 1):
        a = actions[i] if actions is not None else np.zeros(2, np.int64)
        _, r, te, tr, info = env.step(a)
        rewards.append(r)
        out.append(info)
        if te or tr:
            break
    return rewards, out


def paid(out: list[dict], term: str) -> float:
    return float(sum(i[INFO_STEP]["paid"][term] for i in out))


def episode(out: list[dict]) -> dict:
    return out[-1][INFO_EPISODE]
