"""Play one episode and turn the env's privileged `info` into a log record.

The policy receives the observation dict and the episode-start flag only. `info`
(map name, pose, counts, keys) is read here for the log and never passed on.

Success = level exit by two agreeing signals (ADR 0007-recipe Decision 5.1): the
env's `level_exit` (finished, not dead, not timed out) AND a final-step engine
reward of at least `exit_reward_fraction * map_exit_reward`. Disagreement is logged
as `exit_unknown` and is not a success.
Stale fields: on death / exit steps the env repeats the previous observation and
`info` (health, killcount); `final_obs_fresh` False marks this in the record.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    name: str

    def begin_episode(self, seed: int) -> None: ...

    def act(self, obs: Mapping[str, np.ndarray], episode_start: bool) -> np.ndarray: ...


def map_exit_reward(engine_rewards: tuple[tuple[str, float], ...]) -> float | None:
    return dict(engine_rewards).get("map_exit_reward")


def run_episode(env: Any, policy: Policy, seed: int, exit_reward_threshold: float | None) -> dict:
    obs, info = env.reset(seed=seed)
    policy.begin_episode(seed)
    ret, length, start = 0.0, 0, True
    while True:
        action = policy.act(obs, start)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        length += 1
        start = False
        if terminated or truncated:
            break
    level_exit = bool(info.get("level_exit", False))
    reward_ok = exit_reward_threshold is not None and float(reward) >= exit_reward_threshold
    keys = info.get("keys_held", {}) or {}
    return {
        "success": bool(level_exit and reward_ok),
        "exit_unknown": bool(level_exit != reward_ok and not info.get("dead", False)),
        "level_exit_flag": level_exit,
        "final_step_reward": float(reward),
        "return": ret,
        "length": length,
        "death": bool(info.get("dead", False)),
        "timeout": bool(truncated or info.get("timeout", False)),
        "kills": float(info.get("killcount", 0.0)),
        "keys_picked": sorted(k for k, v in keys.items() if v),
        "final_health": float(info.get("health", float("nan"))),
        "final_obs_fresh": bool(info.get("final_obs_fresh", True)),
        "episode_seed_engine": info.get("episode_seed"),
        "map_lump": info.get("map"),
    }
