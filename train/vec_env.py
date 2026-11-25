"""Vector-env interface the trainer is written against, plus an in-process
reference implementation. No torch import (ADR 0004: workers never import torch).

Auto-reset semantics: when env i ends at a step, the returned `obs[i]` and
`infos[i]` are the FIRST observation/info of its next episode; the observation
and info returned by the ending step go to `final_obs[i]` / `final_infos[i]`.
`final_obs_fresh[i]` is read per step and per env from that ending step's
`info["final_obs_fresh"]` (ADR 0007 ruling 5); a missing key counts as stale.
For `env.DoomEnv` it is True at a time-out (the wrapper truncates while the state
is live) and False after death or level exit (the observation is repeated).

Per-env reset options (curriculum map choice, train/curriculum_vec_env.py): `reset`
takes an optional list of one options dict per env, and `step` an optional list of
the options each env uses IF its episode ends at this step (the auto-reset). With
`None` (the default) envs are reset exactly as before, with no options argument.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

ObsBatch = dict[str, np.ndarray]
ResetOptions = dict[str, Any] | None   # one env's `reset(options=...)`; None = no options
FRESH_KEY = "final_obs_fresh"


@dataclass
class VecStep:
    obs: ObsBatch                       # (N, ...) per key; reset obs where an episode ended
    reward: np.ndarray                  # (N,) float32
    terminated: np.ndarray              # (N,) bool: death / exit (no bootstrap)
    truncated: np.ndarray               # (N,) bool: time limit (bootstrap)
    infos: list[dict[str, Any]]         # aligned with obs
    final_obs: list[dict[str, np.ndarray] | None]
    final_infos: list[dict[str, Any] | None]
    final_obs_fresh: np.ndarray         # (N,) bool, False where not ended


class VecEnv(Protocol):
    num_envs: int
    single_observation_space: gym.spaces.Dict
    single_action_space: gym.spaces.MultiDiscrete

    def reset(self, seed: int, options: Sequence[ResetOptions] | None = None) -> tuple[ObsBatch, list[dict[str, Any]]]: ...

    def step(self, actions: np.ndarray, reset_options: Sequence[ResetOptions] | None = None) -> VecStep: ...

    def close(self) -> None: ...


def stack_obs(obs_list: Sequence[dict[str, np.ndarray]]) -> ObsBatch:
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}


def env_seeds(seed: int, n: int) -> list[int]:
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


@dataclass
class OneStep:
    obs: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    final_obs: dict[str, np.ndarray] | None
    final_info: dict[str, Any] | None
    final_fresh: bool


def reset_one(env: gym.Env, seed: int | None = None, options: ResetOptions = None) -> tuple[Any, dict[str, Any]]:
    """`env.reset`, passing `options` only when given (so envs see exactly today's call
    when no curriculum is in use)."""
    kw: dict[str, Any] = {} if seed is None else {"seed": seed}
    if options is not None:
        kw["options"] = options
    return env.reset(**kw)


def per_env(items: Sequence[ResetOptions] | None, n: int, what: str) -> list[ResetOptions]:
    if items is None:
        return [None] * n
    assert len(items) == n, f"{what}: expected one entry per env ({n}), got {len(items)}"
    return list(items)


def step_autoreset(env: gym.Env, action: np.ndarray, reset_options: ResetOptions = None) -> OneStep:
    """Step one env; on episode end keep the final obs/info and reset (with
    `reset_options` when given)."""
    o, r, te, tr, info = env.step(action)
    if te or tr:
        fresh = bool(info.get(FRESH_KEY, False))
        o2, info2 = reset_one(env, None, reset_options)
        return OneStep(o2, float(r), bool(te), bool(tr), info2, o, info, fresh)
    return OneStep(o, float(r), False, False, info, None, None, False)


def collate(steps: Sequence[OneStep]) -> VecStep:
    return VecStep(
        obs=stack_obs([s.obs for s in steps]),
        reward=np.array([s.reward for s in steps], np.float32),
        terminated=np.array([s.terminated for s in steps], bool),
        truncated=np.array([s.truncated for s in steps], bool),
        infos=[s.info for s in steps],
        final_obs=[s.final_obs for s in steps],
        final_infos=[s.final_info for s in steps],
        final_obs_fresh=np.array([s.final_fresh for s in steps], bool),
    )


class SyncVecEnv:
    """Steps N gymnasium envs in turn, in this process."""

    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]], allow_multi_instance: int | None = None) -> None:
        from configs.train_config import SafetyConfig

        from .layout_guard import check_layout, count_engine_instances
        # ADR 0008 Decision 5: several engines in one process count as several instances.
        check_layout(count_engine_instances(env_fns), allow_multi_instance, 1, 0, SafetyConfig())
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed: int, options: Sequence[ResetOptions] | None = None) -> tuple[ObsBatch, list[dict[str, Any]]]:
        opts = per_env(options, self.num_envs, "reset options")
        results = [reset_one(e, s, o) for e, s, o in zip(self.envs, env_seeds(seed, self.num_envs), opts)]
        return stack_obs([r[0] for r in results]), [r[1] for r in results]

    def step(self, actions: np.ndarray, reset_options: Sequence[ResetOptions] | None = None) -> VecStep:
        assert actions.shape[0] == self.num_envs, f"actions expected ({self.num_envs}, heads), got {actions.shape}"
        opts = per_env(reset_options, self.num_envs, "reset_options")
        return collate([step_autoreset(e, actions[i], opts[i]) for i, e in enumerate(self.envs)])

    def close(self) -> None:
        for e in self.envs:
            e.close()
