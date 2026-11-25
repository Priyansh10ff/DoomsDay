"""Picklable env factory that adds reward shaping around another factory (ADR 0010).

Wraps (never replaces) the inner factory, e.g. train.doom_factory.DoomEnvFactory, so
that factory's own refusals still run, and forwards `is_engine_instance` so
train.layout_guard still counts every engine. The reward setup is validated when
the factory is constructed, in the main process, before any engine starts.

No torch, no vizdoom, no `env` import here; the inner factory imports what it needs
when called inside the worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from configs.reward_config import CoverageConfig, RewardConfig

from .validate import check_reward_setup


def per_env_decisions(decisions_done: int, num_envs: int) -> int:
    """The anneal offset for a resumed run: trainer decisions / env count."""
    return int(decisions_done) // max(int(num_envs), 1)


@dataclass(frozen=True)
class ShapedEnvFactory:
    inner: Any                       # picklable zero-argument env factory
    reward: RewardConfig
    coverage: CoverageConfig
    start_decisions: int = 0         # per-env decisions already done (resume)
    # Only for inner factories without an env config (toy envs); otherwise derived.
    max_episode_decisions: int | None = None
    # maps.doors.DoorTable ((lump, doors), ...): required iff reward.door != 0. Built
    # once in the main process from the bundle the engine plays (train/run_doom.py).
    doors: Any = None
    # train.episode_diag.diag_table of the same bundle: adds the per-episode USE / door
    # diagnostics wrapper (logging only; reads the executed action, never changes it).
    diag: Any = None

    def __post_init__(self) -> None:
        cfg = getattr(self.inner, "config", None)
        engine_rewards = tuple(getattr(cfg, "engine_rewards", ()) or ())
        limit = self.max_episode_decisions
        tics, skip = getattr(cfg, "episode_timeout_tics", None), getattr(cfg, "frame_skip", None)
        if limit is None and tics and skip:
            limit = -(-int(tics) // int(skip))  # ceil
        if bool(getattr(cfg, "vanilla_eval", False)):
            raise ValueError("refusing to shape rewards on a vanilla-eval config: evaluation "
                             "runs without any reward code (ADR 0010)")
        check_reward_setup(self.reward, self.coverage, engine_rewards, limit)
        if self.reward.door != 0.0:
            if not self.doors:
                raise ValueError(f"{self.reward.name!r} pays door progress: pass a door table "
                                 f"(maps.doors.door_table of the training bundle)")
            maps = getattr(cfg, "maps", None)
            missing = sorted(set(maps or ()) - {lump for lump, _ in self.doors})
            if missing:
                raise ValueError(f"door table has no geometry for maps {missing[:5]}")

    @property
    def is_engine_instance(self) -> bool:
        return bool(getattr(self.inner, "is_engine_instance", False))

    def __call__(self) -> gym.Env:
        from .wrapper import ShapedRewardEnv
        env = ShapedRewardEnv(self.inner(), self.reward, self.coverage, self.start_decisions,
                              doors=self.doors if self.reward.door != 0.0 else None)
        if self.diag is not None:
            from train.episode_diag import EpisodeDiagEnv
            env = EpisodeDiagEnv(env, self.diag, self.reward.door_near_margin)
        return env
