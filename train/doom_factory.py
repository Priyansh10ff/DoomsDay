"""Picklable env factory for worker processes. Imports env (vizdoom) only when called,
and never torch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass(frozen=True)
class DoomEnvFactory:
    config: Any  # env.config.EnvConfig (frozen dataclass, picklable)
    # Counted by train.layout_guard: each call starts one ViZDoom engine instance.
    is_engine_instance: bool = True

    def __post_init__(self) -> None:
        # `phase2_engine_test` is env-engineer's test-only opt-in for Freedoom Phase 2
        # (held-out) maps; the trainer must never accept it (ADR 0005, 0008).
        if bool(getattr(self.config, "phase2_engine_test", False)):
            raise ValueError("refusing an env config with phase2_engine_test set: Freedoom Phase 2 "
                             "maps are held out and never used for training")

    def __call__(self) -> gym.Env:
        from env import DoomEnv
        return DoomEnv(self.config)
