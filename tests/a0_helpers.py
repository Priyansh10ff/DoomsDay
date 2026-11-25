"""Shared helpers for the agent/ and train/ tests. Importing this caps torch threads
(user's constraint: do not load the laptop hard). No ViZDoom is started here."""
from __future__ import annotations

import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)

# Doom-like sizes per ADR 0001 / 0006, built by hand so tests never start the engine.
DOOM_HEADS = (3, 3, 7, 2, 2, 2)


def doom_like_spaces(n_types: int = 14, with_screen: bool = False, ray_bands: int = 3,
                     max_entities: int = 24) -> tuple[spaces.Dict, spaces.MultiDiscrete]:
    d: dict[str, spaces.Space] = {
        "rays": spaces.Box(0.0, 1.0, (ray_bands, 80), np.float32),
        "entities": spaces.Box(-2.0, 2.0, (max_entities, 8), np.float32),
        "entity_type": spaces.Box(0, n_types - 1, (max_entities,), np.int64),
        "entity_mask": spaces.Box(0, 1, (max_entities,), np.bool_),
        "self": spaces.Box(-1.0, 2.0, (28,), np.float32),
        "prev_action": spaces.Box(0, max(DOOM_HEADS) - 1, (6,), np.int64),
        "modality": spaces.Box(0.0, 1.0, (3,), np.float32),
    }
    if with_screen:
        d["screen"] = spaces.Box(0, 255, (3, 60, 80), np.uint8)
    return spaces.Dict(d), spaces.MultiDiscrete(DOOM_HEADS)


def random_obs(space: spaces.Dict, batch: int, rng: np.random.Generator,
               lead: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    shape = (*lead, batch)
    s = space.spaces
    e = s["entities"].shape[0]
    n_types = int(np.max(s["entity_type"].high)) + 1
    heads = int(np.max(s["prev_action"].high)) + 1
    mask = rng.random((*shape, e)) < 0.5
    out = {
        "rays": rng.random((*shape, *s["rays"].shape), dtype=np.float32),
        "entities": np.where(mask[..., None], rng.uniform(-1, 1, (*shape, *s["entities"].shape)), 0).astype(np.float32),
        "entity_type": np.where(mask, rng.integers(1, n_types, (*shape, e)), 0).astype(np.int64),
        "entity_mask": mask,
        "self": rng.uniform(-1, 1, (*shape, *s["self"].shape)).astype(np.float32),
        "prev_action": rng.integers(0, 2, (*shape, *s["prev_action"].shape)).astype(np.int64) % heads,
        "modality": np.ones((*shape, *s["modality"].shape), np.float32),
    }
    if "screen" in s:
        out["screen"] = rng.integers(0, 256, (*shape, *s["screen"].shape)).astype(np.uint8)
    return out


class FakeDoomEnv(gym.Env):
    """Doom-shaped random env with a time limit, following env.DoomEnv's info contract:
    `final_obs_fresh` on every step; after death the observation is repeated
    (fresh False) and `dead` / `last_step_damage_taken` are set. `stale_timeout=True`
    simulates the fallback case of a repeated observation at a time-out;
    `omit_fresh_key=True` leaves the key out. Reward is 1 per step; `die_at` (if set)
    terminates instead of truncating."""

    def __init__(self, time_limit: int = 7, stale_timeout: bool = False, die_at: int | None = None,
                 n_types: int = 14, omit_fresh_key: bool = False, last_damage: float = 0.0) -> None:
        self.observation_space, self.action_space = doom_like_spaces(n_types=n_types)
        self.time_limit, self.stale_timeout, self.die_at = time_limit, stale_timeout, die_at
        self.omit_fresh_key, self.last_damage = omit_fresh_key, last_damage
        self._t = 0
        self._last: dict[str, np.ndarray] | None = None

    def _obs(self) -> dict[str, np.ndarray]:
        o = random_obs(self.observation_space, 1, self.np_random)
        o = {k: v[0] for k, v in o.items()}
        o["self"][0] = float(self._t)  # marker so tests can tell observations apart
        return o

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        self._last = self._obs()
        return self._last, {"keys_held": {"blue": False, "yellow": False, "red": False}}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._t += 1
        term = self.die_at is not None and self._t >= self.die_at
        trunc = (not term) and self._t >= self.time_limit
        repeated = term or (trunc and self.stale_timeout)
        obs = self._last if repeated else self._obs()
        self._last = obs
        info: dict[str, Any] = {"keys_held": {"blue": True, "yellow": False, "red": False}}
        if not self.omit_fresh_key:
            info["final_obs_fresh"] = not repeated
        if term or trunc:
            exit_ = bool(term and getattr(self, "exit_instead_of_death", False))
            info.update(dead=bool(term) and not exit_, timeout=bool(trunc), level_exit=exit_,
                        last_step_damage_taken=self.last_damage)
        return obs, 1.0, term, trunc, info
