"""Action sources for evaluation.

`NetworkPolicy`: the only path for a trained agent. Each episode gets a fresh
`PolicyRunner` (hidden state zero, sampling generator seeded from the episode seed),
so an episode's actions do not depend on earlier episodes. The observation is passed
as given by the env (batched to B=1); nothing else reaches the model.

`RandomPolicy`: the baseline only. Uniform over each head, seeded per episode.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from agent.inference import PolicyRunner
from agent.model import A0Agent


class NetworkPolicy:
    name = "network"

    def __init__(self, model: A0Agent, argmax_heads: tuple[bool, ...], device: str = "cpu") -> None:
        self.model, self.argmax_heads, self.device = model, argmax_heads, device
        self.runner: PolicyRunner | None = None

    def begin_episode(self, seed: int) -> None:
        self.runner = PolicyRunner(self.model, num_envs=1, argmax_heads=self.argmax_heads, seed=seed,
                                   device=self.device)

    def act(self, obs: Mapping[str, np.ndarray], episode_start: bool) -> np.ndarray:
        assert self.runner is not None, "begin_episode() must be called first"
        batched = {k: np.asarray(v)[None] for k, v in obs.items()}
        a = self.runner.act(batched, np.array([1.0 if episode_start else 0.0], dtype=np.float32))
        assert a.shape == (1, self.model.spec.n_heads), f"action expected (1, {self.model.spec.n_heads})"
        return a[0]


class RandomPolicy:
    name = "random"

    def __init__(self, action_space: spaces.MultiDiscrete) -> None:
        self.nvec = np.asarray(action_space.nvec, dtype=np.int64).reshape(-1)
        self.rng: np.random.Generator | None = None

    def begin_episode(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Mapping[str, np.ndarray], episode_start: bool) -> np.ndarray:
        assert self.rng is not None, "begin_episode() must be called first"
        return self.rng.integers(0, self.nvec).astype(np.int64)
