"""Toy memory task (ADR 0004 validation 2(i); gate and measurement per ADR 0008 Decision 2).

Pure numpy, no ViZDoom. A cue (one of two) is shown in `self` at step 0 only; at
step `answer_step` (the cue-to-answer gap) an "answer now" flag is shown and the
first action head must name the cue (reward `reward_scale`, else 0); the episode
then terminates. Rays and entities are seeded noise. A memoryless policy can do no
better than 0.5. With `dense_cue_label` (arm B2) the env reports the cue identity in
info["keys_held"] at every step, so the existing keys auxiliary head (read from the
GRU state) becomes a dense label for the remembered fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs.model_config import ModelConfig
from configs.train_config import AuxConfig, PPOConfig, RunConfig

TOY_TORCH_THREADS = 4
CUE_KEY_NAMES = ("blue", "yellow")  # cue 0 / cue 1 in the keys-held label (arm B2)


@dataclass(frozen=True)
class ToyConfig:
    answer_step: int = 20          # gap; episode length answer_step + 1
    n_cues: int = 2
    ray_shape: tuple[int, int] = (3, 16)
    max_entities: int = 4
    entity_feat_dim: int = 8
    n_entity_types: int = 14
    self_dim: int = 8
    head_sizes: tuple[int, ...] = (2, 3)
    reward_scale: float = 1.0      # floor measurement multiplies rewards by 0.05
    dense_cue_label: bool = False  # arm B2


class ToyMemoryEnv(gym.Env):
    def __init__(self, cfg: ToyConfig = ToyConfig()) -> None:
        self.cfg = cfg
        e = cfg.max_entities
        self.observation_space = spaces.Dict({
            "rays": spaces.Box(0.0, 1.0, cfg.ray_shape, np.float32),
            "entities": spaces.Box(-2.0, 2.0, (e, cfg.entity_feat_dim), np.float32),
            "entity_type": spaces.Box(0, cfg.n_entity_types - 1, (e,), np.int64),
            "entity_mask": spaces.Box(0, 1, (e,), np.bool_),
            "self": spaces.Box(-1.0, 2.0, (cfg.self_dim,), np.float32),
            "prev_action": spaces.Box(0, max(cfg.head_sizes) - 1, (len(cfg.head_sizes),), np.int64),
            "modality": spaces.Box(0.0, 1.0, (3,), np.float32),
        })
        self.action_space = spaces.MultiDiscrete(cfg.head_sizes)
        self._t = 0
        self._cue = 0
        self._prev = np.zeros(len(cfg.head_sizes), np.int64)

    def _obs(self) -> dict[str, np.ndarray]:
        c, r = self.cfg, self.np_random
        s = np.zeros(c.self_dim, np.float32)
        if self._t == 0:
            s[self._cue] = 1.0
        if self._t == c.answer_step:
            s[c.n_cues] = 1.0
        n_vis = int(r.integers(0, c.max_entities + 1))
        mask = np.zeros(c.max_entities, bool)
        mask[:n_vis] = True
        return {
            "rays": r.random(c.ray_shape, dtype=np.float32),
            "entities": np.where(mask[:, None], r.uniform(-1, 1, (c.max_entities, c.entity_feat_dim)), 0).astype(np.float32),
            "entity_type": np.where(mask, r.integers(1, c.n_entity_types, c.max_entities), 0).astype(np.int64),
            "entity_mask": mask,
            "self": s,
            "prev_action": self._prev.copy(),
            "modality": np.ones(3, np.float32),
        }

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"final_obs_fresh": True}
        if self.cfg.dense_cue_label:
            info["keys_held"] = {k: self._cue == i for i, k in enumerate(CUE_KEY_NAMES)}
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        self._cue = int(self.np_random.integers(self.cfg.n_cues))
        if options and "cue" in options:  # for the init trace diagnostic only
            self._cue = int(options["cue"])
        self._prev = np.zeros(len(self.cfg.head_sizes), np.int64)
        return self._obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, np.int64).reshape(-1)
        reward, terminated = 0.0, False
        if self._t == self.cfg.answer_step:
            reward = self.cfg.reward_scale if int(a[0]) == self._cue else 0.0
            terminated = True
        self._prev = a.copy()
        self._t += 1
        return self._obs(), reward, terminated, False, self._info()


@dataclass(frozen=True)
class ToyRunSpec:
    num_envs: int = 16
    updates: int = 100
    eval_last_episodes: int = 200


@dataclass(frozen=True)
class ToyArm:
    """One single-change arm of ADR 0008 Decision 2. Overrides are field -> value."""
    name: str
    toy: ToyConfig = ToyConfig()
    model: dict[str, Any] = field(default_factory=dict)
    ppo: dict[str, Any] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)


def toy_model_config(recurrent: bool, **overrides: Any) -> ModelConfig:
    """Smaller widths than A0 so the toy runs in about a minute on 4 CPU threads."""
    base = ModelConfig(core_hidden=64, fusion_dim=64, ray_dim=32, entity_token_dim=32,
                       entity_mlp_hidden=32, self_dim=32, self_hidden=32, recurrent=recurrent)
    return replace(base, **overrides)


def toy_run_config(seed: int, spec: ToyRunSpec, lr: float = 1e-3, gamma: float = 0.99,
                   ppo_overrides: dict[str, Any] | None = None,
                   aux_overrides: dict[str, Any] | None = None) -> RunConfig:
    """ADR 0008 toy settings: 16 envs, rollout 32, lr 1e-3, gamma 0.99, base recipe
    (floor 0.1, 3 epochs) unless overridden by an arm."""
    ppo = PPOConfig(rollout_len=32, num_minibatches=4, epochs=3, lr=lr, entropy_coef=0.01, gamma=gamma)
    ppo = replace(ppo, **(ppo_overrides or {}))
    aux = replace(AuxConfig(), **(aux_overrides or {}))
    return replace(RunConfig(), seed=seed, num_workers=1, envs_per_worker=spec.num_envs,
                   torch_threads=TOY_TORCH_THREADS, ppo=ppo, aux=aux)


def run_toy(recurrent: bool, seed: int, spec: ToyRunSpec = ToyRunSpec(), lr: float = 1e-3,
            gamma: float = 0.99, arm: ToyArm = ToyArm("base")) -> dict[str, float]:
    from functools import partial

    from .trainer import Trainer
    from .vec_env import SyncVecEnv

    venv = SyncVecEnv([partial(ToyMemoryEnv, arm.toy) for _ in range(spec.num_envs)])
    trainer = Trainer(venv, toy_model_config(recurrent, **arm.model),
                      toy_run_config(seed, spec, lr, gamma, arm.ppo, arm.aux))
    history = [trainer.train_iteration() for _ in range(spec.updates)]
    rets = np.asarray(trainer.log.returns[-spec.eval_last_episodes:], dtype=np.float64)
    venv.close()
    return {"success": float(np.mean(rets > 0)), "episodes": float(len(trainer.log.returns)),
            "max_first_ratio_dev": max(h["first_minibatch_ratio_max_dev"] for h in history),
            "final_return_scale": float(history[-1]["return_scale"])}


def cue_trace(model_cfg: ModelConfig, seed: int, gap: int, toy: ToyConfig = ToyConfig()) -> float:
    """Debugger's metric (ADR 0008 Decision 3): L2 norm of the hidden-state difference
    at step `gap` between two episodes identical except for the cue, at initialization."""
    import torch

    from agent.model import build_agent
    from agent.obs_spec import spec_from_spaces
    from agent.tensorize import to_tensors

    env = ToyMemoryEnv(replace(toy, answer_step=max(gap, toy.answer_step)))
    model = build_agent(spec_from_spaces(env.observation_space, env.action_space), model_cfg, seed)
    hs = []
    for cue in (0, 1):
        obs, _ = env.reset(seed=seed, options={"cue": cue})
        h = model.initial_state(1)
        with torch.no_grad():
            for t in range(gap + 1):
                batch = {k: v[None] for k, v in obs.items()}
                _, h = model.step(to_tensors(batch), h, torch.tensor([1.0 if t == 0 else 0.0]))
                obs, *_ = env.step(np.zeros(len(toy.head_sizes), np.int64))
        hs.append(h)
    return float((hs[0] - hs[1]).norm())
