"""Inference path: network forward pass, then per-head sample or argmax. Nothing else.

Which heads are argmax and which are sampled is fixed in configuration (ADR 0002);
no code here inspects the observation to change an action.

Play-time decoding temperature `T` (a uniform decoding setting, not a rule): sampled
heads draw from softmax(logits / T), every head with the same T. T == 1.0 skips the
division entirely, so it is bit-identical to plain sampling. Argmax heads are
unaffected (argmax of logits / T equals argmax of logits for T > 0).
"""
from __future__ import annotations

from collections.abc import Mapping

import math

import numpy as np
import torch

from .distributions import FactorizedCategorical
from .model import A0Agent
from .tensorize import to_tensors

DEFAULT_TEMPERATURE = 1.0


def check_temperature(temperature: float) -> float:
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    return t


def tempered_logits(logits: list[torch.Tensor], temperature: float) -> list[torch.Tensor]:
    """Every head divided by the same T; T == 1.0 returns the list unchanged."""
    t = check_temperature(temperature)
    if t == DEFAULT_TEMPERATURE:
        return logits
    return [lg / t for lg in logits]


class PolicyRunner:
    def __init__(self, model: A0Agent, num_envs: int, argmax_heads: tuple[bool, ...], seed: int,
                 device: torch.device | str = "cpu", temperature: float = DEFAULT_TEMPERATURE) -> None:
        assert len(argmax_heads) == model.spec.n_heads, \
            f"argmax_heads needs {model.spec.n_heads} entries, got {len(argmax_heads)}"
        self.temperature = check_temperature(temperature)
        self.model, self.device = model, device
        self.argmax = torch.tensor(argmax_heads, dtype=torch.bool, device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.h = model.initial_state(num_envs, device)

    @torch.no_grad()
    def act(self, obs: Mapping[str, np.ndarray], episode_start: np.ndarray) -> np.ndarray:
        out, self.h = self.model.step(to_tensors(obs, self.device), self.h,
                                      torch.as_tensor(episode_start, dtype=torch.float32, device=self.device))
        dist = FactorizedCategorical(tempered_logits(out.logits, self.temperature))
        a = torch.where(self.argmax, dist.mode(), dist.sample(self.gen))
        return a.cpu().numpy().astype(np.int64)
