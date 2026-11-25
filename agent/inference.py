"""Inference path: network forward pass, then per-head sample or argmax. Nothing else.

Which heads are argmax and which are sampled is fixed in configuration (ADR 0002);
no code here inspects the observation to change an action.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from .distributions import FactorizedCategorical
from .model import A0Agent
from .tensorize import to_tensors


class PolicyRunner:
    def __init__(self, model: A0Agent, num_envs: int, argmax_heads: tuple[bool, ...], seed: int,
                 device: torch.device | str = "cpu") -> None:
        assert len(argmax_heads) == model.spec.n_heads, \
            f"argmax_heads needs {model.spec.n_heads} entries, got {len(argmax_heads)}"
        self.model, self.device = model, device
        self.argmax = torch.tensor(argmax_heads, dtype=torch.bool, device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.h = model.initial_state(num_envs, device)

    @torch.no_grad()
    def act(self, obs: Mapping[str, np.ndarray], episode_start: np.ndarray) -> np.ndarray:
        out, self.h = self.model.step(to_tensors(obs, self.device), self.h,
                                      torch.as_tensor(episode_start, dtype=torch.float32, device=self.device))
        dist = FactorizedCategorical(out.logits)
        a = torch.where(self.argmax, dist.mode(), dist.sample(self.gen))
        return a.cpu().numpy().astype(np.int64)
