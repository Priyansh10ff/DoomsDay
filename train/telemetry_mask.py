"""ADR 0006 trainer-side modality dropout: hide telemetry from the policy input
while the rollout keeps the true rays/entities for auxiliary targets.

A0 trains with probability 0.0; the mechanism exists so A1 can switch it on.
"""
from __future__ import annotations

import numpy as np
import torch

from configs.model_config import ModelConfig

TensorDict = dict[str, torch.Tensor]


def mask_telemetry(obs: TensorDict, dropped: torch.Tensor, cfg: ModelConfig) -> TensorDict:
    """Return a copy of `obs` with telemetry zeroed and flagged absent where `dropped`.

    `dropped` has the leading dims of every obs tensor (e.g. (B,) or (T, B)).
    """
    lead = dropped.shape
    assert obs["rays"].shape[:len(lead)] == lead, \
        f"dropped shape {tuple(lead)} does not lead rays {tuple(obs['rays'].shape)}"
    if not bool(dropped.any()):
        return obs
    d = dropped.bool()

    def expand(x: torch.Tensor) -> torch.Tensor:
        return d.reshape(*lead, *([1] * (x.dim() - len(lead))))

    out = dict(obs)
    out["rays"] = obs["rays"].masked_fill(expand(obs["rays"]), 0.0)
    out["entities"] = obs["entities"].masked_fill(expand(obs["entities"]), 0.0)
    out["entity_type"] = obs["entity_type"].masked_fill(expand(obs["entity_type"]), 0)
    out["entity_mask"] = obs["entity_mask"] & ~expand(obs["entity_mask"])
    s = obs["self"].clone()
    ego = s[..., s.shape[-1] - cfg.self_ego_dims:]
    s[..., s.shape[-1] - cfg.self_ego_dims:] = ego.masked_fill(expand(ego), 0.0)
    out["self"] = s
    m = obs["modality"].clone()
    for idx in (cfg.modality_telemetry_index, cfg.modality_ego_index):
        m[..., idx] = m[..., idx].masked_fill(d, 0.0)
    out["modality"] = m
    return out


class EpisodeDropout:
    """Per-env, per-episode Bernoulli draw from a seeded numpy Generator."""

    def __init__(self, num_envs: int, prob: float, seed: int) -> None:
        self.prob = float(prob)
        self.rng = np.random.default_rng(seed)
        self.dropped = np.zeros(num_envs, bool)

    def on_episode_start(self, starts: np.ndarray) -> np.ndarray:
        draw = self.rng.random(starts.shape[0]) < self.prob
        self.dropped = np.where(starts.astype(bool), draw, self.dropped)
        return self.dropped.copy()
