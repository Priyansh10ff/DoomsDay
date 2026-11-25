"""Observation / action sizes read from gymnasium spaces at runtime (ADR 0006).

Nothing here hard-codes a shape: the env may change ray rows, entity count,
type-table length, or add/remove `screen`, and the model follows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

# Keys the Stage A0 model consumes. `screen` is ignored in A0 (ADR 0003, 0006).
A0_KEYS: tuple[str, ...] = (
    "rays", "entities", "entity_type", "entity_mask", "self", "prev_action", "modality",
)


@dataclass(frozen=True)
class ObsSpec:
    ray_channels: int
    ray_width: int
    max_entities: int
    entity_feat_dim: int
    n_entity_types: int
    self_dim: int
    modality_dim: int
    head_sizes: tuple[int, ...]

    @property
    def n_heads(self) -> int:
        return len(self.head_sizes)


def spec_from_spaces(observation_space: spaces.Dict, action_space: spaces.MultiDiscrete) -> ObsSpec:
    """Build the spec. Type count comes from the `entity_type` Box high + 1, head
    sizes from `action_space.nvec` (the `prev_action` Box high is only a max)."""
    missing = [k for k in A0_KEYS if k not in observation_space.spaces]
    if missing:
        raise KeyError(f"observation space lacks keys required by Stage A0: {missing}")
    s = observation_space.spaces
    rays_shape = s["rays"].shape
    ent_shape = s["entities"].shape
    if len(rays_shape) != 2:
        raise ValueError(f"rays must be (bands, columns), got {rays_shape}")
    if len(ent_shape) != 2:
        raise ValueError(f"entities must be (max_entities, features), got {ent_shape}")
    type_high = int(np.max(s["entity_type"].high))
    nvec = tuple(int(n) for n in np.asarray(action_space.nvec).reshape(-1))
    if s["prev_action"].shape != (len(nvec),):
        raise ValueError(f"prev_action shape {s['prev_action'].shape} != ({len(nvec)},) heads")
    return ObsSpec(
        ray_channels=int(rays_shape[0]),
        ray_width=int(rays_shape[1]),
        max_entities=int(ent_shape[0]),
        entity_feat_dim=int(ent_shape[1]),
        n_entity_types=type_high + 1,
        self_dim=int(s["self"].shape[0]),
        modality_dim=int(s["modality"].shape[0]),
        head_sizes=nvec,
    )
