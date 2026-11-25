"""Observation / action sizes read from gymnasium spaces at runtime (ADR 0006).

Nothing here hard-codes a shape: the env may change ray rows, entity count,
type-table length, or add/remove `screen`, and the model follows.

Optional keys (observation version 3): `egomap`, an ego-centric occupancy map
(C, H, W) uint8. A spec WITHOUT it is a plain `ObsSpec`, whose fields (and so its
`dataclasses.asdict` JSON, string-compared by the checkpoint loaders) are exactly
the v1/v2 ones. A spec WITH it is an `EgoMapObsSpec` (subclass, one extra field).
`ObsSpec(**d)` with a non-None `egomap_shape` in `d` builds an `EgoMapObsSpec`, so
the loaders that rebuild a spec from checkpoint JSON work for both unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

# Keys the Stage A0 model consumes. `screen` is ignored in A0 (ADR 0003, 0006).
A0_KEYS: tuple[str, ...] = (
    "rays", "entities", "entity_type", "entity_mask", "self", "prev_action", "modality",
)
# Optional model input (observation version 3): ego-centric map, (C, H, W) uint8.
EGOMAP_KEY = "egomap"
EGOMAP_NDIM = 3
OPTIONAL_KEYS: tuple[str, ...] = (EGOMAP_KEY,)


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

    def __new__(cls, *args: Any, **kwargs: Any) -> "ObsSpec":
        # Rebuilding from checkpoint JSON (`ObsSpec(**d)`) with an egomap entry gives
        # the subclass; pickle / copy call __new__ with no kwargs and are unaffected.
        if cls is ObsSpec and kwargs.get("egomap_shape") is not None:
            cls = EgoMapObsSpec
        return object.__new__(cls)

    @property
    def n_heads(self) -> int:
        return len(self.head_sizes)

    @property
    def egomap(self) -> tuple[int, int, int] | None:
        """(C, H, W) of the egomap input, or None when the model has none."""
        return None


@dataclass(frozen=True)
class EgoMapObsSpec(ObsSpec):
    egomap_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        shape = tuple(int(x) for x in self.egomap_shape)   # JSON gives a list
        if len(shape) != EGOMAP_NDIM or min(shape) < 1:
            raise ValueError(f"egomap_shape must be (C, H, W) positive, got {self.egomap_shape}")
        object.__setattr__(self, "egomap_shape", shape)

    @property
    def egomap(self) -> tuple[int, int, int] | None:
        return self.egomap_shape  # type: ignore[return-value]


def model_keys(spec: ObsSpec) -> tuple[str, ...]:
    """Observation keys this spec's model consumes, in a fixed order."""
    return A0_KEYS + ((EGOMAP_KEY,) if spec.egomap is not None else ())


def spec_from_spaces(observation_space: spaces.Dict, action_space: spaces.MultiDiscrete) -> ObsSpec:
    """Build the spec. Type count comes from the `entity_type` Box high + 1, head
    sizes from `action_space.nvec` (the `prev_action` Box high is only a max).
    An `egomap` space (C, H, W) makes it an EgoMapObsSpec."""
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
    fields: dict[str, Any] = dict(
        ray_channels=int(rays_shape[0]),
        ray_width=int(rays_shape[1]),
        max_entities=int(ent_shape[0]),
        entity_feat_dim=int(ent_shape[1]),
        n_entity_types=type_high + 1,
        self_dim=int(s["self"].shape[0]),
        modality_dim=int(s["modality"].shape[0]),
        head_sizes=nvec,
    )
    if EGOMAP_KEY in s:
        shape = s[EGOMAP_KEY].shape
        if len(shape) != EGOMAP_NDIM:
            raise ValueError(f"egomap must be (C, H, W), got {shape}")
        return EgoMapObsSpec(**fields, egomap_shape=tuple(int(x) for x in shape))
    return ObsSpec(**fields)
