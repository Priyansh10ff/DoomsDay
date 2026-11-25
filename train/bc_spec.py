"""Observation spec and sensor signature for behaviour cloning, without starting the
engine.

`env.DoomEnv.__init__` starts ViZDoom, so the spec is rebuilt here from the env
module constants and the A0 training config (the same numbers DoomEnv puts into its
observation / action spaces), then cross-checked against every dataset file
(`validate_arrays`). Two sizes are literals inside DoomEnv's space construction
(entity feature width 8, modality width 3); they are mirrored here as named
constants and any drift is caught by the dataset check, which fails loudly.

Sensor signature: the dataset contract carries none. A BC checkpoint therefore
carries the A0 training config's signature, which `configs.a0_env.
generated_maps_env_config` asserts is identical to the generated-maps training
config's. It is recorded as ASSUMED (coordinator ruling 3, 2026-11-25).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from agent.obs_spec import A0_KEYS, ObsSpec

# Mirrors env/doom_env.py DoomEnv.__init__ (literals there): entities (n, 8), modality (3,).
ENV_ENTITY_FEAT_DIM = 8
ENV_MODALITY_DIM = 3

SIGNATURE_SOURCE = ("assumed: configs.a0_env.a0_train_env_config().sensor_signature(), which "
                    "generated_maps_env_config asserts equals the generated-maps training config's; "
                    "NOT read from the dataset (the dataset contract carries no signature)")

# dtypes of the saved arrays (dataset contract) per observation key
OBS_DTYPES: dict[str, np.dtype] = {
    "rays": np.dtype(np.float32), "entities": np.dtype(np.float32), "entity_type": np.dtype(np.int64),
    "entity_mask": np.dtype(np.bool_), "self": np.dtype(np.float32), "prev_action": np.dtype(np.int64),
    "modality": np.dtype(np.float32),
    "egomap": np.dtype(np.uint8),        # observation v3 only (env/egomap.py)
}
# Observation v3 `egomap` per-step shape (mirrors env.egomap.EGOMAP_SHAPE, checked by a test).
EGOMAP_STEP_SHAPE: tuple[int, int, int] = (5, 32, 32)


def egomap_step_shape(obs_version: int) -> tuple[int, int, int] | None:
    """Per-step `egomap` shape under an observation version (None before v3)."""
    return EGOMAP_STEP_SHAPE if obs_version >= 3 else None
LABEL_KEY = "action"
LABEL_DTYPE = np.dtype(np.int64)


class DatasetMismatch(ValueError):
    """A dataset file does not match the observation spec or the dataset contract."""


def a0_training_spec_and_signature() -> tuple[ObsSpec, tuple]:
    """(ObsSpec, sensor signature) of the A0 training env, built engine-free, at the
    observation version configs.a0_env.obs_version() selects (v1, v2 or v3). v3 has the
    same rays / self sizes as v2 plus `egomap` (egomap_step_shape); the signature carries
    the version."""
    from configs.a0_env import a0_train_env_config
    from env import observation as O
    from env.actions import head_sizes
    cfg = a0_train_env_config()
    extra = {}
    ego = egomap_step_shape(cfg.obs_version)
    if ego is not None:   # v3: agent.obs_spec builds an EgoMapObsSpec from a non-None egomap_shape
        extra["egomap_shape"] = ego
    spec = ObsSpec(**extra,
        ray_channels=O.ray_channels(len(cfg.ray_bands), cfg.obs_version), ray_width=O.N_RAY_COLS,
        max_entities=cfg.max_entities, entity_feat_dim=ENV_ENTITY_FEAT_DIM,
        n_entity_types=len(O.ENTITY_TYPES), self_dim=O.self_dim(cfg.obs_version),
        modality_dim=ENV_MODALITY_DIM, head_sizes=tuple(head_sizes(len(cfg.turn_bins))),
    )
    return spec, cfg.sensor_signature()


def expected_step_shapes(spec: ObsSpec) -> dict[str, tuple[int, ...]]:
    """Per-step (no time axis) shape of every A0 observation key and of the label.
    `egomap` (C, H, W) is included when the spec has one (spec.egomap, observation v3)."""
    out = {
        "rays": (spec.ray_channels, spec.ray_width),
        "entities": (spec.max_entities, spec.entity_feat_dim),
        "entity_type": (spec.max_entities,),
        "entity_mask": (spec.max_entities,),
        "self": (spec.self_dim,),
        "prev_action": (spec.n_heads,),
        "modality": (spec.modality_dim,),
        LABEL_KEY: (spec.n_heads,),
    }
    ego = getattr(spec, "egomap", None)
    if ego is not None:
        out["egomap"] = tuple(ego)
    return out


def validate_arrays(arrays: Mapping[str, Any], spec: ObsSpec, where: str) -> int:
    """Check one episode's arrays against the spec; returns its length T. Raises
    DatasetMismatch naming the file, key, and expected vs found shape/dtype."""
    shapes = expected_step_shapes(spec)
    missing = [k for k in shapes if k not in arrays]
    if missing:
        raise DatasetMismatch(f"{where}: missing keys {missing} (has {sorted(arrays)})")
    t = int(np.asarray(arrays[LABEL_KEY]).shape[0])
    if t < 1:
        raise DatasetMismatch(f"{where}: empty episode")
    for k, step_shape in shapes.items():
        a = np.asarray(arrays[k])
        want_dtype = LABEL_DTYPE if k == LABEL_KEY else OBS_DTYPES[k]
        if a.shape != (t, *step_shape):
            raise DatasetMismatch(f"{where}: {k} shape {a.shape}, expected {(t, *step_shape)}")
        if a.dtype != want_dtype:
            raise DatasetMismatch(f"{where}: {k} dtype {a.dtype}, expected {want_dtype}")
    labels = np.asarray(arrays[LABEL_KEY])
    for h, n in enumerate(spec.head_sizes):
        lo, hi = int(labels[:, h].min()), int(labels[:, h].max())
        if lo < 0 or hi >= n:
            raise DatasetMismatch(f"{where}: label head {h} values in [{lo}, {hi}], head has {n} options")
    et = np.asarray(arrays["entity_type"])
    if int(et.min()) < 0 or int(et.max()) >= spec.n_entity_types:
        raise DatasetMismatch(f"{where}: entity_type outside [0, {spec.n_entity_types})")
    assert set(A0_KEYS) <= set(shapes), "every A0 key must be validated"
    return t
