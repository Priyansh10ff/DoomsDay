"""Batched numpy observation dict -> torch tensors for the model keys (screen dropped).

The A0 keys are always converted; optional keys (`egomap`, observation v3) are
converted when present in the observation, so a v1/v2 observation gives exactly the
tensors it always did. `egomap` stays uint8 here; its encoder casts it to float."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from .obs_spec import A0_KEYS, EGOMAP_KEY, OPTIONAL_KEYS

TORCH_DTYPES: dict[str, torch.dtype] = {
    "rays": torch.float32, "entities": torch.float32, "entity_type": torch.long,
    "entity_mask": torch.bool, "self": torch.float32, "prev_action": torch.long,
    "modality": torch.float32, EGOMAP_KEY: torch.uint8,
}
_DTYPES = TORCH_DTYPES  # older name


def torch_dtype(key: str, array: np.ndarray) -> torch.dtype:
    """The model-side dtype of `key`; an unknown key keeps its numpy dtype."""
    if key in TORCH_DTYPES:
        return TORCH_DTYPES[key]
    return torch.from_numpy(np.zeros(0, dtype=np.asarray(array).dtype)).dtype


def present_keys(obs: Mapping[str, object]) -> tuple[str, ...]:
    """A0 keys plus every optional model key found in `obs`."""
    return A0_KEYS + tuple(k for k in OPTIONAL_KEYS if k in obs)


def to_tensors(obs: Mapping[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(obs[k]), device=device).to(TORCH_DTYPES[k])
            for k in present_keys(obs)}
