"""Batched numpy observation dict -> torch tensors for the A0 keys (screen dropped)."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from .obs_spec import A0_KEYS

_DTYPES: dict[str, torch.dtype] = {
    "rays": torch.float32, "entities": torch.float32, "entity_type": torch.long,
    "entity_mask": torch.bool, "self": torch.float32, "prev_action": torch.long,
    "modality": torch.float32,
}


def to_tensors(obs: Mapping[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(obs[k]), device=device).to(_DTYPES[k]) for k in A0_KEYS}
