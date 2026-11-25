"""Grow a model's egomap input by extra channels appended at the end (CLAUDE.md section 7,
"Option B" gap 1: observation v4 = v3 + one egomap channel).

Used by `--init-from` (train/init_from.py) and `--init-from-bc` (train/bc.py) when the
source file's observation spec differs from the new run's ONLY by more egomap channels
(same H, W and every other field). The first egomap conv weight (out, C_old, k, k) is
extended along the input-channel axis with ZERO slices (bias unchanged), so the grown
model computes exactly what the source model computed: the new channels contribute
nothing until training moves those weights. Any other difference is refused by the
callers (they then load strictly, so a key or shape mismatch elsewhere still fails).

The sensor signature may then differ only in its `obs_version` entry, and only upward.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from agent.obs_spec import ObsSpec

EGOMAP_SPEC_FIELD = "egomap_shape"
OBS_VERSION_FIELD = "obs_version"
EGOMAP_CHANNEL_AXIS = 0      # of egomap_shape (C, H, W)
CONV_IN_CHANNEL_AXIS = 1     # of a Conv2d weight (out, in, kh, kw)
FIRST_EGOMAP_CONV_KEY = "egomap.conv.0.weight"   # A0Agent.egomap.conv[0] (checked against the model)


@dataclass(frozen=True)
class EgomapGrowth:
    old_channels: int
    new_channels: int
    key: str = FIRST_EGOMAP_CONV_KEY

    def describe(self) -> str:
        return (f"egomap input grown {self.old_channels}->{self.new_channels} channels: "
                f"'{self.key}' input slices [{self.old_channels}:{self.new_channels}] zero-initialised, "
                f"bias and every other weight copied unchanged")

    def record(self) -> dict[str, Any]:
        return {"kind": "egomap_channel_growth", "key": self.key,
                "old_channels": self.old_channels, "new_channels": self.new_channels,
                "new_slices_init": "zeros"}


def egomap_growth(src_spec_json: str, dst_spec: ObsSpec) -> EgomapGrowth | None:
    """The growth that maps the source spec (checkpoint JSON) onto `dst_spec`, or None
    when the difference is anything other than extra egomap channels appended at the
    end (equal specs also give None: nothing to grow)."""
    src = json.loads(src_spec_json)
    dst = json.loads(json.dumps(asdict(dst_spec)))   # tuples -> lists, as in the file
    if src.get(EGOMAP_SPEC_FIELD) is None or dst.get(EGOMAP_SPEC_FIELD) is None:
        return None
    rest_src = {k: v for k, v in src.items() if k != EGOMAP_SPEC_FIELD}
    rest_dst = {k: v for k, v in dst.items() if k != EGOMAP_SPEC_FIELD}
    if rest_src != rest_dst:
        return None
    (c0, *hw0), (c1, *hw1) = src[EGOMAP_SPEC_FIELD], dst[EGOMAP_SPEC_FIELD]
    if hw0 != hw1 or int(c1) <= int(c0):
        return None
    return EgomapGrowth(int(c0), int(c1))


def obs_version_change(src_sig_json: str, dst_sig_json: str) -> tuple[int, int] | None:
    """(old, new) when the two sensor signatures differ ONLY in their obs_version entry
    and new > old; None otherwise (including when they are equal)."""
    def split(sig_json: str) -> tuple[list[Any], int | None]:
        rest, ver = [], None
        for entry in json.loads(sig_json):
            if isinstance(entry, list) and len(entry) == 2 and entry[0] == OBS_VERSION_FIELD:
                ver = int(entry[1])
            else:
                rest.append(entry)
        return rest, ver
    rest0, v0 = split(src_sig_json)
    rest1, v1 = split(dst_sig_json)
    if rest0 != rest1 or v0 is None or v1 is None or v1 <= v0:
        return None
    return v0, v1


def grow_state(state: dict[str, torch.Tensor], growth: EgomapGrowth,
               target_shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    """Copy of `state` with the first egomap conv weight extended by zero input-channel
    slices to its shape in `target_shapes` (the new model's state-dict shapes)."""
    key = growth.key
    if key not in state or key not in target_shapes:
        raise KeyError(f"egomap growth: {key!r} missing from the source state or the new model")
    w = state[key]
    want = tuple(target_shapes[key])
    assert w.dim() == 4 and w.shape[CONV_IN_CHANNEL_AXIS] == growth.old_channels, \
        f"source {key} expected (out, {growth.old_channels}, kh, kw), got {tuple(w.shape)}"
    expect = (w.shape[0], growth.new_channels, *w.shape[2:])
    assert want == expect, f"new model {key} expected {expect}, got {want}"
    pad_shape = (w.shape[0], growth.new_channels - growth.old_channels, *w.shape[2:])
    grown = torch.cat([w, torch.zeros(pad_shape, dtype=w.dtype, device=w.device)], dim=CONV_IN_CHANNEL_AXIS)
    assert tuple(grown.shape) == want, f"grown {key} expected {want}, got {tuple(grown.shape)}"
    out = dict(state)
    out[key] = grown
    return out


def state_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}
