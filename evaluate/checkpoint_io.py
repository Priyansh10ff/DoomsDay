"""Load a trained A0 model from a checkpoint written by train/checkpoint.py.

The loader itself is imported from `train.checkpoint` (not copied). Model weights are
looked up under the key sets known so far (`trainer.model` today); `_orig_mod.` and
`module.` prefixes are stripped. Loading is strict: every parameter must match.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch

from agent.model import A0Agent
from agent.obs_spec import ObsSpec

MODEL_STATE_PATHS: tuple[tuple[str, ...], ...] = (("trainer", "model"), ("trainer", "learner"),
                                                  ("model",), ("model_state_dict",))
STRIP_PREFIXES: tuple[str, ...] = ("_orig_mod.", "module.")
HASH_CHUNK_BYTES = 1 << 20


@dataclass
class LoadedCheckpoint:
    model: A0Agent
    spec: ObsSpec
    signature_json: str | None
    loaded_from: str
    sha256: str
    obs_spec_json: str
    model_config: dict
    run_config_json: str | None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_loader():
    try:
        from train.checkpoint import LOADED_FROM_KEY, load_checkpoint
    except Exception as e:  # train/ may be mid-edit by another worker
        raise RuntimeError(f"cannot import train.checkpoint.load_checkpoint ({type(e).__name__}: {e}); "
                           f"the evaluator does not carry its own copy of the loader") from e
    return load_checkpoint, LOADED_FROM_KEY


def _find_model_state(blob: dict[str, Any]) -> dict[str, torch.Tensor]:
    for path in MODEL_STATE_PATHS:
        node: Any = blob
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict) and node:
            out = {}
            for k, v in node.items():
                for p in STRIP_PREFIXES:
                    if k.startswith(p):
                        k = k[len(p):]
                out[k] = v
            return out
    raise KeyError(f"no model state found under any of {MODEL_STATE_PATHS}")


def spec_from_json(obs_spec_json: str) -> ObsSpec:
    d = json.loads(obs_spec_json)
    d["head_sizes"] = tuple(d["head_sizes"])
    return ObsSpec(**d)


def load_model(path: str, device: str = "cpu") -> LoadedCheckpoint:
    """Raises (from train.checkpoint) if the checkpoint has no sensor signature."""
    load_checkpoint, loaded_from_key = _import_loader()
    blob = load_checkpoint(path)
    src = blob.get(loaded_from_key, path)
    spec = spec_from_json(blob["obs_spec_json"])
    cfg = blob["model_config"]
    model = A0Agent(spec, cfg)
    model.load_state_dict(_find_model_state(blob), strict=True)
    model.to(device).eval()
    return LoadedCheckpoint(model=model, spec=spec, signature_json=blob.get("sensor_signature_json"),
                            loaded_from=src, sha256=sha256_file(src), obs_spec_json=blob["obs_spec_json"],
                            model_config=dataclasses.asdict(cfg), run_config_json=blob.get("run_config_json"))


def check_spec_matches(spec: ObsSpec, env_spec: ObsSpec) -> None:
    if spec != env_spec:
        raise AssertionError(f"observation/action spec of the EVAL env {env_spec} differs from the "
                             f"checkpoint's {spec}")
