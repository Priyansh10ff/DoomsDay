"""Uniform weight averaging of several checkpoints of the SAME run.

    python -m tools.average_checkpoints --out <path> <ckpt1> <ckpt2> [...]

Every floating-point tensor of the model state is replaced by the uniform mean over
the inputs (accumulated in float64, cast back to its own dtype); non-float tensors
(integer / bool buffers) are taken from the LAST input. Refused when the inputs'
model config, observation spec or sensor signature differ (string-compared, as every
loader here compares them), or when their state-dict keys, shapes or dtypes differ.

The output uses the trainer's blob layout as a weights-only checkpoint (like
`train.checkpoint.save_weights_checkpoint`): `trainer.model` plus the config / spec /
signature JSON copied VERBATIM from the last input, `update_idx` and `decisions_done`
0, and a record under `train.checkpoint.WEIGHTS_ONLY_KEY` with kind
"checkpoint_average" and every source (the file actually read, its sha256, its
update index). So `evaluate.run`, `evaluate.record`, `--init-from` and the ablation
script read it unchanged, while `--resume` refuses it (no optimizer or run state).
Written atomically (`atomic_torch_save`). No engine is started.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from evaluate.checkpoint_io import _find_model_state, sha256_file
from train.checkpoint import (LOADED_FROM_KEY, SHAPING_KEY, WEIGHTS_ONLY_KEY, CheckpointMismatch,
                              atomic_torch_save, canonical_json, load_checkpoint)

AVERAGE_KIND = "checkpoint_average"
MIN_INPUTS = 2
ACCUM_DTYPE = torch.float64
# JSON fields every input must share exactly (the "same run" identity checked here).
IDENTITY_KEYS: tuple[str, ...] = ("model_config_json", "obs_spec_json", "sensor_signature_json")
# JSON fields copied verbatim from the last input into the output (if present).
COPIED_KEYS: tuple[str, ...] = IDENTITY_KEYS + ("run_config_json", SHAPING_KEY)


@dataclass
class Source:
    argument: str
    loaded_from: str
    sha256: str
    update_idx: int | None
    blob: dict[str, Any]
    state: dict[str, torch.Tensor] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"argument": self.argument, "loaded_from": os.path.abspath(self.loaded_from),
                "sha256": self.sha256, "update_idx": self.update_idx}


def read_source(path: str) -> Source:
    blob = load_checkpoint(path)
    src = blob[LOADED_FROM_KEY]
    upd = blob.get("trainer", {}).get("update_idx")
    return Source(path, src, sha256_file(src), None if upd is None else int(upd), blob, _find_model_state(blob))


def check_same_run(sources: list[Source]) -> None:
    if len(sources) < MIN_INPUTS:
        raise ValueError(f"need at least {MIN_INPUTS} checkpoints to average, got {len(sources)}")
    seen: dict[str, str] = {}
    for s in sources:
        key = os.path.normcase(os.path.abspath(s.loaded_from))
        if key in seen:
            raise ValueError(f"{s.argument!r} reads the same file as {seen[key]!r} ({s.loaded_from}); "
                             f"give each checkpoint once")
        seen[key] = s.argument
    ref = sources[0]
    for s in sources[1:]:
        for k in IDENTITY_KEYS:
            if s.blob.get(k) != ref.blob.get(k):
                raise CheckpointMismatch(f"{k} of {s.argument!r} differs from {ref.argument!r}: "
                                         f"{s.blob.get(k)} vs {ref.blob.get(k)}; refusing to average "
                                         f"checkpoints of different runs / models")
        if set(s.state) != set(ref.state):
            diff = sorted(set(s.state) ^ set(ref.state))
            raise CheckpointMismatch(f"model state keys of {s.argument!r} differ from {ref.argument!r}: {diff[:8]}")
        for k, v in s.state.items():
            r = ref.state[k]
            if v.shape != r.shape or v.dtype != r.dtype:
                raise CheckpointMismatch(f"{k}: {s.argument!r} has {tuple(v.shape)} {v.dtype}, "
                                         f"{ref.argument!r} has {tuple(r.shape)} {r.dtype}")


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Uniform mean of every floating-point tensor; non-float tensors from the last state."""
    assert len(states) >= MIN_INPUTS, f"need >= {MIN_INPUTS} states, got {len(states)}"
    out: dict[str, torch.Tensor] = {}
    last = states[-1]
    for k, v in last.items():
        if torch.is_floating_point(v):
            acc = torch.zeros(v.shape, dtype=ACCUM_DTYPE)
            for s in states:
                assert s[k].shape == v.shape, f"{k}: expected shape {tuple(v.shape)}, got {tuple(s[k].shape)}"
                acc += s[k].detach().cpu().to(ACCUM_DTYPE)
            out[k] = (acc / len(states)).to(v.dtype)
        else:
            out[k] = v.detach().cpu().clone()
    return out


def build_blob(sources: list[Source]) -> dict[str, Any]:
    """A fresh blob (never a re-save of what load_checkpoint returned: that holds a
    ModelConfig object and `loaded_from`, which weights_only loading refuses)."""
    check_same_run(sources)
    last = sources[-1]
    record = {"kind": AVERAGE_KIND, "method": "uniform mean of floating-point model tensors (float64 "
              "accumulation); non-float tensors from the last source", "n_sources": len(sources),
              "sources": [s.record() for s in sources], "configs_copied_from": last.record()["loaded_from"]}
    blob: dict[str, Any] = {k: last.blob[k] for k in COPIED_KEYS if k in last.blob}
    blob["trainer"] = {"model": average_states([s.state for s in sources]), "update_idx": 0, "decisions_done": 0}
    blob[WEIGHTS_ONLY_KEY] = canonical_json(record)
    return blob


def average_checkpoints(paths: list[str], out: str) -> dict[str, Any]:
    """Average `paths` into `out`; returns the record stored in the output."""
    sources = [read_source(p) for p in paths]
    out_key = os.path.normcase(os.path.abspath(out))
    for s in sources:
        if out_key in (os.path.normcase(os.path.abspath(s.argument)), os.path.normcase(os.path.abspath(s.loaded_from))):
            raise ValueError(f"--out {out!r} would overwrite the input {s.argument!r}")
    blob = build_blob(sources)
    atomic_torch_save(blob, out)
    return json.loads(blob[WEIGHTS_ONLY_KEY])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.average_checkpoints", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("checkpoints", nargs="+", help=f"{MIN_INPUTS} or more checkpoints of the same run")
    a = p.parse_args(argv)
    if len(a.checkpoints) < MIN_INPUTS:
        p.error(f"need at least {MIN_INPUTS} checkpoints")
    try:
        rec = average_checkpoints(a.checkpoints, a.out)
    except (CheckpointMismatch, ValueError, FileNotFoundError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"averaged {rec['n_sources']} checkpoints -> {a.out}")
    for s in rec["sources"]:
        print(f"  {s['loaded_from']} (update {s['update_idx']}, sha256 {s['sha256'][:12]})")
    print(f"output sha256 {sha256_file(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
