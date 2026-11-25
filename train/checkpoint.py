"""Checkpoint save / resume (ADR 0007 ruling 7 item 3, ADR 0008 points 1 and 7, Decision 5).

- Loaded with `torch.load(weights_only=True)` only. Everything that is not a tensor,
  number, string, list or dict is stored as JSON text (configs, sensor signature,
  NumPy RNG state); a round-trip test decides what passes. No allow-listing.
- Every checkpoint carries the env `sensor_signature()`; resume refuses a checkpoint
  whose signature, model config or observation spec differs.
- Written atomically; the previous file is kept as a second generation (`.prev`).
  The main file is never renamed away: the previous generation is COPIED to `.prev`
  (via its own temp file) and only then is the new file renamed over the main one,
  so a crash at any point leaves a loadable `checkpoint.pt`. `load_checkpoint` also
  falls back to `.prev` when the main file is missing (e.g. written by older code
  that renamed the main file first), and records which file it actually read.
- Callers write only on update boundaries (train/run_loop.py).
- ADR 0011 F6: every checkpoint records the shaping preset (or none) and the env count
  that sets the per-env decision clock of the shaping anneal; `resume` refuses a
  preset different from the stored one, in either direction. A checkpoint written
  before this record existed is read as unshaped, with the env count from its run
  config (`shaping_record`); the record says `legacy: true` then.
- ADR 0011 F3: `keep_numbered_copy` keeps every checkpoint the run loop writes under a
  numbered name in `checkpoints/`, thinned to every second one above a byte budget.
- The health monitor's state (ADR 0011 F4) rides along as JSON text, so its windows,
  run best and event start survive a resume.
- Curriculum runs (`--maps generated`): the scheduler state, with each env's current
  and upcoming choice, rides along as JSON text (`CURRICULUM_KEY`), so a resume
  continues the curriculum. The map source preset is part of the stored run config;
  `resume` refuses a different one (a checkpoint without the field reads as
  "my_way_home").
- Loaded with map_location="cpu"; `Trainer.load_state_dict` moves tensors to the
  trainer's device, so checkpoints load across devices.
- Weights-only checkpoints (`save_weights_checkpoint`, written by behaviour cloning,
  train/bc.py): the same blob keys `evaluate/checkpoint_io.py` and `--init-from` read
  (`trainer.model`, configs, obs spec, sensor signature), no optimizer or RNG state,
  and a `bc_json` record (`WEIGHTS_ONLY_KEY`). `resume` and `checkpoint_resume_info`
  refuse them: such a file starts a fresh PPO run through `--init-from` only.
- RND (`--rnd-int-coef`) and the inverse-dynamics loss (`--inv-dyn-coef`): their
  state rides in the trainer state (`rnd`, `inv_dyn`); `resume` refuses turning
  either switch on or off (`check_switches_match`). Older checkpoints read as off.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
from typing import Any

import torch

from configs.model_config import ModelConfig
from configs.train_config import BCAuxConfig

from .trainer import Trainer

PREV_SUFFIX = ".prev"
TMP_SUFFIX = ".tmp"
LOADED_FROM_KEY = "loaded_from"  # added by load_checkpoint: the file actually read
SHAPING_KEY = "shaping_json"
HEALTH_KEY = "health_monitor_json"
CURRICULUM_KEY = "curriculum_json"
LEGACY_MAPS = "my_way_home"   # map source of checkpoints written before RunConfig.maps
NUMBERED_DIR = "checkpoints"
NUMBERED_PATTERN = "checkpoint-u{update:07d}.pt"
WEIGHTS_ONLY_KEY = "bc_json"   # present only in weights-only (behaviour-cloning) checkpoints


class CheckpointMismatch(RuntimeError):
    pass


def canonical_json(x: Any) -> str:
    """Tuples become lists; key order fixed, so equal content gives equal text."""
    return json.dumps(x, sort_keys=True)


def shaping_of(trainer: Trainer) -> dict[str, Any]:
    return {"preset": trainer.run.shaping, "num_envs": int(trainer.n)}


def save_checkpoint(path: str, trainer: Trainer, sensor_signature: tuple,
                    health_state_json: str | None = None, curriculum_state_json: str | None = None) -> None:
    if not isinstance(sensor_signature, tuple) or not sensor_signature:
        raise ValueError("sensor_signature must be the non-empty tuple from EnvConfig.sensor_signature()")
    blob = {
        SHAPING_KEY: canonical_json(shaping_of(trainer)),
        "trainer": trainer.state_dict(),
        "model_config_json": canonical_json(dataclasses.asdict(trainer.model.cfg)),
        "obs_spec_json": canonical_json(dataclasses.asdict(trainer.spec)),
        "run_config_json": canonical_json(dataclasses.asdict(trainer.run)),
        "sensor_signature_json": canonical_json(sensor_signature),
    }
    if health_state_json is not None:
        blob[HEALTH_KEY] = health_state_json
    if curriculum_state_json is not None:
        blob[CURRICULUM_KEY] = curriculum_state_json
    atomic_torch_save(blob, path)


def atomic_torch_save(blob: dict[str, Any], path: str) -> None:
    """Write `blob` to `path` atomically, keeping the previous file as `.prev`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + TMP_SUFFIX
    torch.save(blob, tmp)
    if os.path.exists(path):
        # keep one older generation; copy (not rename) so `path` never disappears
        prev, prev_tmp = path + PREV_SUFFIX, path + PREV_SUFFIX + TMP_SUFFIX
        shutil.copyfile(path, prev_tmp)
        os.replace(prev_tmp, prev)
    os.replace(tmp, path)                      # atomic: never a half-written checkpoint


def save_weights_checkpoint(path: str, model: torch.nn.Module, model_cfg: ModelConfig, spec: Any,
                            run_config: Any, sensor_signature: tuple, record: dict[str, Any]) -> None:
    """Weights-only checkpoint in the trainer's blob layout (see module docstring).
    `record` (JSON-able) is stored under WEIGHTS_ONLY_KEY and marks the file as not
    resumable."""
    if not isinstance(sensor_signature, tuple) or not sensor_signature:
        raise ValueError("sensor_signature must be the non-empty tuple from EnvConfig.sensor_signature()")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    blob = {
        SHAPING_KEY: canonical_json({"preset": None, "num_envs": int(run_config.num_envs)}),
        "trainer": {"model": state, "update_idx": 0, "decisions_done": 0},
        "model_config_json": canonical_json(dataclasses.asdict(model_cfg)),
        "obs_spec_json": canonical_json(dataclasses.asdict(spec)),
        "run_config_json": canonical_json(dataclasses.asdict(run_config)),
        "sensor_signature_json": canonical_json(sensor_signature),
        WEIGHTS_ONLY_KEY: canonical_json(record),
    }
    atomic_torch_save(blob, path)


def refuse_weights_only(blob: dict[str, Any]) -> None:
    if WEIGHTS_ONLY_KEY in blob:
        raise CheckpointMismatch(
            f"refusing to resume: {blob.get(LOADED_FROM_KEY)!r} is a weights-only (behaviour-cloning) "
            f"checkpoint with no optimizer or run state; start a fresh PPO run from it with --init-from")


def checkpoint_file_to_read(path: str) -> str:
    """`path` if it exists, else its `.prev` generation if that exists. Never `.tmp`
    (it may be half-written)."""
    if os.path.exists(path):
        return path
    prev = path + PREV_SUFFIX
    if os.path.exists(prev):
        return prev
    raise FileNotFoundError(f"no checkpoint at {path!r} and no previous generation {prev!r}")


def load_checkpoint(path: str) -> dict[str, Any]:
    src = checkpoint_file_to_read(path)
    blob = torch.load(src, map_location="cpu", weights_only=True)
    if "sensor_signature_json" not in blob:
        raise CheckpointMismatch(f"checkpoint {src!r} has no sensor signature")
    blob[LOADED_FROM_KEY] = src
    blob["model_config"] = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v
                                          for k, v in json.loads(blob["model_config_json"]).items()})
    return blob


def shaping_record(blob: dict[str, Any]) -> dict[str, Any]:
    """{"preset": name | None, "num_envs": int, "legacy": bool} of a loaded checkpoint."""
    if SHAPING_KEY in blob:
        rec = json.loads(blob[SHAPING_KEY])
        return {"preset": rec["preset"], "num_envs": int(rec["num_envs"]), "legacy": False}
    run = json.loads(blob["run_config_json"])
    return {"preset": run.get("shaping"), "num_envs": int(run["num_workers"]) * int(run["envs_per_worker"]),
            "legacy": True}


def check_shaping_matches(stored: dict[str, Any], requested: str | None, src: str = "") -> None:
    if stored["preset"] != requested:
        raise CheckpointMismatch(
            f"refusing to resume: checkpoint {src!r} was trained with shaping {stored['preset']!r}"
            f"{' (legacy checkpoint, read as unshaped)' if stored['legacy'] else ''}, this run asks for "
            f"{requested!r}. A reward change between runs is not implied by resuming (ADR 0011 F6).")


def maps_of(blob: dict[str, Any]) -> str:
    return json.loads(blob["run_config_json"]).get("maps", LEGACY_MAPS)


def check_maps_matches(stored: str, requested: str, src: str = "") -> None:
    if stored != requested:
        raise CheckpointMismatch(
            f"refusing to resume: checkpoint {src!r} was trained on maps {stored!r}, this run asks for "
            f"{requested!r}. A change of map source is not implied by resuming.")


def bc_aux_of(blob: dict[str, Any]) -> BCAuxConfig:
    """The BC auxiliary-loss settings a checkpoint was trained with (a checkpoint
    written before they existed reads as off)."""
    rec = json.loads(blob["run_config_json"]).get("bc_aux")
    if not rec:
        return BCAuxConfig()
    fields = {f.name for f in dataclasses.fields(BCAuxConfig)}
    return BCAuxConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in rec.items() if k in fields})


def bc_identity(cfg: BCAuxConfig) -> dict[str, Any]:
    return cfg.resume_identity() if cfg.enabled else {"enabled": False}


def check_bc_matches(stored: BCAuxConfig, requested: BCAuxConfig, src: str = "") -> None:
    a, b = bc_identity(stored), bc_identity(requested)
    if a != b:
        raise CheckpointMismatch(
            f"refusing to resume: checkpoint {src!r} was trained with BC auxiliary settings {a}, this run "
            f"asks for {b}. Resuming does not change --bc-data / --bc-coef settings.")


def switches_of(blob: dict[str, Any]) -> dict[str, bool]:
    """RND and inverse-dynamics switches a checkpoint was trained with (a checkpoint
    written before they existed reads as both off)."""
    rc = json.loads(blob["run_config_json"])
    rnd = rc.get("rnd") or {}
    inv = rc.get("inv_dyn") or {}
    return {"rnd": float(rnd.get("int_coef", 0.0)) > 0.0, "inv_dyn": float(inv.get("coef", 0.0)) > 0.0}


def switches_requested(run: Any) -> dict[str, bool]:
    return {"rnd": bool(run.rnd.enabled), "inv_dyn": bool(run.inv_dyn.enabled)}


def check_switches_match(stored: dict[str, bool], requested: dict[str, bool], src: str = "") -> None:
    """Refuse a resume that turns RND (--rnd-int-coef) or the inverse-dynamics loss
    (--inv-dyn-coef) on or off: their networks, normalisers and optimizer state are
    part of the run state. A coefficient change with the switch unchanged is allowed."""
    for name, flag in (("rnd", "--rnd-int-coef"), ("inv_dyn", "--inv-dyn-coef")):
        if stored[name] != requested[name]:
            raise CheckpointMismatch(
                f"refusing to resume: checkpoint {src!r} was trained with {name} "
                f"{'on' if stored[name] else 'off'}, this run asks for {'on' if requested[name] else 'off'} "
                f"({flag}). Start a fresh run (e.g. --init-from) to change it.")


def checkpoint_resume_info(path: str) -> dict[str, Any]:
    """Read before the vec env is built: decisions done, the file read, the shaping
    record (preset and the env count of the per-env decision clock), the map source
    and the curriculum state (JSON text or None)."""
    blob = load_checkpoint(path)
    refuse_weights_only(blob)
    return {"decisions_done": int(blob["trainer"]["decisions_done"]), "loaded_from": blob[LOADED_FROM_KEY],
            **shaping_record(blob), "maps": maps_of(blob), "curriculum_json": blob.get(CURRICULUM_KEY),
            "bc_aux": bc_aux_of(blob), "switches": switches_of(blob)}


def checkpoint_decisions_done(path: str) -> int:
    """Trainer decisions recorded in a checkpoint, read without building a trainer, so
    the env factory (reward-shaping anneal offset) can be made before the vec env."""
    return int(load_checkpoint(path)["trainer"]["decisions_done"])


def resume(trainer: Trainer, path: str, sensor_signature: tuple) -> dict[str, Any]:
    """Restore the trainer, then start a new session with fresh env seeds."""
    blob = load_checkpoint(path)
    refuse_weights_only(blob)
    if blob["sensor_signature_json"] != canonical_json(sensor_signature):
        raise CheckpointMismatch(f"sensor settings differ: checkpoint {blob['sensor_signature_json']} "
                                 f"vs run {canonical_json(sensor_signature)}")
    if blob["model_config"] != trainer.model.cfg:
        raise CheckpointMismatch("model config differs from the checkpoint")
    if blob["obs_spec_json"] != canonical_json(dataclasses.asdict(trainer.spec)):
        raise CheckpointMismatch(f"observation spec differs: {blob['obs_spec_json']}")
    check_shaping_matches(shaping_record(blob), trainer.run.shaping, blob[LOADED_FROM_KEY])
    check_maps_matches(maps_of(blob), trainer.run.maps, blob[LOADED_FROM_KEY])
    check_bc_matches(bc_aux_of(blob), trainer.run.bc_aux, blob[LOADED_FROM_KEY])
    check_switches_match(switches_of(blob), switches_requested(trainer.run), blob[LOADED_FROM_KEY])
    trainer.load_state_dict(blob["trainer"])
    blob["session_env_seed"] = trainer.resume_session()
    return blob


def numbered_checkpoint_path(run_dir: str, update_idx: int) -> str:
    return os.path.join(run_dir, NUMBERED_DIR, NUMBERED_PATTERN.format(update=int(update_idx)))


def numbered_checkpoints(run_dir: str) -> list[str]:
    """Numbered checkpoints of a run, oldest first (the zero-padded update sorts)."""
    return sorted(glob.glob(os.path.join(run_dir, NUMBERED_DIR, "checkpoint-u*.pt")))


def thin_numbered(run_dir: str, budget_bytes: int) -> list[str]:
    """While the numbered set is above `budget_bytes`, keep every second file (oldest
    first) plus the newest; returns the files removed."""
    removed: list[str] = []
    files = numbered_checkpoints(run_dir)
    while len(files) > 2 and sum(os.path.getsize(f) for f in files) > budget_bytes:
        keep = set(files[::2]) | {files[-1]}
        for f in files:
            if f not in keep:
                os.remove(f)
                removed.append(f)
        files = numbered_checkpoints(run_dir)
    return removed


def keep_numbered_copy(main_path: str, run_dir: str, update_idx: int, budget_bytes: int) -> tuple[str, int, list[str]]:
    """Copy the checkpoint just written at `main_path` to its numbered name (via a temp
    file, so a numbered file is never half-written), then thin the set. Returns the
    numbered path, its size in bytes, and the files removed by thinning."""
    dst = numbered_checkpoint_path(run_dir, update_idx)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + TMP_SUFFIX
    shutil.copyfile(main_path, tmp)
    os.replace(tmp, dst)
    size = os.path.getsize(dst)
    return dst, size, thin_numbered(run_dir, budget_bytes)
