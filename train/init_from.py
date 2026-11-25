"""`--init-from`: start a FRESH PPO run with model weights taken from a checkpoint file
(a behaviour-cloning weights-only checkpoint, or any trainer checkpoint).

Fresh means: new optimizer, new return scale, update 0, fresh curriculum, new run
dir. Only the model parameters are copied, with a strict load. Refused when the
file's sensor signature, model config or observation spec differs from the run's.
The source path and its sha256 are recorded in the run config (RunConfig.init_from,
RunConfig.init_from_sha256), hence in every checkpoint the run writes.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any

from .checkpoint import (LOADED_FROM_KEY, WEIGHTS_ONLY_KEY, CheckpointMismatch, canonical_json,
                         load_checkpoint)
from .trainer import Trainer

HASH_CHUNK_BYTES = 1 << 20


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class InitSource:
    path: str            # the argument given
    loaded_from: str     # the file actually read (load_checkpoint may fall back to .prev)
    sha256: str          # of `loaded_from`
    kind: str            # "weights_only" (behaviour cloning) or "trainer_checkpoint"
    blob: dict[str, Any]


def read_init_source(path: str, sensor_signature: tuple) -> InitSource:
    """Load the file and refuse a sensor-signature mismatch (checked before any engine
    starts). Model config and obs spec are checked in `apply_init`, once the trainer
    exists."""
    blob = load_checkpoint(path)
    src = blob[LOADED_FROM_KEY]
    if blob["sensor_signature_json"] != canonical_json(sensor_signature):
        raise CheckpointMismatch(f"--init-from {src!r}: sensor settings differ: file "
                                 f"{blob['sensor_signature_json']} vs run {canonical_json(sensor_signature)}")
    kind = "weights_only" if WEIGHTS_ONLY_KEY in blob else "trainer_checkpoint"
    return InitSource(path, src, sha256_file(src), kind, blob)


def apply_init(trainer: Trainer, src: InitSource) -> None:
    """Copy the model weights into a freshly built trainer (strict), then refresh the
    hybrid actor copy. Nothing else of the file's trainer state is used."""
    blob = src.blob
    if blob["model_config"] != trainer.model.cfg:
        raise CheckpointMismatch(f"--init-from {src.loaded_from!r}: model config differs from this run's")
    if blob["obs_spec_json"] != canonical_json(dataclasses.asdict(trainer.spec)):
        raise CheckpointMismatch(f"--init-from {src.loaded_from!r}: observation spec differs: "
                                 f"{blob['obs_spec_json']} vs {canonical_json(dataclasses.asdict(trainer.spec))}")
    trainer.learner.load_state_dict(blob["trainer"]["model"], strict=True)
    trainer.sync_actor()
