"""`--init-from`: start a FRESH PPO run with model weights taken from a checkpoint file
(a behaviour-cloning weights-only checkpoint, or any trainer checkpoint).

Fresh means: new optimizer, new return scale, update 0, fresh curriculum, new run
dir. Only the model parameters are copied, with a strict load. Refused when the
file's sensor signature, model config or observation spec differs from the run's,
with ONE exception (train/egomap_growth.py): an observation spec that differs only by
extra egomap channels appended at the end (and a sensor signature that differs only
by a higher obs_version) is accepted; the first egomap conv weight then gets zero
input slices for the new channels, so the run starts computing exactly what the file
computed. A second exception (train/goal_head.py): a model config that differs only
by the goal head switched on in the run; the head is then added freshly initialised
(InitSource.goal_head_added) and every other weight is copied. The source path and its sha256 are recorded in the run config
(RunConfig.init_from, RunConfig.init_from_sha256), and any growth in
RunConfig.init_from_growth, hence in every checkpoint the run writes.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any

from .goal_head import add_goal_head_state, goal_head_added
from .egomap_growth import EgomapGrowth, egomap_growth, grow_state, obs_version_change, state_shapes
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
    # (file, run) obs_version when the sensor signatures differ only there (upward);
    # accepted only if apply_init then finds an egomap channel growth.
    obs_version_change: tuple[int, int] | None = None
    # Set by apply_init: True when the run's model has the goal head and the file's
    # does not (train/goal_head.py); the head then starts freshly initialised.
    goal_head_added: bool = False


def read_init_source(path: str, sensor_signature: tuple) -> InitSource:
    """Load the file and refuse a sensor-signature mismatch (checked before any engine
    starts). Model config and obs spec are checked in `apply_init`, once the trainer
    exists."""
    blob = load_checkpoint(path)
    src = blob[LOADED_FROM_KEY]
    run_sig = canonical_json(sensor_signature)
    version_change = None
    if blob["sensor_signature_json"] != run_sig:
        version_change = obs_version_change(blob["sensor_signature_json"], run_sig)
        if version_change is None:
            raise CheckpointMismatch(f"--init-from {src!r}: sensor settings differ: file "
                                     f"{blob['sensor_signature_json']} vs run {run_sig}")
    kind = "weights_only" if WEIGHTS_ONLY_KEY in blob else "trainer_checkpoint"
    return InitSource(path, src, sha256_file(src), kind, blob, version_change)


def growth_record(src: InitSource, growth: EgomapGrowth) -> dict[str, Any]:
    """What was grown, for RunConfig.init_from_growth (JSON-able)."""
    rec = growth.record()
    if src.obs_version_change is not None:
        rec["obs_version_from"], rec["obs_version_to"] = src.obs_version_change
    return rec


def apply_init(trainer: Trainer, src: InitSource) -> EgomapGrowth | None:
    """Copy the model weights into a freshly built trainer (strict), then refresh the
    hybrid actor copy. Nothing else of the file's trainer state is used. Returns the
    egomap growth applied (None when the specs were equal)."""
    blob = src.blob
    try:
        add_head = goal_head_added(blob["model_config"], trainer.model.cfg)
    except ValueError as e:
        raise CheckpointMismatch(f"--init-from {src.loaded_from!r}: {e}") from e
    run_spec = canonical_json(dataclasses.asdict(trainer.spec))
    state = blob["trainer"]["model"]
    growth = None
    if blob["obs_spec_json"] != run_spec:
        growth = egomap_growth(blob["obs_spec_json"], trainer.spec)
        if growth is None:
            raise CheckpointMismatch(f"--init-from {src.loaded_from!r}: observation spec differs: "
                                     f"{blob['obs_spec_json']} vs {run_spec}")
        state = grow_state(state, growth, state_shapes(trainer.learner))
    elif src.obs_version_change is not None:
        raise CheckpointMismatch(f"--init-from {src.loaded_from!r}: obs_version differs "
                                 f"{src.obs_version_change} but the observation spec is the same; refusing")
    if add_head:   # exactly the head's keys from the run's own seeded fresh build
        state = add_goal_head_state(state, trainer.learner)
    src.goal_head_added = add_head
    trainer.learner.load_state_dict(state, strict=True)
    trainer.sync_actor()
    return growth
