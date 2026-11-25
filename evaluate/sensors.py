"""Sensor-signature checks between a checkpoint and the EVAL configuration.

The TRAINED side is rebuilt from the signature stored in the checkpoint; the
EVALUATED side is `configs.a0_env.a0_eval_env_config()` plus harness fields. The
two are compared with `env.config.assert_same_sensors`, so a checkpoint trained with
different sensors (for example screen on) is refused. The evaluated side is NOT
derived from the checkpoint: that would make the check vacuous.
"""
from __future__ import annotations

import json
from typing import Any

from env.config import OPTIONAL_SENSOR_FIELDS, SENSOR_FIELDS, EnvConfig, assert_same_sensors

DERIVED_SENSOR_KEYS: tuple[str, ...] = ("ray_bands",)


class SensorSignatureError(AssertionError):
    """The checkpoint has no usable sensor signature, or it does not match EVAL."""


def canonical_json(x: Any) -> str:
    """Same canonical form as train/checkpoint.py (tuples -> lists, sorted keys)."""
    return json.dumps(x, sort_keys=True)


def _tuplify(x: Any) -> Any:
    return tuple(_tuplify(v) for v in x) if isinstance(x, list) else x


def trained_config_from_signature(signature_json: str | None, base: EnvConfig) -> EnvConfig:
    """`base` with the sensor fields stored in the checkpoint. Raises
    SensorSignatureError if the signature is missing, has unknown or missing fields,
    or does not re-serialise to the same text (e.g. derived ray bands changed)."""
    if not signature_json:
        raise SensorSignatureError("checkpoint has no sensor signature; refusing to evaluate "
                                   "(no fallback: sensors cannot be verified)")
    pairs = json.loads(signature_json)
    stored = {k: _tuplify(v) for k, v in pairs}
    expected = set(SENSOR_FIELDS) | set(DERIVED_SENSOR_KEYS)
    required = expected - set(OPTIONAL_SENSOR_FIELDS)   # obs_version: absent = v1
    if not (required <= set(stored) <= expected):
        raise SensorSignatureError(f"checkpoint sensor fields {sorted(stored)} != current "
                                   f"sensor fields {sorted(expected)}")
    fields = {f: stored[f] for f in SENSOR_FIELDS if f in stored}
    fields.setdefault("obs_version", 1)
    trained = base.replace(name="checkpoint-sensors", **fields)
    if canonical_json(trained.sensor_signature()) != signature_json:
        raise SensorSignatureError(
            f"checkpoint signature {signature_json} does not re-serialise identically "
            f"({canonical_json(trained.sensor_signature())}); derived sensors changed since training")
    return trained


def check_sensors(trained: EnvConfig, evaluated: EnvConfig) -> None:
    """assert_vanilla on the evaluated config, then the sensor comparison."""
    evaluated.assert_vanilla()
    assert_same_sensors(trained, evaluated)
