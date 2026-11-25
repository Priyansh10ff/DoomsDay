"""Select the observation contract (configs.a0_env OBS_VERSION) from the checkpoint.

A checkpoint stores its sensor signature; `obs_version` is in it only when it is not 1.
If $DOOMSDAY_OBS_VERSION is unset, it is set (in this process only) to the
checkpoint's version before any EVAL config is built, so a v1 checkpoint runs on the v1
contract without the caller knowing about the switch. Every other sensor field still
comes from `a0_eval_env_config()` and is compared by `assert_same_sensors`, so the check
is not made vacuous. If the variable is set and disagrees, the run is refused.
"""
from __future__ import annotations

import json
import os

from configs.a0_env import OBS_VERSION_ENV_VAR

DEFAULT_SIGNATURE_OBS_VERSION = 1   # absent from a signature = v1 (env.config.OPTIONAL_SENSOR_FIELDS)


def checkpoint_obs_version(signature_json: str | None) -> int | None:
    if not signature_json:
        return None
    return int(dict(json.loads(signature_json)).get("obs_version", DEFAULT_SIGNATURE_OBS_VERSION))


def pin_obs_version(signature_json: str | None) -> int | None:
    """Returns the version pinned (or None when the signature is missing; the sensor
    check then refuses the checkpoint as before)."""
    v = checkpoint_obs_version(signature_json)
    if v is None:
        return None
    raw = os.environ.get(OBS_VERSION_ENV_VAR, "").strip()
    if raw == "":
        os.environ[OBS_VERSION_ENV_VAR] = str(v)
    elif int(raw) != v:
        raise ValueError(f"{OBS_VERSION_ENV_VAR}={raw} but the checkpoint was trained with observation "
                         f"version {v}; unset the variable or set it to {v}")
    return v
