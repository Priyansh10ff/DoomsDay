"""Stage A0 environment settings (ADR 0006 ruling 4: A0 trains with `screen` off).

`env` is imported lazily because it imports vizdoom; this module never starts the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os

A0_SCREEN_ENABLED: bool = False

# Observation contract used by training, BC/teacher datasets and evaluation (the ONE switch).
#   3 (default): v2 + `egomap` (5, 32, 32) uint8, the ego-centric occupancy map (spatial
#                memory across rooms; env/egomap.py).
#   2: rays (4, 80) = 3 depth bands + line-of-sight door channel; self (34,) =
#                v1 + keys held (b, y, r) + needs key (b, y, r). Engine sector info on,
#                for the door channel only (env/door_cue.py).
#   1: the old contract exactly: rays (3, 80), self (28,), sector info off.
# Override with the environment variable DOOMSDAY_OBS_VERSION=1 or 2 (read on every call,
# so a process and the workers it spawns follow the value set when they build their
# configs; the version then travels inside EnvConfig, env/ never reads the variable).
OBS_VERSION: int = 3
OBS_VERSION_ENV_VAR = "DOOMSDAY_OBS_VERSION"
OBS_VERSION_CHOICES = (1, 2, 3)


def obs_version() -> int:
    raw = os.environ.get(OBS_VERSION_ENV_VAR, "").strip()
    v = int(raw) if raw else OBS_VERSION
    if v not in OBS_VERSION_CHOICES:
        raise ValueError(f"{OBS_VERSION_ENV_VAR}={raw!r}: observation version must be one of "
                         f"{OBS_VERSION_CHOICES}")
    return v


def a0_train_env_config() -> Any:
    """env.config.TRAIN_CONFIG with the screen sensor switched off for Stage A0, at the
    selected observation version."""
    from env.config import TRAIN_CONFIG
    return TRAIN_CONFIG.replace(screen_enabled=A0_SCREEN_ENABLED, obs_version=obs_version())


def a0_eval_env_config() -> Any:
    """env.config.EVAL_CONFIG with the same sensor settings as A0 training."""
    from env.config import EVAL_CONFIG
    return EVAL_CONFIG.replace(screen_enabled=A0_SCREEN_ENABLED, obs_version=obs_version())


# ---------------------------------------------------------------- generated maps (map-forge)
@dataclass(frozen=True)
class GeneratedMapsSettings:
    """Harness-group settings for training on the generated TRAIN bundle
    (maps/bundles/train-all.wad, through maps.loader in training mode).

    Neither value is recorded by maps/splits.json or curriculum/config.json, so both
    are worker choices, NOT measured or ratified:
      * skill 3: EnvConfig's and EVAL's default. The my_way_home TRAIN config carries
        skill 5 from its bundled .cfg, which on maps with monsters is Nightmare
        (fast, respawning monsters).
      * 4200 tics (1050 decisions at frame skip 4) = 2 x my_way_home's 2100. Map
        diagonals in the bundle are 889-2717 units (mean ~1720, measured from the
        VERTEXES lumps); keyed stages need the key detour and a backtrack, so the
        limit is set above my_way_home's, which spans a single comparable map.
    One engine-level value covers the whole bundle (every stage)."""
    skill: int = 3
    episode_timeout_tics: int = 4200
    config_name: str = "train_generated"


GENERATED_MAPS = GeneratedMapsSettings()


def generated_maps_env_config(base: Any, bundle_path: str, lumps: tuple[str, ...],
                              settings: GeneratedMapsSettings = GENERATED_MAPS) -> Any:
    """`base` (the A0 TRAIN config) switched to the generated bundle. Only harness and
    rule fields change: no scenario PWAD/ACS, the bundle as `map_wad`, vanilla engine
    reward knobs (map_exit_reward 1.0 only). Sensor settings are untouched, so the
    sensor signature equals the base config's (asserted)."""
    from env.config import VANILLA_ENGINE_REWARDS
    if not lumps:
        raise ValueError("generated maps: the bundle has no map lumps")
    cfg = base.replace(name=settings.config_name, scenario_wad=None, map_wad=bundle_path,
                       maps=tuple(lumps), skill=settings.skill,
                       episode_timeout_tics=settings.episode_timeout_tics,
                       engine_rewards=VANILLA_ENGINE_REWARDS)
    assert cfg.sensor_signature() == base.sensor_signature(), \
        "generated-maps config must keep the A0 sensor settings"
    cfg.assert_trainable()  # refuses heldout_maps/ (env side, independent of maps.loader)
    return cfg
