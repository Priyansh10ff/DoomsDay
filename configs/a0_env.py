"""Stage A0 environment settings (ADR 0006 ruling 4: A0 trains with `screen` off).

`env` is imported lazily because it imports vizdoom; this module never starts the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os

A0_SCREEN_ENABLED: bool = False

# Observation contract used by training, BC/teacher datasets and evaluation (the ONE switch).
#   4: v3 with `egomap` (6, 32, 32): + channel 5 "locked door known" (env/egomap.py);
#                every other key byte-identical to v3. Opt-in only (DOOMSDAY_OBS_VERSION=4).
#   3 (default): v2 + `egomap` (5, 32, 32) uint8, the ego-centric occupancy map (spatial
#                memory across rooms; env/egomap.py).
#   2: rays (4, 80) = 3 depth bands + line-of-sight door channel; self (34,) =
#                v1 + keys held (b, y, r) + needs key (b, y, r). Engine sector info on,
#                for the door channel only (env/door_cue.py).
#   1: the old contract exactly: rays (3, 80), self (28,), sector info off.
# Override with the environment variable DOOMSDAY_OBS_VERSION=1, 2 or 4 (read on every call,
# so a process and the workers it spawns follow the value set when they build their
# configs; the version then travels inside EnvConfig, env/ never reads the variable).
OBS_VERSION: int = 3
OBS_VERSION_ENV_VAR = "DOOMSDAY_OBS_VERSION"
OBS_VERSION_CHOICES = (1, 2, 3, 4)


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
    One engine-level value covers the whole bundle (every stage).

    Day-run overrides (2026-11-26, Option B gap 2), read by generated_maps_settings():
      * DOOMSDAY_EPISODE_TICS=<int > 0>: episode limit in tics (unset = 4200). Because
        evaluate/map_jobs.py and evaluate/record.py read GENERATED_MAPS.episode_timeout_tics
        for val/held-out generated maps, setting it in an EVALUATION process changes that
        evaluation's time limit too (deliberately: evaluate under the limit trained with).
      * DOOMSDAY_SKILL_MIX="1,2,3,3,4": training env draws the skill uniformly from this
        list at every reset (env RNG); unset/empty = fixed `skill` (3). Training only:
        `skill_mix` is a rule field EVAL must leave empty, and the evaluation configs read
        only `skill`, never the mix."""
    skill: int = 3
    episode_timeout_tics: int = 4200
    config_name: str = "train_generated"
    skill_mix: tuple[int, ...] = ()


EPISODE_TICS_ENV_VAR = "DOOMSDAY_EPISODE_TICS"
SKILL_MIX_ENV_VAR = "DOOMSDAY_SKILL_MIX"


def generated_maps_settings() -> GeneratedMapsSettings:
    """GeneratedMapsSettings with the DOOMSDAY_EPISODE_TICS / DOOMSDAY_SKILL_MIX overrides
    applied (read at call time; unset = the defaults above)."""
    kw: dict[str, Any] = {}
    raw = os.environ.get(EPISODE_TICS_ENV_VAR, "").strip()
    if raw:
        tics = int(raw)
        if tics <= 0:
            raise ValueError(f"{EPISODE_TICS_ENV_VAR}={raw!r}: must be a positive number of tics")
        kw["episode_timeout_tics"] = tics
    raw = os.environ.get(SKILL_MIX_ENV_VAR, "").strip()
    if raw:
        mix = tuple(int(x) for x in raw.replace(";", ",").split(",") if x.strip())
        if not mix or any(not 1 <= k <= 5 for k in mix):
            raise ValueError(f"{SKILL_MIX_ENV_VAR}={raw!r}: a comma list of skills 1-5")
        kw["skill_mix"] = mix
    return GeneratedMapsSettings(**kw)


GENERATED_MAPS = generated_maps_settings()   # import-time snapshot (evaluation reads .skill/.episode_timeout_tics)


def generated_maps_env_config(base: Any, bundle_path: str, lumps: tuple[str, ...],
                              settings: GeneratedMapsSettings | None = None) -> Any:
    """`base` (the A0 TRAIN config) switched to the generated bundle. Only harness and
    rule fields change: no scenario PWAD/ACS, the bundle as `map_wad`, vanilla engine
    reward knobs (map_exit_reward 1.0 only). Sensor settings are untouched, so the
    sensor signature equals the base config's (asserted). `settings` None = resolved now
    from the environment (generated_maps_settings())."""
    from env.config import VANILLA_ENGINE_REWARDS
    if settings is None:
        settings = generated_maps_settings()
    if not lumps:
        raise ValueError("generated maps: the bundle has no map lumps")
    cfg = base.replace(name=settings.config_name, scenario_wad=None, map_wad=bundle_path,
                       maps=tuple(lumps), skill=settings.skill,
                       episode_timeout_tics=settings.episode_timeout_tics,
                       skill_mix=tuple(settings.skill_mix),
                       engine_rewards=VANILLA_ENGINE_REWARDS)
    assert cfg.sensor_signature() == base.sensor_signature(), \
        "generated-maps config must keep the A0 sensor settings"
    cfg.assert_trainable()  # refuses heldout_maps/ (env side, independent of maps.loader)
    return cfg
