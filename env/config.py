"""Named environment configurations: TRAIN and EVAL (ADR 0001, 0002, amended by 0006).

Every EnvConfig field belongs to exactly one of four groups. `FIELD_GROUPS` lists
them and a test checks that every field is listed once.

1. VANILLA RULES (`RULE_FIELDS`). These are what "unmodified engine, vanilla rules,
   no mods" means. Under EVAL each must equal its vanilla value, and
   `assert_vanilla()` rejects any difference: no PWAD/ACS, no console variables or
   cheats, no altered engine reward knobs, no whole-level god-view info, no extra
   training knobs, and a stock IWAD.
2. HARNESS SETTINGS (`HARNESS_FIELDS`): map list, skill, episode time limit. The
   evaluation harness decides these. They are not mods, so they may take any legal
   value. `assert_vanilla()` checks only that the value is legal (skill 1-5, at
   least one map, a positive timeout).
3. SENSOR / ACTION PLUMBING (`SENSOR_FIELDS`): frame skip, HUD, turn bins, entity
   cap, screen on/off, observation version. These are the agent's own sensors and
   controls, not game rules, so either value is vanilla. They must be IDENTICAL between the
   configuration a model was trained with and the one it is evaluated with;
   `sensor_signature()` / `assert_same_sensors()` enforce that. The ray bands are
   derived from `render_hud` (ADR 0006) and are not a field.
4. HOST PLUMBING (`HOST_FIELDS`): name, vanilla_eval flag, the Windows timer
   request. These do not touch the game.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import struct
from dataclasses import dataclass, field

import vizdoom as vzd

# ADR 0002 turn bins, as TURN_LEFT_RIGHT_DELTA values in degrees per tic.
# Probe result: positive = turn right (clockwise, ANGLE decreases).
TURN_BINS: tuple[float, ...] = (-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0)

# ADR 0006 ruling 1: ray bands are row offsets [start, end) from the horizon row.
RAY_BAND_OFFSETS: tuple[tuple[int, int], ...] = ((-20, -4), (-4, 4), (4, 20))


def horizon_row(render_hud: bool) -> int:
    """The single place the horizon row is defined (ADR 0006). Probe: the HUD-on
    image is the HUD-off image shifted up exactly 10 rows; the direct floor fit
    converges to 49.1 (HUD on) and 59.1 (HUD off) once quantization-biased near
    pixels are excluded (docs/reference/engine-probes.md R2.1, small sample)."""
    return 50 if render_hud else 60


def ray_bands_for(render_hud: bool) -> tuple[tuple[int, int], ...]:
    h = horizon_row(render_hud)
    return tuple((h + a, h + b) for a, b in RAY_BAND_OFFSETS)


# Engine reward knobs ViZDoom exposes. Only these names may appear in
# `engine_rewards`. EVAL may only carry the bundled Freedoom value below.
ENGINE_REWARD_KNOBS = (
    "living_reward", "death_penalty", "kill_reward", "item_reward", "secret_reward",
    "frag_reward", "hit_reward", "hit_taken_penalty", "damage_made_reward",
    "damage_taken_penalty", "health_reward", "armor_reward", "map_exit_reward",
)
VANILLA_ENGINE_REWARDS: tuple[tuple[str, float], ...] = (("map_exit_reward", 1.0),)

STOCK_IWADS = ("freedoom1", "freedoom2")

# Environment variable overriding the timer request of any config:
# "0" turns it off, a positive integer sets the period in ms.
TIMER_ENV_VAR = "DOOMSDAY_WIN_TIMER_MS"


_MAP_DATA_LUMPS = frozenset({"THINGS", "TEXTMAP"})  # first lump after a Doom / UDMF map marker


@functools.lru_cache(maxsize=64)
def _wad_map_names_cached(path: str, size: int, mtime_ns: int) -> frozenset[str]:
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) != 12 or head[:4] not in (b"IWAD", b"PWAD"):
            raise ValueError(f"{path!r} is not a WAD file")
        n, off = struct.unpack("<ii", head[4:])
        f.seek(off)
        dir_bytes = f.read(16 * n)
    if len(dir_bytes) != 16 * n:
        raise ValueError(f"{path!r}: truncated WAD directory")
    names = [struct.unpack_from("<ii8s", dir_bytes, 16 * i)[2].split(b"\0", 1)[0]
             .decode("ascii", "replace").upper() for i in range(n)]
    return frozenset(names[i] for i in range(n - 1) if names[i + 1] in _MAP_DATA_LUMPS)


def wad_map_names(path: str) -> frozenset[str]:
    """Upper-case names of the map markers in a WAD (a lump followed by THINGS or
    TEXTMAP). Reads the directory only; nothing in the WAD is run."""
    try:
        st = os.stat(path)
        return _wad_map_names_cached(os.path.realpath(path), st.st_size, st.st_mtime_ns)
    except OSError as e:
        raise ValueError(f"cannot read WAD {path!r}: {e}") from e


def iwad_path(name: str) -> str:
    """Resolve a stock IWAD name to the copy bundled with the vizdoom package."""
    if name not in STOCK_IWADS:
        raise ValueError(f"unknown stock IWAD {name!r}; expected one of {STOCK_IWADS}")
    return os.path.join(os.path.dirname(vzd.__file__), f"{name}.wad")


@dataclass(frozen=True)
class EnvConfig:
    # --- host plumbing ---
    name: str
    vanilla_eval: bool
    # Windows 1 ms timer request (ADR 0006 ruling 3): on for training, off for
    # evaluation and inference. None = off. Overridden by $DOOMSDAY_WIN_TIMER_MS.
    win_timer_ms: int | None = None
    # Engine tests only (tests/env_test_helpers.py): lets a non-vanilla config load the
    # Freedoom Phase 2 campaign maps, whose geometry the env tests were probed on.
    # Never set by training code; trainers should refuse a config with it set. It does
    # NOT lift the heldout_maps/ refusal.
    phase2_engine_test: bool = False
    # --- harness settings ---
    maps: tuple[str, ...] = ("MAP01",)
    skill: int = 3
    episode_timeout_tics: int = 126000
    # ADR 0007 ruling 6: maps/IWAD supplied by the evaluation harness, loaded as given.
    map_wad: str | None = None                 # WAD supplying the maps (harness only)
    iwad_file: str | None = None               # harness IWAD path; None = stock `iwad`
    # --- sensor / action plumbing (must match between training and inference) ---
    frame_skip: int = 4
    render_hud: bool = True
    turn_bins: tuple[float, ...] = TURN_BINS
    max_entities: int = 24
    screen_enabled: bool = True
    # Observation contract (env/observation.py OBS_VERSIONS). 1 = rays (3, 80), self (28,).
    # 2 = v1 + line-of-sight door channel (rays row 4) + key bits (self 34,), and the
    # engine's sector info switched on for the door channel ONLY (env/door_cue.py).
    # Default 1 here; configs/a0_env.py OBS_VERSION decides what training/eval use.
    obs_version: int = 1
    # --- vanilla rules (must stay at these values under EVAL) ---
    iwad: str = "freedoom2"
    scenario_wad: str | None = None            # PWAD, may carry ACS
    game_args: tuple[str, ...] = ()            # console variables / cheats
    objects_info: bool = False                 # whole-level god-view -> info only
    sectors_info: bool = False                 # whole-level god-view -> info only
    # (obs_version 2 turns the engine's sector info on by itself, for the door channel
    # only; see engine_sectors_enabled. This field stays forbidden under EVAL.)
    engine_rewards: tuple[tuple[str, float], ...] = VANILLA_ENGINE_REWARDS
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def ray_bands(self) -> tuple[tuple[int, int], ...]:
        return ray_bands_for(self.render_hud)

    @property
    def door_channel(self) -> bool:
        return self.obs_version >= 2

    @property
    def engine_sectors_enabled(self) -> bool:
        """Whether the engine is asked for sector info. Under v2 it is needed by the
        line-of-sight door channel; the raw sector list then never leaves
        env/door_cue.py (no observation field, no info{} key; tested)."""
        return bool(self.sectors_info or self.door_channel)

    def replace(self, **kw) -> "EnvConfig":
        return dataclasses.replace(self, **kw)

    def sensor_signature(self) -> tuple:
        """Everything that shapes what the policy sees or how its action is applied.

        `obs_version` is appended only when it is not 1, so the v1 signature is
        byte-identical to the one stored in checkpoints made before it existed (those
        checkpoints keep resuming / evaluating under v1, and never under v2)."""
        sig = tuple((f, getattr(self, f)) for f in SENSOR_FIELDS if f not in OPTIONAL_SENSOR_FIELDS)
        sig += (("ray_bands", self.ray_bands),)
        if self.obs_version != 1:
            sig += (("obs_version", self.obs_version),)
        return sig

    def effective_timer_ms(self) -> int | None:
        raw = os.environ.get(TIMER_ENV_VAR)
        if raw is not None and raw.strip() != "":
            v = int(raw)
            return v if v > 0 else None
        return self.win_timer_ms if self.win_timer_ms and self.win_timer_ms > 0 else None

    def assert_vanilla(self) -> None:
        """Rule fields must be vanilla, harness fields must be legal; sensor and host
        fields may take any value (see module docstring). Raises AssertionError."""
        problems = []
        if not self.vanilla_eval:
            problems.append("vanilla_eval is False")
        # vanilla rules
        if self.iwad not in STOCK_IWADS:
            problems.append(f"iwad {self.iwad!r} is not a stock IWAD")
        if self.scenario_wad is not None:
            problems.append(f"scenario_wad set ({self.scenario_wad!r}); PWADs may carry ACS")
        if tuple(self.game_args) not in EVAL_ALLOWED_GAME_ARGS:
            problems.append(f"game_args {self.game_args!r} not allowed: only () or exactly "
                            f"('+viz_nocheat 1',)")
        if self.objects_info:
            problems.append("objects_info enabled (whole-level god-view)")
        # Sector info is a whole-level god-view. It is allowed under EVAL ONLY as the
        # input of the line-of-sight door channel (obs_version 2): there it is filtered
        # to door faces that agree with the depth buffer, and only the (80,) per-column
        # cue reaches the observation. The `sectors_info` field (raw sector data for
        # info{} / logging) stays forbidden.
        if self.sectors_info:
            problems.append("sectors_info enabled (whole-level god-view)")
        if self.obs_version not in (1, 2):
            problems.append(f"obs_version {self.obs_version!r} not in (1, 2)")
        if tuple(self.engine_rewards) != VANILLA_ENGINE_REWARDS:
            problems.append(f"engine_rewards altered: {self.engine_rewards!r}")
        if self.extra:
            problems.append(f"extra training knobs present: {sorted(self.extra)}")
        # harness settings: legal values only
        if not (1 <= int(self.skill) <= 5):
            problems.append(f"skill {self.skill} outside 1-5")
        if not self.maps or not all(isinstance(m, str) and m for m in self.maps):
            problems.append("maps must be a non-empty tuple of map names")
        if self.episode_timeout_tics <= 0:
            problems.append("episode_timeout_tics must be positive")
        if self.iwad_file is not None and not os.path.isfile(self.iwad_file):
            problems.append(f"iwad_file {self.iwad_file!r} does not exist")
        if self.map_wad is not None:
            if not os.path.isfile(self.map_wad):
                problems.append(f"map_wad {self.map_wad!r} does not exist")
            if _under_training_dirs(self.map_wad):
                problems.append(f"map_wad {self.map_wad!r} is under a training-map folder "
                                f"{TRAINING_MAP_DIRS}")
        if problems:
            raise AssertionError(f"config {self.name!r} is not vanilla: " + "; ".join(problems))

    def validate(self) -> None:
        for k, _ in self.engine_rewards:
            if k not in ENGINE_REWARD_KNOBS:
                raise ValueError(f"unknown engine reward knob {k!r}")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if self.obs_version not in (1, 2):
            raise ValueError(f"obs_version {self.obs_version!r} not in (1, 2)")
        if self.map_wad is not None and self.scenario_wad is not None:
            raise ValueError("map_wad and scenario_wad cannot both be set")
        if self.vanilla_eval:
            self.assert_vanilla()
        else:
            self.assert_trainable()

    def assert_trainable(self) -> None:
        """ADR 0008 (held-out split, training side). A non-vanilla config may not load
        anything under heldout_maps/ (no exemption), nor the Freedoom Phase 2 campaign
        maps (freedoom2 IWAD with no scenario/map WAD) unless `phase2_engine_test` is
        set by an engine test. Raises ValueError before any engine starts."""
        for field_name in ("scenario_wad", "map_wad", "iwad_file"):
            p = getattr(self, field_name)
            if p is not None and _under_repo_dirs(p, (HELDOUT_MAP_DIR,)):
                raise ValueError(f"{self.name!r}: {field_name} {p!r} is under {HELDOUT_MAP_DIR}/; "
                                 f"held-out maps are never loaded by a training config")
        iwad_is_phase2 = (os.path.basename(self.iwad_file).lower() == "freedoom2.wad"
                          if self.iwad_file is not None else self.iwad == "freedoom2")
        if iwad_is_phase2 and self.pwad_path() is None and not self.phase2_engine_test:
            raise ValueError(f"{self.name!r}: Freedoom Phase 2 campaign maps {self.maps} are the "
                             f"held-out set and are never trained on (ADR 0005, 0008)")
        # Audit E1 (probe `pwad_map_fallback`): a map name absent from the PWAD loads the
        # IWAD's map of that name, e.g. MAP02 over my_way_home.wad or the bundle loads
        # freedoom2 MAP02 (held-out). So every configured map must be in the PWAD.
        allowed = self.allowed_maps()
        missing = [m for m in self.maps if str(m).upper() not in allowed]
        if missing:
            raise ValueError(f"{self.name!r}: maps {missing} are not in {self.pwad_path()!r}; a training "
                             f"config may only load map lumps of its own PWAD (an absent name falls "
                             f"through to the IWAD's map)")

    def allowed_maps(self) -> frozenset[str] | None:
        """Map names a reset may select (upper case). None = any (EVAL: the harness
        decides). Training: the map markers of the PWAD, or, with no PWAD, the
        config's own `maps` tuple."""
        if self.vanilla_eval:
            return None
        pw = self.pwad_path()
        if pw is not None:
            return wad_map_names(pw)
        return frozenset(str(m).upper() for m in self.maps)

    def check_reset_map(self, name, allowed: frozenset[str] | None = None) -> None:
        """Raise ValueError if a training config may not load map `name`."""
        if self.vanilla_eval:
            return
        allowed = self.allowed_maps() if allowed is None else allowed
        if not isinstance(name, str) or name.upper() not in allowed:
            raise ValueError(f"{self.name!r}: reset map {name!r} is not a map of this training "
                             f"config ({self.pwad_path() or 'maps tuple'}); refused (audit E1)")

    def game_path(self) -> str:
        return self.iwad_file if self.iwad_file is not None else iwad_path(self.iwad)

    def pwad_path(self) -> str | None:
        """The one PWAD to load: the training scenario, or the harness's map WAD."""
        return self.scenario_wad if self.scenario_wad is not None else self.map_wad


# Under EVAL, game_args may be empty or exactly this one argument (ADR 0007 ruling 6:
# it only removes information). Exact tuple match: nothing else is whitelisted.
EVAL_ALLOWED_GAME_ARGS = ((), ("+viz_nocheat 1",))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Folders owned by map-forge for training content (CLAUDE.md). No file under them
# may be passed as an EVAL `map_wad` (ADR 0007 ruling 6).
TRAINING_MAP_DIRS = ("maps", "curriculum")
# Held-out maps (ADR 0008): generated/ and external/ subfolders. EVAL may load from
# here; no training (non-vanilla) config may.
HELDOUT_MAP_DIR = "heldout_maps"


def _under_training_dirs(path: str) -> bool:
    return _under_repo_dirs(path, TRAINING_MAP_DIRS)


def _under_repo_dirs(path: str, dirs) -> bool:
    p = os.path.normcase(os.path.realpath(path))
    for d in dirs:
        root = os.path.normcase(os.path.realpath(os.path.join(REPO_ROOT, d)))
        try:
            if os.path.commonpath([p, root]) == root:
                return True
        except ValueError:  # different drives
            continue
    return False


RULE_FIELDS = ("iwad", "scenario_wad", "game_args", "objects_info", "sectors_info",
               "engine_rewards", "extra")
HARNESS_FIELDS = ("maps", "skill", "episode_timeout_tics", "map_wad", "iwad_file")
SENSOR_FIELDS = ("frame_skip", "render_hud", "turn_bins", "max_entities", "screen_enabled",
                 "obs_version")
# Sensor fields that appear in the signature only when not at their v1 value.
OPTIONAL_SENSOR_FIELDS = ("obs_version",)
HOST_FIELDS = ("name", "vanilla_eval", "win_timer_ms", "phase2_engine_test")
FIELD_GROUPS = {"rule": RULE_FIELDS, "harness": HARNESS_FIELDS, "sensor": SENSOR_FIELDS,
                "host": HOST_FIELDS}


def assert_same_sensors(trained: EnvConfig, evaluated: EnvConfig) -> None:
    """A model may only be run with the sensor settings it was trained with."""
    a, b = dict(trained.sensor_signature()), dict(evaluated.sensor_signature())
    # Union of keys: obs_version is present only under v2, so a v1-vs-v2 pair differs here.
    diff = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    if diff:
        raise AssertionError(f"sensor settings differ between {trained.name!r} and "
                             f"{evaluated.name!r}: {diff}")


EVAL_CONFIG = EnvConfig(name="eval", vanilla_eval=True, win_timer_ms=None)


# ---------------------------------------------------------------- bundled scenarios
def _parse_vizdoom_cfg(path: str) -> dict[str, str]:
    """Plain `key = value` lines of a ViZDoom .cfg (comments and `{...}` lists skipped)."""
    out, depth = {}, 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            depth += line.count("{") - line.count("}")
            if depth or "=" not in line or "{" in line or "}" in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            out[k.lower()] = v
    return out


def bundled_scenario_config(name: str, cfg_name: str, **overrides) -> EnvConfig:
    """TRAIN config from a bundled ViZDoom scenario .cfg. Taken from the cfg: the
    scenario WAD, map, skill, episode timeout and every reward knob the cfg sets
    (passed through unshaped). Reward knobs the cfg does not set are NOT added: in
    particular no map_exit_reward unless the cfg has one (my_way_home pays its goal
    through ACS and then exits; adding map_exit_reward would pay it twice, probe
    `mwh_goal`). Our sensor/action settings are kept, not the cfg's (render_hud,
    resolution, buttons), so the policy sees the same sensors as under EVAL. The
    cfg's episode_start_time is not applied (DoomEnv has no such field)."""
    cfg_path = os.path.join(vzd.scenarios_path, f"{cfg_name}.cfg")
    kv = _parse_vizdoom_cfg(cfg_path)
    wad = os.path.join(vzd.scenarios_path, kv["doom_scenario_path"])
    rewards = tuple((k, float(kv[k])) for k in ENGINE_REWARD_KNOBS if k in kv)
    base = dict(
        name=name, vanilla_eval=False, win_timer_ms=1,
        scenario_wad=wad, maps=(kv.get("doom_map", "MAP01").upper(),),
        skill=int(kv.get("doom_skill", 3)),
        episode_timeout_tics=int(kv.get("episode_timeout", 126000)),
        engine_rewards=rewards,
    )
    base.update(overrides)
    return EnvConfig(**base)


# TRAIN (ADR 0007 ruling 7, first run): bundled `my_way_home`, bundled reward only.
# No Freedoom Phase 2 map is trained on (ADR 0005).
TRAIN_CONFIG = bundled_scenario_config("train", "my_way_home")
