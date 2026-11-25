"""Evaluation on a map file the harness supplies (`--wad` / `--iwad`), e.g. a judge's
unseen map.

* The map file is loaded as given on the EVAL config (`a0_eval_env_config()`), with only
  harness fields changed (map_wad, iwad_file, maps, skill, timeout). The same
  `assert_vanilla` / `assert_same_sensors` checks as the split paths run on it.
* Map lumps: the `--map` names, or every map marker found in the file's directory
  (`env.config.wad_map_names`, which reads the directory only; nothing in the WAD runs).
  A `--map` that is not a lump of the file is refused: the engine would silently load
  the IWAD's map of that name instead (audit E1), and EVAL does not check it.
* Held-out guard: the file that supplies the maps (the PWAD, or the IWAD when there is
  no PWAD) is refused if it is under heldout_maps/ or its SHA-256 is any held-out hash
  in maps/splits.json, unless the caller acknowledges it with `--split heldout`
  (`heldout_ack`). maps/loader.py is used as is, not changed.
* One MapJob per lump (`maps=(lump,)`), in sorted lump order, so map_index and the
  episode seeds are stable.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from configs.a0_env import a0_eval_env_config
from env.config import STOCK_IWADS, iwad_path, wad_map_names
from maps import loader as L

from .checkpoint_io import sha256_file
from .config import EvalRunConfig
from .map_jobs import MapJob, MapPlan

EXTERNAL_SPLIT = "external"
EXTERNAL_STAGE = "external"
EXTERNAL_HELDOUT_STAGE = "external_heldout"   # marker: a held-out file run with --split heldout
WAD_MAGIC_IWAD = b"IWAD"
WAD_MAGIC_LEN = 4


@dataclass(frozen=True)
class ExternalDefaults:
    """Harness defaults for a map we have never seen. Not measured; chosen so a full
    vanilla level is not cut short: skill 3 is EVAL's default (Hurt Me Plenty), and
    21000 tics = 10 minutes of game time (5250 decisions at frame skip 4)."""
    skill: int = 3
    timeout_tics: int = 21_000


EXTERNAL_DEFAULTS = ExternalDefaults()


@dataclass(frozen=True)
class ExternalTarget:
    wad: str | None                 # PWAD supplying the maps (None = IWAD-only run)
    iwad: str | None                # IWAD path or stock name (None = EVAL's stock IWAD)
    maps: tuple[str, ...] = ()      # empty = every map lump in the map file
    skill: int = EXTERNAL_DEFAULTS.skill
    timeout_tics: int = EXTERNAL_DEFAULTS.timeout_tics
    heldout_ack: bool = False       # True only when the caller passed --split heldout

    def __post_init__(self) -> None:
        if self.wad is None and self.iwad is None:
            raise ValueError("external target needs --wad and/or --iwad")
        if not (1 <= int(self.skill) <= 5):
            raise ValueError(f"--skill {self.skill} outside 1-5")
        if self.timeout_tics <= 0:
            raise ValueError("--timeout-tics must be positive")


@dataclass(frozen=True)
class ExternalEvalRunConfig(EvalRunConfig):
    """EvalRunConfig whose split is "external". The other fields are validated by
    EvalRunConfig's own rules (evaluate/config.py is not changed)."""
    split: str = EXTERNAL_SPLIT

    def __post_init__(self) -> None:
        if self.split != EXTERNAL_SPLIT:
            raise ValueError(f"ExternalEvalRunConfig split must be {EXTERNAL_SPLIT!r}")
        EvalRunConfig(**{**asdict(self), "split": "val"})  # runs the shared field checks


class ExternalMapError(ValueError):
    """The supplied map file or map names cannot be evaluated."""


def resolve_iwad(arg: str | None) -> tuple[str, str | None]:
    """(EnvConfig.iwad stock name, EnvConfig.iwad_file). A path to an existing file is
    loaded as given; a bare stock name (freedoom1 / freedoom2, with or without .wad)
    that is not a file resolves to the copy bundled with vizdoom."""
    base = a0_eval_env_config()
    if arg is None:
        return base.iwad, None
    stem = os.path.splitext(os.path.basename(arg))[0].lower()
    if os.path.isfile(arg):
        with open(arg, "rb") as f:
            magic = f.read(WAD_MAGIC_LEN)
        if magic != WAD_MAGIC_IWAD:
            raise ExternalMapError(f"--iwad {arg!r} is not an IWAD (header {magic!r})")
        return (stem if stem in STOCK_IWADS else base.iwad), os.path.abspath(arg)
    if stem in STOCK_IWADS:
        return stem, None
    raise ExternalMapError(f"--iwad {arg!r}: no such file, and not a stock IWAD name {STOCK_IWADS}")


def map_source_file(target: ExternalTarget) -> str:
    """The file whose map lumps are played: the PWAD, else the IWAD."""
    if target.wad is not None:
        if not os.path.isfile(target.wad):
            raise ExternalMapError(f"--wad {target.wad!r} does not exist")
        return os.path.abspath(target.wad)
    name, path = resolve_iwad(target.iwad)
    return path if path is not None else iwad_path(name)


def discover_lumps(path: str) -> tuple[str, ...]:
    """Every map lump in a WAD, sorted (stable map_index / seeds)."""
    try:
        names = wad_map_names(path)
    except ValueError as e:
        raise ExternalMapError(str(e)) from e
    return tuple(sorted(names))


def select_lumps(path: str, requested: tuple[str, ...]) -> tuple[str, ...]:
    """Requested names (upper-cased, order kept, duplicates dropped) checked against the
    file's lumps, or all lumps when none are requested. Refuses an empty result."""
    found = discover_lumps(path)
    if not found:
        raise ExternalMapError(f"{path!r} contains no map lumps")
    if not requested:
        return found
    want = tuple(dict.fromkeys(m.upper() for m in requested))
    missing = [m for m in want if m not in found]
    if missing:
        raise ExternalMapError(f"maps {missing} are not lumps of {path!r} (it has {list(found)}); "
                               f"refused: the engine would load the IWAD's map of that name instead")
    return want


def check_heldout(path: str, ack: bool, splits: L.Splits) -> bool:
    """True if `path` is a held-out file. Raises HeldOutAccessError unless `ack`."""
    sha = sha256_file(path)
    held = L.under_heldout_dir(path) or sha in splits.heldout_hashes()
    if held and not ack:
        raise L.HeldOutAccessError(
            f"{path!r} is a HELD-OUT map file (under {L.HELDOUT_DIR}/ or a held-out sha256 in "
            f"maps/splits.json). Refused. To evaluate it deliberately, add --split heldout.")
    return held


def _split_notes(sha: str, splits: L.Splits) -> list[str]:
    for s in ("train", "val"):
        for _, e in splits.entries(s):
            if e["sha256"] == sha:
                return [f"sha256 equals {s}-split map {e['id']} (not an unseen map)"]
    return []


def plan_external(target: ExternalTarget, max_maps: int | None = None,
                  splits: L.Splits | None = None) -> MapPlan:
    splits = splits or L.load_splits()
    src = map_source_file(target)
    held = check_heldout(src, target.heldout_ack, splits)
    lumps = select_lumps(src, target.maps)
    skipped: list[str] = []
    if max_maps is not None:
        skipped = [f"{m}: beyond --max-maps {max_maps}" for m in lumps[max_maps:]]
        lumps = lumps[:max_maps]
    iwad_name, iwad_file = resolve_iwad(target.iwad)
    sha = sha256_file(src)
    notes = _split_notes(sha, splits)
    if held:
        notes.append("held-out file, run with --split heldout acknowledgement")
    if target.wad is None:
        notes.append("IWAD-only run: maps come from the IWAD itself")
    base = a0_eval_env_config()
    jobs = []
    for lump in lumps:
        cfg = base.replace(name=f"eval_{EXTERNAL_SPLIT}", map_wad=(src if target.wad is not None else None),
                           iwad=iwad_name, iwad_file=iwad_file, maps=(lump,), skill=int(target.skill),
                           episode_timeout_tics=int(target.timeout_tics))
        jobs.append(MapJob(map_id=f"{os.path.basename(src)}:{lump}",
                           stage=EXTERNAL_HELDOUT_STAGE if held else EXTERNAL_STAGE,
                           split=EXTERNAL_SPLIT, env_config=cfg, wad_sha256=sha, source_file=src,
                           staged=False, notes=list(notes)))
    return MapPlan(jobs=jobs, skipped=skipped, staging_dir=None,
                   time_limit_source=(f"--skill {target.skill}, --timeout-tics {target.timeout_tics} "
                                      f"(defaults: {asdict(EXTERNAL_DEFAULTS)})"))
