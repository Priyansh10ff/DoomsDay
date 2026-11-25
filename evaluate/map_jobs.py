"""Which maps an evaluation plays, and the EVAL config for each.

* `val` / `heldout`: generated maps from maps/splits.json, resolved through
  `maps.loader.resolve` in mode "val" / "eval" (hash-verified). Held-out maps are
  resolved here and nowhere on a training path. External Freedoom sets are skipped
  (listed as not evaluated).
* `my_way_home`: the bundled training scenario loaded as a harness `map_wad`. It is
  IN-DISTRIBUTION (the map A0 trained on) and carries its own ACS goal script, so it
  is a sanity check, not a vanilla held-out result.

Val WADs live under maps/val/, and `EnvConfig.assert_vanilla` refuses any `map_wad`
under maps/ (a folder rule that also covers val). Without editing env/, val maps are
STAGED: copied to a temporary folder outside the repository, re-hashed against
splits.json after the copy, and refused if the hash matches any train-split map.
The staging is recorded in the summary.

Time limits: no map record carries one. Generated maps use
`configs.a0_env.GENERATED_MAPS` (skill 3, 4200 tics); my_way_home uses the skill and
time limit of the A0 training config (taken from its bundled .cfg).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field

from configs.a0_env import GENERATED_MAPS, a0_eval_env_config, a0_train_env_config
from env.config import EnvConfig
from maps import loader as L

from .checkpoint_io import sha256_file
from .config import LOADER_MODE, MY_WAY_HOME_ID, MY_WAY_HOME_STAGE, EvalRunConfig


@dataclass
class MapJob:
    map_id: str                 # log only
    stage: str                  # log only
    split: str
    env_config: EnvConfig
    wad_sha256: str | None
    source_file: str | None     # file as listed (before staging)
    staged: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class MapPlan:
    jobs: list[MapJob]
    skipped: list[str]          # entries not evaluated, with the reason
    staging_dir: str | None
    time_limit_source: str

    def cleanup(self) -> None:
        if self.staging_dir and os.path.isdir(self.staging_dir):
            shutil.rmtree(self.staging_dir, ignore_errors=True)


def _train_hashes(splits: L.Splits) -> set[str]:
    return {e["sha256"] for _, e in splits.entries("train")}


def _stage_copy(src: str, expected_sha: str, staging_dir: str, train_hashes: set[str]) -> str:
    if expected_sha in train_hashes:
        raise L.HeldOutAccessError(f"{src}: hash equals a TRAIN-split map; refused for evaluation")
    dst = os.path.join(staging_dir, os.path.basename(src))
    shutil.copyfile(src, dst)
    got = sha256_file(dst)
    if got != expected_sha:
        raise L.SplitIntegrityError(f"staged copy {dst} sha256 {got} != recorded {expected_sha}")
    return dst


def my_way_home_job(run: EvalRunConfig) -> MapJob:
    train = a0_train_env_config()
    wad = train.scenario_wad
    cfg = a0_eval_env_config().replace(
        name="eval_my_way_home", map_wad=wad, maps=train.maps, skill=train.skill,
        episode_timeout_tics=run.timeout_tics_override or train.episode_timeout_tics)
    return MapJob(map_id=MY_WAY_HOME_ID, stage=MY_WAY_HOME_STAGE, split="my_way_home", env_config=cfg,
                  wad_sha256=sha256_file(wad), source_file=wad,
                  notes=["in-distribution: the map A0 trained on",
                         "PWAD carries its ACS goal script (pays reward, then exits); not a vanilla held-out map",
                         "EVAL engine rewards: no living reward, map_exit_reward 1.0 on top of the ACS reward; "
                         "returns are not comparable to training logs"])


def plan_maps(run: EvalRunConfig, splits: L.Splits | None = None) -> MapPlan:
    if run.split == "my_way_home":
        return MapPlan(jobs=[my_way_home_job(run)], skipped=[], staging_dir=None,
                       time_limit_source="a0_train_env_config() (bundled my_way_home.cfg): skill and timeout")
    splits = splits or L.load_splits()
    mode = LOADER_MODE[run.split]
    entries = [e for s, e in splits.entries(run.split)]
    skipped = [f"{e['id']}: external {e.get('source')} set, not evaluated by this tool"
               for e in splits.data.get("external_eval", []) if e["split"] == run.split]
    if run.max_maps is not None:
        skipped += [f"{e['id']}: beyond --max-maps {run.max_maps}" for e in entries[run.max_maps:]]
        entries = entries[:run.max_maps]
    staging = tempfile.mkdtemp(prefix="doomsday-eval-") if run.split == "val" else None
    train_hashes = _train_hashes(splits)
    jobs: list[MapJob] = []
    try:
        for e in entries:
            path = L.resolve(e["id"], mode, splits)          # split assert + hash check
            split_of, _ = splits.by_id(e["id"])
            if split_of != run.split:
                raise L.SplitIntegrityError(f"{e['id']} is in split {split_of}, not {run.split}")
            staged = staging is not None
            wad = _stage_copy(path, e["sha256"], staging, train_hashes) if staged else path
            cfg = a0_eval_env_config().replace(
                name=f"eval_{run.split}", map_wad=wad, maps=(e["map"],), skill=GENERATED_MAPS.skill,
                episode_timeout_tics=run.timeout_tics_override or GENERATED_MAPS.episode_timeout_tics)
            notes = [f"staged copy of {e['file']} (env refuses map_wad under maps/)"] if staged else []
            jobs.append(MapJob(map_id=e["id"], stage=e["stage"], split=run.split, env_config=cfg,
                               wad_sha256=e["sha256"], source_file=e["file"], staged=staged, notes=notes))
    except BaseException:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return MapPlan(jobs=jobs, skipped=skipped, staging_dir=staging,
                   time_limit_source=(f"configs.a0_env.GENERATED_MAPS: skill {GENERATED_MAPS.skill}, "
                                      f"{GENERATED_MAPS.episode_timeout_tics} tics (no per-map limit in the records)"))
