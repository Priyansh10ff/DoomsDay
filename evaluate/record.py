"""Record the trained policy playing, as video of what the player sees.

    python -m evaluate.record --checkpoint <ckpt> --split val --map-id <id> --out <dir>
        [--episodes N] [--seed S] [--argmax] [--timeout-tics N] [--skill N]
    python -m evaluate.record --checkpoint <ckpt> --wad <map.wad> --map MAP01 --out <dir> ...
    python -m evaluate.record --checkpoint <ckpt> --iwad <iwad> --map MAP05 --out <dir> ...

Same EVAL config, checks (`assert_vanilla`, `assert_same_sensors` against the
checkpoint's signature) and policy path (`NetworkPolicy`, `run_episode`) as
`python -m evaluate.run`. The only difference: the env is built with
`render_mode="rgb_array"` (which changes nothing in the engine setup: the screen buffer
is always rendered at 160x120) and wrapped in `RecordingEnv`, a pass-through proxy that
copies `env.render()` after each reset/step. The observation the model receives is the
env's own, untouched; frames go to the video only. Frames are 160x120 and are upscaled
with nearest-neighbour when written.

Writes per episode `episode_XXX.mp4` / `.gif` (or PNG frames if no writer works) and
`episode_XXX.json` (outcome), plus `record_summary.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from agent.obs_spec import spec_from_spaces
from configs.a0_env import GENERATED_MAPS, a0_eval_env_config
from env.doom_env import DoomEnv
from maps import loader as L

from .checkpoint_io import check_spec_matches, load_model
from .config import POLICY_ARGMAX, POLICY_SAMPLE, EvalRunConfig
from .episode import map_exit_reward, run_episode
from .external import EXTERNAL_DEFAULTS, ExternalEvalRunConfig, ExternalTarget, plan_external
from .map_jobs import MapJob, MapPlan, _stage_copy, _train_hashes
from .obs_select import pin_obs_version
from .policies import NetworkPolicy
from .run import _jsonable, verify_configs
from .video import VideoConfig, write_video

TICRATE = 35                      # Doom game tics per second
RECORD_SPLITS = ("val",)          # generated splits the recorder accepts (never held-out)
DEFAULT_EPISODES = 1


class RecordingEnv:
    """Pass-through proxy around a DoomEnv built with render_mode="rgb_array".
    `reset` / `step` return exactly what the env returns; afterwards the current screen
    (H, W, 3) is appended to `frames`. On a terminal step the engine has no state and
    `render()` returns None: no frame is added."""

    def __init__(self, env: DoomEnv) -> None:
        assert env.render_mode == "rgb_array", "RecordingEnv needs render_mode='rgb_array'"
        self.env = env
        self.frames: list[np.ndarray] = []

    def reset(self, **kw: Any) -> tuple[Mapping[str, np.ndarray], dict]:
        out = self.env.reset(**kw)
        self.frames = []
        self._grab()
        return out

    def step(self, action: np.ndarray) -> tuple:
        out = self.env.step(action)
        self._grab()
        return out

    def _grab(self) -> None:
        f = self.env.render()
        if f is not None:
            self.frames.append(np.ascontiguousarray(f, dtype=np.uint8))


def val_plan(map_id: str, skill: int | None, timeout_tics: int | None) -> MapPlan:
    """One generated val map, hash-verified and staged outside the repo (as map_jobs)."""
    splits = L.load_splits()
    split, e = splits.by_id(map_id)
    if split != "val":
        raise L.HeldOutAccessError(f"{map_id} is in split {split!r}; the recorder plays val maps only")
    path = L.resolve(map_id, "val", splits)
    staging = tempfile.mkdtemp(prefix="doomsday-record-")
    try:
        wad = _stage_copy(path, e["sha256"], staging, _train_hashes(splits))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    cfg = a0_eval_env_config().replace(
        name="eval_val", map_wad=wad, maps=(e["map"],),
        skill=GENERATED_MAPS.skill if skill is None else skill,
        episode_timeout_tics=GENERATED_MAPS.episode_timeout_tics if timeout_tics is None else timeout_tics)
    job = MapJob(map_id=map_id, stage=e["stage"], split="val", env_config=cfg, wad_sha256=e["sha256"],
                 source_file=e["file"], staged=True, notes=[f"staged copy of {e['file']}"])
    return MapPlan(jobs=[job], skipped=[], staging_dir=staging, time_limit_source="GENERATED_MAPS or --timeout-tics")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evaluate.record", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=RECORD_SPLITS, default=None)
    p.add_argument("--map-id", help="generated map id (with --split val)")
    p.add_argument("--wad")
    p.add_argument("--iwad")
    p.add_argument("--map", action="append", default=[], dest="maps", metavar="MAPxx")
    p.add_argument("--skill", type=int, default=None)
    p.add_argument("--timeout-tics", type=int, default=None)
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    p.add_argument("--seed", type=int, default=EvalRunConfig.base_seed)
    p.add_argument("--device", default=EvalRunConfig.device)
    p.add_argument("--argmax", action="store_const", dest="policy", const=POLICY_ARGMAX, default=POLICY_SAMPLE)
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args(argv)
    ext = a.wad is not None or a.iwad is not None
    if ext == (a.split is not None):
        p.error("give exactly one of --split val --map-id <id>, or --wad/--iwad --map MAPxx")
    if a.split is not None and not a.map_id:
        p.error("--split val needs --map-id")
    if ext and a.map_id:
        p.error("--map-id is for --split val; use --map with --wad/--iwad")
    if not ext and a.maps:
        p.error("--map needs --wad or --iwad")
    if a.episodes < 1:
        p.error("--episodes must be >= 1")
    return a


def build_plan(a: argparse.Namespace) -> MapPlan:
    if a.split is not None:
        return val_plan(a.map_id, a.skill, a.timeout_tics)
    target = ExternalTarget(
        wad=a.wad, iwad=a.iwad, maps=tuple(a.maps),
        skill=EXTERNAL_DEFAULTS.skill if a.skill is None else a.skill,
        timeout_tics=EXTERNAL_DEFAULTS.timeout_tics if a.timeout_tics is None else a.timeout_tics)
    plan = plan_external(target)
    if len(plan.jobs) != 1:
        raise SystemExit(f"the recorder plays one map; pick one with --map from "
                         f"{[j.env_config.maps[0] for j in plan.jobs]}")
    return plan


def record(a: argparse.Namespace) -> dict:
    loaded = load_model(a.checkpoint, a.device)
    pin_obs_version(loaded.signature_json)          # before any EVAL config is built
    common = dict(episodes_per_map=a.episodes, policy=a.policy, base_seed=a.seed, device=a.device)
    run = EvalRunConfig(split="val", **common) if a.split else ExternalEvalRunConfig(**common)
    torch.set_num_threads(run.torch_threads)
    video_cfg = VideoConfig(write_mp4=not a.no_mp4, write_gif=not a.no_gif)
    plan = build_plan(a)
    try:
        verdict = verify_configs(plan, loaded)
        job = plan.jobs[0]
        os.makedirs(a.out, exist_ok=True)
        fps = TICRATE / job.env_config.frame_skip
        thr = map_exit_reward(job.env_config.engine_rewards)
        thr = None if thr is None else thr * run.exit_reward_fraction
        episodes = []
        env = DoomEnv(job.env_config, render_mode="rgb_array")   # the only engine instance
        try:
            check_spec_matches(loaded.spec, spec_from_spaces(env.observation_space, env.action_space))
            policy = NetworkPolicy(loaded.model, run.argmax_heads(loaded.spec.n_heads), run.device)
            rec_env = RecordingEnv(env)
            for ei in range(run.episodes_per_map):
                seed = run.episode_seed(0, ei)
                rec = run_episode(rec_env, policy, seed, thr)
                stem = os.path.join(a.out, f"episode_{ei:03d}")
                videos = write_video(rec_env.frames, stem, fps, video_cfg)
                row = {"map_id": job.map_id, "stage": job.stage, "split": job.split, "episode_index": ei,
                       "seed": seed, "policy": run.policy, **rec, "frames": len(rec_env.frames),
                       "fps": fps, "frame_size": list(rec_env.frames[0].shape), "videos": videos}
                with open(stem + ".json", "w", encoding="utf-8") as f:
                    json.dump(_jsonable(row), f, indent=1, sort_keys=True)
                episodes.append(row)
        finally:
            env.close()
        summary = {"episodes": episodes, "verification": verdict,
                   "checkpoint": {"argument": a.checkpoint, "sha256": loaded.sha256,
                                  "sensor_signature_json": loaded.signature_json},
                   "map": {"map_id": job.map_id, "lumps": list(job.env_config.maps),
                           "source_file": job.source_file, "wad_sha256": job.wad_sha256,
                           "skill": job.env_config.skill,
                           "timeout_tics": job.env_config.episode_timeout_tics, "notes": job.notes},
                   "recording": {"source": "env.render() of the EVAL env, 160x120 RGB, after each decision",
                                 "video_config": video_cfg.__dict__},
                   "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(a.out, "record_summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=1, sort_keys=True)
        return summary
    finally:
        plan.cleanup()


def main(argv: list[str] | None = None) -> int:
    s = record(parse_args(argv))
    for e in s["episodes"]:
        print(f"episode {e['episode_index']}: success {e['success']} death {e['death']} timeout {e['timeout']} "
              f"length {e['length']} -> {e['videos']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
