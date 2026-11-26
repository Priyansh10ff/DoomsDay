"""Demo dashboard: video of what the trained agent sees, remembers, attends to and does.

    python -m tools.dashboard --checkpoint <ckpt> --split val --map-id <id> --out <dir>
        [--episodes N] [--seed S] [--temperature T] [--argmax] [--timeout-tics N] [--skill N] [--live]
    python -m tools.dashboard --checkpoint <ckpt> --wad <map.wad> --map MAP01 --out <dir> ...

READ-ONLY TOOL. Same EVAL config, checks (`verify_configs`: assert_vanilla and the
checkpoint's sensor signature; `check_spec_matches`), map staging (`evaluate.record`'s
`build_plan`), episode seeds and policy path (`NetworkPolicy` -> `PolicyRunner` ->
`run_episode`) as `python -m evaluate.record`. Differences, all display-side:
  * the model is wrapped in `tools.dashboard_probe.ModelProbe`, which forwards the
    observation tensors unchanged to the model's own `step` and copies its outputs and
    the entity pooling-attention weights (forward hook, recomputed with the same module
    and checked against the model's own pooled vector to 1e-6);
  * the env is wrapped in `DashboardEnv`, the recorder's pass-through render proxy; it
    composes one dashboard frame per decision just before that decision's action is
    sent to the engine, and streams it to the MP4 writer (and optionally a live window).

Per episode: `episode_XXX.mp4`, `episode_XXX.gif` (downscaled, strided) and
`episode_XXX.json`; plus `dashboard_summary.json`. Panels: game screen with entity
attention boxes and the door-cue strip; egomap; rolling value; chosen action;
stage / step / health / keys; predicted sub-goal only if the model has a goal head.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from agent.obs_spec import EGOMAP_KEY, spec_from_spaces
from env.doom_env import DoomEnv
from evaluate.checkpoint_io import check_spec_matches, load_model
from evaluate.config import POLICY_ARGMAX, POLICY_SAMPLE, EvalRunConfig
from evaluate.episode import map_exit_reward, run_episode
from evaluate.external import ExternalEvalRunConfig
from evaluate.obs_select import pin_obs_version
from evaluate.policies import NetworkPolicy
from evaluate.record import RECORD_SPLITS, TICRATE, RecordingEnv, build_plan
from evaluate.run import _jsonable, add_temperature_arg, check_temperature_arg, decode_fields, verify_configs

from .dashboard_compose import FrameData, compose
from .dashboard_config import DEFAULT_GOAL_NAMES, GOAL_DISPLAY, GOAL_NAME_ATTRS, DashboardConfig
from .dashboard_probe import ModelProbe, StepCapture
from .dashboard_writer import EpisodeWriter, LiveView

DEFAULT_EPISODES = 1
DOOR_ROW = -1                 # v2: the door channel is the last row of `rays`
V2_RAY_EXTRA = 1


def goal_names() -> tuple[str, ...]:
    """Display names of the goal head's outputs, in id order: configs/goal_modes.py's
    GOAL_MODES translated by GOAL_DISPLAY (an unknown id keeps its own name); the
    default list if that module is absent."""
    try:
        mod = importlib.import_module("configs.goal_modes")
    except ImportError:
        return DEFAULT_GOAL_NAMES
    for attr in GOAL_NAME_ATTRS:
        val = getattr(mod, attr, None)
        if isinstance(val, (tuple, list)) and val and all(isinstance(s, str) for s in val):
            return tuple(GOAL_DISPLAY.get(s, s.replace("_", " ")) for s in val)
    return DEFAULT_GOAL_NAMES


def make_policy(model: torch.nn.Module, argmax_heads: tuple[bool, ...], device: str,
                temperature: float) -> tuple[NetworkPolicy, ModelProbe]:
    """`evaluate.run.network_policy` with the probe in the model's place: the same
    NetworkPolicy, argmax heads, device and temperature (applied by PolicyRunner)."""
    probe = ModelProbe(model, temperature)
    return NetworkPolicy(probe, argmax_heads, device, temperature=temperature), probe  # type: ignore[arg-type]


def door_columns(rays: np.ndarray, ray_bands: int) -> np.ndarray | None:
    """The v2 door row of `rays`, or None when the rays carry depth bands only."""
    return rays[DOOR_ROW] if rays.shape[0] == ray_bands + V2_RAY_EXTRA else None


def frame_data(cap: StepCapture, screen: np.ndarray | None, info: Mapping[str, Any], action: np.ndarray,
               values: list[float], turn_bins: tuple[float, ...], ray_bands: int, title: str, step: int,
               names: tuple[str, ...]) -> FrameData:
    o = cap.obs
    a = np.asarray(action, np.int64).reshape(-1)
    probs = np.array([p[i] for p, i in zip(cap.probs, a)], np.float64)
    goal = None
    if cap.goal_logits is not None:
        g = torch.softmax(torch.as_tensor(cap.goal_logits, dtype=torch.float64), dim=-1).numpy()
        k = int(np.argmax(g))
        goal = (names[k] if k < len(names) else f"goal {k}", float(g[k]))
    em = o.get(EGOMAP_KEY)
    return FrameData(
        screen=screen, door_cols=door_columns(o["rays"], ray_bands), egomap=em,
        entities=o["entities"], entity_type=o["entity_type"], entity_mask=o["entity_mask"].astype(bool),
        attn=cap.attn, attn_empty=cap.attn_empty, values=values, action=a, action_probs=probs,
        turn_bins=turn_bins, title=title, step=step,
        health=None if info.get("health") is None else float(info["health"]),
        keys_held=dict(info.get("keys_held", {}) or {}), goal=goal)


class DashboardEnv(RecordingEnv):
    """The recorder's pass-through proxy, extended: `reset` / `step` still return exactly
    what the env returns. Before each action reaches the engine, the dashboard frame of
    that decision (screen, the policy's own input, its outputs, the action) is composed
    and handed to `sink`. Frames are not accumulated."""

    def __init__(self, env: DoomEnv, probe: ModelProbe, sink: Callable[[np.ndarray, FrameData], None],
                 cfg: DashboardConfig, title: str, names: tuple[str, ...]) -> None:
        super().__init__(env)
        self.probe, self.sink, self.cfg, self.title, self.names = probe, sink, cfg, title, names
        self.turn_bins = tuple(env.cfg.turn_bins)
        self.ray_bands = len(env.cfg.ray_bands)
        self.latest: np.ndarray | None = None
        self.info: Mapping[str, Any] = {}
        self.values: list[float] = []
        self.decisions = 0
        self.last_data: FrameData | None = None
        self.attn_steps = 0

    def _grab(self) -> None:                           # keep only the newest frame
        f = self.env.render()
        self.latest = None if f is None else np.ascontiguousarray(f, dtype=np.uint8)

    def reset(self, **kw: Any) -> tuple[Mapping[str, np.ndarray], dict]:
        obs, info = super().reset(**kw)
        self.info, self.values, self.decisions, self.last_data, self.attn_steps = info, [], 0, None, 0
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        cap = self.probe.last
        assert cap is not None, "policy.act() must run before env.step()"
        self.values.append(cap.value)
        self.attn_steps += cap.attn is not None
        d = frame_data(cap, self.latest, self.info, action, self.values, self.turn_bins, self.ray_bands,
                       self.title, self.decisions, self.names)
        self.sink(compose(d, self.cfg), d)
        self.last_data = d
        self.decisions += 1
        out = self.env.step(action)
        self._grab()
        self.info = out[4]
        return out


def outcome_text(rec: Mapping[str, Any]) -> str:
    if rec["success"]:
        return "LEVEL EXIT"
    if rec["death"]:
        return "DIED"
    if rec["timeout"]:
        return "TIME LIMIT"
    return "EPISODE END"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.dashboard", description=__doc__,
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
    add_temperature_arg(p)
    p.add_argument("--live", action="store_true", help="also show the composite in a window")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args(argv)
    check_temperature_arg(p, a)
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


def run_dashboard(a: argparse.Namespace, cfg: DashboardConfig | None = None) -> dict:
    cfg = cfg or DashboardConfig()
    loaded = load_model(a.checkpoint, a.device)
    pin_obs_version(loaded.signature_json)          # before any EVAL config is built
    common = dict(episodes_per_map=a.episodes, policy=a.policy, base_seed=a.seed, device=a.device,
                  temperature=a.temperature)
    run = EvalRunConfig(split="val", **common) if a.split else ExternalEvalRunConfig(**common)
    torch.set_num_threads(run.torch_threads)
    names = goal_names()
    live = LiveView.open(cfg) if a.live else None
    live_note = None if not a.live else ("live window open" if live else LiveView.unavailable_reason())
    if a.live and live is None:
        print(f"live view unavailable: {live_note}; writing files only", file=sys.stderr)
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
            policy, probe = make_policy(loaded.model, run.argmax_heads(loaded.spec.n_heads), run.device,
                                        run.temperature)
            try:
                for ei in range(run.episodes_per_map):
                    seed = run.episode_seed(0, ei)
                    stem = os.path.join(a.out, f"episode_{ei:03d}")
                    title = (f"DoomsDay agent | {job.map_id} | stage {job.stage} | ep {ei} | seed {seed} | "
                             f"{run.policy}" + ("" if run.policy == POLICY_ARGMAX else f" T={run.temperature:g}"))
                    writer = EpisodeWriter(stem, fps, cfg, mp4=not a.no_mp4, gif=not a.no_gif)

                    def sink(img: np.ndarray, d: FrameData, _w: EpisodeWriter = writer) -> None:
                        _w.add(img)
                        if live is not None:
                            live.show(img)
                    dash = DashboardEnv(env, probe, sink, cfg, title, names)
                    try:
                        rec = run_episode(dash, policy, seed, thr)
                        if dash.last_data is not None:          # outcome card on the last frame
                            last = dash.last_data
                            last.banner = outcome_text(rec)
                            card = compose(last, cfg)
                            for _ in range(max(1, int(round(cfg.end_card_seconds * fps)))):
                                sink(card, last)
                    finally:
                        videos = writer.close()
                    row = {"map_id": job.map_id, "stage": job.stage, "split": job.split, "episode_index": ei,
                           "seed": seed, **decode_fields(run), **rec,
                           "decisions_drawn": dash.decisions, "attention_steps": dash.attn_steps,
                           "attention_max_abs_diff": probe.max_attn_diff,
                           "attention_mismatches_total": probe.attn_mismatches,
                           "goal_head": bool(dash.last_data is not None and dash.last_data.goal is not None),
                           "fps": fps, "frame_size": [cfg.height, cfg.width, 3], "videos": videos}
                    with open(stem + ".json", "w", encoding="utf-8") as f:
                        json.dump(_jsonable(row), f, indent=1, sort_keys=True)
                    episodes.append(row)
            finally:
                probe.close()
        finally:
            env.close()
        summary = {"episodes": episodes, "verification": verdict, "live": live_note,
                   "temperature_applied_by": "evaluate NetworkPolicy -> agent.inference.PolicyRunner",
                   "checkpoint": {"argument": a.checkpoint, "sha256": loaded.sha256,
                                  "sensor_signature_json": loaded.signature_json},
                   "map": {"map_id": job.map_id, "lumps": list(job.env_config.maps),
                           "source_file": job.source_file, "wad_sha256": job.wad_sha256,
                           "skill": job.env_config.skill,
                           "timeout_tics": job.env_config.episode_timeout_tics, "notes": job.notes},
                   "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(a.out, "dashboard_summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=1, sort_keys=True)
        return summary
    finally:
        plan.cleanup()
        if live is not None:
            live.close()


def main(argv: list[str] | None = None) -> int:
    s = run_dashboard(parse_args(argv))
    for e in s["episodes"]:
        print(f"episode {e['episode_index']}: success {e['success']} death {e['death']} timeout {e['timeout']} "
              f"length {e['length']} attention max diff {e['attention_max_abs_diff']:.2g} -> {e['videos']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
