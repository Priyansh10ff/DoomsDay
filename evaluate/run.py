"""Evaluation entry point.

    python -m evaluate.run --checkpoint <path> --split val|heldout|my_way_home
        [--episodes-per-map N] [--max-maps M] [--out <dir>] [--sample|--argmax]
        [--random-policy] [--temperature T] [--seed S] [--device cpu]

--temperature T (sampling only; default 1.0 = plain sampling, exclusive with --argmax
and --random-policy): every head samples from softmax(logits / T). It is recorded in
each results row and in the summary's run_config.

    python -m evaluate.run --checkpoint <path> --wad <map.wad> [--map MAP01 ...]
        [--iwad <iwad path or freedoom1|freedoom2>] [--skill N] [--timeout-tics N] ...
    python -m evaluate.run --checkpoint <path> --iwad freedoom2.wad --map MAP05 ...

A map file the harness supplies (`--wad` / `--iwad`, see evaluate/external.py) runs on
the same EVAL config, checks and policy path; with no --map every map lump in the
file is played. Rows are written with split "external". A held-out file is refused
unless --split heldout is also given (as an explicit acknowledgement).

Order of checks, all before any episode: maps resolved and hash-verified; checkpoint
loaded (refused without a sensor signature, no fallback); for every map config
`assert_vanilla` and `assert_same_sensors(trained-from-checkpoint, a0_eval_env_config)`;
the env's observation/action spec equals the checkpoint's. One engine instance at a
time (each map's env is closed before the next is built).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from typing import Any

import gymnasium
import numpy as np
import torch
import vizdoom as vzd

from agent.inference import DEFAULT_TEMPERATURE, check_temperature
from agent.obs_spec import spec_from_spaces
from configs.a0_env import a0_eval_env_config
from env.doom_env import DoomEnv

from .checkpoint_io import LoadedCheckpoint, check_spec_matches, load_model, sha256_file
from .config import POLICY_ARGMAX, POLICY_RANDOM, POLICY_SAMPLE, SPLIT_CHOICES, EvalRunConfig
from .episode import map_exit_reward, run_episode
from .external import EXTERNAL_DEFAULTS, ExternalEvalRunConfig, ExternalTarget, plan_external
from .map_jobs import MapPlan, plan_maps
from .obs_select import pin_obs_version
from .policies import NetworkPolicy, RandomPolicy
from .sensors import check_sensors, trained_config_from_signature
from .stats import summarize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "reports", "eval")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (x != x):
        return None
    return x


def decode_fields(run: EvalRunConfig) -> dict:
    """Per-row decoding record: the policy and its temperature (None for the random
    baseline, which has no network heads)."""
    return {"policy": run.policy, "temperature": None if run.policy == POLICY_RANDOM else run.temperature}


def network_policy(run: EvalRunConfig, loaded: LoadedCheckpoint) -> NetworkPolicy:
    return NetworkPolicy(loaded.model, run.argmax_heads(loaded.spec.n_heads), run.device,
                         temperature=run.temperature)


def verify_configs(plan: MapPlan, loaded: LoadedCheckpoint | None) -> str:
    """Raises on any vanilla or sensor failure. Returns a one-line verdict."""
    if loaded is None:
        for job in plan.jobs:
            job.env_config.assert_vanilla()
        return "random policy: assert_vanilla passed on every map config; no checkpoint, sensor check not applicable"
    trained = trained_config_from_signature(loaded.signature_json, a0_eval_env_config())
    for job in plan.jobs:
        check_sensors(trained, job.env_config)
    return "assert_vanilla and assert_same_sensors(checkpoint signature, a0_eval_env_config) passed on every map config"


def evaluate(run: EvalRunConfig, out_dir: str, loaded: LoadedCheckpoint | None,
             checkpoint_arg: str | None = None, plan: MapPlan | None = None) -> dict:
    """`plan`: a prebuilt map plan (external maps); None = plan the run's split."""
    if (loaded is None) != (run.policy == POLICY_RANDOM):
        raise ValueError("a checkpoint is required unless --random-policy is given (and not allowed with it)")
    torch.set_num_threads(run.torch_threads)
    plan = plan if plan is not None else plan_maps(run)
    try:
        verdict = verify_configs(plan, loaded)
        os.makedirs(out_dir, exist_ok=True)
        rows: list[dict] = []
        t0 = time.time()
        with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
            for mi, job in enumerate(plan.jobs):
                env = DoomEnv(job.env_config)       # the only engine instance alive
                try:
                    if loaded is not None:
                        check_spec_matches(loaded.spec, spec_from_spaces(env.observation_space, env.action_space))
                        policy = network_policy(run, loaded)
                    else:
                        policy = RandomPolicy(env.action_space)
                    thr = map_exit_reward(job.env_config.engine_rewards)
                    thr = None if thr is None else thr * run.exit_reward_fraction
                    for ei in range(run.episodes_per_map):
                        seed = run.episode_seed(mi, ei)
                        rec = run_episode(env, policy, seed, thr)
                        row = {"map_id": job.map_id, "stage": job.stage, "split": job.split,
                               "map_index": mi, "episode_index": ei, "seed": seed,
                               **decode_fields(run), **rec}
                        rows.append(row)
                        f.write(json.dumps(_jsonable(row)) + "\n")
                        f.flush()
                finally:
                    env.close()
        wall = time.time() - t0
        summary = {
            "summary": summarize(rows, run.ci_z),
            "interval": "Wilson score interval, 95 %",
            "verification": verdict,
            "meta": _meta(run, plan, loaded, checkpoint_arg, wall),
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=1, sort_keys=True)
        return summary
    finally:
        plan.cleanup()


def _meta(run: EvalRunConfig, plan: MapPlan, loaded: LoadedCheckpoint | None,
          checkpoint_arg: str | None, wall: float) -> dict:
    iwads = {j.env_config.game_path() for j in plan.jobs}
    return {
        "run_config": dataclasses.asdict(run),
        "seed_rule": "seed = base_seed + map_index * seed_stride_map + episode_index; env.reset(seed=seed); "
                     "policy generator seeded with the same value, fresh per episode",
        "reproducibility_note": "per-episode engine reproducibility under set_seed is not established by any "
                                "probe (ADR 0007-recipe Decision 5.2)",
        "argmax_heads": None if loaded is None else list(run.argmax_heads(loaded.spec.n_heads)),
        "checkpoint": None if loaded is None else {
            "argument": checkpoint_arg, "loaded_from": loaded.loaded_from, "sha256": loaded.sha256,
            "sensor_signature_json": loaded.signature_json, "obs_spec_json": loaded.obs_spec_json,
            "model_config": loaded.model_config,
            "run_config": json.loads(loaded.run_config_json) if loaded.run_config_json else None},
        "maps": [{"map_id": j.map_id, "stage": j.stage, "lumps": list(j.env_config.maps),
                  "source_file": j.source_file, "wad_sha256": j.wad_sha256, "staged": j.staged,
                  "notes": j.notes, "env_config": _jsonable(dataclasses.asdict(j.env_config))}
                 for j in plan.jobs],
        "maps_skipped": plan.skipped,
        "time_limit_source": plan.time_limit_source,
        "iwad_sha256": {p: sha256_file(p) for p in sorted(iwads)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "vizdoom": getattr(vzd, "__version__", "unknown"), "numpy": np.__version__,
                     "gymnasium": gymnasium.__version__},
        "engine_instances": 1,
        "wall_seconds": wall,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evaluate.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--split", choices=SPLIT_CHOICES, default=None,
                   help="generated split; with --wad/--iwad only 'heldout' is accepted, as the "
                        "acknowledgement that a held-out file may be run")
    add_external_args(p)
    p.add_argument("--episodes-per-map", type=int, default=EvalRunConfig.episodes_per_map)
    p.add_argument("--max-maps", type=int, default=None)
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=EvalRunConfig.base_seed)
    p.add_argument("--device", default=EvalRunConfig.device)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sample", action="store_const", dest="policy", const=POLICY_SAMPLE)
    g.add_argument("--argmax", action="store_const", dest="policy", const=POLICY_ARGMAX)
    g.add_argument("--random-policy", action="store_const", dest="policy", const=POLICY_RANDOM)
    p.set_defaults(policy=POLICY_SAMPLE)
    add_temperature_arg(p)
    a = p.parse_args(argv)
    check_target_args(p, a)
    check_temperature_arg(p, a)
    return a


def add_temperature_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, default=None, metavar="T",
                   help="sampled actions from softmax(logits / T), the same T for every head "
                        f"(default {DEFAULT_TEMPERATURE} = plain sampling); not with --argmax / --random-policy")


def check_temperature_arg(p: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    """--temperature needs sampling; fills the default (1.0) when it is not given."""
    if a.temperature is not None:
        if a.policy != POLICY_SAMPLE:
            p.error(f"--temperature cannot be combined with --{a.policy.replace('random', 'random-policy')}")
        try:
            check_temperature(a.temperature)
        except ValueError as e:
            p.error(str(e))
    else:
        a.temperature = DEFAULT_TEMPERATURE


def add_external_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wad", help="map WAD (PWAD) to evaluate on, loaded as given on the vanilla engine")
    p.add_argument("--iwad", help="IWAD path, or freedoom1 / freedoom2 (bundled with vizdoom); "
                                  "alone (no --wad) its own maps are played")
    p.add_argument("--map", action="append", default=[], dest="maps", metavar="MAPxx",
                   help="map lump to play (repeatable); default: every map lump in the file")
    p.add_argument("--skill", type=int, default=None,
                   help=f"external maps only (default {EXTERNAL_DEFAULTS.skill})")
    p.add_argument("--timeout-tics", type=int, default=None,
                   help=f"external maps only (default {EXTERNAL_DEFAULTS.timeout_tics})")


def is_external(a: argparse.Namespace) -> bool:
    return a.wad is not None or a.iwad is not None


def check_target_args(p: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    """--split and --wad/--iwad are exclusive, except --split heldout, which with
    --wad/--iwad only acknowledges that a held-out file may be run."""
    if is_external(a):
        if a.split not in (None, "heldout"):
            p.error(f"--split {a.split} cannot be combined with --wad/--iwad "
                    f"(only --split heldout, to acknowledge a held-out file)")
        return
    if a.split is None:
        p.error("one of --split or --wad/--iwad is required")
    for flag, val in (("--map", a.maps), ("--skill", a.skill), ("--timeout-tics", a.timeout_tics)):
        if val:
            p.error(f"{flag} needs --wad or --iwad")


def external_target(a: argparse.Namespace) -> ExternalTarget:
    return ExternalTarget(
        wad=a.wad, iwad=a.iwad, maps=tuple(a.maps),
        skill=EXTERNAL_DEFAULTS.skill if a.skill is None else a.skill,
        timeout_tics=EXTERNAL_DEFAULTS.timeout_tics if a.timeout_tics is None else a.timeout_tics,
        heldout_ack=(a.split == "heldout"))


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    ext = is_external(a)
    common = dict(episodes_per_map=a.episodes_per_map, max_maps=a.max_maps, policy=a.policy,
                  base_seed=a.seed, device=a.device, temperature=a.temperature)
    run = ExternalEvalRunConfig(**common) if ext else EvalRunConfig(split=a.split, **common)
    if run.policy == POLICY_RANDOM and a.checkpoint:
        raise SystemExit("--random-policy takes no --checkpoint")
    if run.policy != POLICY_RANDOM and not a.checkpoint:
        raise SystemExit("--checkpoint is required (or --random-policy for the baseline)")
    loaded = load_model(a.checkpoint, run.device) if a.checkpoint else None
    if loaded is not None:
        pin_obs_version(loaded.signature_json)  # before any EVAL config is built
    plan = plan_external(external_target(a), a.max_maps) if ext else None
    out = a.out or os.path.join(DEFAULT_OUT_ROOT, f"{time.strftime('%Y%m%d-%H%M%S')}-{run.policy}-{run.split}")
    s = evaluate(run, out, loaded, a.checkpoint, plan=plan)
    o = s["summary"]["overall"]
    tag = f" T={run.temperature}" if run.temperature != DEFAULT_TEMPERATURE else ""
    print(f"{run.split} {run.policy}{tag}: {o['successes']}/{o['episodes']} success "
          f"(Wilson 95% {o['success_ci95_wilson'][0]:.3f}-{o['success_ci95_wilson'][1]:.3f}), "
          f"death {o['death_rate']:.3f}, timeout {o['timeout_rate']:.3f}, mean length {o['mean_length']:.1f}, "
          f"exit_unknown {o['exit_unknown']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
