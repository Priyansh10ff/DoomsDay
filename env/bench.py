"""Single-environment throughput benchmark (one DoomGame, short run).

Usage: python -m env.bench [--decisions N] [--cap S] [--timer MS|0] [--screen 1|0]
                           [--per-key] [--obs-version 1|2] [--generated LUMP] [--out file.json]
Random actions, EVAL sensors (freedoom2 MAP01, frame skip 4); --generated runs one
generated TRAIN map instead (TRAIN config, timer as given). Reports decisions/s,
step latency p50/p90/p99, the share of steps >= 14 ms (the 15.6 ms Windows timer
mode), power source and active power scheme, peak RAM of Python and vizdoom.exe.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import psutil

from env import EVAL_CONFIG, DoomEnv
from env import observation as O
from env.config import TIMER_ENV_VAR
from env.doom_env import split_vars


def peak_mb(p: psutil.Process) -> float:
    mi = p.memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2**20


def power_info() -> dict:
    out = {}
    try:
        b = psutil.sensors_battery()
        out["on_ac_power"] = None if b is None else bool(b.power_plugged)
        out["battery_percent"] = None if b is None else round(b.percent)
    except Exception as e:  # pragma: no cover
        out["battery_error"] = str(e)
    try:  # read-only query of the active power scheme
        r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=5)
        out["power_scheme"] = r.stdout.strip().split("(")[-1].rstrip(")") if r.stdout else r.stderr.strip()
    except Exception as e:  # pragma: no cover
        out["power_scheme"] = f"unavailable: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=20000)
    ap.add_argument("--cap", type=float, default=25.0, help="wall-clock cap of the timed loop (s)")
    ap.add_argument("--timer", type=int, default=1, help="Windows timer request in ms, 0 = off")
    ap.add_argument("--screen", type=int, default=1)
    ap.add_argument("--per-key", action="store_true")
    ap.add_argument("--obs-version", type=int, default=1)
    ap.add_argument("--generated", default=None, help="lump of the generated TRAIN bundle")
    ap.add_argument("--out")
    a = ap.parse_args()
    if os.environ.get(TIMER_ENV_VAR):
        raise SystemExit(f"unset {TIMER_ENV_VAR} for benchmarking; it overrides --timer")

    cfg = EVAL_CONFIG.replace(win_timer_ms=a.timer or None, screen_enabled=bool(a.screen),
                              obs_version=a.obs_version)
    if a.generated:
        from configs.a0_env import generated_maps_env_config
        from env.config import TRAIN_CONFIG
        from maps.loader import training_bundle
        bundle, lumps, _ = training_bundle()
        base = TRAIN_CONFIG.replace(screen_enabled=bool(a.screen), obs_version=a.obs_version,
                                    win_timer_ms=a.timer or None)
        cfg = generated_maps_env_config(base, bundle, tuple(sorted(lumps))).replace(maps=(a.generated,))
    proc = psutil.Process()
    res = {"timer_ms": a.timer or None, "screen_enabled": bool(a.screen), "config": cfg.name,
           "obs_version": cfg.obs_version,
           "iwad": cfg.iwad, "map": cfg.maps[0], "frame_skip": cfg.frame_skip,
           "resolution": "160x120", "n_envs": 1, "processes": 1,
           "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(), **power_info()}
    env = DoomEnv(cfg)
    try:
        res["timer_active"] = env._timer.active
        res["child_processes"] = [c.name() for c in proc.children(recursive=True)]
        env.action_space.seed(0)
        env.reset(seed=0)
        for _ in range(50):  # warm-up
            env.step(env.action_space.sample())
        steps = episodes = 0
        lat = []
        cpu0 = proc.cpu_times()
        t0 = time.perf_counter()
        while steps < a.decisions and time.perf_counter() - t0 < a.cap:
            ts = time.perf_counter()
            obs, r, te, tr, info = env.step(env.action_space.sample())
            lat.append(time.perf_counter() - ts)
            steps += 1
            if te or tr:
                episodes += 1
                env.reset()
        dt = time.perf_counter() - t0
        cpu1 = proc.cpu_times()
        lat_ms = np.array(lat) * 1000
        res["full_env"] = {
            "decisions": steps, "seconds": round(dt, 2),
            "decisions_per_s": round(steps / dt, 1),
            "tics_per_s": round(steps * cfg.frame_skip / dt, 1),
            "episodes_ended": episodes,
            "step_ms_p50_p90_p99": [round(float(x), 2) for x in np.percentile(lat_ms, [50, 90, 99])],
            "share_steps_ge_14ms": round(float((lat_ms >= 14.0).mean()), 4),
            "python_cpu_s": round((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system), 2),
        }
        if a.per_key:
            g = env.game
            rng = np.random.default_rng(1)
            states = []
            for _ in range(200):
                g.set_action([1.0, 0, 0, 0, float(rng.choice(cfg.turn_bins)), 0, 0, 0])
                g.advance_action(4, True)
                if g.is_episode_finished():
                    g.new_episode()
                states.append(g.get_state())
            tr_ = O.EntityTracker()
            cost = {}

            def timeit(name, fn):
                t = time.perf_counter()
                for s in states:
                    fn(s)
                cost[name] = round((time.perf_counter() - t) / len(states) * 1e6, 1)

            timeit("rays", lambda s: O.rays(s.depth_buffer, cfg.ray_bands))
            timeit("entities(+type,mask)", lambda s: O.entities(
                s.labels, s.labels_buffer, s.depth_buffer, tr_, cfg.max_entities))
            timeit("self(+ego,split_vars)", lambda s: O.self_vector(
                *(lambda ov: (ov.health, ov.armor, ov.selected_ammo, 0.0, ov.selected_weapon,
                              ov.weapons, O.ego_motion(*ov.vel, ov.heading, 0.0)))(
                    split_vars(s.game_variables)[0])))
            timeit("screen", lambda s: O.screen_small(s.screen_buffer))
            if cfg.door_channel:
                from env import door_cue as D
                dt_ = D.DoorTracker()
                mid = np.full((O.N_RAY_COLS,), 20.0)
                timeit("door:read_sectors", lambda s: [(x.ceiling_height, x.floor_height) for x in s.sectors])
                timeit("door:faces_now", lambda s: dt_.faces_now(s.sectors))
                timeit("door:cast", lambda s: D.door_channel(dt_.faces_now(s.sectors), 0.0, 0.0, 0.0, mid))
                res["n_sectors"] = len(states[-1].sectors)
            res["per_key_us"] = cost
        res["peak_ram_mb"] = {"python": round(peak_mb(proc), 1)}
        for c in proc.children(recursive=True):
            res["peak_ram_mb"][c.name()] = round(peak_mb(c), 1)
    finally:
        env.close()
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
