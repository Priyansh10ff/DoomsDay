"""Replay saved teacher episodes and compare labels bit for bit (TRAINING-TIME ONLY).

    python -m teachers.replay_check --data data/teacher/bc_v4map --per-stage 4 [--stages S0 S1 S2]

For each chosen indexed episode, re-runs the teacher on the same map with the same job
seed and noise (one engine) and checks that the labels (`action`, `executed_action`)
are identical to the npz (exit status), and reports which saved observation keys differ
(an env change since the dataset was collected shows up there, not in the labels) and
whether the outcome (success) matches. Used
to pin that the combat teacher leaves S0-S2 (no monsters) bit-identical.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def replay(data: str, stages: tuple[str, ...], per_stage: int) -> dict:
    from env import DoomEnv
    from maps.loader import resolve
    from teachers.collect import Job, env_config, run_episode, training_maps
    from teachers.navigator import load_route
    with open(os.path.join(data, "index.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    bundle, l2i, _ = training_maps()
    id2lump = {v: k for k, v in l2i.items()}
    chosen, seen_maps = [], set()
    for st in stages:
        sel = [r for r in rows if r["stage"] == st and r["map_id"] not in seen_maps]
        for r in sel:
            if sum(c["stage"] == st for c in chosen) >= per_stage:
                break
            if r["map_id"] in seen_maps:
                continue
            chosen.append(r)
            seen_maps.add(r["map_id"])
    env = DoomEnv(env_config(bundle, tuple(l2i)))
    out = []
    try:
        for r in chosen:
            route = load_route(resolve(r["map_id"], "train"), map_id=r["map_id"])
            ep = int(r["file"][3:9])
            job = Job(ep, id2lump[r["map_id"]], r["map_id"], r["stage"], int(r["seed"]), float(r["noise"]))
            arrays, summ = run_episode(env, route, job, 3)
            with np.load(os.path.join(data, r["file"]), allow_pickle=False) as z:
                same = {k: bool(np.array_equal(z[k], arrays[k])) for k in z.files if k in arrays}
            out.append({"file": r["file"], "map_id": r["map_id"], "stage": r["stage"],
                        "success_saved": r["success"], "success_now": summ["success"],
                        "length_saved": r["length"], "length_now": summ["length"],
                        "labels_identical": same.get("action", False) and same.get("executed_action", False),
                        "identical": all(same.values()), "differs": [k for k, v in same.items() if not v]})
    finally:
        env.close()
    return {"data": data, "episodes": out,
            "all_identical": all(e["identical"] for e in out),
            "labels_identical": all(e["labels_identical"] for e in out),
            "success_now": sum(e["success_now"] for e in out), "n": len(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stages", nargs="+", default=["S0", "S1", "S2"])
    ap.add_argument("--per-stage", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = replay(a.data, tuple(a.stages), a.per_stage)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"}))
    for e in res["episodes"]:
        print(e)
    sys.exit(0 if res["labels_identical"] else 1)
