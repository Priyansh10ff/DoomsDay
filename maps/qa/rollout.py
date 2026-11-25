"""200-step uniform-random rollouts through DoomEnv on sample maps (map QA only).

Train and val maps run with a TRAIN-type config that carries NO mods (no ACS, no
game_args, vanilla engine rewards); held-out maps run through EVAL_CONFIG (vanilla),
because training configs refuse heldout_maps/. One engine instance at a time.
    python -m maps.qa.rollout [--per-stage 3] [--steps 200] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from maps.loader import REPO_ROOT, load_splits, resolve

STAGES = ("S0", "S1", "S2", "S3")


def rollout(path: str, split: str, steps: int, seed: int) -> dict:
    from env import DoomEnv
    from env import observation as O
    from env.config import EVAL_CONFIG, TRAIN_CONFIG, VANILLA_ENGINE_REWARDS
    if split == "heldout":
        cfg = EVAL_CONFIG.replace(name="qa-heldout", map_wad=path, maps=("MAP01",))
    else:
        cfg = TRAIN_CONFIG.replace(name=f"qa-{split}", scenario_wad=None, map_wad=path,
                                   maps=("MAP01",), skill=3, engine_rewards=VANILLA_ENGINE_REWARDS,
                                   episode_timeout_tics=4200)
    out = {"crash": None, "steps": 0, "finite": True, "terminated": False, "truncated": False,
           "entity_types_seen": {}, "reward": 0.0}
    env = None
    try:
        env = DoomEnv(cfg)
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        seen = Counter()
        for t in range(steps):
            for k, v in obs.items():
                if np.issubdtype(np.asarray(v).dtype, np.floating) and not np.all(np.isfinite(v)):
                    out["finite"] = False
            m = np.asarray(obs["entity_mask"], bool)
            for ty in np.asarray(obs["entity_type"])[m]:
                seen[O.ENTITY_TYPES[int(ty)]] += 1
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            out["reward"] += float(r)
            out["steps"] = t + 1
            if term or trunc:
                out["terminated"], out["truncated"] = bool(term), bool(trunc)
                break
        out["entity_types_seen"] = dict(seen)
    except Exception as e:  # recorded, not hidden
        out["crash"] = f"{type(e).__name__}: {e}"
    finally:
        if env is not None:
            env.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stage", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "runs",
                                                  "2026-11-25-maps-batch1-rollouts.json"))
    a = ap.parse_args(argv)
    sp = load_splits()
    results = []
    plan = [("train", st, a.per_stage) for st in STAGES] + [("heldout", "S3", 1), ("val", "S2", 1)]
    for split, stage, n in plan:
        ents = [e for _, e in sp.entries(split) if e["stage"] == stage][:n]
        for i, e in enumerate(ents):
            mode = {"train": "train", "val": "val", "heldout": "eval"}[split]
            path = resolve(e["id"], mode, sp)
            r = rollout(path, split, a.steps, seed=1000 + i)
            r.update({"id": e["id"], "split": split, "stage": stage,
                      "placed": e["counts"]})
            results.append(r)
            print(json.dumps(r))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
