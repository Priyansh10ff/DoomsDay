"""Observation v3 pin (env-engineer, 2026-11-26): v3 must stay byte-identical while v4
(locked-door egomap channel) is added. The golden file was captured from the code at
HEAD 3bb7a24 BEFORE any v4 edit (`python tests/obs_v3_golden.py --write`).

Two sequences, hashed per observation key per step (sha256 of the raw array bytes plus
dtype and shape):
  * unit: env.egomap.EgoMap update/crop on synthetic readings (no engine);
  * engine: the synthetic corridor with a blue-locked door (special 26), walked up to the
    door and USE pressed there without the key (needs-key pulses), then turns and backs off.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "obs_v3_golden.json")

FWD = [1, 0, 3, 0, 0, 0]
STILL = [0, 0, 3, 0, 0, 0]
USE = [0, 0, 3, 0, 1, 0]
BACK = [2, 0, 3, 0, 0, 0]
R16 = [0, 0, 5, 0, 0, 0]
PULSES: list = []   # capture only: needs-key pulse per tail step
L40 = [0, 0, 0, 0, 0, 0]


def _h(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()[:24]


def unit_hashes() -> list[str]:
    from env import egomap as E
    from env import observation as O
    m = E.EgoMap()
    t = np.abs(E._COL_TAN)
    rng = np.random.default_rng(7)
    out = []
    ents = np.zeros((24, 8), np.float32)
    types = np.zeros(24, np.int64)
    mask = np.zeros(24, bool)
    ents[0, E.U_COL], ents[0, E.D_COL] = 0.2, (E.DEPTH_PER_UNIT * 240.0 + E.DEPTH_OFFSET) / 255.0
    types[0], mask[0] = O.T["key-blue"], True
    for k in range(30):
        half = 64.0 + 8 * (k % 3)
        fwd = np.where(t > 1e-9, half / np.maximum(t, 1e-9), np.inf)
        fwd = np.minimum(fwd, 200.0 + 25 * k)
        lv = np.clip(np.rint(E.DEPTH_PER_UNIT * fwd + E.DEPTH_OFFSET), 1, 70).astype(np.float64)
        lv[rng.integers(0, 80, 5)] = 0
        door = (np.abs(E._COL_TAN) < 0.2 * (k % 2)).astype(np.float32)
        held = [k > 20, False, False]
        m.update(13.0 + 17.0 * k, -5.0 + 3.0 * k, 10.0 + 11.0 * k, lv, door, ents, types, mask, held)
        out.append(_h(m.crop()))
    return out


def engine_run(tmp_dir: str, actions: list | None = None):
    """(actions, per-step {key: hash}) on the locked-door corridor under observation v3.
    With `actions` None the driver chooses them (capture); otherwise they are replayed."""
    sys.path.insert(0, HERE)
    from egomap_maps import corridor_config, corridor_wad
    from env import DoomEnv
    wad = corridor_wad(os.path.join(tmp_dir, "locked_v3.wad"), door_special=26)
    env = DoomEnv(corridor_config(wad, obs_version=3))
    steps, acts = [], []
    try:
        obs, info = env.reset(seed=0)
        steps.append({k: _h(v) for k, v in sorted(obs.items())})
        if actions is None:
            plan = []
            while info["pos_x"] < 580 and len(plan) < 40:
                plan.append(FWD)
                obs, *_r, info = env.step(np.array(FWD))
                steps.append({k: _h(v) for k, v in sorted(obs.items())})
            tail = [STILL, USE, STILL, USE, USE, STILL, R16, USE, L40, BACK, BACK, FWD, USE, STILL]
            for a in tail:
                obs, *_r, info = env.step(np.array(a))
                steps.append({k: _h(v) for k, v in sorted(obs.items())})
                PULSES.append(bool(obs["self"][-3:].any()))
            acts = plan + tail
        else:
            for a in actions:
                obs, *_r, info = env.step(np.array(a))
                steps.append({k: _h(v) for k, v in sorted(obs.items())})
            acts = actions
    finally:
        env.close()
    return acts, steps


if __name__ == "__main__" and "--write" in sys.argv:
    import tempfile
    sys.path.insert(0, os.path.dirname(HERE))
    with tempfile.TemporaryDirectory() as d:
        acts, steps = engine_run(d)
    data = {"source": "HEAD 3bb7a24, before v4 edits", "unit": unit_hashes(),
            "engine_actions": acts, "engine": steps}
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)
    print("wrote", GOLDEN, len(acts), "actions; pulses", PULSES)
