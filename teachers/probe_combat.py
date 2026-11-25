"""Engine probes for the combat teacher (TRAINING-TIME ONLY; one engine, run by hand).

    python -m teachers.probe_combat fire      # shots per attack pattern (pistol, no monsters)
    python -m teachers.probe_combat corpses   # what a monster's label looks like after it dies

`fire`: on an S0 TRAIN map (no monsters), stand still and execute an attack pattern for
N decisions (frame skip 4); count shots as pistol-ammo decrements. Answers: does a HELD
attack keep firing, and which press pattern fires most often.
`corpses`: on S3 TRAIN maps, turn toward the nearest visible monster label and hold
attack when aligned; log every monster label (category, box, own pixels) per decision
around each KILLCOUNT increase, to find a state-only "is it alive" test.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

PATTERNS = {"hold": [1], "1-0": [1, 0], "1-1-0": [1, 1, 0], "1-0-0": [1, 0, 0],
            "1-1-1-0": [1, 1, 1, 0], "1-1-1-1-0": [1, 1, 1, 1, 0]}


def _env():
    from teachers.collect import env_config, training_maps
    from env import DoomEnv
    bundle, lump_to_id, lump_to_stage = training_maps()
    return DoomEnv(env_config(bundle, tuple(lump_to_id))), lump_to_id, lump_to_stage


def probe_fire(n: int = 60) -> dict:
    env, l2i, l2s = _env()
    lump = sorted(l for l in l2i if l2s[l] == "S0")[0]
    out = {}
    try:
        for name, pat in PATTERNS.items():
            obs, info = env.reset(seed=1, options={"map": lump})
            ammo0 = list(info["ammo"])
            shots, trace = 0, []
            prev = list(info["ammo"])
            for t in range(n):
                a = np.array([0, 0, 3, pat[t % len(pat)], 0, 0], np.int64)
                obs, r, term, trunc, info = env.step(a)
                cur = list(info["ammo"])
                d = int(round(sum(prev) - sum(cur)))
                shots += max(d, 0)
                trace.append(d)
                prev = cur
            out[name] = {"shots": shots, "decisions": n, "per_trace": "".join(str(x) for x in trace),
                         "ammo_start": ammo0, "ammo_end": list(info["ammo"])}
    finally:
        env.close()
    return out


def _monster_labels(state):
    buf = state.labels_buffer
    rows = []
    for lab in state.labels:
        if lab.object_category != "Monster":
            continue
        x, y, w, h = lab.x, lab.y, lab.width, lab.height
        pix = int((buf[y:y + h, x:x + w] == lab.value).sum()) if w > 0 and h > 0 else 0
        rows.append({"id": int(lab.object_id), "name": lab.object_name, "w": int(w), "h": int(h),
                     "pix": pix, "x": round(lab.object_position_x), "y": round(lab.object_position_y),
                     "z": round(lab.object_position_z), "vx": round(lab.object_velocity_x, 1),
                     "vy": round(lab.object_velocity_y, 1)})
    return rows


def probe_corpses(n_maps: int = 4, steps: int = 400) -> list:
    env, l2i, l2s = _env()
    lumps = sorted(l for l in l2i if l2s[l] == "S3")[:n_maps]
    bins = env.cfg.turn_bins
    logs = []
    try:
        for lump in lumps:
            obs, info = env.reset(seed=3, options={"map": lump})
            hist = []
            kc = info["killcount"]
            for t in range(steps):
                st = env.game.get_state()
                mons = [m for m in _monster_labels(st) if m["pix"] > 0]
                x, y, ang = info["pos_x"], info["pos_y"], info["angle"]
                a = np.array([0, 0, 3, 0, 0, 0], np.int64)
                if mons:
                    m = min(mons, key=lambda m: math.hypot(m["x"] - x, m["y"] - y))
                    err = (math.degrees(math.atan2(m["y"] - y, m["x"] - x)) - ang + 180) % 360 - 180
                    a[2] = int(np.argmin([abs(err + 4 * b) for b in bins]))
                    a[3] = 1 if abs(err) < 4 else 0
                else:
                    a[2] = 2   # slow left scan
                hist.append({"t": t, "kc": kc, "hp": info["health"], "mons": mons, "act": a.tolist()})
                obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    break
                if info["killcount"] > kc:
                    logs.append({"map": l2i[lump], "t": t, "kill": info["killcount"],
                                 "before": hist[-3:],
                                 "after": [{"t": t + 1 + k} for k in range(0)]})
                    after = []
                    for k in range(6):   # watch the corpse for a few decisions, standing still
                        st2 = env.game.get_state()
                        after.append(_monster_labels(st2))
                        obs, r, term, trunc, info = env.step(np.array([0, 0, 3, 0, 0, 0], np.int64))
                        if term or trunc:
                            break
                    logs[-1]["after"] = after
                    kc = info["killcount"]
                    if term or trunc:
                        break
            logs.append({"map": l2i[lump], "end_t": t, "dead": bool(info.get("dead")),
                         "kills": info["killcount"]})
    finally:
        env.close()
    return logs


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "fire"
    res = probe_fire() if what == "fire" else probe_corpses()
    print(json.dumps(res, indent=1))
