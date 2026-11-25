"""Third pass for bc_v2 vs bc_v3: blocked / stationary states, S0 split by exit kind,
door-channel vs depth (did the observation change between the two collections?).

The closed-loop v3 failures end wedged at a wall (final coordinates 16 units off a grid
line) with move agreement ~0.05 and turn agreement ~0.02, so this measures, per
dataset, what the teacher labels when the player is blocked or not moving, and how
often a label equals the previous executed action there (the copy shortcut).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ZT = 3
N_PER_STAGE = 400


def kinds_from_closedloop() -> dict:
    out = {}
    for v in ("v2", "v3"):
        p = ROOT / f"reports/eval/2026-11-26-bc-{v}-usew8-closedloop/episodes.partial.jsonl"
        for l in open(p):
            e = json.loads(l)
            if e["stage"] == "S0":
                out[e["map_id"]] = e["anchor_kind"]
    return out


def main() -> None:
    kinds = kinds_from_closedloop()
    res = {}
    for name in ("bc_v2", "bc_v3"):
        d = ROOT / "data/teacher" / name
        idx = [json.loads(l) for l in open(d / "index.jsonl")]
        seen = defaultdict(int)
        g = defaultdict(lambda: defaultdict(list))
        for e in idx:
            st = e["stage"]
            if seen[st] >= N_PER_STAGE:
                continue
            seen[st] += 1
            z = np.load(d / e["file"])
            s, a, pa, r = z["self"], z["action"], z["prev_action"], z["rays"]
            T = len(a)
            spd = np.abs(s[:, 24])
            centre = r[:, 0, 36:44].min(1)                     # depth channel, centre columns
            blocked = (pa[:, 0] == 1) & (spd < 0.02)
            blocked[0] = False
            still = (spd < 0.02) & (np.abs(s[:, 25]) < 0.02) & (np.abs(s[:, 27]) < 0.005)
            still[0] = False
            keys = [st]
            if st == "S0":
                keys.append("S0_" + kinds.get(e["map_id"], "unknown"))
            for k in keys:
                h = g[k]
                h["n"].append(T)
                h["blocked"].append(blocked)
                h["blocked_lab_fwd"].append(a[blocked, 0] == 1)
                h["blocked_lab_turn0"].append(a[blocked, 2] == ZT)
                h["blocked_lab_fwd_turn0"].append((a[blocked, 0] == 1) & (a[blocked, 2] == ZT))
                h["still"].append(still)
                h["still_copy_all"].append((a[still] == pa[still]).all(1))
                h["still_copy_move"].append(a[still, 0] == pa[still, 0])
                h["still_copy_turn"].append(a[still, 2] == pa[still, 2])
                near = centre < 0.05
                h["near_wall"].append(near)
                h["near_lab_fwd_turn0"].append((a[near, 0] == 1) & (a[near, 2] == ZT))
                h["near_lab_turn_in_place"].append((a[near, 0] != 1) & (a[near, 2] != ZT))
                h["move"].append(a[:, 0] == 1)
                h["turn0"].append(a[:, 2] == ZT)
                h["last15_fwd"].append(a[-15:, 0] == 1)
                h["last15_turn0"].append(a[-15:, 2] == ZT)
                h["centre_depth"].append(centre)
                # door channel vs depth on lit columns
                lit = r[:, 3, :] > 0
                if lit.any():
                    h["door_minus_depth"].append((r[:, 3, :] - r[:, 0, :])[lit])
                    for row in lit:
                        if row.any():
                            w = np.diff(np.concatenate([[0], row.astype(int), [0]]))
                            h["lit_run_width"].append(np.nonzero(w == -1)[0] - np.nonzero(w == 1)[0])
            del z
        out = {}
        for k, h in sorted(g.items()):
            cat = lambda key: np.concatenate(h[key]) if h[key] else np.zeros(0)
            m = lambda key: round(float(cat(key).mean()), 4) if cat(key).size else None
            o = {"episodes": len(h["n"]), "len_mean": round(float(np.mean(h["n"])), 1),
                 "blocked_share": m("blocked"), "P(label fwd|blocked)": m("blocked_lab_fwd"),
                 "P(label turn0|blocked)": m("blocked_lab_turn0"),
                 "P(label fwd&turn0|blocked)": m("blocked_lab_fwd_turn0"),
                 "still_share": m("still"), "P(label==prev all heads|still)": m("still_copy_all"),
                 "P(move==prev|still)": m("still_copy_move"), "P(turn==prev|still)": m("still_copy_turn"),
                 "near_wall_share(centre depth<0.05)": m("near_wall"),
                 "P(fwd&turn0|near wall)": m("near_lab_fwd_turn0"),
                 "P(turn in place|near wall)": m("near_lab_turn_in_place"),
                 "fwd_share": m("move"), "turn0_share": m("turn0"),
                 "last15_fwd": m("last15_fwd"), "last15_turn0": m("last15_turn0"),
                 "centre_depth_p10_p50": [round(float(np.percentile(cat("centre_depth"), q)), 4) for q in (10, 50)]}
            if h["door_minus_depth"]:
                dd = cat("door_minus_depth")
                o["door_minus_depth_mean_p10_p90"] = [round(float(dd.mean()), 4),
                                                      round(float(np.percentile(dd, 10)), 4),
                                                      round(float(np.percentile(dd, 90)), 4)]
                lw = cat("lit_run_width")
                o["lit_run_width_mean_p50"] = [round(float(lw.mean()), 2), int(np.median(lw))]
            out[k] = o
        res[name] = out
    p = ROOT / "reports/eval/2026-11-26-bc-v3-regression-wall.json"
    p.write_text(json.dumps(res, indent=1))
    for k in res["bc_v2"]:
        print("=====", k)
        for f in res["bc_v2"][k]:
            print(f"{f}: v2 {res['bc_v2'][k][f]} | v3 {res['bc_v3'][k].get(f)}")


if __name__ == "__main__":
    main()
