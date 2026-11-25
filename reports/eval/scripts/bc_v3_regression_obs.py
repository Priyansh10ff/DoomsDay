"""Observation-side checks for bc_v2 vs bc_v3 (companion of bc_v3_regression.py).

Per stage: ray channel statistics, self-vector differences, and whether the ego turn
delta in self[27] at step t matches the executed action of t-1 (correct) or of t (a
one-step leak of the label). One episode at a time; first N episodes per stage.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TURN_VAL = np.array([-10, -4, -1, 0, 1, 4, 10], dtype=np.float64)
FS = 4
N_PER_STAGE = 300


def main() -> None:
    res = {}
    for name in ("bc_v2", "bc_v3"):
        d = ROOT / "data/teacher" / name
        idx = [json.loads(l) for l in open(d / "index.jsonl")]
        seen = defaultdict(int)
        acc = defaultdict(lambda: defaultdict(list))
        for e in idx:
            st = e["stage"]
            if seen[st] >= N_PER_STAGE:
                continue
            seen[st] += 1
            z = np.load(d / e["file"])
            r, s, ex, a = z["rays"], z["self"], z["executed_action"], z["action"]
            g = acc[st]
            g["ray_mean"].append(r.mean((0, 2)))                    # per channel
            g["ray_nonzero"].append((r != 0).mean((0, 2)))
            g["ray_centre"].append(r[:, :, 36:44].mean((0, 2)))
            g["self_mean"].append(s.mean(0))
            g["ent_count"].append(z["entity_mask"].sum(1).mean())
            g["ent_types"].append(np.bincount(z["entity_type"][z["entity_mask"]].ravel(), minlength=10)[:10])
            g["modality"].append(z["modality"].mean(0))
            obs_turn = s[1:, 27] * 180.0                            # degrees, heading change
            exp_prev = -FS * TURN_VAL[ex[:-1, 2]]                    # from executed t-1
            exp_same = -FS * TURN_VAL[ex[1:, 2]]                     # from executed t (leak)
            exp_lab = -FS * TURN_VAL[a[1:, 2]]                       # from label t (leak)
            g["turn_match_prev"].append(np.abs(obs_turn - exp_prev) < 1.0)
            g["turn_match_same_exec"].append(np.abs(obs_turn - exp_same) < 1.0)
            g["turn_match_label"].append(np.abs(obs_turn - exp_lab) < 1.0)
            # forward speed sign vs executed move
            g["fwd_speed_when_exec_fwd_prev"].append(s[1:, 24][ex[:-1, 0] == 1])
            del z
        out = {}
        for st, g in sorted(acc.items()):
            out[st] = {
                "episodes": seen[st],
                "ray_channel_mean": np.round(np.mean(g["ray_mean"], 0), 4).tolist(),
                "ray_channel_nonzero": np.round(np.mean(g["ray_nonzero"], 0), 4).tolist(),
                "ray_channel_centre_mean": np.round(np.mean(g["ray_centre"], 0), 4).tolist(),
                "self_mean": np.round(np.mean(g["self_mean"], 0), 3).tolist(),
                "entities_per_step": round(float(np.mean(g["ent_count"])), 3),
                "entity_type_counts_per_ep": np.round(np.mean(g["ent_types"], 0), 2).tolist(),
                "modality_mean": np.round(np.mean(g["modality"], 0), 3).tolist(),
                "turn_obs_matches_exec_t-1": round(float(np.concatenate(g["turn_match_prev"]).mean()), 4),
                "turn_obs_matches_exec_t": round(float(np.concatenate(g["turn_match_same_exec"]).mean()), 4),
                "turn_obs_matches_label_t": round(float(np.concatenate(g["turn_match_label"]).mean()), 4),
                "fwd_speed_mean_after_exec_fwd": round(float(np.concatenate(g["fwd_speed_when_exec_fwd_prev"]).mean()), 4),
            }
        res[name] = out
    p = ROOT / "reports/eval/2026-11-26-bc-v3-regression-obs.json"
    p.write_text(json.dumps(res, indent=1))
    for st in ("S0", "S1", "S2"):
        print("=====", st)
        for k in res["bc_v2"][st]:
            print(f"{k}\n  v2 {res['bc_v2'][st][k]}\n  v3 {res['bc_v3'][st][k]}")


if __name__ == "__main__":
    main()
