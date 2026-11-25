"""Fourth pass for bc_v2 vs bc_v3: door-channel geometry binned by centre depth, what the
student sees at USE=1 steps, and kNN turn-label consistency restricted to steps with a
lit door channel at short depth. 400 episodes per door stage (S1, S2).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
N_PER_STAGE = 400
BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 1.01)
RNG = np.random.default_rng(1)


def knn_turn(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    sq = (X * X).sum(1)
    agree = []
    for i0 in range(0, len(X), 512):
        q = X[i0:i0 + 512]
        d = sq[i0:i0 + 512, None] + sq[None, :] - 2 * q @ X.T
        d[np.arange(len(q)), i0 + np.arange(len(q))] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        agree.append((Y[nn] == Y[i0:i0 + 512, None]).mean(1))
    return round(float(np.concatenate(agree).mean()), 3)


def main() -> None:
    res = {}
    for name in ("bc_v2", "bc_v3"):
        d = ROOT / "data/teacher" / name
        idx = [json.loads(l) for l in open(d / "index.jsonl")]
        seen = defaultdict(int)
        g = defaultdict(lambda: defaultdict(list))
        for e in idx:
            st = e["stage"]
            if st == "S0" or seen[st] >= N_PER_STAGE:
                continue
            seen[st] += 1
            z = np.load(d / e["file"])
            r, a, s = z["rays"], z["action"], z["self"]
            h = g[st]
            centre_depth = r[:, 0, 36:44].min(1)
            centre_door = (r[:, 3, 36:44] > 0).mean(1)
            lit = r[:, 3, :] > 0
            width = lit.sum(1)
            for b0, b1 in zip(BINS[:-1], BINS[1:]):
                m = (centre_depth >= b0) & (centre_depth < b1) & (centre_door > 0)
                h[f"width_bin_{b0}-{b1}"].append(width[m])
            u = a[:, 4] == 1
            h["use_centre_door"].append(centre_door[u])
            h["use_centre_depth"].append(centre_depth[u])
            h["use_centre_door_lit_all8"].append(centre_door[u] == 1.0)
            # same quantities on non-USE steps with a lit centre at short depth (would-be false positives)
            near_lit = (centre_door == 1.0) & (centre_depth < 0.05)
            h["near_lit_share"].append(near_lit)
            h["use_given_near_lit"].append(u[near_lit])
            # kNN sample: steps with lit centre door at short depth
            pick = np.nonzero(near_lit)[0]
            if len(pick):
                pick = RNG.choice(pick, size=min(len(pick), 15), replace=False)
                feat = np.concatenate([r[pick].reshape(len(pick), 4, 20, 4).mean(3).reshape(len(pick), -1),
                                       s[pick]], 1)
                h["knn_X"].append(feat)
                h["knn_Y"].append(a[pick, 2])
                h["knn_M"].append(a[pick, 0])
            del z
        out = {}
        for st, h in sorted(g.items()):
            cat = lambda k: np.concatenate(h[k]) if h[k] else np.zeros(0)
            o = {"lit_width_by_centre_depth_bin(mean, n)": {
                k.replace("width_bin_", ""): [round(float(cat(k).mean()), 2) if cat(k).size else None, int(cat(k).size)]
                for k in h if k.startswith("width_bin_")}}
            o["at_USE=1: centre door-lit fraction mean"] = round(float(cat("use_centre_door").mean()), 3)
            o["at_USE=1: all 8 centre columns lit"] = round(float(cat("use_centre_door_lit_all8").mean()), 3)
            o["at_USE=1: centre depth p10/p50/p90"] = [round(float(np.percentile(cat("use_centre_depth"), q)), 4) for q in (10, 50, 90)]
            o["share of steps with centre lit & depth<0.05"] = round(float(cat("near_lit_share").mean()), 4)
            o["P(USE=1 | centre lit & depth<0.05)"] = round(float(cat("use_given_near_lit").mean()), 4)
            X, Y, M = cat("knn_X"), cat("knn_Y"), cat("knn_M")
            o["knn_on_lit_near_steps"] = {"n": int(len(Y)), "turn_agree": knn_turn(X, Y),
                                          "move_agree": knn_turn(X, M),
                                          "turn_label_entropy_bits": round(float(-(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(Y, minlength=7) / len(Y))), 3)}
            out[st] = o
        res[name] = out
    p = ROOT / "reports/eval/2026-11-26-bc-v3-regression-door.json"
    p.write_text(json.dumps(res, indent=1))
    for st in ("S1", "S2"):
        print("=====", st)
        for k in res["bc_v2"][st]:
            print(f"{k}\n  v2 {res['bc_v2'][st][k]}\n  v3 {res['bc_v3'][st][k]}")


if __name__ == "__main__":
    main()
