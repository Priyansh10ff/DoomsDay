"""Data-only comparison of the teacher datasets bc_v2 and bc_v3 (why BC on v3 plays worse).

CPU-light: one episode file at a time, only labels / small per-step features kept in
memory, a few thousand ray samples for a nearest-neighbour label-consistency probe.
No engine, no torch. Usage: python reports/eval/scripts/bc_v3_regression.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = {"v2": ROOT / "data/teacher/bc_v2", "v3": ROOT / "data/teacher/bc_v3"}
HEADS = ("move", "strafe", "turn", "attack", "use", "speed")
SIZES = (3, 3, 7, 2, 2, 2)
ZT = 3  # zero-turn bin index of TURN_BINS (-10,-4,-1,0,1,4,10)
TURN_VAL = np.array([-10, -4, -1, 0, 1, 4, 10])
RNG = np.random.default_rng(0)


def runs(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs."""
    out, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def load(name: str):
    d = DATA[name]
    idx = [json.loads(l) for l in open(d / "index.jsonl")]
    per = defaultdict(lambda: defaultdict(list))
    shapes, dtypes = {}, {}
    self_sum = defaultdict(lambda: 0.0)
    self_sq = defaultdict(lambda: 0.0)
    self_n = defaultdict(int)
    ray_samples = defaultdict(list)  # stage -> list of (feat, label, kind)
    const_check = {}
    for e in idx:
        z = np.load(d / e["file"])
        st = e["stage"]
        a, ex, pa = z["action"], z["executed_action"], z["prev_action"]
        T = len(a)
        if not shapes:
            shapes = {k: z[k].shape[1:] for k in z.files}
            dtypes = {k: str(z[k].dtype) for k in z.files}
        for k in z.files:
            v = z[k]
            lo, hi = (float(v.min()), float(v.max())) if v.size else (0, 0)
            c = const_check.get(k)
            const_check[k] = (min(lo, c[0]), max(hi, c[1])) if c else (lo, hi)
        g = per[st]
        g["len"].append(T)
        g["action"].append(a)
        g["exec"].append(ex)
        g["prev"].append(pa)
        g["unstick"].append(z["label_unstick"])
        g["done_ok"].append(bool(z["done"][-1]) and int(z["done"].sum()) == 1)
        g["success"].append(bool(e["success"]))
        # prev_action alignment: prev[t] must equal executed[t-1]
        g["prev_align"].append(float((pa[1:] == ex[:-1]).all(1).mean()) if T > 1 else 1.0)
        g["prev0"].append(pa[0])
        s = z["self"].astype(np.float64)
        self_sum[st] = self_sum[st] + s.sum(0)
        self_sq[st] = self_sq[st] + (s * s).sum(0)
        self_n[st] += T
        # ray samples for the label-consistency probe: depth channel 0, 80 columns -> 20
        if RNG.random() < 0.35:
            r = z["rays"].astype(np.float32)  # (T, 4, 80)
            feat = np.concatenate([r[:, :, :].reshape(T, 4, 20, 4).mean(3).reshape(T, -1),
                                   s[:, :].astype(np.float32)], 1)
            pick = RNG.choice(T, size=min(T, 12), replace=False)
            for t in pick:
                ray_samples[st].append((feat[t], a[t].copy(), pa[t].copy()))
        del z
    return idx, per, shapes, dtypes, self_sum, self_sq, self_n, ray_samples, const_check


def head_share(A: np.ndarray, h: int) -> list[float]:
    return [round(float((A[:, h] == k).mean()), 4) for k in range(SIZES[h])]


def stage_stats(g) -> dict:
    A = np.concatenate(g["action"]); E = np.concatenate(g["exec"]); P = np.concatenate(g["prev"])
    U = np.concatenate(g["unstick"])
    L = np.array(g["len"])
    out = {"episodes": len(L), "steps": int(L.sum()), "len_mean": round(float(L.mean()), 1),
           "len_p50": int(np.median(L)), "len_p90": int(np.percentile(L, 90)),
           "success": round(float(np.mean(g["success"])), 3),
           "done_only_last": round(float(np.mean(g["done_ok"])), 3),
           "prev_action_aligned_to_exec_t-1": round(float(np.mean(g["prev_align"])), 4),
           "prev_action_t0_nonzero": round(float(np.mean([np.any(p != np.array([0, 0, ZT, 0, 0, 0])) for p in g["prev0"]])), 3),
           "label_unstick_share": round(float(U.mean()), 4)}
    for h, n in enumerate(HEADS):
        out[f"share_{n}"] = head_share(A, h)
    # jitter: label change between consecutive steps within an episode
    jit = defaultdict(list); copy = defaultdict(list); exdiff = defaultdict(list)
    tip, idle, mv0_runs, turn_flip, zero_turn_fwd = [], [], [], [], []
    for a, ex, pa in zip(g["action"], g["exec"], g["prev"]):
        if len(a) > 1:
            for h, n in enumerate(HEADS):
                jit[n].append(a[1:, h] != a[:-1, h])
            jit["any"].append((a[1:] != a[:-1]).any(1))
        for h, n in enumerate(HEADS):
            copy[n].append(a[:, h] == pa[:, h])
            exdiff[n].append(a[:, h] != ex[:, h])
        copy["all"].append((a == pa).all(1))
        exdiff["any"].append((a != ex).any(1))
        nomove = (a[:, 0] != 1) & (a[:, 1] == 0)
        tip.append(nomove & (a[:, 2] != ZT))
        idle.append(nomove & (a[:, 2] == ZT) & (a[:, 4] == 0) & (a[:, 0] == 0))
        mv0_runs += runs(a[:, 0] != 1)
        tv = TURN_VAL[a[:, 2]]
        nz = (tv[1:] != 0) & (tv[:-1] != 0)
        turn_flip.append(nz & (np.sign(tv[1:]) != np.sign(tv[:-1])))
    out["jitter"] = {k: round(float(np.concatenate(v).mean()), 4) for k, v in jit.items()}
    out["copy_prev_acc"] = {k: round(float(np.concatenate(v).mean()), 4) for k, v in copy.items()}
    out["exec_ne_label"] = {k: round(float(np.concatenate(v).mean()), 4) for k, v in exdiff.items()}
    fwd = A[:, 0] == 1
    out["turn_nonzero_share"] = round(float((A[:, 2] != ZT).mean()), 4)
    out["turn_nonzero_while_not_fwd_share_of_all"] = round(float(((A[:, 2] != ZT) & ~fwd).mean()), 4)
    out["share_of_turn_labels_without_fwd"] = round(float(((A[:, 2] != ZT) & ~fwd).sum() / max(1, (A[:, 2] != ZT).sum())), 4)
    out["turn_in_place_share"] = round(float(np.concatenate(tip).mean()), 4)
    out["idle_share(no move,no turn,no use)"] = round(float(np.concatenate(idle).mean()), 4)
    out["turn_sign_flip_rate(consecutive nonzero)"] = round(float(np.concatenate(turn_flip).mean()), 4)
    r = np.array(mv0_runs) if mv0_runs else np.zeros(1)
    out["not_fwd_runs"] = {"count_per_ep": round(len(mv0_runs) / len(L), 2), "mean": round(float(r.mean()), 2),
                           "p90": int(np.percentile(r, 90)), "max": int(r.max()),
                           "steps_in_runs>=5": round(float(r[r >= 5].sum() / A.shape[0]), 4),
                           "steps_in_runs>=10": round(float(r[r >= 10].sum() / A.shape[0]), 4)}
    # turn magnitude by move state
    for nm, m in (("fwd", fwd), ("not_fwd", ~fwd)):
        out[f"turn_share_given_{nm}"] = [round(float(v), 4) for v in
                                         np.bincount(A[m, 2], minlength=7) / max(1, m.sum())]
    # window around each USE=1 label
    W = 8
    win = np.zeros((2 * W + 1, 4)); cnt = np.zeros(2 * W + 1)
    for a in g["action"]:
        for t in np.nonzero(a[:, 4] == 1)[0]:
            for o in range(-W, W + 1):
                if 0 <= t + o < len(a):
                    r_ = a[t + o]
                    win[o + W] += [r_[0] == 1, r_[0] == 0 and r_[2] != ZT, r_[0] == 0 and r_[2] == ZT, r_[0] == 2]
                    cnt[o + W] += 1
    out["use_window"] = {"offsets": list(range(-W, W + 1)),
                         "fwd": [round(float(v), 3) for v in win[:, 0] / np.maximum(cnt, 1)],
                         "turn_in_place": [round(float(v), 3) for v in win[:, 1] / np.maximum(cnt, 1)],
                         "stand_still": [round(float(v), 3) for v in win[:, 2] / np.maximum(cnt, 1)],
                         "back": [round(float(v), 3) for v in win[:, 3] / np.maximum(cnt, 1)]}
    out["use_presses_per_ep"] = round(float((A[:, 4] == 1).sum() / len(L)), 2)
    return out


def knn_consistency(samples, k: int = 5) -> dict:
    """Label agreement with the k nearest other samples in (pooled rays + self) space,
    separately for samples whose move label is fwd vs not. Low agreement on similar
    observations = labels the observation does not determine."""
    if len(samples) < 50:
        return {}
    X = np.stack([s[0] for s in samples]).astype(np.float32)
    Y = np.stack([s[1] for s in samples])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    n = len(X)
    agree_move, agree_turn = np.zeros(n), np.zeros(n)
    sq = (X * X).sum(1)
    for i0 in range(0, n, 512):
        q = X[i0:i0 + 512]
        d = sq[i0:i0 + 512, None] + sq[None, :] - 2 * q @ X.T
        for j in range(len(q)):
            d[j, i0 + j] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        agree_move[i0:i0 + 512] = (Y[nn, 0] == Y[i0:i0 + 512, None, 0]).mean(1)
        agree_turn[i0:i0 + 512] = (Y[nn, 2] == Y[i0:i0 + 512, None, 2]).mean(1)
    fwd = Y[:, 0] == 1
    tip = (Y[:, 0] != 1) & (Y[:, 2] != ZT)
    return {"n": int(n),
            "move_agree_all": round(float(agree_move.mean()), 3),
            "turn_agree_all": round(float(agree_turn.mean()), 3),
            "turn_agree_fwd": round(float(agree_turn[fwd].mean()), 3),
            "turn_agree_not_fwd": round(float(agree_turn[~fwd].mean()), 3) if (~fwd).any() else None,
            "move_agree_not_fwd": round(float(agree_move[~fwd].mean()), 3) if (~fwd).any() else None,
            "turn_agree_turn_in_place": round(float(agree_turn[tip].mean()), 3) if tip.any() else None,
            "n_not_fwd": int((~fwd).sum()), "n_turn_in_place": int(tip.sum())}


def main() -> None:
    res = {}
    for name in ("v2", "v3"):
        idx, per, shapes, dtypes, ss, sq, sn, rs, cc = load(name)
        r = {"shapes": {k: list(v) for k, v in shapes.items()}, "dtypes": dtypes,
             "field_ranges": {k: [round(v[0], 3), round(v[1], 3)] for k, v in cc.items()},
             "stages": {}}
        for st in sorted(per):
            s = stage_stats(per[st])
            mean = ss[st] / sn[st]
            s["self_mean"] = [round(float(v), 3) for v in mean]
            s["self_std"] = [round(float(v), 3) for v in np.sqrt(np.maximum(sq[st] / sn[st] - mean ** 2, 0))]
            s["knn"] = knn_consistency(rs[st])
            r["stages"][st] = s
        res[name] = r
        per.clear(); rs.clear()
        print(f"done {name}", file=sys.stderr)
    out = ROOT / "reports/eval/2026-11-26-bc-v3-regression.json"
    out.write_text(json.dumps(res, indent=1))
    print(out)


if __name__ == "__main__":
    main()
