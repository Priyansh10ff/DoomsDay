"""Gap analysis of 2026-11-25 training logs (read-only, CPU-light, no engine).

Usage: python reports/eval/scripts/gap_analysis.py > <out.txt>
Streams episodes.jsonl / metrics.jsonl line by line; a partial last line (live run) is skipped.
Map features come from maps/train|val/<id>.json. Path doors are counted on the room graph
from start room to exit room (S2: start -> key room -> exit) -- analysis only.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from collections import defaultdict, deque

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = {
    "gen1": "reports/runs/20261125-nogit-45275c79",
    "n1A": "reports/runs/20261125-night1-A-dagger-doors",
    "n1C": "reports/runs/20261125-night1-C-s1-doors",
}


def jl(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line in a live file


_map_cache = {}


def mapinfo(mid):
    if mid in _map_cache:
        return _map_cache[mid]
    split = mid.split("-")[1]
    p = os.path.join(ROOT, "maps", split, mid + ".json")
    if not os.path.exists(p):
        _map_cache[mid] = None
        return None
    d = json.load(open(p, encoding="utf-8"))
    g = d["graph"]
    adj = defaultdict(list)
    for c in g["corridors"]:
        adj[c["a"]].append((c["b"], c["door"]))
        adj[c["b"]].append((c["a"], c["door"]))

    def path(s, t):
        prev = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for v, dr in adj[u]:
                if v not in prev:
                    prev[v] = (u, dr)
                    q.append(v)
        out, u = [], t
        while prev.get(u):
            pu, dr = prev[u]
            out.append((pu, u, dr))
            u = pu
        return out[::-1]

    rooms = g["rooms"]

    def ctr(i):
        r = rooms[i]
        return (r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)

    legs = []
    if g.get("key_room") is not None:
        legs = path(g["start_room"], g["key_room"]) + path(g["key_room"], g["exit_room"])
    else:
        legs = path(g["start_room"], g["exit_room"])
    pdoors = sum(1 for _, _, dr in legs if dr)
    dist = sum(abs(ctr(a)[0] - ctr(b)[0]) + abs(ctr(a)[1] - ctr(b)[1]) for a, b, _ in legs)
    all_doors = sum(1 for c in g["corridors"] if c["door"])
    first_door_hop = next((i for i, (_, _, dr) in enumerate(legs) if dr), None)
    info = dict(stage=d["stage"], rooms=len(rooms), doors=all_doors, path_doors=pdoors,
                side_doors=all_doors - sum(1 for _, _, dr in path(g["start_room"], g["exit_room"]) if dr) if g.get("key_room") is None else None,
                locked=d["counts"].get("locked_doors", 0), loops=d["counts"].get("loops", 0),
                exit_kind=g["exit_kind"], hops=len(legs), dist=dist,
                monsters=d["counts"].get("monsters", 0),
                first_door_hop=first_door_hop,
                start_room_deg=len(adj[g["start_room"]]))
    _map_cache[mid] = info
    return info


# Fixed snapshot of the live run (night1-C): only rows with update_idx <= this are used.
CUTOFF = {"n1C": 329}


def load(run):
    rows = []
    for e in jl(RUNS[run] + "/episodes.jsonl"):
        if run in CUTOFF and e["update_idx"] > CUTOFF[run]:
            continue
        s = e.get("shaping_episode") or {}
        rows.append(dict(u=e["update_idx"], dec=e["decision"], succ=bool(e["success"]),
                         to=bool(e["timeout"]), dead=bool(e["dead"]), L=e["length"],
                         stage=e.get("stage"), mid=e.get("map_id"), rep=e.get("replay"),
                         cells=s.get("cells_visited"), dp=s.get("doors_passed"), dm=s.get("doors_in_map"),
                         p_exit=s.get("paid/level_exit", 0.0), p_time=s.get("paid/time", 0.0),
                         p_door=s.get("paid/door", 0.0), intr=s.get("intrinsic", 0.0),
                         intr_raw=s.get("intrinsic_raw", 0.0), coef=s.get("intrinsic_coef"),
                         p_keys=s.get("paid/keycard", 0.0), ret=e.get("return")))
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def pct(k, n):
    return f"{k}/{n} ({100*k/n:.1f}%)" if n else "0/0"


def q(xs, ps=(0.25, 0.5, 0.75)):
    if not xs:
        return "-"
    xs = sorted(xs)
    return "/".join(f"{xs[min(len(xs)-1, int(p*len(xs)))]:.0f}" for p in ps)


def section(t):
    print(f"\n## {t}\n")


def main():
    data = {r: load(r) for r in RUNS}
    for r, rows in data.items():
        print(f"{r}: {len(rows)} episodes, updates {rows[0]['u']}..{rows[-1]['u']}, decisions to {rows[-1]['dec']}")

    # 1. per stage
    section("1. Per-run, per-stage outcome")
    print("| run | stage | episodes | success | timeout | dead | len succ p25/50/75 |")
    print("|---|---|---|---|---|---|---|")
    for r, rows in data.items():
        for sg in ("S0", "S1", "S2", "S3"):
            x = [e for e in rows if e["stage"] == sg]
            if not x:
                continue
            k = sum(e["succ"] for e in x)
            print(f"| {r} | {sg} | {len(x)} | {pct(k, len(x))} | {pct(sum(e['to'] for e in x), len(x))} | "
                  f"{pct(sum(e['dead'] for e in x), len(x))} | {q([e['L'] for e in x if e['succ']])} |")

    # gen1 split by phase (before/after promotion at update 521)
    section("1b. gen1 S0 before/after promotion (update 521), and by quarter")
    for r in ("gen1",):
        rows = data[r]
        for lab, f in (("S0 u<521", lambda e: e["stage"] == "S0" and e["u"] < 521),
                       ("S0 u>=521 (replay)", lambda e: e["stage"] == "S0" and e["u"] >= 521),
                       ("S1 last 500", None)):
            if f is None:
                x = [e for e in rows if e["stage"] == "S1"][-500:]
            else:
                x = [e for e in rows if f(e)]
            k = sum(e["succ"] for e in x)
            print(f"- {lab}: {pct(k, len(x))}")

    # per map
    section("1c. Per-map success (all runs, per stage); maps never solved")
    for r, rows in data.items():
        bym = defaultdict(lambda: [0, 0, 0])
        for e in rows:
            b = bym[e["mid"]]
            b[0] += 1
            b[1] += e["succ"]
            b[2] += e["to"]
        print(f"\n### {r}\n")
        print("| map | eps | succ | timeout | rooms | doors(all/path) | loops | exit | hops | dist | start deg |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for mid in sorted(bym):
            n, k, t = bym[mid]
            m = mapinfo(mid) or {}
            print(f"| {mid} | {n} | {k} ({100*k/n:.0f}%) | {t} | {m.get('rooms')} | {m.get('doors')}/{m.get('path_doors')} | "
                  f"{m.get('loops')} | {m.get('exit_kind')} | {m.get('hops')} | {m.get('dist')} | {m.get('start_room_deg')} |")
        for sg in ("S0", "S1", "S2", "S3"):
            ms = [mid for mid in bym if mid.startswith(sg)]
            if not ms:
                continue
            never = sorted(mid for mid in ms if bym[mid][1] == 0 and bym[mid][0] >= 5)
            print(f"- {r} {sg}: {len(ms)} maps seen, {len(never)} never solved with >=5 eps: {', '.join(never)}")

    # feature association (pool per run/stage)
    section("1d. Success vs map features (episode-weighted)")
    feats = ("rooms", "doors", "path_doors", "loops", "exit_kind", "hops", "start_room_deg")
    for r, rows in data.items():
        for sg in ("S0", "S1"):
            x = [e for e in rows if e["stage"] == sg]
            if len(x) < 100:
                continue
            print(f"\n{r} {sg} (n={len(x)}):")
            for f in feats:
                g = defaultdict(lambda: [0, 0, set()])
                for e in x:
                    m = mapinfo(e["mid"])
                    if not m:
                        continue
                    v = m[f]
                    g[v][0] += 1
                    g[v][1] += e["succ"]
                    g[v][2].add(e["mid"])
                s = "; ".join(f"{v}: {g[v][1]}/{g[v][0]} ({100*g[v][1]/g[v][0]:.1f}%, {len(g[v][2])} maps)" for v in sorted(g, key=str))
                print(f"- {f}: {s}")
            # distance bins
            g = defaultdict(lambda: [0, 0])
            for e in x:
                m = mapinfo(e["mid"])
                b = int(m["dist"] // 15) * 15
                g[b][0] += 1
                g[b][1] += e["succ"]
            print("- path dist (grid units, bin 15): " + "; ".join(f"{b}-{b+15}: {g[b][1]}/{g[b][0]} ({100*g[b][1]/g[b][0]:.1f}%)" for b in sorted(g)))

    # 2. door funnel
    section("2. Door funnel (runs with doors_passed)")
    for r in ("n1A", "n1C"):
        rows = [e for e in data[r] if e["dm"] is not None]
        for sg in ("S1", "S2"):
            x = [e for e in rows if e["stage"] == sg]
            if not x:
                continue
            print(f"\n### {r} {sg} (n={len(x)})\n")
            mx = max(e["dm"] for e in x)
            print("| doors passed >= k | episodes | share |")
            print("|---|---|---|")
            for k in range(0, mx + 1):
                c = sum(e["dp"] >= k for e in x)
                print(f"| {k} | {c} | {100*c/len(x):.1f}% |")
            c = sum(e["succ"] for e in x)
            print(f"| exit | {c} | {100*c/len(x):.1f}% |")
            # vs path doors
            print("\n| doors passed vs path doors | eps | success | timeout |")
            print("|---|---|---|---|")
            g = defaultdict(lambda: [0, 0, 0])
            for e in x:
                m = mapinfo(e["mid"])
                key = ("0" if e["dp"] == 0 else ("<path" if e["dp"] < m["path_doors"] else ">=path"))
                g[key][0] += 1
                g[key][1] += e["succ"]
                g[key][2] += e["to"]
            for key in ("0", "<path", ">=path"):
                if key in g:
                    print(f"| {key} | {g[key][0]} | {pct(g[key][1], g[key][0])} | {pct(g[key][2], g[key][0])} |")
            fails_all = [e for e in x if not e["succ"] and e["dp"] >= mapinfo(e["mid"])["path_doors"]]
            print(f"\nFailures with doors passed >= path doors: {len(fails_all)} of {sum(not e['succ'] for e in x)} failures "
                  f"(cells p25/50/75 {q([e['cells'] for e in fails_all])})")
            # time course
            print("\n| update bucket | eps | success | dp>=1 | mean dp | mean cells |")
            print("|---|---|---|---|---|---|")
            ub = defaultdict(list)
            for e in x:
                ub[e["u"] // 50 * 50].append(e)
            for b in sorted(ub):
                y = ub[b]
                print(f"| {b}-{b+49} | {len(y)} | {pct(sum(e['succ'] for e in y), len(y))} | "
                      f"{100*sum(e['dp']>=1 for e in y)/len(y):.1f}% | {st.mean(e['dp'] for e in y):.2f} | {st.mean(e['cells'] for e in y):.1f} |")

    # 3. coverage
    section("3. Coverage: cells_visited (p25/50/75), success vs failure")
    print("| run | stage | succ n | succ cells | fail n | fail cells | fail timeout cells | fail len p50 |")
    print("|---|---|---|---|---|---|---|---|")
    for r, rows in data.items():
        for sg in ("S0", "S1", "S2"):
            x = [e for e in rows if e["stage"] == sg and e["cells"] is not None]
            if not x:
                continue
            s = [e["cells"] for e in x if e["succ"]]
            f = [e for e in x if not e["succ"]]
            print(f"| {r} | {sg} | {len(s)} | {q(s)} | {len(f)} | {q([e['cells'] for e in f])} | "
                  f"{q([e['cells'] for e in f if e['to']])} | {q([e['L'] for e in f], (0.5,))} |")
    # low-coverage timeouts (stuck)
    print()
    for r, rows in data.items():
        for sg in ("S0", "S1"):
            f = [e for e in rows if e["stage"] == sg and e["to"] and e["cells"] is not None]
            if not f:
                continue
            for thr in (5, 10, 20):
                c = sum(e["cells"] <= thr for e in f)
                print(f"- {r} {sg} timeouts with cells <= {thr}: {pct(c, len(f))}")

    # 5. reward accounting
    section("5. Reward accounting per episode (means; paid values, as the learner saw them)")
    print("| run | stage | outcome | n | exit | door | time | coverage (coef*raw) | total shaped |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r, rows in data.items():
        for sg in ("S0", "S1"):
            for lab, f in (("success", lambda e: e["succ"]), ("fail", lambda e: not e["succ"])):
                x = [e for e in rows if e["stage"] == sg and f(e)]
                if not x:
                    continue
                print(f"| {r} | {sg} | {lab} | {len(x)} | {st.mean(e['p_exit'] for e in x):.3f} | {st.mean(e['p_door'] or 0 for e in x):.3f} | "
                      f"{st.mean(e['p_time'] for e in x):.3f} | {st.mean(e['intr'] for e in x):.3f} | {st.mean(e['ret'] for e in x):.3f} |")
    # coef schedule
    for r, rows in data.items():
        cs = [e["coef"] for e in rows if e["coef"] is not None]
        print(f"- {r} intrinsic_coef first {cs[0]:.3f}, after 200 eps {cs[min(199,len(cs)-1)]:.3f}, last {cs[-1]:.3f}")
    # hack check: failed episodes with shaped return > a short success?
    for r, rows in data.items():
        f = [e for e in rows if not e["succ"]]
        pos = [e for e in f if e["ret"] > 0]
        print(f"- {r}: failed episodes with positive shaped return {pct(len(pos), len(f))}; max {max((e['ret'] for e in f), default=0):.3f}; "
              f"min success return {min((e['ret'] for e in rows if e['succ']), default=0):.3f}")

    # 4. dynamics from metrics
    section("4. Learning dynamics (metrics.jsonl), bucket means")
    keys = ["entropy_head0", "entropy_head1", "entropy_head2", "entropy_head3", "entropy_head4", "entropy_head5",
            "approx_kl", "clipfrac", "explained_variance", "v_loss", "return_scale", "grad_norm",
            "action_share_head0_opt1", "action_share_head0_opt0", "action_share_head4_opt1", "action_share_head3_opt1",
            "action_share_head2_opt3", "action_share_head5_opt1", "recent_success"]
    short = ["H_mv", "H_st", "H_tn", "H_at", "H_use", "H_spd", "kl", "clip", "EV", "vloss", "rscale", "gnorm",
             "fwd", "mv_none", "use", "atk", "turn0", "spd", "rsucc"]
    for r in RUNS:
        b = defaultdict(lambda: defaultdict(list))
        bs = 25 if r == "n1C" else (100 if r == "n1A" else 200)
        for m in jl(RUNS[r] + "/metrics.jsonl"):
            u = int(m.get("update_idx", 0))
            if r in CUTOFF and u > CUTOFF[r]:
                continue
            for k in keys:
                v = m.get(k)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    b[u // bs * bs][k].append(v)
        print(f"\n### {r} (bucket {bs} updates)\n")
        print("| upd | " + " | ".join(short) + " |")
        print("|" + "---|" * (len(short) + 1))
        for u in sorted(b):
            print(f"| {u} | " + " | ".join(f"{st.mean(b[u][k]):.3f}" if b[u][k] else "-" for k in keys) + " |")


if __name__ == "__main__":
    main()
