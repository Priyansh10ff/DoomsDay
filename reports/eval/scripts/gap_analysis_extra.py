"""Extra tables for the 2026-11-25 gap analysis: success over time, val-set comparison."""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from gap_analysis import load, mapinfo, pct, ROOT, jl  # load() applies the n1C cutoff

print("## A. Success per 100 updates (S0 / S1) and switch-vs-pad on S0")
for r in ("gen1", "n1A", "n1C"):
    rows = load(r)
    b = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for e in rows:
        k = e["u"] // 100 * 100
        b[k][e["stage"]][0] += 1; b[k][e["stage"]][1] += e["succ"]
        if e["stage"] == "S0":
            ek = mapinfo(e["mid"])["exit_kind"]
            b[k]["S0-" + ek][0] += 1; b[k]["S0-" + ek][1] += e["succ"]
    print(f"\n### {r}\n\n| upd | S0 | S0 pad | S0 switch | S1 |\n|---|---|---|---|---|")
    for k in sorted(b):
        c = b[k]
        print(f"| {k} | " + " | ".join(pct(c[s][1], c[s][0]) if c[s][0] else "-" for s in ("S0", "S0-pad", "S0-switch", "S1")) + " |")

print("\n## B. Validation sets (reports/eval/2026-11-25-bc-v1/*/results.jsonl)\n")
pols = ("random", "bc_sample", "ppo_gen_run1")
res = {p: list(jl(f"reports/eval/2026-11-25-bc-v1/{p}/results.jsonl")) for p in pols}
maps = sorted({e["map_id"] for p in pols for e in res[p]})
print("| map | exit | rooms | path doors | " + " | ".join(pols) + " |\n|---|---|---|---|" + "---|" * len(pols))
tot = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for m in maps:
    mi = mapinfo(m) or {}
    cells = []
    for p in pols:
        x = [e for e in res[p] if e["map_id"] == m]
        k = sum(bool(e["success"]) for e in x)
        tot[p][m[:2]][0] += len(x); tot[p][m[:2]][1] += k
        cells.append(f"{k}/{len(x)}")
    print(f"| {m} | {mi.get('exit_kind')} | {mi.get('rooms')} | {mi.get('path_doors')} | " + " | ".join(cells) + " |")
print("\n| policy | " + " | ".join(("S0", "S1", "S2", "S3")) + " |\n|---|---|---|---|---|")
for p in pols:
    print(f"| {p} | " + " | ".join(pct(tot[p][s][1], tot[p][s][0]) for s in ("S0", "S1", "S2", "S3")) + " |")
for p in pols:
    x = res[p]
    print(f"- {p}: seeds {sorted({e['seed'] for e in x})[:6]}..., policy field {set(e.get('policy') for e in x)}, deaths {sum(bool(e['death']) for e in x)}, timeouts {sum(bool(e['timeout']) for e in x)}, exit_unknown {sum(bool(e['exit_unknown']) for e in x)}")
