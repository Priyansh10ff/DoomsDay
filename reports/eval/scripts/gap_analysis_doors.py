"""Door funnel restricted to maps whose doors are all on the start->exit path, matched
switch-vs-pad comparison, failure coverage by exit kind (2026-11-25 gap analysis)."""
import os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from gap_analysis import load, mapinfo, pct, q, wilson

c = load("n1C")
s1 = [e for e in c if e["stage"] == "S1"]
onpath = sorted({e["mid"] for e in s1 if mapinfo(e["mid"])["doors"] == mapinfo(e["mid"])["path_doors"] and mapinfo(e["mid"])["path_doors"] > 0})
print("## Door funnel, n1C S1, maps whose every door is on the path:", ", ".join(onpath))
for lab, sel in (("all updates", lambda e: True), ("u<150", lambda e: e["u"] < 150), ("u>=150", lambda e: e["u"] >= 150)):
    x = [e for e in s1 if e["mid"] in onpath and sel(e)]
    print(f"\n### {lab} (n={len(x)})\n\n| stage of funnel | episodes | share | Wilson 95% |\n|---|---|---|---|")
    n = len(x)
    for k in (1, 2, 3):
        cnt = sum(e["dp"] >= k for e in x)
        lo, hi = wilson(cnt, n)
        print(f"| passed >= {k} door(s) | {cnt} | {100*cnt/n:.1f}% | {lo:.3f}-{hi:.3f} |")
    allp = [e for e in x if e["dp"] >= mapinfo(e["mid"])["path_doors"]]
    k = sum(e["succ"] for e in x); lo, hi = wilson(k, n)
    print(f"| passed all path doors | {len(allp)} | {100*len(allp)/n:.1f}% | |")
    print(f"| exit | {k} | {100*k/n:.1f}% | {lo:.3f}-{hi:.3f} |")
    print(f"\nAfter all path doors: {pct(sum(e['succ'] for e in allp), len(allp))} exit; timeouts {sum(e['to'] for e in allp)}; failed cells p25/50/75 {q([e['cells'] for e in allp if not e['succ']])}")
    print(f"Success with fewer doors than path doors (should be 0): {sum(e['succ'] and e['dp'] < mapinfo(e['mid'])['path_doors'] for e in x)}")
    by1 = defaultdict(lambda: [0, 0, 0])
    for e in x:
        pdn = mapinfo(e["mid"])["path_doors"]
        by1[pdn][0] += 1; by1[pdn][1] += e["dp"] >= 1; by1[pdn][2] += e["succ"]
    for pdn in sorted(by1):
        print(f"- path doors {pdn}: n {by1[pdn][0]}, >=1 door {by1[pdn][1]}, exit {by1[pdn][2]}")
m16 = [e for e in s1 if e["mid"] == "S1-train-101016"]
print(f"\nS1-train-101016 (1 door, off the path): {pct(sum(e['succ'] for e in m16), len(m16))} exit, dp>=1 in {sum(e['dp']>=1 for e in m16)}")
other = [e for e in s1 if e["mid"] not in onpath and e["mid"] != "S1-train-101016"]
print(f"S1 maps with side doors (n={len(other)}): dp>=1 {pct(sum(e['dp']>=1 for e in other), len(other))}, exit {pct(sum(e['succ'] for e in other), len(other))}, "
      f"failed with dp>=path doors (could be side doors) {sum((not e['succ']) and e['dp']>=mapinfo(e['mid'])['path_doors'] for e in other)}")

print("\n## Matched switch vs pad on S0 (3 rooms, 2 hops)")
SW = ("S0-train-100002", "S0-train-100004", "S0-train-100020")
PD = ("S0-train-100006", "S0-train-100007", "S0-train-100009")
for r in ("gen1", "n1A", "n1C"):
    rows = [e for e in load(r) if e["stage"] == "S0"]
    for lab, ms in (("switch", SW), ("pad", PD)):
        x = [e for e in rows if e["mid"] in ms]
        k = sum(e["succ"] for e in x); lo, hi = wilson(k, len(x))
        f = [e["cells"] for e in x if not e["succ"]]
        print(f"- {r} {lab} {ms}: {pct(k, len(x))} (Wilson {lo:.3f}-{hi:.3f}); failed cells p25/50/75 {q(f)}; success len p50 {q([e['L'] for e in x if e['succ']], (0.5,))}")
    for ek in ("pad", "switch"):
        f = [e for e in rows if not e["succ"] and mapinfo(e["mid"])["exit_kind"] == ek]
        print(f"  {r} all S0 {ek} failures: n {len(f)}, cells p25/50/75 {q([e['cells'] for e in f])}")
