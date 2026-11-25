"""Ablation (cumulative progression) table and checkpoint selection on the VAL split.

    python reports/eval/scripts/ablation_table.py [--manifest M] [--dry-run] [--aggregate-only]
        [--no-reuse] [--episodes-per-map E] [--seed S] [--out DIR]
    python reports/eval/scripts/ablation_table.py --select reports/runs/<run> [--decode sample|argmax]
        [--screen-episodes 3] [--final-episodes 15] [--top 2] [--seed S] [--out DIR] [--dry-run]

Ablation mode: for each manifest row (in manifest order, never sorted) runs
`python -m evaluate.run --split val --episodes-per-map E --seed S --sample|--argmax
--out <out>/<name>` (or --random-policy), ONE engine at a time, sequentially. The same
E and S for every row, so episodes are paired on (map_id, seed). A row is skipped when
its output directory is already complete (summary.json present, protocol and checkpoint
sha256 match, rows == maps x E), so the script is resumable. An earlier result directory
is reused only on an EXACT protocol match (full run_config fields below, checkpoint
sha256 of the file on disk now, every val map id + wad sha256 equal to maps/splits.json,
complete row count); reused paths are referenced, not copied, and flagged in the table
because they were produced by older env/ code.

DOOMSDAY_OBS_VERSION is set per row in the child process only, from the checkpoint's
stored sensor signature (evaluate.obs_select.checkpoint_obs_version); a manifest
obs_version that disagrees refuses the row.

The held-out split is refused in both modes (held-out results are for reporting only,
never for selection or comparison-driven decisions).

Select mode: screens every numbered checkpoint in <run>/checkpoints/ at E=3, re-runs
the top 2 at E=15 on the same base seed (the E=15 seeds contain the screening seeds),
and tests the top-2 difference both on all 15 episodes and on episode_index >= 3 only
(the episodes not used for screening).

Statistics: Wilson 95 % score intervals (evaluate.stats.wilson_interval); paired
comparisons use McNemar's exact test (two-sided binomial on the discordant pairs) and a
map-cluster bootstrap of the success-rate difference (maps resampled with replacement,
fixed RNG seed), because episodes cluster within 20 maps.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from evaluate.stats import wilson_interval  # noqa: E402

Z95 = 1.959963984540054
SPLIT = "val"
OBS_ENV = "DOOMSDAY_OBS_VERSION"
DEFAULT_MANIFEST = os.path.join(REPO, "reports", "eval", "ablation_manifest.json")
# Result directories whose wall time / lengths calibrate the duration estimate (task spec).
TIMING_DIRS = ("reports/eval/2026-11-26-val", "reports/eval/2026-11-25-bc-v1")
BOOT_RESAMPLES = 10_000
BOOT_SEED = 20261126
STAGES = ("S0", "S1", "S2", "S3")
SEED_STRIDE_MAP = 10_000            # evaluate.config.EvalRunConfig default; checked on reuse
CHECKPOINT_RE = re.compile(r"checkpoint-u(\d+)\.pt$")


# ----------------------------------------------------------------------------- helpers
def rel(p: str) -> str:
    p = os.path.abspath(p)
    try:
        return os.path.relpath(p, REPO).replace("\\", "/")
    except ValueError:
        return p


def absp(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def sha256(path: str) -> str:
    from evaluate.checkpoint_io import sha256_file
    return sha256_file(absp(path))


def val_maps() -> list[dict]:
    with open(os.path.join(REPO, "maps", "splits.json"), encoding="utf-8") as f:
        d = json.load(f)
    return [{"id": m["id"], "stage": m["stage"], "sha256": m["sha256"]} for m in d["splits"][SPLIT]]


def ckpt_obs_version(path: str) -> int | None:
    """Version from the checkpoint's own sensor signature (no engine)."""
    from evaluate.obs_select import checkpoint_obs_version
    from train.checkpoint import load_checkpoint
    blob = load_checkpoint(absp(path))
    return checkpoint_obs_version(blob.get("sensor_signature_json"))


def read_rows(d: str) -> list[dict]:
    p = os.path.join(absp(d), "results.jsonl")
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_summary(d: str) -> dict | None:
    p = os.path.join(absp(d), "summary.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- protocol match
def protocol_mismatch(d: str, *, policy: str, episodes: int, seed: int, ckpt_sha: str | None,
                      maps: list[dict]) -> str | None:
    """None when directory `d` holds a complete result of exactly this protocol."""
    s = read_summary(d)
    if s is None or "meta" not in s:
        return "no summary.json with meta"
    m = s["meta"]
    rc = m.get("run_config", {})
    want = {"split": SPLIT, "episodes_per_map": episodes, "policy": policy, "base_seed": seed,
            "seed_stride_map": SEED_STRIDE_MAP, "max_maps": None, "exit_reward_fraction": 1.0,
            "timeout_tics_override": None}
    for k, v in want.items():
        if rc.get(k, "<absent>") != v:
            return f"run_config.{k}={rc.get(k, '<absent>')!r}, want {v!r}"
    got_sha = (m.get("checkpoint") or {}).get("sha256")
    if got_sha != ckpt_sha:
        return f"checkpoint sha256 {str(got_sha)[:12]} != {str(ckpt_sha)[:12]}"
    got_maps = sorted((x["map_id"], x["wad_sha256"]) for x in m.get("maps", []))
    if got_maps != sorted((x["id"], x["sha256"]) for x in maps):
        return "val map ids / wad hashes differ from maps/splits.json"
    n = len(read_rows(d))
    if n != len(maps) * episodes:
        return f"{n} result rows, want {len(maps) * episodes}"
    return None


def find_reuse(search_dirs: list[str], **kw) -> str | None:
    for root in search_dirs:
        for sp in sorted(glob.glob(os.path.join(absp(root), "**", "summary.json"), recursive=True)):
            d = os.path.dirname(sp)
            if protocol_mismatch(d, **kw) is None:
                return rel(d)
    return None


# ----------------------------------------------------------------------------- timing
def timing_model() -> dict:
    """Seconds per episode (wall incl. engine start-up per map) and per decision, from
    complete summaries in TIMING_DIRS, split random vs network."""
    acc: dict[str, list[tuple[float, int, int, str]]] = {"random": [], "network": []}
    for root in TIMING_DIRS:
        for sp in sorted(glob.glob(os.path.join(REPO, root, "*", "summary.json"))):
            s = read_summary(os.path.dirname(sp))
            if not s or "meta" not in s:
                continue
            rows = read_rows(os.path.dirname(sp))
            if not rows:
                continue
            kind = "random" if s["meta"]["run_config"]["policy"] == "random" else "network"
            acc[kind].append((s["meta"]["wall_seconds"], len(rows), sum(r["length"] for r in rows),
                              rel(os.path.dirname(sp))))
    out = {}
    for kind, xs in acc.items():
        if not xs:
            out[kind] = None
            continue
        per_ep = [w / n for w, n, _, _ in xs]
        out[kind] = {"sources": [x[3] for x in xs], "sec_per_episode_mean": sum(w for w, *_ in xs) / sum(n for _, n, _, _ in xs),
                     "sec_per_episode_min": min(per_ep), "sec_per_episode_max": max(per_ep),
                     "sec_per_decision": sum(w for w, *_ in xs) / sum(L for _, _, L, _ in xs)}
    return out


def fmt_dur(s: float) -> str:
    return f"{s / 3600:.2f} h ({s / 60:.0f} min)"


# ----------------------------------------------------------------------------- running
def eval_cmd(ckpt: str | None, decode: str, episodes: int, seed: int, out: str) -> list[str]:
    cmd = [sys.executable, "-m", "evaluate.run"]
    if ckpt is None:
        cmd += ["--random-policy"]
    else:
        cmd += ["--checkpoint", ckpt, "--" + decode]
    return cmd + ["--split", SPLIT, "--episodes-per-map", str(episodes), "--seed", str(seed), "--out", out]


def run_one(cmd: list[str], obs_version: int | None, out_dir: str) -> int:
    env = dict(os.environ)
    env.pop(OBS_ENV, None)
    if obs_version is not None:
        env[OBS_ENV] = str(obs_version)     # child process only
    log = absp(out_dir).rstrip("/\\") + ".log"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print(f"[run] {OBS_ENV}={obs_version} {' '.join(cmd)}  (log {rel(log)})", flush=True)
    with open(log, "w", encoding="utf-8") as lf:
        lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {OBS_ENV}={obs_version} {' '.join(cmd)}\n")
        lf.flush()
        return subprocess.call(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)


# ----------------------------------------------------------------------------- stats
def agg(rows: list[dict]) -> dict:
    n = len(rows)
    s = sum(bool(r["success"]) for r in rows)
    lo, hi = wilson_interval(s, n, Z95) if n else (float("nan"), float("nan"))
    lens = [r["length"] for r in rows]
    slen = [r["length"] for r in rows if r["success"]]
    return {"episodes": n, "successes": s, "success_rate": s / n if n else None,
            "wilson95": [lo, hi] if n else None,
            "mean_length": sum(lens) / n if n else None,
            "mean_length_successful": sum(slen) / len(slen) if slen else None,
            "deaths": sum(bool(r["death"]) for r in rows),
            "timeouts": sum(bool(r["timeout"]) for r in rows),
            "exit_unknown": sum(bool(r.get("exit_unknown")) for r in rows),
            "mean_kills": sum(r.get("kills", 0.0) for r in rows) / n if n else None}


def summarize_rows(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["stage"]].append(r)
    return {"overall": agg(rows), "per_stage": {k: agg(by[k]) for k in sorted(by)}}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(rows_a: list[dict], rows_b: list[dict], min_episode_index: int = 0) -> dict:
    """B minus A on shared (map_id, seed) keys."""
    ka = {(r["map_id"], r["seed"]): r for r in rows_a if r["episode_index"] >= min_episode_index}
    kb = {(r["map_id"], r["seed"]): r for r in rows_b if r["episode_index"] >= min_episode_index}
    shared = sorted(set(ka) & set(kb))
    seed_engine_mismatch = [k for k in shared
                            if ka[k].get("episode_seed_engine") != kb[k].get("episode_seed_engine")]
    only_a, only_b = len(set(ka) - set(kb)), len(set(kb) - set(ka))
    both = sum(ka[k]["success"] and kb[k]["success"] for k in shared)
    a_only = sum(ka[k]["success"] and not kb[k]["success"] for k in shared)   # A wins
    b_only = sum(kb[k]["success"] and not ka[k]["success"] for k in shared)   # B wins
    n = len(shared)
    diff = (b_only - a_only) / n if n else None
    by_map = defaultdict(list)
    for k in shared:
        by_map[k[0]].append(int(kb[k]["success"]) - int(ka[k]["success"]))
    maps = sorted(by_map)
    rng = random.Random(BOOT_SEED)
    boots = []
    if maps:
        for _ in range(BOOT_RESAMPLES):
            pick = [maps[rng.randrange(len(maps))] for _ in maps]
            num = sum(sum(by_map[m]) for m in pick)
            den = sum(len(by_map[m]) for m in pick)
            boots.append(num / den)
        boots.sort()
    ci = [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if boots else None
    return {"pairs": n, "maps": len(maps), "only_in_a": only_a, "only_in_b": only_b,
            "episode_seed_engine_mismatches": len(seed_engine_mismatch),
            "both_success": both, "a_only_success": a_only, "b_only_success": b_only,
            "success_diff_b_minus_a": diff, "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
            "map_bootstrap_ci95_diff": ci, "min_episode_index": min_episode_index}


# ----------------------------------------------------------------------------- formatting
def f_rate(a: dict | None) -> str:
    if not a or not a["episodes"]:
        return "not measured"
    lo, hi = a["wilson95"]
    return f"{a['successes']}/{a['episodes']} ({a['success_rate'] * 100:.1f}%, {lo * 100:.1f}-{hi * 100:.1f})"


def f_num(x, nd=1) -> str:
    return "not measured" if x is None else f"{x:.{nd}f}"


def f_p(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"


def f_paired(name_a: str, name_b: str, p: dict) -> str:
    d = p["success_diff_b_minus_a"]
    ci = p["map_bootstrap_ci95_diff"]
    warn = []
    if p["only_in_a"] or p["only_in_b"]:
        warn.append(f"unpaired: {p['only_in_a']} only in A, {p['only_in_b']} only in B")
    if p["episode_seed_engine_mismatches"]:
        warn.append(f"{p['episode_seed_engine_mismatches']} engine-seed mismatches")
    return (f"| {name_a} | {name_b} | {p['pairs']} ({p['maps']} maps) | {p['a_only_success']} | "
            f"{p['b_only_success']} | {p['both_success']} | "
            f"{'not measured' if d is None else f'{d * 100:+.1f} pp'} | "
            f"{'not measured' if ci is None else f'{ci[0] * 100:+.1f} to {ci[1] * 100:+.1f} pp'} | "
            f"{f_p(p['mcnemar_exact_p'])} | {'; '.join(warn) or '-'} |")


PAIRED_HEADER = ("| A | B | pairs | A only | B only | both | B - A success | map-bootstrap 95% CI | "
                 "McNemar exact p | pairing notes |\n|---|---|---|---|---|---|---|---|---|---|")


# ----------------------------------------------------------------------------- ablation mode
def plan_rows(man: dict, episodes: int, seed: int, out_root: str, no_reuse: bool, check_ckpt: bool):
    maps = val_maps()
    search = [] if no_reuse else man.get("reuse_search_dirs", [])
    plan = []
    for row in man["rows"]:
        p = {"row": row, "status": row.get("status", "ready"), "obs_version": None,
             "out": rel(os.path.join(out_root, row["name"])), "action": None, "detail": "",
             "ckpt_sha": None, "cmd": None}
        ck = row.get("checkpoint")
        if p["status"] == "pending" or ck is None:
            p["action"], p["status"] = "pending", "pending"
            p["detail"] = row.get("note", "")
            plan.append(p)
            continue
        is_random = ck == "random"
        policy = "random" if is_random else row["decode"]
        if not is_random:
            if not os.path.isfile(absp(ck)):
                p["action"], p["detail"] = "blocked", f"checkpoint file missing: {ck}"
                plan.append(p)
                continue
            p["ckpt_sha"] = sha256(ck)
            if check_ckpt:
                v = ckpt_obs_version(ck)
                if v is None:
                    p["action"], p["detail"] = "blocked", "checkpoint has no sensor signature"
                    plan.append(p)
                    continue
                if row.get("obs_version") is not None and int(row["obs_version"]) != v:
                    p["action"] = "blocked"
                    p["detail"] = f"manifest obs_version {row['obs_version']} != checkpoint signature {v}"
                    plan.append(p)
                    continue
                p["obs_version"] = v
        kw = dict(policy=policy, episodes=episodes, seed=seed, ckpt_sha=p["ckpt_sha"], maps=maps)
        if protocol_mismatch(p["out"], **kw) is None:
            p["action"], p["src"] = "done", p["out"]
        else:
            reuse = find_reuse(search, **kw)
            if reuse:
                p["action"], p["src"] = "reuse", reuse
            else:
                p["action"], p["src"] = "run", p["out"]
                p["cmd"] = eval_cmd(None if is_random else ck, row.get("decode", "sample"),
                                    episodes, seed, p["out"])
        plan.append(p)
    return plan, maps


def estimate(plan, n_maps: int, episodes: int, tm: dict) -> tuple[float, list[str]]:
    total, lines = 0.0, []
    for p in plan:
        if p["action"] != "run":
            continue
        kind = "random" if p["row"]["checkpoint"] == "random" else "network"
        t = tm.get(kind)
        if t is None:
            lines.append(f"  {p['row']['name']}: no timing data for {kind}")
            continue
        sec = n_maps * episodes * t["sec_per_episode_mean"]
        lo = n_maps * episodes * t["sec_per_episode_min"]
        hi = n_maps * episodes * t["sec_per_episode_max"]
        total += sec
        lines.append(f"  {p['row']['name']}: {n_maps * episodes} episodes x {t['sec_per_episode_mean']:.1f} s "
                     f"= {fmt_dur(sec)} (range {lo / 60:.0f}-{hi / 60:.0f} min)")
    return total, lines


def write_ablation_report(man, plan, maps, episodes, seed, out_root, reused_caveat: str) -> dict:
    loaded = {}
    for p in plan:
        if p["action"] in ("done", "reuse"):
            loaded[p["row"]["name"]] = read_rows(p["src"])
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol": {
        "split": SPLIT, "episodes_per_map": episodes, "base_seed": seed, "seed_stride_map": SEED_STRIDE_MAP,
        "n_maps": len(maps), "seed_rule": "seed = base_seed + map_index * 10000 + episode_index"},
        "rows": [], "paired": []}
    names = [p["row"]["name"] for p in plan]
    for p in plan:
        r = p["row"]
        e = {"name": r["name"], "checkpoint": r.get("checkpoint"), "checkpoint_sha256": p["ckpt_sha"],
             "obs_version": p["obs_version"] if p["obs_version"] is not None else r.get("obs_version"),
             "decode": "random" if r.get("checkpoint") == "random" else r.get("decode"),
             "note": r.get("note", ""), "action": p["action"], "source": p.get("src"), "detail": p["detail"]}
        if r["name"] in loaded:
            e.update(summarize_rows(loaded[r["name"]]))
        res["rows"].append(e)
    measured = [n for n in names if n in loaded]
    pairs = []
    if measured:
        base = measured[0]
        for a, b in zip(measured, measured[1:]):
            pairs.append((a, b, "adjacent"))
        for b in measured[2:]:
            pairs.append((base, b, f"vs {base}"))
    for a, b, kind in pairs:
        res["paired"].append({"a": a, "b": b, "kind": kind, **paired(loaded[a], loaded[b])})

    # markdown
    L = [f"# Cumulative progression table (val split), {res['written']}", "",
         "This is a **cumulative progression**, not a one-switch ablation: adjacent rows differ in more "
         "than one component (training method, observation version, loss weighting). Rows are in "
         "manifest order and are not sorted.", "",
         f"Protocol: split `{SPLIT}` ({len(maps)} maps, 5 per stage S0-S3), {episodes} episodes per map, "
         f"base seed {seed}, seed = base + map_index x 10000 + episode_index; every row uses the same "
         f"(map, seed) pairs. Success = level exit (two agreeing engine signals, `evaluate/episode.py`). "
         f"Interval: Wilson 95% score interval. One engine at a time; vanilla EVAL config checked by "
         f"`evaluate.run` (assert_vanilla, assert_same_sensors) before any episode.", ""]
    if any(p["action"] == "reuse" for p in plan):
        L += [f"**Reused results:** {reused_caveat}", ""]
    L += ["## Success by stage", "",
          "| row | obs | decode | S0 | S1 | S2 | S3 | overall | source |", "|---|---|---|---|---|---|---|---|---|"]
    for e in res["rows"]:
        ps = e.get("per_stage", {})
        src = {"done": f"ran: `{e['source']}`", "reuse": f"REUSED: `{e['source']}`",
               "pending": "pending (not run)", "run": "not run yet", "blocked": f"blocked: {e['detail']}"}[e["action"]]
        L.append(f"| {e['name']} | {'-' if e['obs_version'] is None else 'v' + str(e['obs_version'])} | {e['decode']} | "
                 + " | ".join(f_rate(ps.get(s)) for s in STAGES)
                 + f" | {f_rate(e.get('overall'))} | {src} |")
    L += ["", "## Overall episode metrics", "",
          "Mean length is in decisions over all episodes (failed episodes count at their full length, "
          "timeouts at the limit). Deaths and timeouts are episode counts.", "",
          "| row | episodes | mean length | mean length (successful) | deaths | timeouts | exit unknown | mean kills |",
          "|---|---|---|---|---|---|---|---|"]
    for e in res["rows"]:
        o = e.get("overall")
        if not o:
            L.append(f"| {e['name']} | not measured | not measured | not measured | not measured | "
                     f"not measured | not measured | not measured |")
            continue
        L.append(f"| {e['name']} | {o['episodes']} | {f_num(o['mean_length'])} | "
                 f"{f_num(o['mean_length_successful'])} | {o['deaths']}/{o['episodes']} | "
                 f"{o['timeouts']}/{o['episodes']} | {o['exit_unknown']} | {f_num(o['mean_kills'], 2)} |")
    L += ["", "## Per-stage length and deaths", "",
          "Cell = mean length (decisions, all episodes) / deaths (episode count) / episodes.", "",
          "| row | S0 | S1 | S2 | S3 |", "|---|---|---|---|---|"]
    for e in res["rows"]:
        ps = e.get("per_stage", {})
        cells = [("not measured" if not ps.get(s) else
                  f"{ps[s]['mean_length']:.1f} / {ps[s]['deaths']} / {ps[s]['episodes']}") for s in STAGES]
        L.append(f"| {e['name']} | " + " | ".join(cells) + " |")
    L += ["", "## Paired comparisons (same map and seed)", "",
          "B - A = difference in success rate on shared (map, seed) pairs. McNemar: exact two-sided "
          f"binomial test on the discordant pairs. Bootstrap: {BOOT_RESAMPLES} resamples of maps with "
          f"replacement (RNG seed {BOOT_SEED}), percentile interval; episodes on one map are not "
          "independent, so trust the bootstrap interval over McNemar when they disagree. No correction "
          "for multiple comparisons.", "", PAIRED_HEADER]
    for c in res["paired"]:
        L.append(f_paired(c["a"], c["b"], c))
    if not res["paired"]:
        L.append("| not measured | | | | | | | | | |")
    L += ["", "## Rows", ""]
    for e in res["rows"]:
        L.append(f"- **{e['name']}**: `{e['checkpoint']}` sha256 {str(e['checkpoint_sha256'])[:12]}; {e['note']}")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "ablation_table.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_root, "ablation_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[write] {rel(os.path.join(out_root, 'ablation_table.md'))} and .json")
    return res


def ablation_main(a) -> int:
    with open(absp(a.manifest), encoding="utf-8") as f:
        man = json.load(f)
    proto = man.get("protocol", {})
    if proto.get("split", SPLIT) != SPLIT:
        raise SystemExit(f"split {proto.get('split')!r} refused: this script evaluates only on '{SPLIT}' "
                         f"(held-out is for final reporting, never for comparison or selection)")
    episodes = a.episodes_per_map or proto["episodes_per_map"]
    seed = a.seed if a.seed is not None else proto["seed"]
    out_root = absp(a.out or man["out_dir"])
    plan, maps = plan_rows(man, episodes, seed, out_root, a.no_reuse, check_ckpt=True)
    tm = timing_model()
    total, lines = estimate(plan, len(maps), episodes, tm)
    print(f"Protocol: split={SPLIT} maps={len(maps)} E={episodes} seed={seed} out={rel(out_root)}")
    for p in plan:
        r = p["row"]
        print(f"- {r['name']:<28} [{p['action']}] obs={p['obs_version'] if p['obs_version'] is not None else r.get('obs_version')} "
              f"{p.get('src') or ''} {p['detail']}")
        if p["cmd"]:
            print(f"    {OBS_ENV}={p['obs_version'] if p['obs_version'] is not None else '(unset)'} {' '.join(p['cmd'])}")
    for k, t in tm.items():
        if t:
            print(f"timing[{k}]: {t['sec_per_episode_mean']:.2f} s/episode (min {t['sec_per_episode_min']:.2f}, "
                  f"max {t['sec_per_episode_max']:.2f}), {t['sec_per_decision'] * 1000:.1f} ms/decision, "
                  f"from {', '.join(t['sources'])}")
    print("Estimate per row to run (pending rows: not measured):")
    print("\n".join(lines) or "  nothing to run")
    print(f"Estimated total: {fmt_dur(total)}")
    if a.dry_run:
        return 0
    if not a.aggregate_only:
        for p in plan:
            if p["action"] != "run":
                continue
            rc = run_one(p["cmd"], p["obs_version"], p["out"])
            r = p["row"]
            pol = "random" if r["checkpoint"] == "random" else r["decode"]
            bad = protocol_mismatch(p["out"], policy=pol, episodes=episodes, seed=seed,
                                    ckpt_sha=p["ckpt_sha"], maps=maps)
            if rc != 0 or bad:
                print(f"[fail] {r['name']}: exit {rc}; {bad}. Stopping (resume re-runs this row).")
                p["action"], p["detail"] = "blocked", f"run failed (exit {rc}; {bad})"
                break
            p["action"] = "done"
    write_ablation_report(man, plan, maps, episodes, seed, out_root, man.get("reuse_caveat", ""))
    return 0


# ----------------------------------------------------------------------------- select mode
def select_main(a) -> int:
    run_dir = absp(a.select)
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint-u*.pt")),
                 key=lambda p: int(CHECKPOINT_RE.search(p).group(1)))
    if not cks:
        raise SystemExit(f"no numbered checkpoints in {rel(run_dir)}/checkpoints")
    seed = a.seed if a.seed is not None else 11
    out_root = absp(a.out or os.path.join("reports", "eval",
                                          f"select-{os.path.basename(run_dir.rstrip('/\\'))}"))
    maps = val_maps()
    tm = timing_model()["network"]
    entries = []
    for ck in cks:
        name = os.path.basename(ck)[:-3]
        v = ckpt_obs_version(ck)
        entries.append({"name": name, "ckpt": rel(ck), "sha": sha256(ck), "obs": v})
    n_scr = len(maps) * a.screen_episodes
    est = len(entries) * n_scr * tm["sec_per_episode_mean"] + a.top * len(maps) * a.final_episodes * tm["sec_per_episode_mean"]
    print(f"Select: {len(entries)} checkpoints from {rel(run_dir)}, decode={a.decode}, seed={seed}, "
          f"screen E={a.screen_episodes}, final top {a.top} at E={a.final_episodes}; out {rel(out_root)}")
    print(f"Estimated (upper bound, nothing reused): {fmt_dur(est)} at {tm['sec_per_episode_mean']:.1f} s/episode")

    def stage(E: int, sub: str, items: list[dict]) -> dict[str, list[dict]]:
        out = {}
        for e in items:
            d = rel(os.path.join(out_root, sub, e["name"]))
            kw = dict(policy=a.decode, episodes=E, seed=seed, ckpt_sha=e["sha"], maps=maps)
            if protocol_mismatch(d, **kw) is not None:
                cmd = eval_cmd(e["ckpt"], a.decode, E, seed, d)
                print(f"    {OBS_ENV}={e['obs']} {' '.join(cmd)}")
                if a.dry_run:
                    continue
                rc = run_one(cmd, e["obs"], d)
                bad = protocol_mismatch(d, **kw)
                if rc != 0 or bad:
                    raise SystemExit(f"{e['name']} failed: exit {rc}; {bad} (resume re-runs it)")
            out[e["name"]] = read_rows(d)
        return out

    print("Screening:")
    scr = stage(a.screen_episodes, f"screen-e{a.screen_episodes}", entries)
    if a.dry_run:
        print(f"Final: top {a.top} of the screen at E={a.final_episodes} (commands depend on the screen)")
        return 0

    def key(n):  # rank: successes desc, deaths asc, then later checkpoint first
        s = agg(scr[n])
        return (-s["successes"], s["deaths"], -int(CHECKPOINT_RE.search(n + ".pt").group(1)))
    ranked = sorted(scr, key=key)
    top = [e for n in ranked[:a.top] for e in entries if e["name"] == n]
    cut = agg(scr[ranked[a.top - 1]])["successes"] if len(ranked) >= a.top else None
    ties = [n for n in ranked[a.top:] if agg(scr[n])["successes"] == cut]
    print("Final:")
    fin = stage(a.final_episodes, f"final-e{a.final_episodes}", top)
    names = [e["name"] for e in top]
    fa = {n: agg(fin[n]) for n in names}
    winner = max(names, key=lambda n: (fa[n]["successes"], -fa[n]["deaths"]))
    comp_all = paired(fin[names[0]], fin[names[1]]) if len(names) == 2 else None
    comp_new = paired(fin[names[0]], fin[names[1]], min_episode_index=a.screen_episodes) if len(names) == 2 else None
    res = {"run": rel(run_dir), "decode": a.decode, "seed": seed, "screen_episodes": a.screen_episodes,
           "final_episodes": a.final_episodes,
           "screen": {n: {**summarize_rows(scr[n]), "checkpoint": next(e for e in entries if e["name"] == n)} for n in scr},
           "ranking_rule": "screen successes desc, then deaths asc, then later checkpoint",
           "ties_at_cut": ties, "final": {n: summarize_rows(fin[n]) for n in names},
           "winner": winner, "paired_all": comp_all, "paired_fresh_only": comp_new}
    L = [f"# Checkpoint selection on val: `{rel(run_dir)}`", "",
         f"decode {a.decode}; base seed {seed}; screening {a.screen_episodes} episodes/map x {len(maps)} maps; "
         f"top {a.top} re-run at {a.final_episodes} episodes/map on the same base seed (its first "
         f"{a.screen_episodes} episodes per map are the screening seeds). Val only; held-out never used for selection.", "",
         "## Screen", "", "| checkpoint | obs | S0 | S1 | S2 | S3 | overall | deaths |", "|---|---|---|---|---|---|---|---|"]
    for n in sorted(scr, key=lambda n: int(CHECKPOINT_RE.search(n + ".pt").group(1))):
        s = summarize_rows(scr[n])
        obs = next(e["obs"] for e in entries if e["name"] == n)
        L.append(f"| {n} | v{obs} | " + " | ".join(f_rate(s["per_stage"].get(x)) for x in STAGES)
                 + f" | {f_rate(s['overall'])} | {s['overall']['deaths']}/{s['overall']['episodes']} |")
    if ties:
        L += ["", f"Ties at the cut (same screen successes as the last selected, not re-run): {', '.join(ties)}"]
    L += ["", "## Final", "", "| checkpoint | S0 | S1 | S2 | S3 | overall | deaths |", "|---|---|---|---|---|---|---|"]
    for n in names:
        s = summarize_rows(fin[n])
        L.append(f"| {n} | " + " | ".join(f_rate(s["per_stage"].get(x)) for x in STAGES)
                 + f" | {f_rate(s['overall'])} | {s['overall']['deaths']}/{s['overall']['episodes']} |")
    if comp_all:
        L += ["", "## Top-2 paired test", "", PAIRED_HEADER, f_paired(names[0], names[1], comp_all),
              f_paired(names[0] + " (ep>=%d)" % a.screen_episodes, names[1] + " (ep>=%d)" % a.screen_episodes, comp_new)]
        sig = comp_new["mcnemar_exact_p"] < 0.05 and comp_new["map_bootstrap_ci95_diff"] and \
            not (comp_new["map_bootstrap_ci95_diff"][0] <= 0 <= comp_new["map_bootstrap_ci95_diff"][1])
        L += ["", f"Winner (most final successes, then fewer deaths): **{winner}**. Difference on the "
              f"fresh episodes (not used for screening) is {'significant' if sig else 'NOT significant'} "
              f"(needs McNemar p < 0.05 and a bootstrap interval excluding 0)."]
        res["significant_on_fresh"] = bool(sig)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "select.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_root, "select.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[write] {rel(os.path.join(out_root, 'select.md'))}; winner {winner}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--episodes-per-map", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="output root (default: manifest out_dir; fixed, so a "
                                                "resume after midnight continues the same directory)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, commands and duration estimate only")
    p.add_argument("--aggregate-only", action="store_true", help="no engine: tabulate complete/reused results")
    p.add_argument("--no-reuse", action="store_true", help="never reuse earlier result directories")
    p.add_argument("--split", default=SPLIT, help="only 'val' is accepted (held-out is refused)")
    p.add_argument("--select", metavar="RUN_DIR", help="checkpoint selection over RUN_DIR/checkpoints")
    p.add_argument("--decode", choices=("sample", "argmax"), default="sample")
    p.add_argument("--screen-episodes", type=int, default=3)
    p.add_argument("--final-episodes", type=int, default=15)
    p.add_argument("--top", type=int, default=2)
    a = p.parse_args(argv)
    if a.split != SPLIT:
        raise SystemExit(f"--split {a.split} refused: only '{SPLIT}' (held-out results are for final "
                         f"reporting only, never for comparison or checkpoint selection)")
    return select_main(a) if a.select else ablation_main(a)


if __name__ == "__main__":
    sys.exit(main())
