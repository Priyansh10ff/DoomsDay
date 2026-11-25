"""ADR 0011 Decision 2 health monitor and E0 log forensics, applied offline to a run's logs.

Stdlib only. Reads metrics.jsonl and episodes.jsonl; starts no engine, no training, no model.

Usage:
    python -I reports/eval/scripts/adr0011_monitor_forensics.py RUN_DIR OUT_DIR [--p0-successes 28 --p0-episodes 200]

Writes to OUT_DIR:
    adr0011_levels_primary.csv   one row per metrics update: inputs, booleans, level (primary reading)
    adr0011_levels_variants.csv  per update: level under every variant reading (see VARIANTS)
    e0_per_update_400_1200.csv   v_loss, explained_variance, grad_norm, return_scale, entropies, updates 400..1200
    e0_entropy_per_update.csv    per update: each head's entropy, fraction of its maximum, lowest-fraction head
    e0_summary.json              every summary figure quoted in the report

Readings (all stated in the report, section "Ambiguities"):
  * Alignment: the monitor at metrics update u sees the first `episodes_done[u]` episodes of episodes.jsonl
    (file order = decision order), i.e. the episodes stamped update_idx <= u-1. Checked on every line.
  * Success is an integer count over the trailing 500 finished episodes; thresholds are compared on counts:
    0.5 -> 250; "0.15 below run best" -> count <= best - 75; "below p0 + 0.10" -> count < 500*(p0+0.10)
    computed exactly as a fraction (28/200 + 1/10 = 0.24 -> count < 120).
  * Primary: full windows only (no success value before 500 episodes; no 100-/50-update mean before 100/50
    updates exist); run best = max of full-window success over updates <= u, current included;
    head maxima = exact ln(k); strict "<" for "below"; event segments = maximal contiguous runs of updates
    where the event condition holds; Stop = an event segment still active 600 updates after its start
    (i.e. active at start+600).
  * Level precedence for the single `level` column: stop > event > warning > watch > none. Watch, warning
    and event are also written as independent booleans.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from fractions import Fraction

HEAD_NAMES = ["move", "strafe", "turn", "attack", "use", "speed"]
HEAD_OPTIONS = [3, 3, 7, 2, 2, 2]
EXACT_MAX = [math.log(k) for k in HEAD_OPTIONS]
ROUNDED_MAX = [1.10, 1.10, 1.95, 0.69, 0.69, 0.69]  # as printed in ADR 0011
WINDOW_EP = 500
WATCH_UPDATES = 100
WARN_UPDATES = 50
STOP_AFTER = 600


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rolling_mean(xs, n, partial):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= n:
            s -= xs[i - n]
        k = min(i + 1, n)
        out.append(s / k if (partial or i + 1 >= n) else None)
    return out


def segments(flags):
    """Maximal contiguous runs of True, as (start_update, end_update), 1-based update indices."""
    segs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i + 1
        if not f and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(flags)))
    return segs


def run_monitor(metrics, ep_success, *, head_max, partial, p0_frac, label):
    n_up = len(metrics)
    sum_max = sum(head_max)
    ent_heads = [[m[f"entropy_head{h}"] for m in metrics] for h in range(6)]
    ent_sum = [sum(ent_heads[h][i] for h in range(6)) for i in range(n_up)]
    mean100 = rolling_mean(ent_sum, WATCH_UPDATES, partial)
    mean50 = [rolling_mean(ent_heads[h], WARN_UPDATES, partial) for h in range(6)]
    # prefix success counts
    pref = [0]
    for s in ep_success:
        pref.append(pref[-1] + (1 if s else 0))
    p0_line = Fraction(p0_frac) + Fraction(1, 10)  # exact
    rows, best_num, best_den = [], None, None
    for i, m in enumerate(metrics):
        u = i + 1
        n_ep = int(m["episodes_done"])
        if n_ep >= WINDOW_EP:
            cnt, den = pref[n_ep] - pref[n_ep - WINDOW_EP], WINDOW_EP
        elif partial and n_ep > 0:
            cnt, den = pref[n_ep], n_ep
        else:
            cnt, den = None, None
        if cnt is not None:
            if best_num is None or Fraction(cnt, den) > Fraction(best_num, best_den):
                best_num, best_den = cnt, den
        succ = None if cnt is None else Fraction(cnt, den)
        best = None if best_num is None else Fraction(best_num, best_den)
        watch = (mean100[i] is not None and succ is not None
                 and mean100[i] < 0.5 * sum_max and succ < Fraction(1, 2))
        warn_heads = [h for h in range(6) if mean50[h][i] is not None and mean50[h][i] < 0.1 * head_max[h]]
        warning = bool(warn_heads)
        drop = succ is not None and best is not None and (best - succ) >= Fraction(15, 100)
        low = succ is not None and succ < p0_line
        event = warning and (drop or low)
        rows.append(dict(update=u, episodes_seen=n_ep, succ_count=cnt, succ_den=den,
                         succ=None if succ is None else float(succ),
                         run_best=None if best is None else float(best),
                         ent_sum=ent_sum[i], ent_sum_mean100=mean100[i],
                         **{f"mean50_{HEAD_NAMES[h]}": mean50[h][i] for h in range(6)},
                         watch=watch, warning=warning, warn_heads="+".join(HEAD_NAMES[h] for h in warn_heads),
                         drop_cond=drop, low_cond=low, event=event))
    ev_flags = [r["event"] for r in rows]
    segs = segments(ev_flags)
    # Stop, contiguous reading
    stop_contig = []
    for s, e in segs:
        if e >= s + STOP_AFTER:
            stop_contig.append((s, s + STOP_AFTER))
    # Stop, merged reading: first onset to last active
    stop_merged = None
    if segs:
        s0, elast = segs[0][0], segs[-1][1]
        if elast >= s0 + STOP_AFTER and ev_flags[s0 + STOP_AFTER - 1]:
            stop_merged = s0 + STOP_AFTER
    stop_at = stop_contig[0][1] if stop_contig else None
    for r in rows:
        u = r["update"]
        if stop_at is not None and u >= stop_at:
            lvl = "stop"
        elif r["event"]:
            lvl = "event"
        elif r["warning"]:
            lvl = "warning"
        elif r["watch"]:
            lvl = "watch"
        else:
            lvl = "none"
        r["level"] = lvl
    summary = dict(label=label,
                   watch_segments=segments([r["watch"] for r in rows]),
                   warning_segments=segments([r["warning"] for r in rows]),
                   warning_segments_per_head={HEAD_NAMES[h]: segments([HEAD_NAMES[h] in r["warn_heads"].split("+")
                                                                        for r in rows]) for h in range(6)},
                   event_segments=segs,
                   event_segments_detail=[dict(start=s, end=e, length=e - s + 1,
                                               heads_at_start=rows[s - 1]["warn_heads"],
                                               heads_during=sorted({h for r in rows[s - 1:e] for h in r["warn_heads"].split("+") if h}),
                                               cond_at_start=("drop" if rows[s - 1]["drop_cond"] else "") + ("+low" if rows[s - 1]["low_cond"] else ""),
                                               succ_at_start=rows[s - 1]["succ_count"], best_at_start=rows[s - 1]["run_best"],
                                               succ_at_end=rows[e - 1]["succ_count"],
                                               succ_after_end=rows[e]["succ_count"] if e < len(rows) else None,
                                               warn_heads_after_end=rows[e]["warn_heads"] if e < len(rows) else None)
                                          for s, e in segs],
                   stop_contiguous=stop_contig, stop_merged_first_onset=stop_merged,
                   event_active_at_last_update=ev_flags[-1],
                   first_merged_span=(segs[0][0], segs[-1][1]) if segs else None)
    return rows, summary


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    run_dir, out_dir = argv[1], argv[2]
    p0_s, p0_n = 28, 200
    if "--p0-successes" in argv:
        p0_s = int(argv[argv.index("--p0-successes") + 1])
    if "--p0-episodes" in argv:
        p0_n = int(argv[argv.index("--p0-episodes") + 1])
    p0 = Fraction(p0_s, p0_n)
    os.makedirs(out_dir, exist_ok=True)
    metrics = load_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    eps = load_jsonl(os.path.join(run_dir, "episodes.jsonl"))
    S = {"inputs": dict(run_dir=run_dir, n_updates=len(metrics), n_episodes=len(eps), p0=f"{p0_s}/{p0_n}")}

    # ---------- integrity / alignment checks
    assert [int(m["update_idx"]) for m in metrics] == list(range(1, len(metrics) + 1))
    dec = [e["decision"] for e in eps]
    S["episodes_sorted_by_decision"] = all(a <= b for a, b in zip(dec, dec[1:]))
    cnt_by_ep_update = {}
    for e in eps:
        cnt_by_ep_update[e["update_idx"]] = cnt_by_ep_update.get(e["update_idx"], 0) + 1
    cum, mism = 0, []
    for i, m in enumerate(metrics):
        cum += cnt_by_ep_update.get(i, 0)  # episodes stamped update_idx i are known after metrics update i+1
        if cum != int(m["episodes_done"]):
            mism.append((i + 1, cum, m["episodes_done"]))
    S["alignment_mismatches"] = mism[:10]
    S["alignment_mismatch_count"] = len(mism)
    S["success_equals_level_exit"] = all(e["success"] == e["level_exit"] for e in eps)
    S["entropy_field_minus_head_sum_max_abs"] = max(abs(m["entropy"] - sum(m[f"entropy_head{h}"] for h in range(6)))
                                                    for m in metrics)
    S["head_max_exact"] = EXACT_MAX
    S["sum_max_exact"] = sum(EXACT_MAX)
    S["half_sum_max_exact"] = sum(EXACT_MAX) / 2

    succ = [bool(e["success"]) for e in eps]

    # ---------- monitor variants
    VARIANTS = {
        "primary_exact_full": dict(head_max=EXACT_MAX, partial=False),
        "rounded_full": dict(head_max=ROUNDED_MAX, partial=False),
        "exact_partial": dict(head_max=EXACT_MAX, partial=True),
        "rounded_partial": dict(head_max=ROUNDED_MAX, partial=True),
    }
    results = {}
    for name, kw in VARIANTS.items():
        rows, summ = run_monitor(metrics, succ, p0_frac=p0, label=name, **kw)
        results[name] = (rows, summ)
    S["monitor"] = {k: v[1] for k, v in results.items()}

    prim = results["primary_exact_full"][0]
    cols = list(prim[0].keys())
    with open(os.path.join(out_dir, "adr0011_levels_primary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(prim)
    with open(os.path.join(out_dir, "adr0011_levels_variants.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = list(VARIANTS)
        w.writerow(["update"] + names)
        for i in range(len(metrics)):
            w.writerow([i + 1] + [results[n][0][i]["level"] for n in names])
    S["variant_level_differences_vs_primary"] = {
        n: [i + 1 for i in range(len(metrics)) if results[n][0][i]["level"] != prim[i]["level"]][:50]
        for n in VARIANTS if n != "primary_exact_full"}
    S["variant_level_difference_counts"] = {
        n: sum(1 for i in range(len(metrics)) if results[n][0][i]["level"] != prim[i]["level"])
        for n in VARIANTS if n != "primary_exact_full"}

    # min summed-entropy update and the per-update monitor values at the backtest's rows
    imin = min(range(len(metrics)), key=lambda i: metrics[i]["entropy"])
    S["min_entropy"] = dict(update=imin + 1, entropy=metrics[imin]["entropy"],
                            turn=metrics[imin]["entropy_head2"], move=metrics[imin]["entropy_head0"])

    # ---------- first trailing-500 success reaching 0.5 (episode granularity)
    pref = [0]
    for s in succ:
        pref.append(pref[-1] + s)
    def first_cross(pred):
        for i in range(WINDOW_EP - 1, len(eps)):
            c = pref[i + 1] - pref[i + 1 - WINDOW_EP]
            if pred(c):
                e = eps[i]
                return dict(episode_index_0based=i, episode_number_1based=i + 1, count=c,
                            episode_update_idx_stamp=e["update_idx"], aligned_metrics_update=e["update_idx"] + 1,
                            episode_decision_stamp=e["decision"],
                            trained_decisions_at_aligned_update=(e["update_idx"] + 1) * 2048,
                            half_of_episode_decision_stamp=Fraction(e["decision"], 2))
        return None
    ge = first_cross(lambda c: c >= 250)
    gt = first_cross(lambda c: c > 250)
    for d in (ge, gt):
        if d:
            d["half_of_episode_decision_stamp"] = float(d["half_of_episode_decision_stamp"])
    S["first_trailing500_success_ge_0.5"] = ge
    S["first_trailing500_success_gt_0.5"] = gt
    # update-granularity (what the monitor would see)
    S["first_metrics_update_with_trailing500_ge_250"] = next((r["update"] for r in prim if r["succ_count"] is not None and r["succ_count"] >= 250), None)
    S["first_metrics_update_with_trailing500_gt_250"] = next((r["update"] for r in prim if r["succ_count"] is not None and r["succ_count"] > 250), None)
    # non-overlapping blocks of 500 (the run report's method)
    blocks = []
    for b in range(0, len(eps) // WINDOW_EP):
        lo, hi = b * WINDOW_EP, (b + 1) * WINDOW_EP
        blocks.append(dict(episodes=f"{lo}-{hi}", success=(pref[hi] - pref[lo]) / WINDOW_EP,
                           end_decision=eps[hi - 1]["decision"]))
    S["blocks_500"] = blocks
    S["first_block_ge_0.5"] = next((b for b in blocks if b["success"] >= 0.5), None)
    S["check_1213904_times_2"] = 1213904 * 2
    # last 500
    S["last500_success"] = (pref[-1] - pref[-1 - WINDOW_EP]) / WINDOW_EP

    # ---------- backtest table re-computation (per-100-update blocks, lesson 11 alignment)
    ep_by_aligned = {}
    for idx, e in enumerate(eps):
        ep_by_aligned.setdefault(e["update_idx"] + 1, []).append(idx)
    bt = []
    for lo in range(1, len(metrics) + 1, 100):
        hi = min(lo + 99, len(metrics))
        ms = metrics[lo - 1:hi]
        idxs = [j for u in range(lo, hi + 1) for j in ep_by_aligned.get(u, [])]
        row = dict(updates=f"{lo}-{hi}", n_updates=len(ms), n_episodes=len(idxs),
                   success=(sum(succ[j] for j in idxs) / len(idxs)) if idxs else None,
                   ent_sum=statistics.fmean(m["entropy"] for m in ms))
        for h in range(6):
            row[HEAD_NAMES[h]] = statistics.fmean(m[f"entropy_head{h}"] for m in ms)
        lv = [prim[u - 1]["level"] for u in range(lo, hi + 1)]
        row["levels_in_block"] = {k: lv.count(k) for k in ("none", "watch", "warning", "event", "stop") if lv.count(k)}
        bt.append(row)
    S["per_100_update_blocks"] = bt
    # the ADR's run-best figures 0.41 and 0.654 vs trailing-window run best
    S["run_best_trailing_at"] = {str(u): prim[u - 1]["run_best"] for u in (700, 800, 900, 950, 1000, 1100, 1200, 1300, 1814)}
    S["run_best_trailing_count_at"] = {}

    # ---------- which head fell first
    fell = {}
    for h in range(6):
        raw = [m[f"entropy_head{h}"] for m in metrics]
        m50 = rolling_mean(raw, WARN_UPDATES, False)
        d = {}
        for frac in (0.5, 0.25, 0.1):
            line = frac * EXACT_MAX[h]
            d[f"raw_first_below_{frac}"] = next((i + 1 for i, x in enumerate(raw) if x < line), None)
            d[f"mean50_first_below_{frac}"] = next((i + 1 for i, x in enumerate(m50) if x is not None and x < line), None)
        d["raw_min"] = min(raw)
        d["raw_min_update"] = raw.index(min(raw)) + 1
        fell[HEAD_NAMES[h]] = d
    S["head_first_crossings"] = fell
    with open(os.path.join(out_dir, "e0_entropy_per_update.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["update"] + [f"ent_{n}" for n in HEAD_NAMES] + [f"frac_{n}" for n in HEAD_NAMES] + ["lowest_frac_head"])
        lowest_counts = {}
        for i, m in enumerate(metrics):
            e = [m[f"entropy_head{h}"] for h in range(6)]
            fr = [e[h] / EXACT_MAX[h] for h in range(6)]
            lo_h = HEAD_NAMES[min(range(6), key=lambda h: fr[h])]
            if 400 <= i + 1 <= 1200:
                lowest_counts[lo_h] = lowest_counts.get(lo_h, 0) + 1
            w.writerow([i + 1] + e + fr + [lo_h])
    S["lowest_fraction_head_counts_updates_400_1200"] = lowest_counts

    # ---------- E0: health metrics 400..1200
    keys = ["v_loss", "explained_variance", "grad_norm", "return_scale", "approx_kl", "clipfrac", "entropy"]
    with open(os.path.join(out_dir, "e0_per_update_400_1200.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["update"] + keys + [f"entropy_head{h}" for h in range(6)] + ["trailing500_succ_count", "level"])
        for u in range(400, 1201):
            m = metrics[u - 1]
            w.writerow([u] + [m[k] for k in keys] + [m[f"entropy_head{h}"] for h in range(6)]
                       + [prim[u - 1]["succ_count"], prim[u - 1]["level"]])
    e0b = []
    for lo in range(400, 1201, 100):
        hi = min(lo + 99, 1200)
        ms = metrics[lo - 1:hi]
        r = dict(updates=f"{lo}-{hi}")
        for k in ("v_loss", "explained_variance", "grad_norm", "return_scale"):
            vals = [m[k] for m in ms]
            r[k] = dict(mean=statistics.fmean(vals), min=min(vals), max=max(vals), median=statistics.median(vals))
        e0b.append(r)
    S["e0_health_blocks_400_1200"] = e0b
    rs = [m["return_scale"] for m in metrics]
    S["return_scale_max_ratio_after_50"] = max(rs[i] / rs[i - 1] for i in range(50, len(rs)))

    # ADR 0009 ten-fold trigger: value loss or grad_norm jumping >10x "when the first goal rewards arrive"
    first_goal_ep = next(i for i, s in enumerate(succ) if s)
    first_goal_update = eps[first_goal_ep]["update_idx"] + 1
    tf = dict(first_goal_episode_index=first_goal_ep, first_goal_aligned_update=first_goal_update,
              first_25_updates={k: [metrics[i][k] for i in range(25)] for k in ("v_loss", "grad_norm")})
    for k in ("v_loss", "grad_norm"):
        v = [m[k] for m in metrics]
        ratios = [(i + 1, v[i] / v[i - 1]) for i in range(1, len(v))]
        tf[f"{k}_consecutive_ratio_gt10"] = [(u, r) for u, r in ratios if r > 10]
        tf[f"{k}_max_consecutive_ratio"] = max(ratios, key=lambda t: t[1])
        tf[f"{k}_max_consecutive_ratio_updates_400_1200"] = max((t for t in ratios if 400 <= t[0] <= 1200), key=lambda t: t[1])
        # vs update 1 (the pre-goal baseline)
        tf[f"{k}_updates_gt10x_update1_first100"] = [i + 1 for i in range(100) if v[i] > 10 * v[0]]
        # settle: first update after the jump at which the value is back under 10x update 1
        tf[f"{k}_ratio_to_update1_at_2_to_25"] = [v[i] / v[0] for i in range(1, 25)]
    S["adr0009_tenfold"] = tf

    # ---------- per env / per worker success and successful-episode lengths, before / during / after
    segs = results["primary_exact_full"][1]["event_segments"]
    if segs:
        ev_start, ev_end = segs[0][0], segs[-1][1]
    else:
        ev_start = ev_end = None
    S["phase_definition"] = dict(event_first_start=ev_start, event_last_end=ev_end,
                                 before=f"aligned update 1..{ev_start - 1}" if ev_start else None,
                                 during=f"aligned update {ev_start}..{ev_end}" if ev_start else None,
                                 after=f"aligned update {ev_end + 1}..{len(metrics)}" if ev_start else None)

    def phase(e):
        u = e["update_idx"] + 1
        if ev_start is None:
            return "all"
        return "before" if u < ev_start else ("during" if u <= ev_end else "after")
    per_env, per_worker, lens = {}, {}, {}
    for e in eps:
        p = phase(e)
        k = (p, e["env_index"])
        per_env.setdefault(k, [0, 0])
        per_env[k][0] += e["success"]
        per_env[k][1] += 1
        kw = (p, e["env_index"] // 8)
        per_worker.setdefault(kw, [0, 0])
        per_worker[kw][0] += e["success"]
        per_worker[kw][1] += 1
        if e["success"]:
            lens.setdefault(p, []).append(e["length"])
    S["per_env"] = {f"{p}|env{env}": dict(succ=s, n=n, rate=s / n) for (p, env), (s, n) in sorted(per_env.items())}
    S["per_worker"] = {f"{p}|worker{wk}": dict(succ=s, n=n, rate=s / n) for (p, wk), (s, n) in sorted(per_worker.items())}

    def qs(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(math.floor(q * (len(v) - 1) + 0.5)))]  # nearest-rank on (n-1)
    S["success_lengths"] = {p: dict(n=len(v), mean=statistics.fmean(v), median=statistics.median(v),
                                    p10=qs(v, 0.10), p25=qs(v, 0.25), p75=qs(v, 0.75), p90=qs(v, 0.90),
                                    min=min(v), max=max(v)) for p, v in lens.items()}
    # per-phase overall success
    tot = {}
    for e in eps:
        p = phase(e)
        tot.setdefault(p, [0, 0])
        tot[p][0] += e["success"]
        tot[p][1] += 1
    S["phase_success"] = {p: dict(succ=s, n=n, rate=s / n) for p, (s, n) in tot.items()}
    # per-env during: spread
    dur = [v["rate"] for k, v in S["per_env"].items() if k.startswith("during|")]
    if dur:
        S["per_env_during_spread"] = dict(min=min(dur), max=max(dur), median=statistics.median(dur))

    # matched 300-update windows either side of the event (in addition to the full before/after phases)
    if ev_start is not None:
        near = {"near_before": (max(1, ev_start - 300), ev_start - 1), "during": (ev_start, ev_end),
                "near_after": (ev_end + 1, min(len(metrics), ev_end + 300))}
        out = {}
        for name, (lo, hi) in near.items():
            sel = [e for e in eps if lo <= e["update_idx"] + 1 <= hi]
            v = [e["length"] for e in sel if e["success"]]
            wk = {}
            for e in sel:
                wk.setdefault(e["env_index"] // 8, [0, 0])
                wk[e["env_index"] // 8][0] += e["success"]
                wk[e["env_index"] // 8][1] += 1
            out[name] = dict(updates=f"{lo}..{hi}", n_episodes=len(sel), succ=sum(e["success"] for e in sel),
                             rate=sum(e["success"] for e in sel) / len(sel),
                             per_worker={f"worker{k}": dict(succ=a, n=b, rate=a / b) for k, (a, b) in sorted(wk.items())},
                             success_lengths=dict(n=len(v), mean=statistics.fmean(v), median=statistics.median(v),
                                                  p10=qs(v, 0.10), p25=qs(v, 0.25), p75=qs(v, 0.75), p90=qs(v, 0.90),
                                                  min=min(v), max=max(v)))
        S["matched_300_update_windows"] = out

    with open(os.path.join(out_dir, "e0_summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1, default=str)
    print(json.dumps({k: S[k] for k in ("alignment_mismatch_count", "entropy_field_minus_head_sum_max_abs",
                                         "first_trailing500_success_ge_0.5", "first_trailing500_success_gt_0.5",
                                         "phase_definition")}, indent=1, default=str))
    for n, (_, s) in results.items():
        print(n, "events", s["event_segments"], "stop", s["stop_contiguous"], s["stop_merged_first_onset"],
              "active_at_end", s["event_active_at_last_update"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
