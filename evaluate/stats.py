"""Aggregate episode records: success rate with a Wilson 95 % score interval."""
from __future__ import annotations

import math
from collections import defaultdict


def wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(rows: list[dict], z: float) -> dict:
    n = len(rows)
    s = sum(r["success"] for r in rows)
    lo, hi = wilson_interval(s, n, z)
    succ_len = [r["length"] for r in rows if r["success"]]
    return {
        "episodes": n,
        "successes": s,
        "success_rate": s / n if n else float("nan"),
        "success_ci95_wilson": [lo, hi],
        "exit_unknown": sum(r["exit_unknown"] for r in rows),
        "mean_length": _mean([r["length"] for r in rows]),
        "mean_length_successful": _mean(succ_len),
        "death_rate": _mean([float(r["death"]) for r in rows]),
        "timeout_rate": _mean([float(r["timeout"]) for r in rows]),
        "mean_return": _mean([r["return"] for r in rows]),
        "mean_kills": _mean([r["kills"] for r in rows]),
    }


def summarize(rows: list[dict], z: float) -> dict:
    by_stage, by_map = defaultdict(list), defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
        by_map[r["map_id"]].append(r)
    return {
        "overall": aggregate(rows, z),
        "per_stage": {k: aggregate(v, z) for k, v in sorted(by_stage.items())},
        "per_map": {k: aggregate(v, z) for k, v in sorted(by_map.items())},
    }
