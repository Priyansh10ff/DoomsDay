"""ADR 0008 Decision 2 toy runs: Gate A, Measurement B arms, the floor measurement.

    python -m train.toy_experiments --out <results.jsonl> --budget-min 85

One run at a time, 4 torch threads, below-normal priority (user's load limit). Runs
in the order ADR 0008 and the coordinator give; stops when the time budget is used
(a run is only started if its estimated time still fits) and after a Gate A failure
(ADR 0008: "If it fails on any seed: stop, hand to debugger"). Each result is
appended to the JSONL file as soon as it finishes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from typing import Any

from . import process_control as PC
from .toy_memory import ToyArm, ToyConfig, ToyRunSpec

SEEDS_8 = tuple(range(8))
SEEDS_3 = (0, 1, 2)
GATE_A_UPDATES = 100
MEASURE_B_UPDATES = 300
GATE_A_RECURRENT_MIN = 0.95
GATE_A_MEMORYLESS_BAND = 0.1
RATIO_DEV_MAX = 1e-5
B_SUCCESS = 0.9
B_REQUIRED_SEEDS = 7
FLOOR_REWARD_SCALE = 0.05


@dataclass(frozen=True)
class Plan:
    group: str
    arm: ToyArm
    recurrent: bool
    updates: int
    seeds: tuple[int, ...]
    est_s_per_run: float


def gap(g: int, **toy_kw: Any) -> ToyConfig:
    return replace(ToyConfig(), answer_step=g, **toy_kw)


B1_MODEL = {"gru_update_bias_init": True}


def plans() -> list[Plan]:
    """Order: Gate A, B0 at gap 20, then the arms as ADR 0008 lists them, gap 60, floor."""
    a4 = ToyArm("A", gap(4))
    out = [
        Plan("gateA", a4, True, GATE_A_UPDATES, SEEDS_8, 30),
        Plan("gateA-memoryless", a4, False, GATE_A_UPDATES, SEEDS_8, 30),
        Plan("B@20", ToyArm("B0", gap(20)), True, MEASURE_B_UPDATES, SEEDS_8, 85),
        Plan("B@20", ToyArm("B1", gap(20), model=B1_MODEL), True, MEASURE_B_UPDATES, SEEDS_8, 85),
        Plan("B@20", ToyArm("B2", gap(20, dense_cue_label=True), model=B1_MODEL), True, MEASURE_B_UPDATES, SEEDS_8, 85),
        Plan("B@20", ToyArm("E", gap(20), ppo={"epochs": 2}), True, MEASURE_B_UPDATES, SEEDS_8, 65),
        Plan("B@20", ToyArm("I", gap(20), model={"init_scheme": "orthogonal"}), True, MEASURE_B_UPDATES, SEEDS_8, 85),
        Plan("B@60", ToyArm("B1", gap(60), model=B1_MODEL), True, MEASURE_B_UPDATES, SEEDS_8, 85),
        Plan("B@60", ToyArm("B2", gap(60, dense_cue_label=True), model=B1_MODEL), True, MEASURE_B_UPDATES, SEEDS_8, 85),
    ]
    for floor in (0.01, 0.1, 1.0):
        out.append(Plan("floor", ToyArm(f"floor{floor}", gap(4, reward_scale=FLOOR_REWARD_SCALE),
                                        ppo={"return_scale_min": floor}),
                        True, GATE_A_UPDATES, SEEDS_3, 30))
    return out


def gate_a_failed(results: list[dict[str, Any]]) -> bool:
    for r in results:
        if r["group"] == "gateA" and r["success"] < GATE_A_RECURRENT_MIN:
            return True
        if r["group"] == "gateA-memoryless" and abs(r["success"] - 0.5) > GATE_A_MEMORYLESS_BAND:
            return True
        if r["group"].startswith("gateA") and r["max_first_ratio_dev"] >= RATIO_DEV_MAX:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    a.add_argument("--budget-min", type=float, default=85.0)
    a.add_argument("--only", default=None, help="comma-separated plan groups/arms to run, e.g. 'gateA'")
    ns = a.parse_args(argv)
    PC.set_below_normal()
    import torch

    from .toy_memory import TOY_TORCH_THREADS, run_toy
    torch.set_num_threads(TOY_TORCH_THREADS)
    budget_s = ns.budget_min * 60.0
    t0 = time.monotonic()
    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    wanted = set(ns.only.split(",")) if ns.only else None
    for p in plans():
        tag = f"{p.group}/{p.arm.name}"
        if wanted and p.group not in wanted and p.arm.name not in wanted and tag not in wanted:
            continue
        for seed in p.seeds:
            if gate_a_failed(results):
                skipped.append(f"{tag} seed {seed} (gate A failed)")
                continue
            if time.monotonic() - t0 + p.est_s_per_run > budget_s:
                skipped.append(f"{tag} seed {seed} (budget)")
                continue
            ts = time.monotonic()
            r = run_toy(p.recurrent, seed, ToyRunSpec(updates=p.updates), arm=p.arm)
            row = {"group": p.group, "arm": p.arm.name, "gap": p.arm.toy.answer_step, "recurrent": p.recurrent,
                   "updates": p.updates, "seed": seed, **r, "seconds": round(time.monotonic() - ts, 1)}
            results.append(row)
            with open(ns.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    with open(ns.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"skipped": skipped, "elapsed_min": round((time.monotonic() - t0) / 60, 1)}) + "\n")
    print(f"skipped: {len(skipped)}; elapsed {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
