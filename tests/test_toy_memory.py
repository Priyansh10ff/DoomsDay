"""Gate A (ADR 0008 Decision 2): trainer correctness on the toy memory task.

Gap 4, 100 updates, base recipe (floor 0.1, 3 epochs, built init). Pass: every
recurrent seed >= 0.95; every memoryless seed within 0.1 of 0.5; first-minibatch
ratio deviation < 1e-5 in every run.

All 8 seeds take about 8 minutes, too slow for the suite, so pytest runs a subset
(recurrent seeds 0 and 1, memoryless seed 0). The full 8-seed gate is
`python -m train.toy_experiments --only gateA,gateA-memoryless --out <file>`
(results in reports/runs/2026-11-25-adr0008-toy-gate-and-memory.md).

Measurement B (gaps 20 and 60) is not a pytest gate: it does not block pipeline runs.
"""
import time

import pytest

import a0_helpers  # noqa: F401
from train.toy_experiments import (GATE_A_MEMORYLESS_BAND, GATE_A_RECURRENT_MIN, GATE_A_UPDATES,
                                   RATIO_DEV_MAX, plans)
from train.toy_memory import ToyRunSpec, run_toy

GATE_A = next(p for p in plans() if p.group == "gateA")
PYTEST_RECURRENT_SEEDS = (0, 1)
PYTEST_MEMORYLESS_SEEDS = (0,)


def test_gate_a_definition_matches_adr_0008() -> None:
    assert GATE_A.arm.toy.answer_step == 4 and GATE_A.updates == GATE_A_UPDATES == 100
    assert GATE_A.seeds == tuple(range(8)) and GATE_A.arm.model == {} and GATE_A.arm.ppo == {}
    assert GATE_A_RECURRENT_MIN == 0.95 and GATE_A_MEMORYLESS_BAND == 0.1 and RATIO_DEV_MAX == 1e-5


@pytest.mark.parametrize("seed", PYTEST_RECURRENT_SEEDS)
def test_gate_a_recurrent(seed: int) -> None:
    t0 = time.perf_counter()
    r = run_toy(True, seed, ToyRunSpec(updates=GATE_A.updates), arm=GATE_A.arm)
    print(f"gate A recurrent seed {seed}: {r} in {time.perf_counter() - t0:.1f}s")
    assert r["max_first_ratio_dev"] < RATIO_DEV_MAX
    assert r["success"] >= GATE_A_RECURRENT_MIN


@pytest.mark.parametrize("seed", PYTEST_MEMORYLESS_SEEDS)
def test_gate_a_memoryless(seed: int) -> None:
    r = run_toy(False, seed, ToyRunSpec(updates=GATE_A.updates), arm=GATE_A.arm)
    print(f"gate A memoryless seed {seed}: {r}")
    assert r["max_first_ratio_dev"] < RATIO_DEV_MAX
    assert abs(r["success"] - 0.5) <= GATE_A_MEMORYLESS_BAND
