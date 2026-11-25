# 2026-11-25: ADR 0008 toy runs (Gate A, Measurement B, floor)

## How these runs were made

- **Code:** `python -m train.toy_experiments --budget-min 84`, plan in `train/toy_experiments.py`.
- **Machine load:** one process, 4 torch threads, below-normal priority (`train/process_control.py`), one run at a time.
- **Toy settings:** 16 envs, rollout 32, lr 1e-3, gamma 0.99, core 64.
- **Base recipe:** value-scale floor 0.1, 3 epochs, init as built, value_coef 0.5.
- **Score:** success is the fraction of correct answers over the last 200 episodes.
- **Ratio check:** the first-minibatch ratio deviation was at most 9.5e-7 in every run.
- **Time:** 83.8 minutes of compute in total; runs that did not fit the budget were skipped, listed at the end.

## Gate A: gap 4, 100 updates, 8 seeds. PASS

Pass rule: every recurrent seed at or above 0.95; every memoryless seed within 0.1 of 0.5.

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Recurrent | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Memoryless | 0.500 | 0.515 | 0.560 | 0.470 | 0.450 | 0.520 | 0.490 | 0.520 |

All 8 recurrent seeds are at or above 0.95, all 8 memoryless seeds are within 0.1 of 0.5, and the ratio deviation stays below 1e-5. Mean final return scale: 0.325 (recurrent) and 0.434 (memoryless).

In pytest, `tests/test_toy_memory.py` runs a subset (recurrent seeds 0 and 1, memoryless seed 0); all 8 seeds take about 7.4 minutes.

## Measurement B: 300 updates, 8 seeds per arm

Pass rule: at least 7 of 8 seeds at or above 0.9.

### Gap 20 (within one 32-step rollout)

| Arm | Change | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | Seeds at or above 0.9 | Mean final scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | base | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 | 0.980 | **8/8** | 0.253 |
| B1 | update-gate retention bias | 1.000 | 0.945 | 0.995 | 1.000 | 0.950 | 0.990 | 1.000 | 1.000 | **8/8** | 0.286 |
| B2 | B1 plus dense cue label | 0.965 | 0.995 | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 | **8/8** | 0.269 |
| E | 2 epochs | 1.000 | 0.995 | 1.000 | 0.995 | 1.000 | 0.995 | 1.000 | 1.000 | **8/8** | 0.276 |
| I | orthogonal init | 1.000 | 0.990 | 0.990 | 1.000 | 0.990 | 0.985 | 1.000 | 1.000 | **8/8** | 0.256 |

How B2 was built: `ToyConfig.dense_cue_label` makes the toy env report the cue in `info["keys_held"]` at every step. The existing keys head (on the GRU state, weight 0.1) then learns that dense label, so this arm tests the keys-head mechanism itself.

### Gap 60 (longer than the rollout: the cue and the answer never share a backpropagation window)

| Arm | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | Seeds at or above 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 0.550 | 0.460 | 0.480 | 0.510 | 0.520 | 0.530 | 0.480 | 0.495 | **0/8 (FAIL)** |
| B2 | 0.445 | 0.480 | 0.480 | 0.520 | not run | not run | not run | not run | **0/4 (FAIL on all seeds run)** |

At gap 60 both arms stay at chance. The mean final return scale was about 0.185.

### Reading against ADR 0008's adoption rule (no default was changed)

- **B0 now passes at gap 20.** It fails only at gap 60, where B0 itself was not run. Compared with the 0007-rulings runs, this moved with the floor change 1.0 to 0.1 and 300 updates.
- **B1 and B2:** each scored 8/8 against B0's 8/8. The rule requires "higher", so the tie keeps the base. **Not adopted.**
- **E and I:** each scored 8/8 against 8/8. ADR 0008 says the change is adopted if the count "is not lower than B0's (for E and I)". It also says "Ties keep the base". These conflict for exactly this case, so this is a **question for advisor**. Its reversal clause ("E or I beats B0 by 2 seeds or more") is not met.
- **Ceiling effect:** at gap 20 every arm is at the ceiling, so this measurement cannot separate them. The per-seed scores above are the only finer signal (lowest per arm: B0 0.980, B1 0.945, B2 0.965, E 0.995, I 0.985).
- **Gap 60 fails for B1 and B2.** Per ADR 0008, no long-horizon memory claim can be made, and any key-and-door curriculum stage stays blocked.

## Cue trace at initialization (ADR 0008 validation 2)

Metric: L2 difference of the hidden state at step 20 between two episodes that differ only in the cue. Seed 0, toy network, `train.toy_memory.cue_trace`.

| Init | Step-20 trace |
|---|---|
| Base | 1.34e-6 (debugger reported 1.5e-6) |
| Retention bias on GRUCell bias chunk 1 (B1) | 4.59e-4 (about 340 times higher) |
| Same bias on chunk 0 | 1.20e-5 |
| Same bias on chunk 2 | 2.07e-6 |
| Orthogonal init (I) | 1.68e-5 |

Chunk 1 is therefore the update gate that keeps state. `tests/test_agent_init_schemes.py` checks this.

## Floor measurement (gap 4, rewards times 0.05, 100 updates, 3 seeds): mostly not run

| Floor | s0 | s1 | s2 |
|---|---|---|---|
| 0.01 | 1.000 | 1.000 | not run |
| 0.1 | not run | not run | not run |
| 1.0 | not run | not run | not run |

Mean final scale for floor 0.01: 0.019.

## Skipped for budget

- B@60 B2, seeds 4 to 7.
- Floor 0.01 seed 2; floor 0.1 and floor 1.0, all seeds.

The runner checks whether each run's estimated time still fits the budget. B@60 B2 seeds 4 to 7 (about 85 s each) did not fit, but the shorter floor runs (about 30 s) did. So floor 0.01 seeds 0 and 1 ran after B2 was cut, which departs slightly from the stated order.
