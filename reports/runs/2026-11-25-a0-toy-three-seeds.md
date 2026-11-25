# 2026-11-25: Toy memory gate on three seeds (ADR 0007 validation 1): FAILED

Code: `train/toy_memory.py`, run in one process with 4 torch threads, sequentially.

Setup:
- Task: cue at step 0, recall at step 20. Episode length 21, rollout 32, 16 envs, reduced widths (core 64).
- Trainer settings after ADR 0007: value-scale floor 1.0, monster target min(count, 8)/8, damage label read from the final info.
- Success is the mean reward over the last 200 episodes.
- Gate: every recurrent run at or above 0.9, and every memoryless run within 0.1 of 0.5.

| Seed | Recurrent, 1e-3, gamma 0.99, 100 updates | Memoryless, 1e-3, gamma 0.99, 100 updates | Recurrent, 3e-4, gamma 0.997, constant, 350 updates |
|---|---|---|---|
| 0 | 0.605 | 0.505 | 0.775 |
| 1 | 0.485 | 0.49 | 0.57 |
| 2 | 0.495 | 0.495 | 0.51 |

Result:
- No recurrent run reaches 0.9, so the gate fails on every seed.
- The memoryless runs behave as expected (0.49 to 0.505).
- The first-minibatch ratio deviation is at most 5e-7 in all runs, so the recompute path is consistent.

Ablation: value-scale floor 0.01 (the value before ADR 0007), recurrent, 1e-3, 100 updates.

| Seed | Floor 1.0 (ADR 0007) | Floor 0.01 |
|---|---|---|
| 0 | 0.605 | 0.87 |
| 1 | 0.485 | 0.745 |

Exact repro: seed 0 with floor 0.01 and `monster_count_cap=4` gives 0.985, identical to the pre-ADR-0007 result. With 4 toy entities, a cap of 4 equals the old count/4 target. So the two ADR 0007 value changes fully explain the drop on seed 0, and the damage-label change has no effect on this task.

Seed 1 fails even at floor 0.01 (0.745), so the earlier single-seed pass was read too confidently.

Hypothesis for `advisor`: any answer bias makes returns depend on the cue. The value loss then trains the memory directly, and the 1.0 floor weakens exactly that signal.

Reading:
- On this task returns lie in [0, 1], with a standard deviation near 0.5. With a floor of 1.0 the targets are no longer scaled up, which effectively cuts the value-loss weight by about 4.
- Learning is slower and very sensitive to seed and small changes even at floor 0.01. The earlier pass at 100 updates was near the edge of the budget.
- This does not point to a correctness bug: the ratio checks pass, and sequence replay matches stepwise.
- These are hypotheses for `advisor`, not established. Per ADR 0007's reversal clause, DOOM training stays blocked until `advisor` rules.
