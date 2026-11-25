# 2026-11-25: Stage A0 trainer check on the toy memory task (no ViZDoom)

Code: `train/toy_memory.py`, test `tests/test_toy_memory.py`. CPU, `torch.set_num_threads(4)`.

Task: a cue (one of two) is shown at step 0. At step 20 the first action head must name it (reward 1, else 0). Episode length is 21, rollout length 32, so episodes straddle rollouts. 16 envs, 100 updates, about 51k decisions. The model is the A0 network at reduced widths (core 64, fusion 64). The memoryless baseline is the same network with its hidden state zeroed every step.

Toy settings: lr 1e-3, gamma 0.99, constant lr. Other settings as in ADR 0004 (clip 0.2, 3 epochs, 4 minibatches, entropy 0.01).

| Run | Success (last 200 episodes) | Max abs(ratio - 1), first minibatch |
|---|---|---|
| Recurrent | 0.985 | 7e-7 |
| Memoryless | 0.505 | 5e-7 |

Wall time for both runs: 72 s.

Diagnostics, recurrent only, seed 0:

| lr | gamma | Updates | Schedule | Success |
|---|---|---|---|---|
| 3e-4 | 0.997 | 100 | decay to 0 | 0.51 |
| 3e-4 | 0.997 | 300 | decay to 0 | 0.475 |
| 3e-4 | 0.99 | 150 | constant | 0.455 |
| 1e-3 | 0.997 | 100 | constant | 0.83 |
| 3e-4 | 0.997 | 350 | constant | 0.995 |

Reading: at the ADR 0004 values (lr 3e-4, gamma 0.997), the trainer solves the task, but slowly. It needs about 350 updates at a constant learning rate, against 100 at lr 1e-3. With linear decay to 0 over 300 updates it stayed at chance. That points to learning-rate and budget sensitivity, not a trainer bug.

The gamma effect is unclear. These are single-seed runs.

None of this is evidence about ViZDoom.

A0 parameter count is 474,474. This was computed on Doom-like spaces: rays 3x80, 24x8 entities, 14 entity types, self 28, heads (3, 3, 7, 2, 2, 2), screen off.
