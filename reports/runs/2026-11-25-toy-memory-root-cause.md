# 2026-11-25: Toy memory gate failure, root cause (debugger)

Question: is the failing toy gate (recurrent 0.605 vs 0.9 needed) a PPO bug, a task/budget problem, or hyperparameter sensitivity?

Answer: **no trainer bug found.** The 20-step cue lag is not learnable from the answer-step gradient alone at this network size and init within the budget. The memory is learned through the **value loss** (value head gradient through the GRU). ADR 0007's scale floor of 1.0 weakens that signal by about 3 to 4 times on this task (return std is about 0.29). That accounts for most of the drop. The worker found that the monster-cap change also moved seed 0 from 0.87 to 0.985 at floor 0.01; that residual was not isolated here. Even with the old signal strength the gate is not met on all seeds at 100 updates (see section 5 for longer runs). This is a design and budget question for `advisor`.

Setup: CPU, 4 torch threads, one run at a time, toy settings from `train/toy_memory.py` (16 envs, rollout 32, lr 1e-3, gamma 0.99, 100 updates) unless stated. Success = mean reward over the last 200 episodes. Scripts lived in the session scratchpad, outside the repo. Total compute was about 17 minutes.

## 1. Init diagnostic (seed 0, no training)

- Flipping the cue changes the fused GRU input by 0.09 (L2 norm). The noise changes it by 1.63 from one step to the next.
- The update gate z is about 0.50, so the cue difference in h halves each step: 2.5e-2 at t=0, 2.4e-3 at t=4, 3.2e-4 at t=8, 1.5e-6 at t=20.
- So at init the gradient from the answer step back to the cue is about 1e-5 of its size at lag 0.

## 2. Supervised BPTT on the same network (full 21-step episodes, h0 = 0, 64 episodes per step, lr 1e-3)

| Loss | Lag | Seed | Steps | Final acc |
|---|---|---|---|---|
| CE at last step only | 1 (T=2) | 0 | 100 | 1.00 |
| CE at last step only | 4 (T=5) | 0 | 400 | 1.00 (jumps between 100 and 200) |
| CE at last step only | 9 (T=10) | 0 | 100 | 0.56 (inconclusive: fewer steps than lag 4 needed) |
| CE at last step only | 20 (T=21) | 0 | 500 | 0.48 (loss 0.693) |
| CE at last step only | 20 | 1 | 300 | 0.41 |
| CE at last step only | 20 | 2 | 300 | 0.50 |
| CE at every step (dense) | 20 | 0 / 1 / 2 | 300 | 1.00 / 1.00 / 1.00 (all by step 200) |

Even with clean labels and 19k to 32k episodes, a loss only at the answer step did not learn the 20-step lag within 500 steps. A per-lag (dense) signal learns it on every seed. In PPO only the value loss supplies a per-lag signal: once the answer has any bias, V(h_t) depends on the cue at every t.

## 3. PPO positive control: `ToyConfig(answer_step=4)`, floor 1.0, otherwise unchanged (episodes still straddle rollouts)

| Seed | 0 | 1 | 2 |
|---|---|---|---|
| Success | 1.000 | 0.995 | 1.000 |

The hidden-state carry across rollouts, episode-start masks, GAE and replay all work. The current code solves a lag the network can learn.

## 4. PPO at lag 20: value-pathway ablations (success; learning curve = last-100 mean every 25 updates)

| Config | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Floor 1.0, value_coef 0.5 (current ADR 0007) | 0.605 | 0.485 | 0.495 |
| Floor 0.01, value_coef 0.5 (pre-ADR 0007) | 0.870 | 0.745 | 0.690 |
| Floor 0.01, value head on `h.detach()` | 0.510 | 0.470 | 0.495 |
| Floor 1.0, value_coef 2.0 | 0.960 | 0.645 | 0.555 |
| Floor 1.0, value_coef 0.5, rollout 42 (episode-aligned), 76 updates | 0.495 | 0.515 | 0.690 |

- Floor 0.01 gives a return scale of about 0.29, so targets are multiplied by about 3.4.
- When the value head cannot send gradient into the GRU, learning fails on all seeds, even with a strongly biased answer (P(answer 0) reached 0.08 to 0.90). The policy gradient alone does not build the memory.
- Order of events, from probes every 25 updates: the cue difference in V over steps 1 to 19 rises before or with the cue difference in the answer logits. Seed 0 at floor 0.01, update 50: dval 1.87, dlogit 1.09, P(answer 0) 0.16. Seed 1, update 75: dval 0.043, dlogit 0.004.
- Raising value_coef 4 times at floor 1.0 restores seed 0 (0.96) but not seeds 1 and 2 (0.645, 0.555). Curves were still rising at update 100, for example seed 1 [0.6, 0.46, 0.65, 0.73].
- Aligning rollouts with episodes (so the cue and answer are always in one BPTT window) did not help clearly at this budget.

## 5. Budget: current config (floor 1.0, value_coef 0.5), 300 updates

| Seed | Success (last 200) | First update where the last-100 mean is at least 0.9 (checked every 25) |
|---|---|---|
| 0 | 0.935 | 125 |
| 1 | 0.970 | 250 |
| 2 | 0.495 | never (stays 0.42 to 0.51 throughout) |

More updates fix 2 of 3 seeds. The onset is a sudden jump whose timing depends on the seed, which is consistent with waiting for a chance answer bias to start the value-loss path. Seed 2 shows no onset in 300 updates.

Total compute for this investigation: about 25 minutes.

## Ruled out

- Hidden-state handling across rollouts and minibatch sequence slicing: the lag-4 control passes, and the first-minibatch ratio deviation is below 1e-6.
- GAE termination: the lag-4 control passes. The truncation/bootstrap path is not exercised by the toy (it only terminates); `tests/test_train_gae.py` covers it.
- Advantage normalization: masked per-minibatch mean/std over valid steps, by inspection of `train/ppo.py`, plus the lag-4 control.
- Return-scale arithmetic: inert at floor 1.0 (scale = 1.000).
- Aux losses: weights 0.1 on near-constant targets (keys invalid, damage 0, monsters 0 or 0.125). Aux off at floor 1.0, value_coef 0.5: 0.745 / 0.490 / 0.580 (vs 0.605 / 0.485 / 0.495 with aux on). That is within the seed spread; no seed reaches the gate, so aux weighting is not the cause.

## For `advisor`

The gate as specified (every seed at 0.9 or above at lag 20, 100 updates) is not met by the current or the pre-ADR 0007 configuration. The original 0.985 was a favourable seed.

Options to rule on, none applied here:
- Restore the value signal strength (floor or value_coef).
- Shorten the toy lag or give it a larger budget.
- Change GRU init or input scale so the cue survives the lag at init (for example an update-gate bias toward keeping state).
- Accept that, in DOOM, memory will mostly be learned through the value and aux losses.
