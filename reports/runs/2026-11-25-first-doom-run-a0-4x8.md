# First DOOM training run: Stage A0 on `my_way_home`, 4 x 8

Run folder: `reports/runs/20261125-nogit-7a675759/` (`stdout.log`, `metrics.jsonl`, `episodes.jsonl`, `config.json`, `checkpoint.pt`, `checkpoint.pt.prev`).
Baseline data: `reports/runs/2026-11-25-random-baseline.json`.
Written by the coordinator session from the log files; every number below is computed from those files.

## Verdict (ADR 0009 rule)

**By ADR 0009 as written: no verdict.** The ADR says "no verdict" when sampling stays below 50 decisions/s per engine instance for more than 10 minutes. With 32 engines this run sampled about 17 to 21 decisions/s per engine for its whole length (540 to 660 in total), so that condition was tripped by the choice of layout, not by a fault.

**Under the corrected throughput rule (half the benchmarked rate for the chosen layout; benchmark 520 decisions/s for 4 x 8, so the line is 260): learning, on the training scenario.** That correction was given by `advisor` in a reply and is **not yet recorded in any decision file**; until it is (planned as ADR 0011), the "learning" reading rests on an unrecorded rule. The learning conditions themselves are all met:

| ADR 0009 condition | Required | Measured | Met |
|---|---|---|---|
| Decisions | at least 1,000,000 | 3,715,072 trained | yes |
| Success, last 500 episodes, minus random | at least +0.15 | 0.726 - 0.140 = **+0.586** | yes |
| Mean return, last 500 episodes, minus random | at least +0.15 | 0.616 - (-0.052) = **+0.668** | yes |
| Success rising (by thirds of the run) | rising | 0.335 (first third) to 0.617 (last third); first 500 episodes 0.168, last 500 0.726; **not monotonic**, see below | yes, with a caveat |
| Sampling per engine instance | at least 50 decisions/s (as written) | about 17 to 21 | **no as written**; yes under the unrecorded correction |

Standard errors: last-500 success 0.020, last-500 return 0.023; baseline return 0.028 (200 episodes), baseline success about 0.025.

**What this does not show.** `my_way_home` is one map with random start positions and no enemies, doors or keys. The result says the observation pipeline, model, PPO and run infrastructure work together and the policy learns to reach the goal on that map. It says nothing about unseen maps, combat, or keys and doors. No held-out evaluation exists yet. One seed only.

## Setup

- Config: `configs/ops/a0_4x8_main.json` (hash `7a675759`), launched with `python -u -m configs.ops_launch`.
- 4 workers x 8 envs = 32 engine instances, 4 torch threads, rollout 64, below-normal priority, 8 GiB cap, checkpoint every 600 s, stop after 8400 s.
- Reward: the scenario's built-in sparse reward only (goal +1, living reward -0.0001 per tic). No shaping, no exploration bonus.
- Model: Stage A0, 474,474 parameters, `screen` off.
- Code version: none (`nogit`, the repository has 0 commits).
- **Correction to `config.json`:** its description says the layout was "recommended by trainer-ops". That is wrong. trainer-ops recommended 4 x 2; the coordinator chose 4 x 8. The JSON is left unedited because the run name contains its hash.
- **Order deviation:** ADR 0009 asks for the random baseline before launch. It was measured after the run (same config, so the comparison is still valid).

## Outcome

- Started 08:04:45, stopped by `max_seconds` at 10:25:02, exit code 0, stop latency 0.03 s.
- 1,814 updates, 10,060 episodes, 3,715,072 decisions trained (the last finished episode is stamped 3,714,880). 14 checkpoints written.
- Engine processes alive after shutdown: 0. Deaths: 0 (the scenario has no enemies). Every failure is a time-out at 525 decisions.

## Random-policy baseline

Uniform random actions (`env.action_space.sample()`), same TRAIN config, one engine, 200 episodes, 96,034 decisions, 451.5 s.

| Metric | Value |
|---|---|
| Success (level exit) | 0.140 |
| Mean return | -0.052 (std 0.393) |
| Mean length | 480.2 decisions (median 525 = time-out) |
| Time-out rate | 0.860 |

The run's own first 100 episodes (untrained network): success 0.22, return +0.037.

## Curve, 500-episode windows

| Episodes | Decisions at end | Success | Mean return | Mean length | Time-out |
|---|---|---|---|---|---|
| 0-500 | 239,776 | 0.168 | -0.018 | 466.2 | 0.832 |
| 500-1000 | 450,272 | 0.306 | +0.138 | 420.0 | 0.694 |
| 1000-1500 | 647,904 | 0.372 | +0.214 | 395.2 | 0.628 |
| 1500-2000 | 841,856 | 0.410 | +0.255 | 386.4 | 0.590 |
| 2000-2500 | 1,039,104 | 0.354 | +0.196 | 394.7 | 0.646 |
| 2500-3000 | 1,236,768 | 0.366 | +0.208 | 394.6 | 0.634 |
| 3000-3500 | 1,434,176 | 0.368 | +0.209 | 396.8 | 0.632 |
| 3500-4000 | 1,639,808 | 0.288 | +0.124 | 409.4 | 0.712 |
| 4000-4500 | 1,854,624 | 0.220 | +0.051 | 427.6 | 0.780 |
| 4500-5000 | 2,075,200 | 0.176 | -0.002 | 445.6 | 0.824 |
| 5000-5500 | 2,271,296 | 0.294 | +0.136 | 395.2 | 0.706 |
| 5500-6000 | 2,427,808 | 0.654 | +0.530 | 311.4 | 0.346 |
| 6000-6500 | 2,593,824 | 0.600 | +0.468 | 330.8 | 0.400 |
| 6500-7000 | 2,747,424 | 0.624 | +0.501 | 308.0 | 0.376 |
| 7000-7500 | 2,910,752 | 0.550 | +0.419 | 327.1 | 0.450 |
| 7500-8000 | 3,066,016 | 0.614 | +0.490 | 311.2 | 0.386 |
| 8000-8500 | 3,227,904 | 0.608 | +0.481 | 321.6 | 0.392 |
| 8500-9000 | 3,402,944 | 0.562 | +0.423 | 352.1 | 0.438 |
| 9000-9500 | 3,559,264 | 0.620 | +0.496 | 311.1 | 0.380 |
| 9500-10000 | 3,698,464 | 0.718 | +0.606 | 279.5 | 0.282 |

Last 500 episodes: success 0.726, return +0.616. Per-env success over the last 2,000 episodes ranges from 0.47 to 0.77 across the 32 envs.

## The mid-run collapse (the main thing to investigate)

Success plateaued near 0.37, then **fell back to 0.176, close to the random 0.14**, between episodes 3,500 and 5,000, and recovered sharply to 0.65 after episode 5,500. Policy entropy over the same period (mean per 100 updates; maximum possible is about 6.22):

| Updates | Entropy mean | Entropy min | move head | turn head |
|---|---|---|---|---|
| 1-100 | 5.62 | 5.31 | 0.98 | 1.67 |
| 401-500 | 5.05 | 4.33 | 0.99 | 1.64 |
| 501-600 | 3.42 | 2.64 | 0.58 | 1.42 |
| 601-700 | 1.65 | 0.88 | 0.30 | 0.69 |
| 701-800 | 1.22 | 0.37 | 0.44 | 0.30 |
| 801-900 | 1.16 | 0.66 | 0.27 | 0.21 |
| 901-1000 | 0.52 | 0.12 | 0.11 | 0.02 |
| 1001-1100 | 1.11 | 0.57 | 0.28 | 0.01 |
| 1201-1300 | 2.15 | 1.74 | 0.58 | 0.05 |
| 1701-1800 | 2.51 | 2.21 | 0.17 | 0.30 |

Reading: entropy began falling (around update 500) **before** success fell (around episode 3,500, update 700). The policy became nearly deterministic (entropy minimum 0.122 at update 953, where turn-head entropy was 0.0019 out of a possible 1.95; 0.02 is the 901-1000 average), and success recovered while the turn head was still almost fixed. ADR 0009's entropy-collapse trigger only covers the first 50 updates, so nothing fired. This looks like a premature collapse onto one behaviour, not a learned skill. It then recovered without intervention. **The cause is not established.** Candidates to check, none tested: entropy coefficient too low for a sparse reward; the turn head locking onto one bin (spinning or never turning); return-scale normalisation interacting with a falling success rate. A second seed is needed to know whether the collapse or the recovery is the typical behaviour.

## Health metrics (means over six equal parts of the run)

| Metric | 1 | 2 | 3 | 4 | 5 | 6 | Min | Max |
|---|---|---|---|---|---|---|---|---|
| Entropy | 5.56 | 4.52 | 1.31 | 1.16 | 2.44 | 2.50 | 0.122 | 6.22 |
| Explained variance | 0.790 | 0.827 | 0.806 | 0.831 | 0.719 | 0.728 | -4.07 | 0.999 |
| Approx. KL | 0.0069 | 0.0056 | 0.0030 | 0.0034 | 0.0063 | 0.0061 | 0.0001 | 0.0255 |
| Clip fraction | 0.100 | 0.075 | 0.038 | 0.042 | 0.082 | 0.080 | 0 | 0.317 |
| Gradient norm | 1.10 | 0.82 | 0.63 | 0.68 | 0.92 | 0.94 | 0.16 | 1.70 |
| Return scale | 0.139 | 0.189 | 0.184 | 0.193 | 0.217 | 0.226 | 0.100 | 0.230 |
| Sampling, decisions/s | 661 | 653 | 639 | 643 | 574 | 540 | 457 | 826 |

- No non-finite value in any metric. `valid_frac` 1.0 throughout. Stale time-outs: 0.
- Auxiliary losses (keys, damage, monster count) are near zero because the scenario has no keys, damage or monsters; they were not exercised.
- Sampling speed fell from about 660 to about 540 decisions/s over the run. Not explained (shorter episodes mean more resets; untested).

## Laptop load

- Process-tree memory: 2.47 to 2.62 GiB (2.65 to 2.82 GB; cap 8 GiB, never approached).
- Free system memory: minimum 3.52 GiB (3,777 MB), typically 4.1 to 4.4 GiB.
- All 37 processes ran at below-normal priority. CPU was 27% at the one point it was sampled (08:21); not logged continuously.

## Not done / open

- No held-out or vanilla-EVAL evaluation of the checkpoint.
- One seed. The collapse and recovery are unexplained.
- Rewards and the coverage bonus (`rewards/`, `intrinsic/`) were not used.
- Baseline measured after launch instead of before.
- The corrected throughput rule is not in any decision file (for `advisor`, ADR 0011), and ADR 0009's entropy trigger does not cover a late collapse.
- `train/running_stats.py:4` has a comment saying the value-scale floor is 1.0; the config uses 0.1 (for `worker`).

_Corrected 2026-11-25 after `teacher`'s review found the throughput condition missing from the verdict and four wrong figures._
