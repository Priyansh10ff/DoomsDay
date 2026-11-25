# 2026-11-25: Pre-training checks, single engine instance (ADR 0008 validation 5)

Author: trainer-ops. One engine instance at a time, 2 torch threads, below-normal priority. **All run names use `nogit`**: the repository has no commits, so no run is tied to an exact code version, and the whole working tree is uncommitted.

**Code changed during this session.** `worker` edited `train/trainer.py` (07:43:11), `train/run_loop.py` (07:43:51) and `train/doom_factory.py` (07:44:02) to add per-episode logging. The time-out smoke run and the first time-out probe ran on the code from before that edit, and the smoke run's directory has no `episodes.jsonl`. Every later run used the edited code. The time-out probe was re-run on the edited code and gave identical results. None of my runs failed on an import.

## How runs were launched

`train.run_doom` has no flag for the episode time limit or for the worker design, so I added a launcher, `configs/ops_launch.py`, with one JSON config per run in `configs/ops/`. It does three things:

- It substitutes `a0_train_env_config` and `SafetyConfig` inside `train.run_doom` for its own process only.
- It prints and asserts the effective values.
- It copies the JSON to `<run>/config.json` and calls `run_doom.main()`.

No file outside `configs/` was changed. Run name = `20261125-nogit-<first 8 hex of the SHA-256 of the JSON>`.

## a. Time-out smoke: PASS

Config `configs/ops/timeout_smoke.json` (`episode_timeout_tics` 200, which is 50 decisions at frame skip 4; 120 s). Run `20261125-nogit-2d4e952a`.

**Which path fired.** A scratchpad probe ran two `Trainer.collect()` rollouts of 128 steps at 1x1 with the same environment settings:

| Rollout | Truncated at step | Terminated | `valid` all True | `bootstrap_value` at the truncated steps | Non-zero bootstrap elsewhere | Stale |
|---|---|---|---|---|---|---|
| 0 | 49, 99 | none | yes | 0.0428, 0.0651 | 0 | 0 |
| 1 | 21, 71, 121 | none | yes | -0.0121, 0.0486, 0.0305 | 0 | 0 |

Episode lengths were exactly 50. The wrapper's own limit fired, which is the fresh-final-observation path: truncated, `valid=True`, and a bootstrap value computed from V(final observation). The engine's backstop limit never fired.

**Trainer run.** 250 updates (16,000 decisions) in 120 s.
- Stale 0 on every update; summary `stale_timeouts_total` 0; `valid_frac` 1.0 throughout.
- 323 episodes finished, about the 320 expected from 50-decision time-outs. `recent_return` peaked at 0.081, so roughly one episode reached the goal.
- Sampling: median 177 decisions/s (118 to 259); update median 617/s.
- Process tree 409 MiB (429,174,784 B); lowest system available memory 5.44 GiB.
- Priority: all three processes below normal, none set directly.
- Stop 0.015 s after the request; checkpoint written; 0 engines after shutdown.

**Metrics.** No NaN or inf.
- Entropy fell from 6.22 to 2.90, with a minimum of 2.00, within 250 updates. This is a fast early decline, but not a collapse.
- KL peaked at 0.011; clip fraction peaked at 0.19; value loss had one spike to 13.4.
- Explained variance was mostly strongly negative (last -43.7). The resume leg of `20261125-nogit-8653f5c1`, at the standard 2100-tic limit, also logged -36.1 at update 107. **Cause not established** (low return variance is one candidate). Routed to `worker`.
- This is a plumbing test; nothing of value was learned.

## b. Hard-kill test (ADR 0008 ruling 6): no survivors with either design

Method:
1. Start a 1x1 run in the background and wait for training updates (at least 40 lines in `metrics.jsonl`).
2. Record the main process and all its descendants, with PID and create time.
3. Kill only the main process with `psutil.Process(pid).kill()`, not a tree kill.
4. Count recorded PIDs still alive with the same create time at 1, 3, 5, 10, 15 and 20 s.
5. Count `vizdoom.exe` system-wide.

| Design | Run | Recorded (main, worker, engine) | Survivors at 1 s to 20 s | `vizdoom.exe` after |
|---|---|---|---|---|
| Non-daemon (`worker_daemon=False`, the default) | `20261125-nogit-a9f0b2fb` | 32512, 32692, 22168 | 0 | 0 |
| Daemon (`worker_daemon=True`) | `20261125-nogit-7cfb625c` | 8852, 14144, 12840 | 0 | 0 |

Both designs leave no survivors. The parent-poll loop closes the game within a second either way. Per ruling 6, the target design, **non-daemon**, stays as the default. Neither run reached its 10-minute checkpoint, so neither has one, which is expected for a hard kill. Logs: `hardkill.log` and `hardkill_pids.json` in each run directory.

**Incident.** My first PID-finding one-liner matched its own command line, which contained the search strings. It killed its own parent shell (bash 17196) and so itself, exit 15. The trainer was untouched, and the test was redone with explicit PIDs. No process I did not start was affected.

## c. Interrupt then resume: PASS, with real OS Ctrl-C untested

**OS-level Ctrl-C was not delivered.** A helper that would have started the trainer in its own hidden console and sent `CTRL_C_EVENT` there was blocked by a security hook:

> Prismor blocked this action: [CRITICAL] Blocks reverse shells, bash -i /dev/tcp, crontab injection (rule: rce-canary)

I stopped that action and did not work around it. Two things are therefore still untested and need the user at a terminal: (1) that Ctrl-C from the console reaches the trainer's handler, and (2) whether console Ctrl-C reaches the engine processes directly.

**What was tested.** Run `20261125-nogit-8653f5c1` (config `configs/ops/interrupt_resume.json`).

- **Interrupt.** A scratchpad probe ran the real `run_training` at 1x1 with its own `StopController`. A timer thread called `stop.on_sigint()`, the same function the SIGINT handler calls, at a random 19.20 s.
  - The interrupt arrived while update 41 was in progress. The update finished, then `checkpoint written (interrupt) at update 41` and `stopped (interrupt) 0.11s after the request`.
  - The checkpoint loads with `weights_only=True`: `update_idx` 41, session 0, equal to the last line of `metrics.jsonl`.
  - Stale 0; 0 engines after close.
  - Log: `interrupt.log`.
- **Resume.** The real launcher with `--resume <run>/checkpoint.pt`, 30 s: `resumed ... at update 41, session 1, env seed 3964924996`, a new session seed, not the run seed 0. It continued to update 107, checkpoint written, previous kept as `checkpoint.pt.prev`, 0 engines. Log: `resume_stdout.log`.

## Issues routed

- `worker`: `run_doom` takes no config file and exposes neither `episode_timeout_tics` nor `worker_daemon`. trainer-ops needs a `--config` path, so no substitution is needed.
- `worker`: the trainer writes no TensorBoard or W&B logs, only `metrics.jsonl` and `episodes.jsonl`. `tensorboard` is not installed.
- `worker`: there is no validation evaluation, so no best-by-val checkpoint exists. `map-forge` and `evaluator`: training uses the single fixed map `my_way_home`, and there is no val split yet.
