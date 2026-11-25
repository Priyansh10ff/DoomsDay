# 2026-11-25: Stage A0 single-instance DOOM smoke runs (1 worker x 1 env)

Entry point: `python -m train.run_doom`, TRAIN config `my_way_home`, `screen_enabled=False`. One engine at a time; every run stopped well under 20 s. These runs test plumbing only; they train nothing of value.

## Earlier runs (before ADR 0008), numbers copied from the command output

| Run | Command (beyond run dir) | Result |
|---|---|---|
| 1 | `--workers 1 --envs-per-worker 1 --minibatches 1 --rollout-len 32 --max-updates 1` | update 1 done; sampling 119 decisions/s, update 535/s; process tree 394 MiB at the update, peak 413,466,624 B (394 MiB); checkpoint written; stale time-outs 0; 4.7 s wall |
| 2 | same, `--resume <run 1 checkpoint>` | resumed at update 1, ran update 2; sampling 180/s; tree 395 MiB, peak 413,921,280 B; checkpoint written |
| 3 | `--rollout-len 540` (just above 2100 tics / frame skip 4 = 525) | 1 episode finished, return 0.954: the agent reached the goal (termination), **not a time-out**; sampling 187/s; peak 482,807,808 B (**460 MiB**); stale time-outs 0; 8.2 s wall |

All three runs had the same priority reading: main `python.exe`, worker `python.exe` and `vizdoom.exe` all `below_normal=True`, with `set_directly` empty. After each run, the count of engine processes alive after shutdown was 0.

## Rerun after ADR 0008 (bare layout: 1 x 1, 2 torch threads, minibatches 1)

Command: `python -m train.run_doom --rollout-len 32 --max-updates 1 --max-seconds 20 --run-dir <scratch>`, then the same with `--resume <checkpoint>`.

| Item | Run A | Run B (resume) |
|---|---|---|
| Planned layout printed | 1 engine instance, 1 worker, 2 torch threads, estimate 467 MiB | same |
| Priority | main, worker, `vizdoom.exe` below normal; `set_directly` empty | same |
| Update index after the run | 1 | 2 (resumed at 1; session 1; env seed 3964924996) |
| Value scale | 0.100 (new floor) | 0.100 |
| Sampling / update rate (decisions/s) | 17 / 343 | 18 / 400 |
| Process-tree peak | 412,086,272 B (393 MiB) | 414,420,992 B (395 MiB) |
| Lowest system available memory seen | 3.45 GB | 3.31 GB |
| Stale time-outs | 0 | 0 |
| Checkpoint | `checkpoint.pt` written | written, previous kept as `checkpoint.pt.prev` |
| Time from stop request to stop | 0.016 s | 0.016 s |
| Engines alive after shutdown | 0 | 0 |
| Wall time | 8.3 s | 8.6 s |

## Open points

- **Sampling rate dropped from 119-187 to 17-18 decisions/s.** The cause has not been determined.
  - It is not the worker's new parent-polling loop: a toy subprocess env measured 3376 steps/s polling every 1.0 s against 3346 steps/s at 1 ms.
  - At the time of the rerun, another `python.exe` was using about 48% CPU and system available memory was 3.4 GB of 15.4 GB.
  - The timer variance recorded by `env-engineer` (ADR 0006 item 3) is a candidate.
  - Not investigated further, to stay within the one-instance, 20-second limit.
- **No DOOM time-out has been exercised**: all stale counts of 0 are vacuous. ADR 0008 Decision 4 assigns a short-episode-limit time-out smoke run to `trainer-ops`.
- **Priority inheritance by the engine is still unproven**, because the worker lowers its engine's priority directly before the main process checks.
- **Ctrl-C reaching the engines directly, hard-kill survivors, and interrupt-then-resume with an engine** were not tested here. They are in ADR 0008 validation item 5 (`trainer-ops`).
