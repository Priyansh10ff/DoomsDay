# Single-environment throughput after ADR 0006 (timer on/off, screen on/off)

Date: 2026-11-25. Run by env-engineer for ADR 0006 validation item 3.

## Setup

- **Command:** `python -m env.bench --timer {1|0} --screen {1|0} [--per-key] --out <json>` (`env/bench.py`).
- **Instances:** one `DoomEnv` and its `vizdoom.exe` child per run. Runs were sequential with a 3 s gap. There was no vectorization and never more than one instance.
- **Config:** EVAL sensors (freedoom2 MAP01, skill 3, 160x120, HUD on, frame skip 4), with ADR 0006 changes:
  - USE is held for the whole decision, so every step is one engine round trip.
  - Ray bands are horizon offsets.
- **Actions:** uniform random over the six heads, after 50 warm-up decisions.
- **Run length:** each timed loop stops at 20,000 decisions or 25 s of wall time.
- **Power:** AC power for every run (battery 93%), Windows power scheme "Turbo", read with `powercfg /getactivescheme`.
- **Machine:** Intel64 Family 6 Model 198, 24 logical CPUs; Windows 11; Python 3.12.10; vizdoom 1.3.3.
- `$DOOMSDAY_WIN_TIMER_MS` was unset; the script refuses to run if it is set.

## Results

"≥14 ms" is the share of steps at or above 14 ms, which marks the 15.6 ms timer mode.

| Run | Timer | Screen | Decisions | Seconds | Decisions/s | Step ms p50 / p90 / p99 | ≥14 ms |
|---|---|---|---|---|---|---|---|
| on-1 | 1 ms | on | 19002 | 25.00 | 760.1 | 1.02 / 1.68 / 6.36 | 0.0001 |
| on-2 | 1 ms | on | 20000 | 22.45 | 890.8 | 0.96 / 1.39 / 3.66 | 0.0000 |
| on-3 | 1 ms | on | 20000 | 23.05 | 867.7 | 0.97 / 1.40 / 3.97 | 0.0001 |
| on-4 | 1 ms | on | 20000 | 22.81 | 876.9 | 0.96 / 1.43 / 3.80 | 0.0001 |
| on-5 | 1 ms | on | 20000 | 21.38 | 935.7 | 0.94 / 1.33 / 3.47 | 0.0000 |
| off-1 | off | on | 2079 | 25.01 | 83.1 | 1.41 / 46.40 / 63.77 | 0.3353 |
| off-2 | off | on | 2319 | 25.06 | 92.5 | 1.47 / 46.34 / 63.77 | 0.2794 |
| off-3 | off | on | 1872 | 25.04 | 74.8 | 1.45 / 56.52 / 63.88 | 0.3515 |
| off-4 | off | on | 2045 | 25.01 | 81.8 | 1.41 / 47.43 / 63.88 | 0.3178 |
| off-5 | off | on | 5092 | 25.06 | 203.2 | 1.11 / 13.27 / 63.33 | 0.0966 |
| extra-1 | 1 ms | **off** | 20000 | 17.08 | 1170.8 | 0.72 / 1.04 / 3.28 | 0.0001 |
| extra-2 | 1 ms | **off** | 20000 | 16.05 | 1246.0 | 0.70 / 0.99 / 2.70 | 0.0001 |

**Summary:**

| Condition | Runs | Decisions/s |
|---|---|---|
| Timer on, screen on | 5 | median 877, range 760-936 |
| Timer off, screen on | 5 | median 83, range 75-203 |
| Timer on, screen off | 2 | 1171 and 1246 |

**Per-key extraction cost** (run on-1, µs per state, 200 states): screen 236, rays 80, self 7, entities 3.

**Peak RAM per run:**

| Process | Peak working set |
|---|---|
| Python | 47 to 67 MB (67 MB in the run that also captured 200 states for the per-key timings) |
| `vizdoom.exe` | 72 to 73 MB |

**CPU:** the Python process used 20.0 CPU-seconds in run on-1's 25 s loop, about 0.8 of one core. Child-process CPU was not measured.

## Reading

- **Timer off:** the latency distribution is clearly quantised. The median step is fast (1.4 ms), but 28-35% of steps wait one or more 15.6 ms ticks: p90 is about 46.8 ms (3 ticks) and p99 about 63.8 ms (4 ticks). The cause is the timer, as found in round 1, now with percentiles.
- **Timer on:** no run showed the 15.6 ms mode in more than 0.01% of steps. The round-1 "75/s with the timer on" anomaly did not recur in 5 runs, so ADR 0006's reverse criterion ("a 15.6 ms mode in more than a small share of steps") is not triggered.
- **Run off-5 (203/s):** this timer-off run was faster than the other four. Its ≥14 ms share was 9.7% against about 31%. A plausible cause is another process on the machine raising the system timer resolution at that moment. That is **not verified**, so this run shows that timer-off throughput depends on what else is running.
- **Compared with round 1:** timer-on runs rose from 368-703 to 760-936 decisions/s. Two code changes happened in between: the split USE advance was removed (ADR 0006 ruling 2) and the second `get_state()` per step was removed. Neither was measured on its own, and power state was not recorded in round 1, so **how much each change contributed is not known**.
- **Against ADR 0006's estimate:** turning the timer off made the env about ten times slower (median 83 against 877 decisions/s), not "about five times".
- **Screen off:** saves about 0.24 ms per decision, about +35% throughput (877 to about 1200 decisions/s), consistent with the per-key cost.
- **Not measured:** vectorized or multi-process throughput, and child-process CPU.

Raw JSON per run was written to the session scratchpad (`bench2/t{timer}_s{screen}_r{n}.json`). It is not copied into the repo.

## Addendum: after the round-3 ray fix (same day)

Two 15 s runs (timer on, screen on, AC power, "Turbo"), after `invalid_depth_mask` was added to `rays()`, the entity depth was changed to use valid pixels only, and the time limit was moved into the wrapper:

| Run | Decisions/s | Step ms p50 / p90 / p99 | ≥14 ms |
|---|---|---|---|
| r3-1 | 359.2 | 2.12 / 4.15 / 10.31 | 0.0002 |
| r3-2 | 377.3 | 1.91 / 4.12 / 10.56 | 0.0 |

**Not attributed to the change.** The mask plus rays, timed in isolation on synthetic buffers, cost 57 µs per decision. Per-key costs of code that did not change also rose against run on-1 (screen 236 to 301 µs, entities 2.5 to 16 µs, self 7 to 10 µs), which points to a slower machine state. During these runs another `python.exe` used about 34% of one core, and the system was at 11.5% CPU. The run-to-run cause is not isolated: no run without the change was made in the same machine state. Treat about 360 decisions/s as the current conservative single-env figure until a same-state A/B run is done.

**Still owed:** a clean before/after comparison (round-2 code against round-3 code, alternating runs, same machine state, no other heavy process) is still needed to settle whether the drop from a median of 877 to 359-377 decisions/s is the change or the machine. It was deliberately not run during round 3, because another agent was using about 3.5 cores for experiments.
