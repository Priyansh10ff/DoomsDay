# Single-environment throughput, env slice 1

Date: 2026-11-25. Command: `python -m env.bench 3000 <out.json> <timer_ms>` (`env/bench.py`).

## Setup

- **Environment:** one `DoomEnv`, 1 process, plus its `vizdoom.exe` child. No vectorization.
- **Config:** `EVAL_CONFIG`, i.e. freedoom2 MAP01, skill 3, 160x120, HUD on, frame skip 4.
- **Observation:** all contract keys on, including the pixel key `screen` (3,60,80).
- **Actions:** uniform random over the six heads (MultiDiscrete [3,3,7,2,2,2]); USE is split 1 tic + 3 tics on about half the steps.
- **Warm-up:** 50 decisions before timing.
- **Machine:** laptop CPU "Intel64 Family 6 Model 198", 24 logical CPUs; Windows 11; Python 3.12.10; vizdoom 1.3.3.

## Results

| Run | Timer request | Decisions | Full env decisions/s | Tics/s | Raw engine decisions/s |
|---|---|---|---|---|---|
| A | 1 ms | 3000 | **368.4** | 1474 | 582.9 |
| B | 1 ms | 3000 | **419.1** | 1677 | 845.8 |
| C | 1 ms | 5000 | 75.0 (unexplained) | 300 | 830.3 |
| D | none | 3000 | **76.1** | 304 | 70.0 |
| E | 1 ms | 3000 | **703.3** | 2813 | 1667.1 |

- "Raw engine" is the same game with the same buffers, one `advance_action(4)` and one `get_state()` per decision, and no extraction.
- Run E is the final code, with a 40 s wall-clock cap added to the timed loop. Its step latency was p50 1.15 ms, p90 1.65 ms and p99 6.84 ms.
- Run C ran before the double-`get_state` fix and before latency logging. Its full-env phase was slow even though the raw-engine phase of the same process was fast.
- **Cause unexplained.** Without latency percentiles for that run, it is not known whether its steps were quantised at 15.6 ms.
- Run-to-run variance with the timer request is large (368 to 703). Treat about 370 decisions/s as the conservative figure.
- Runs A to D did not have the wall-clock cap. Run C took 66.7 s, above the 60 s guideline.
- A separate diagnostic loop, the same env without the fix and then with it, gave 50 and 611 decisions/s.

**Per-key extraction cost** (microseconds per state, mean over 200 captured states, run A):

| Key | Cost (µs) |
|---|---|
| `screen` (2x2 area average) | 239 |
| `rays` | 36 |
| `self` (including the variable split and ego-motion) | 8 |
| `entities` + `entity_type` + `entity_mask` | 4 |
| state field access | 1 |

- About 1.2 labels were visible per state on average (maximum 8 to 11), so the entity cost will grow on busier maps.

**Peak RAM:**

| Process | Peak working set |
|---|---|
| Python | 64.9 MB |
| `vizdoom.exe` | 71.6 MB |

## Reading

- **The engine round trip dominates.** Extraction totals about 0.29 ms per decision, against roughly 2.5 ms per decision overall.
- **Against the ADR 0001 threshold** (reverse below 250 decisions/s): with the timer request, the env clears it in 3 of 4 runs (368 to 703). Without it, the env is at 76.
- **Cheapest next win:** `screen` is 80% of extraction. Making it optional, or computing it only when the pixel branch is on, would save about 0.24 ms per decision.
- **Vectorized throughput was not measured.** The user limited this run to one instance.
