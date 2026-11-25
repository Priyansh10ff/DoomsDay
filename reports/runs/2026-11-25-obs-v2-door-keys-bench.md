# Observation v2 (door channel + key bits): step-cost benchmark, 2026-11-25

- **Setup:** single environment, one engine, 1 process, frame skip 4, 160x120, HUD on, screen (pixels) **off** (A0 sensors), Windows 1 ms timer on, random actions. Measured after the night1 run had stopped, so no other engines were running.
- **Vectorized:** not measured. The task allowed at most one engine of our own.

## Throughput, `python -m env.bench --cap 12 --screen 0 --obs-version {1,2} [--generated DD0023]`

| Map | v1 decisions/s | v2 decisions/s | Change | v1 step p50/p90/p99 (ms) | v2 step p50/p90/p99 (ms) |
|---|---|---|---|---|---|
| generated S1-train-101004 (lump DD0023, 12 sectors) | 1106 | 826 | **-25%** | 0.83 / 1.17 / 3.44 | 0.98 / 1.54 / 6.41 |
| freedoom2 MAP01 (EVAL config, 211 sectors, 2,036 lines) | 859 | 185 | **-78%** | 0.96 / 1.50 / 5.74 | 4.72 / 8.99 / 14.46 |

## Interleaved A/B step latency (3 rounds x 800 steps per version, p50)

| Map | v1 p50 (ms) | v2 p50 (ms) | Ratio |
|---|---|---|---|
| DD0023 (S1) | 0.80 | 0.98 | 1.23 |
| DD0040 (S2) | 0.81 | 0.98 | 1.22 |
| MAP01 | 0.92 | 4.06 | 4.42 |

The same A/B measurement, taken earlier while the 32-engine run was live, gave 1.23 / 1.26 / 5.47.

## Where the time goes (MAP01)

- Engine side, sector info on, no Python observation code: `advance_action` rises from 0.95 to 3.35 ms and `get_state` from 0.02 to 1.04 ms.
- Python side (`--per-key`): reading sector heights takes about 0.27 ms, the door tracker loop 0.19 ms, and the cast 0.28 ms.
- The cost scales with the number of sectors and lines on the map. It cannot be paid every N decisions only: `set_sectors_info_enabled(False)` after `init()` has no effect.
- On generated training maps the extra is about 0.18 ms per decision. About 0.08 ms of that is Python.

Raw JSON: written to the session scratchpad only, not kept.
