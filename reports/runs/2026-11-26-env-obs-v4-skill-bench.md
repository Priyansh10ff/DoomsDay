# Env benchmark: observation v4 (locked-door egomap channel) and per-episode skill, 2026-11-26

Measured by env-engineer while the 24-engine night-3b run was live (PID 27344), so absolute numbers are noisy; only one engine of ours at a time. Vectorized throughput was **not** measured (one-engine limit while the run is live).

Setup: generated TRAIN map S2-train-102001, 1 process x 1 env, frame skip 4, pixels off, Windows timer 1 ms, random actions (attack off), 3 interleaved rounds of 800 decisions per version (script: env-engineer scratchpad `bench_v4.py`).

| | step p50 | step p90 | mean decisions/s |
|---|---|---|---|
| v3, run 1 | 1.42 ms | 4.69 ms | 437 |
| v4, run 1 | 1.73 ms | 6.93 ms | 325 |
| v3, run 2 | 1.54 ms | 5.82 ms | 387 |
| v4, run 2 | 1.56 ms | 6.93 ms | 345 |

Isolated cost of the egomap code (no engine, 3000 updates, update + crop p50): v3 0.24-0.26 ms, v4 with no mark 0.24 ms, v4 with one kept mark 0.25-0.27 ms. So v4 adds about 0.01-0.02 ms per decision per remembered locked door and nothing otherwise; the run-1 gap is load noise from the live run (run 2 p50 differs by 0.02 ms).

Reset (new_episode) p50: same skill 9.7 / 14.5 ms, with a skill switch (console `skill`) 9.8 / 13.1 ms (two runs): no measurable cost. A full engine re-init would cost about 0.9-1.2 s.
