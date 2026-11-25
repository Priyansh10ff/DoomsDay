# Door cue occlusion gate (audit 02 F1 / R9): cost and effect, 2026-11-26 ~02:30

Change: `env/door_cue.py` adds an occlusion gate to the v2 door cue (`rays[3]`, and so egomap
channel 3). A column lights only if the depth check passes AND no eye-level-blocking line of
the level is crossed by the column's ray more than 1 unit before the matched door face
(blocking = one-sided, or two-sided with an opening that does not clear the band
eye +- 0.044 x distance; blockers nearer than ~11.7 units, which read depth level 0 and are
masked, are left to the depth check). Scripts: `reports/runs/scripts/door_gate_probe.py`
(old-vs-new on real frames), `reports/runs/scripts/door_gate_ab_bench.py` (A/B wrapper around
`tools/env_bench`).

## Effect on real frames (obs v2, final constants)

| Set | Frames | Lit old | Lit new | New not in old | Removed | Removed with residual > -1.5 (likely visible) |
|---|---|---|---|---|---|---|
| 7 generated TRAIN maps (teacher, 2 seeds x <= 500) | 998 | 17,607 | 17,601 | 0 | 6 | 0 |
| freedoom2 MAP01 (random, forward-biased, 3 seeds x <= 1,500) | 2,299 | 4,566 | 4,490 | 0 | 76 | 31 |

Generated maps: all 6 removed columns are the audit's case occurring in play (S2-train-102001,
a one-sided wall 24-27 units in front of the face, residual -3.3, inside the old tolerance).
MAP01: most removed columns sit at -2 .. -5 levels at 470-500 units behind stair/ledge lines
or one-sided walls (leaks). The 31 near-zero ones are door-recess corner columns (a wall end
3-9 units before the face at the bin centre), blockers at 12-14 units, and ledges whose edge is
within ~10 units of eye level at ~290 units; the 2-D bin-centre ray and band-height model are
conservative there (0.7% of MAP01 lit columns).

Tuning on MAP01 (same seeds): band slope 0.05 + view-bob slack 8 removed 93 likely-visible
columns; 0.05 without bob 32; 0.035 without bob 20. Chosen: 0.044 (3.5 rows / 80 px focal,
upper end of the derived range), no bob slack.

Residual of faces that are the geometric first blocker, beyond 150 units: generated n = 2,082,
p5 -0.52, p50 +0.20, p99 +0.80, 16 in -6 .. -2; MAP01 41 in -6 .. -2 of 1,647. The R5.2
cluster persists among geometrically visible faces, so the 5% relative tolerance is kept;
occlusion is carried by the gate.

## Throughput (single env, interleaved A/B, random actions, obs v2, frame skip 4, screen off,
## 4,000 decisions or 15 s; machine shared with other agents' engines, whole-step numbers noisy)

| Map | Door code ms/decision old -> new | decisions/s old | decisions/s new | step p50 ms old -> new |
|---|---|---|---|---|
| generated DD0040 (S2-train-102001) | 0.11-0.12 -> 0.18-0.20 | 975, 879 | 826, 708 | 0.88-0.98 -> 0.98-1.07 |
| freedoom2 MAP01 (211 sectors, 1,274 lines) | 0.32-0.43 -> 0.50-0.60 | 202, 196 | 156, 157 | 4.0-4.4 -> 5.3 |

Door code cost: +0.07 ms per decision on generated maps, +0.17 ms on MAP01 (timed directly).
The whole-step drop measured (about 10-20%) is larger than the door-code delta explains and
varied run to run with other engines on the machine; an earlier 3-rep run of a slightly
heavier version showed 0-10%. `--per-key` on MAP01: scene_now 271 us (one pass over sectors,
line table cached per episode), cast + gate 40 us. Vectorised throughput not measured
(one-engine limit tonight).
