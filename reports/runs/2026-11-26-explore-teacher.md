# Exploration teacher: build, measurement, bc_v6_explore collection (2026-11-26)

## What was built
- `teachers/explore.py` (new, training-only): `ExploreMap` (static WAD geometry), `Knowledge`
  (per-episode seen cells from 80 view rays over the 90-degree FOV at the student's column
  angles, capped at 450 units, the egomap sensor range; rays blocked by one-sided lines and
  by the faces of doors whose live gap is below 40; cells beyond a wall hit become known
  blocked; keys seen when their cell is seen; exit seen when the switch line is hit or a
  pad cell is seen; locked doors learned only from the env's needs-key pulse after an
  executed USE, attributed to the door under the 64-unit use trace at the press pose),
  and `ExploreController(TeacherController)`: goal = exit (seen, reachable through known
  space) > seen key not held > known locked door with key held > cheapest frontier (path
  cost + up to 8 cells for a frontier behind; single-cell unknown pockets ignored;
  passable seen doors with unknown space beside them count as frontiers). The goal is kept
  until resolved / timed out; frontiers reached but unresolved, and timed-out goals, are
  blacklisted. Dijkstra over KNOWN cells only; the controller runs on a dynamic Route over
  the known grid (door anchors on the path, then the goal). USE labels use lock knowledge
  limited to known-locked doors. Combat unchanged.
- `teachers/collect.py`: `--teacher route|explore` (default route; route commands unchanged).
  Episode summaries gain `teacher` and, for explore, replans/goals/seen share/locks learned.
- `tests/test_teachers_explore.py`: 9 tests (plans never touch unseen cells; rays stop at
  closed doors and pass open ones; exit ignored until seen; lock learned only from a pulse
  after a press, not pressed again without the key, pressed with it; `--teacher` default).

## Measurement (train-all-b2, maps S{0..3}-train-10s001..020, 1 episode each, noise 0)
| limit | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 8400 tics: success / mean length (decisions) / time-outs | 20/20, 70, 0 | 20/20, 136, 0 | 20/20, 230, 0 | 20/20, 312, 0 |
| 4200 tics (default) | 20/20, 70, 0 | 20/20, 136, 0 | 20/20, 230, 0 | 20/20, 312, 0 |
| 8400 tics, noise 0.2 (seed 7) | 20/20, 128, 0 | 20/20, 267, 0 | 20/20, 362, 0 | 20/20, 508, 0 |
| route teacher, 8400, noise 0 (reference) | 20/20, 40 | 20/20, 58 | 20/20, 101 | 20/20, 148 |

Explore episodes are 1.75-2.3x the route teacher's length; mean share of open cells seen
0.90 (S0), 0.92 (S1), 0.98 (S2, S3). Locks learned from the pulse: 6 on S2, 6 on S3 (the
other maps found the key before trying the locked door); those 12 presses are the
"ineffective" presses. S3: 71 kills, 0 deaths. No deaths or time-outs anywhere. The first
version met the target, so no fix iteration was needed. 2 engines, ~284 decisions/s.

## Collection
`DOOMSDAY_OBS_VERSION=4 DOOMSDAY_TRAIN_BUNDLE=train-all-b2 DOOMSDAY_EPISODE_TICS=8400 python -m
teachers.collect --teacher explore --stages S0 S1 S2 S3 --episodes-per-map 4 --out
data/teacher/bc_v6_explore --noise 0.2 --workers 2 --seed 2032`, 04:51-05:44, complete, no
errors, no memory stop or worker reduction (min available 4.36 GiB). egomap (T, 6, 32, 32)
confirmed in the saved files.

| stage | success | mean length (successes) | time-outs | deaths |
|---|---|---|---|---|
| S0 | 880/880 | 151 | 0 | 0 |
| S1 | 880/880 | 253 | 0 | 0 |
| S2 | 879/880 | 485 | 1 | 0 |
| S3 | 469/480 | 591 | 5 | 6 |

3108 episodes saved (failures dropped), 1,058,075 decisions, 228 MB.

## Update 2026-11-26 (morning): goal-mode labels and the door-loop fix
Datasets collected from now on differ from bc_v6_explore (which has no `goal_mode` and
the loop below); bc_v6_explore stays valid.
- Goal-mode label: every teacher decision reports one id (`teachers/goal_modes.py`, copy
  `configs/goal_modes.py`, equality tested): 0 explore, 1 get_key, 2 go_locked_door,
  3 go_exit, 4 fight, 5 open_door, 6 recover. Priority: combat label 4; unstick label 6;
  a labelled door press, or the current anchor is a door in its align/push phase, 5;
  else the leg (route: next key / locked door / exit anchor -> 1 / 2 / 3; explore:
  goal frontier/doorf/spin -> 0, key 1, lockdoor 2, exit 3). Read-only: labels are
  unchanged. `collect.py` saves `goal_mode` int64 (T,) per npz, `goal_mode_counts` per
  episode, `goal_mode_share` per stage@noise and `goal_mode_share_by_stage` in
  summary.json. Route smoke (S2-train-102001 + S3-train-103001, noise 0.2): S2 get_key
  0.14, go_locked_door 0.41, go_exit 0.03, open_door 0.43; S3 adds fight 0.16. Door
  handling (wait for the door, push through) is about 11 decisions per door.
- Door loop, cause found: `TeacherController._crossed` is a half-plane test with no
  lateral bound. On S3-train-140040 the lockdoor/10 door anchor was planned with the
  player 682 units away, 507 across the door axis, and counted as crossed at once; it
  completed on all 3 re-plans every decision and the label was a no-op until the 500-
  decision timeout, then doorf/10, blacklist clear, repeat. The proposed "blacklist a
  crossed door frontier" fix was tried first and rejected: open-loop replay changed 9 of
  45 recorded non-looping episodes; a stall-triggered variant changed none but turned
  standing still into spinning (no success).
- Fix applied (explore only, `ExploreConfig.crossed_max_across = 128`, 0 = old): a door
  also needs to be within 128 units of the door centre across its axis to count as
  crossed. Open-loop replay of 45 recorded episodes (15 each S1-S3, noise 0.2, engine
  inputs recorded with the old code): identical labels on all 44 with no stall; the one
  that changed (S3-train-140026) had a 27-decision stall of the same kind. Closed loop,
  same seeds, of the 6 bc_v6_explore door time-outs: 5 now succeed (S3-train-140040 x3,
  S2-train-130040, S3-train-140079); S3-train-140076 still times out (a different loop:
  lockdoor/9 pressed, opens, closes again, 1338 open_door decisions; not fixed).
  Pinned by `tests/test_teachers_explore.py` with 4 recorded non-looping episodes
  (`tests/fixtures/explore_nonloop_states.json.gz`, 956 steps).

## Known issues (not fixed: the dataset was collected with this code)
- Door-frontier ping-pong: the 6 time-outs (S2-train-130040, S3-train-140040 x3,
  S3-train-140079, S3-train-140076) end with the same door goal (`doorf/N` or `lockdoor/N`)
  re-planned and its door anchor completed every decision. Likely an unseeable pocket
  beside the door keeps it a frontier, and the goal is re-issued after each crossing,
  which resets its timeout. Proposed fix: blacklist a door frontier once its door anchor
  is crossed. Not applied, so the code matches the dataset.
- About one blacklisted frontier per episode in the measurement; not inspected with a
  trace.
- The exit is recognized from the switch line or pad cells, as specified. The student's
  telemetry observation (rays, entities, egomap) has no exit-specific channel, so
  exit-directed labels still rest on a cue the student sees only in pixels, or not at all.
- DAgger (student-driven) labelling is unit-tested only (no `--student` engine run).
