# Teacher combat (S3) and S3 teacher dataset, 2026-11-26

Training-time only. The scripted teacher (`teachers/`) gained a combat branch. Aiming
uses atan2 on privileged world positions. That is allowed only because nothing under
`agent/`, `env/` or `evaluate/` imports `teachers`, directly or transitively
(`tests/test_teachers_isolation.py`).

## What was built

`teachers/combat.py`:
- Visibility follows the env's entity rule. A monster counts when its label has the
  `Monster` category and at least one own pixel in the labels buffer.
- Target choice: the nearest monster, with melee-only monsters weighted at 0.6 x distance.
- Aim tolerance: max(2.5 degrees, the target's angular half-size).
- Projectile threat: closest approach within 40 tics and nearer than 44 units.

`teachers/controller.py` has a combat branch that runs before the route.
- **Turn:** the nearest turn bin toward the target.
- **ATTACK:** held while the heading error is within tolerance, or while a fine (4 degree)
  correction reaches it this decision.
- **Move:** no forward motion. Back off (if there is room) from any monster within 96
  units, or from a melee-only monster within 140 units.
- **Strafe:** away from an incoming projectile.
- **Route state:** frozen while engaged. The route resumes as soon as no visible live
  monster remains, and the teacher never chases.
- **No ammo** (or fist selected): the teacher keeps to the route.
- **Give-up:** after 80 engaged decisions on one target. Never triggered in the
  measurement runs; once in the dataset.

`teachers/collect.py`:
- `--stages S3` is accepted.
- `info` gains `monsters` and `projectiles` (privileged, labelling only).
- The npz gains a `label_combat` (T,) bool. This is an extra key: `train/bc_data.py`
  loads only the keys it knows.
- New per-episode and per-stage counters: attack labels, combat decisions, kills, deaths,
  timeouts, damage taken, monsters seen, give-ups.

Two engine probes back the design: `teachers/probe_combat.py`.
- **`fire`** (pistol, 60 decisions): held attack 16 shots; tap 1-0 12, 1-0-0 13, 1-1-0 15,
  1-1-1-0 14, 1-1-1-1-0 15. A held attack keeps firing, so the label holds.
- **`corpses`** (2 kills, S3-train-103004): on the decision after the kill the label left
  the `Monster` category on all 6 decisions watched. So the category is the liveness test.
- **Shotgun** (from the saved S3 episodes, not probed directly): single held decisions
  used 0.93 shells; held runs of 8-15 decisions used 1.38. That fits a refire cycle of
  about 11 decisions.

## Measurements (teacher drives, all 20 S3 TRAIN maps)

| noise | episodes | success | deaths | kills | monsters seen | damage taken | attack-label share | mean length |
|---|---|---|---|---|---|---|---|---|
| 0 (seed 11, 2/map) | 40 | 40 | 0 | 131 | 131 | 246 | 0.222 | 148 |
| 0.2 (seed 12, 4/map) | 80 | 80 | 0 | 259 | 264 | 1323 | 0.117 | 301 |

The summaries are `measure_s3_noise0.json` and `measure_s3_noise0.2.json`.

## S0-S2 unchanged

- **Fake env:** 24 FakeDoomEnv episodes (4 seeds x 3 start angles x noise 0 / 0.3) were
  captured before the change (`tests/fixtures/teacher_labels_pre_combat.json`). They are
  bit-identical after it.
- **Real routes:** 6 real S0-S2 TRAIN routes x 120 synthetic states
  (`tests/fixtures/teacher_labels_pre_combat_maps.json`, `tests/teacher_golden.py`) are
  bit-identical, with and without an empty `monsters` list.
- **Engine replay:** `python -m teachers.replay_check --data data/teacher/bc_v4map
  --per-stage 4` re-ran 12 saved episodes (4 maps per stage). `action` and
  `executed_action` were identical in 12 of 12, and 12 of 12 succeeded.
- **Observation mismatch (not caused by the teacher):** in 4 of the 12 replays (S1 and
  S2 door maps) the saved `rays` and `egomap` differ from the current env. `env/door_cue.py`
  and `env/doom_env.py` were edited at 01:42-01:45 tonight (occlusion gate, audit 02 F1),
  after `bc_v4map` was collected. This belongs to `env-engineer`; see the dataset section.

## Dataset `data/teacher/bc_v4map_s3`

Command: `python -m teachers.collect --stages S3 --episodes-per-map 50 --out
data/teacher/bc_v4map_s3 --noise 0.2 --workers 2 --seed 2029`, observation v3 (the
default). It ran 01:44:53-01:57:38 (764 s): complete, no errors, no low-memory stop, and
a minimum of 3.25 GiB available.

**Results:**
- 1,000 episodes, 984 succeeded and saved, 303,332 decisions saved, 66 MB on disk.
- 16 deaths (all failures were deaths, mostly drained by imps or shotgun guys; 0 timeouts).
- 3,240 kills of 3,338 monsters seen, 18,740 damage taken in total.
- 2 give-ups.

**Labels:**
- Attack-label share 0.113 (34,814 labels).
- 60,459 combat decisions.
- USE share 0.029.
- In the first 250 episodes: 611 dodge strafes and 779 back-off labels, so both branches
  fired in the engine.

**Contract:** same keys, shapes and dtypes as `bc_v4map`, plus `label_combat`. The summary
copy is `dataset_bc_v4map_s3_summary.json`.

**Observation version caveat (owner: `env-engineer`):**
- `env/door_cue.py` was edited at 01:45:22 and again at 01:53:15, while this collection
  was running. Its workers had imported env at about 01:45:00.
- A replay with the current env (`teachers.replay_check --data data/teacher/bc_v4map_s3
  --stages S3 --per-stage 3`): labels identical 3 of 3; saved `rays` (the door channel)
  differ in 1 of 3.
- `bc_v4map` likewise differs from the current env in 4 of 12 episodes.
- So both datasets carry the door channel as it was before tonight's occlusion-gate
  edits. If the gate is final, re-collect both (about 13 minutes each, 2 engines) before
  BC, or accept the mismatch.

## Not done

- Pickup detour to health or ammo (optional): not built. The teacher keeps its route.
- Liveness by category rests on two observed kills. The 0.2-noise data (3,240 kills, 2
  give-ups) is consistent with it.
- `--stages` still defaults to S0 S1 S2 (`DEFAULT_STAGES`), so existing invocations are
  unchanged; S3 must be asked for.
