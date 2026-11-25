# Gap analysis of the 2026-11-25 training logs: where and why episodes fail

Evaluator, 2026-11-25 (late evening). Log analysis only: no engine, no GPU, no training,
no episodes run. Nothing here is a held-out result; every number is from TRAIN maps
(training episodes, sampled actions, shaping on) or from the three existing VAL runs.
Read-only on code and on the live run; nothing committed.

**Inputs**

| tag | run folder | what | used |
|---|---|---|---|
| gen1 | `reports/runs/20261125-nogit-45275c79/` | PPO from scratch, `full_level`, curriculum S0 -> S1 (promotion at update 521) | 4,817 episodes, updates 3-2260, 4.63M decisions |
| n1A | `reports/runs/20261125-night1-A-dagger-doors/` | PPO from DAgger r1, `full_level_doors`, stayed on S0 | 2,869 episodes, updates 0-1398, 2.15M decisions |
| n1C | `reports/runs/20261125-night1-C-s1-doors/` (LIVE) | PPO from DAgger r1, `full_level_doors`, start S1 (20 % S0 replay) | **snapshot: rows with `update_idx <= 329`** (660 episodes, 648k decisions) |
| val | `reports/eval/2026-11-25-bc-v1/{random,bc_sample,ppo_gen_run1}/results.jsonl` | 20 val maps x 3 episodes, seed 11, vanilla EVAL | 60 episodes each |

Scripts (re-runnable, stream the jsonl line by line):
`reports/eval/scripts/gap_analysis.py` (main tables), `gap_analysis_extra.py` (success over
time, val), `gap_analysis_doors.py` (door funnel on on-path maps, matched switch/pad).
Full outputs: `reports/eval/2026-11-25-gap-analysis/tables.md`, `tables_extra.md`,
`tables_doors.md`. Map features joined from `maps/{train,val}/<id>.json`; "path doors" =
doors on the room-graph path start -> exit (S2: start -> key -> exit), computed for
analysis only.

Notes on the data:
* No deaths in any training episode (S0/S1 have no monsters). Every failure is a
  1,050-decision time-out. Combat is not measured anywhere here.
* `doors_passed` counts ANY door, including side-branch doors off the path, and the logs
  carry neither which door nor when. So "how long after the last door the episode still
  failed" **cannot be measured** from these logs.
* n1C resumed at update 277 (memory-guard stop; 4x8 -> 4x6). The first resume attempt
  failed on a sensor-signature mismatch (`obs_version 2` in the current default vs v1 in
  the checkpoint, `stderr_resume_231249.log`); the second succeeded, so it ran under v1.
  `env/observation.py` and `configs/a0_env.py` were modified at 22:55-22:56, after n1C
  started (22:45), so post-resume episodes ran on newer v1 code; I cannot verify bit
  equality without the engine. The trends below start well before 277, so they are not
  a resume artifact. The `recent_success` metric resets at resume (0.628 at 275-299);
  it is not cited.
* Training success is measured with sampled actions on maps the policy trains on; it is
  an upper bound on unseen-map performance, not an estimate of it.

---

## Ranked gaps

### Gap 1. The agent cannot press an exit switch (USE on a wall). Largest single loss on S0

Evidence:
* n1A S0 (DAgger start, 2,869 episodes): **pad exits 1,156/1,577 (73.3 %) vs switch exits
  88/1,292 (6.8 %)**. Matched on layout (3 rooms, 2 hops): pad maps 100006/7/9 **411/432
  (95.1 %, Wilson 0.927-0.968)** vs switch maps 100002/4/20 **30/431 (7.0 %, 0.049-0.098)**
  (`tables_doors.md`).
* Same in gen1 (PPO from scratch): matched pad 197/303 (65.0 %) vs switch 99/301 (32.9 %);
  all S0 pad 43.2 % vs switch 19.8 % (`tables.md` 1d). n1C S0: pad 49/77 vs switch 3/63.
* At the DAgger start (n1A updates 0-99) switch success is **0/79**; PPO lifted it to
  30/96 (31.2 %) at updates 500-599, then lost it again (8.2 %, 4.0 %, ... 3.7 % at
  1300-1398) while pad success stayed 55-85 % (`tables_extra.md` A).
* S1: switch-exit maps 0/154 in n1C, 7/843 (0.8 %) in gen1; pad 11.2 % / 7.1 %.
* Failed switch episodes cover more cells than successes: n1A switch failures median 36
  cells (p25/75 24/48) vs 21 for all S0 successes; they are not pinned near the start.
* 9 of the 20 S0 TRAIN maps, 3 of 5 S0 val maps and 12 of the 20 val maps have switch exits.

Likely causes (competing; the logs cannot separate them):
(a) the imitation start: the DAgger report (`reports/eval/2026-11-25-bc-dagger.md` s4.1)
shows the teacher's USE labels are gated (0.9 % of student-driven S1 labels, measured for
DOORS); that the same gating hurts switches is my inference, not measured. It fits the
0/79 switch success at the n1A start.
(b) not only the teacher: gen1 never saw a teacher and still shows the gap (matched pad
65.0 % vs switch 32.9 %), so PPO alone also finds switches hard.
(c) observation: all three runs ran observation v1 (signatures without `obs_version`).
The door channel exists only in v2 (`env/config.py`), and `env/observation.py` /
`env/door_cue.py` contain no exit-switch cue (only a locked door/switch message pulse in
v2). In v1 a switch line may be indistinguishable from a wall in the rays; not verified.
After the switch is found, PPO does not hold on to it (Gap 3).
Quick test: TRAIN-map closed-loop on the 3 matched switch maps vs the 3 pad maps (the
`teachers.collect --no-save` measurement, same seeds) for any USE-label fix or USE
auxiliary/bonus; target is closing the 95 % vs 7 % gap. A cheap diagnostic without
retraining: count USE 0->1 edges within N decisions of the switch in failed episodes
(needs a per-step trace, which the logs do not have).

### Gap 2. The first door stops ~80 % of S1 episodes; a second door almost never opens

Evidence (n1C S1, restricted to the 9 maps whose every door is on the path, 231 episodes,
`tables_doors.md`):

| funnel | episodes | share (Wilson 95 %) |
|---|---|---|
| passed >= 1 door | 48 | 20.8 % (0.160-0.265) |
| passed >= 2 doors | 4 | 1.7 % (0.007-0.044) |
| passed all path doors | 24 | 10.4 % |
| exit | 11 | 4.8 % (0.027-0.083) |

* By path-door count: 1 door 10/105 exits, **2 doors 1/102**, 3 doors 0/24.
* No episode reached the exit with fewer doors than the path needs (consistency check of
  the door term: 0 violations).
* Coverage confirms "blocked, not wandering": S1 failures median 14 cells (p25/75 9/19)
  vs successes 19; 37.8 % of n1C S1 time-outs visit <= 10 cells (gen1: 36.3 %).
* The one S1 map whose door is off the path (101016) is solved 20/27 (74.1 %) in n1C and
  74/140 in gen1, i.e. S1 maps are otherwise within reach of the navigation skill.
* gen1 S1 by path doors: 0 doors 52.9 %, 1 door 6.4 %, 2 doors 1.2 %, 3 doors 0.4 %.
* Val S1-S3: 0/45 for every policy.

Likely causes: as Gap 1 (gated USE labels for doors are measured in the DAgger
report; gen1, with no teacher, is no better: 6.4 % at 1 path door, 1.2 % at 2), plus v1
having no door channel. No v2 run exists to compare. Quick test: the on-path door funnel above (passed >= 1 / >= 2) per 25 updates is the
fastest signal; it moves within 50 updates (see Gap 3).

### Gap 3. PPO fine-tuning erodes what BC/DAgger gave, starting within ~150 updates

Evidence, n1C (live, cutoff 329):
* Door passing on S1 per 50 updates (all S1 maps): 39.0 %, 41.5 %, 40.7 %, **27.9 %
  (150-199), 8.0 % (200-249), 1.6 % (250-299)**, 2.6 % (300-329). On-path maps only:
  30.0 % before update 150 vs 12.4 % after. S1 success 7.8 % / 6.1 % / 12.8 % / 11.6 % /
  6.8 % / 3.2 % / 2.6 % over the same buckets (n 77-88 per bucket up to 249; 63 in
  250-299; 38 in 300-329). S0 replay success 40.0 % / 41.3 % / 35.1 % per 100 updates,
  then 2/12 (16.7 %) at 300-329.
* This decline starts before the resume at 277 (updates 200-249 already at 8.0 %).
* Action drift (25-update means, `tables.md` 4): USE share 0.014 (warm-up) -> 0.155-0.219
  (100-199) -> still 0.170 at 200-224 -> 0.116 (225-249) -> **0.065 (250-274)**. So door
  passing fell (27.9 % at 150-199, 8.0 % at 200-249) BEFORE the USE share did; the
  collapse is not explained by fewer USE presses alone. Zero-turn share 0.18 -> 0.35
  (25-74) -> **0.012-0.10 from 100 on, 0.012 at 250-299**; strafe-head entropy 0.44 ->
  **0.016-0.10 from 100 on** (strafe almost always "none"); forward share 0.39 -> 0.84
  (250-274).
* From about update 268 the forward share reverses (0.915 at 268, 0.585 at 277, 0.835 at
  278 just after the resume, 0.337 at 290) and attack-head entropy rises from 0 (0.08 at
  268 to 0.31 at 290). Cause unknown; the shift starts before the resume at 277 and
  continues after it.
* Which head moved first (per update, `metrics.jsonl` updates 15-40): the first PPO
  update after the 20-update value warm-up (update 21) had approx_kl 0.155 and clip
  fraction 0.587. Over updates 21-28 the move head changed most: "no move" share 0.55 ->
  0.19, with zero-turn 0.17 -> 0.51 and USE 0.01 -> 0.07. The later collapse is in turn
  (zero-turn to ~0.01), strafe (entropy -> 0.02) and then USE (from ~200).
* Critic not ready: explained variance (25-update means) negative for updates 0-149
  (-0.56 to -0.79 in 25-99, -0.13 at 125-149), reaching 0.39-0.65 only at 200-274.
  Clip fraction 0.16-0.26 and KL ~0.013-0.024 throughout (gen1 from scratch: clip
  0.07-0.11, KL 0.005-0.008).
* n1A shows the same shape on S0: success 40.4 % -> 53.2 % (update 500) -> 31.1 %
  (1300-1398); switch success peaked at 31.2 % then fell to 3.7 %.
* gen1 after promotion to S1: S0 replay success 60.5 % (500-599) -> 11.6-25 % for most
  later buckets; forward share 0.96 -> 0.13-0.41; S1 never above 9.8 % after 600.

Likely cause (inference, not proven): policy steps too large for a fine-tune (clip
0.2+, first update KL 0.155) driven by a critic with negative explained variance,
with sparse door/exit signal; entropy bonus 0.01 summed over six heads does not keep
the USE/turn heads alive. Quick test: two short arms from the same DAgger r1 start on S1
(~150 updates each; n1C did 277 updates in about 22 minutes at 4x8, 22:45-23:07): lower LR / tighter clip or a longer value warm-up
(until EV > 0), optionally a KL-to-BC penalty; compare door-pass >= 1 on on-path maps
per 25 updates and USE share. Numbered n1C checkpoints on disk: `checkpoints/checkpoint-u0000143.pt` (before the
door-pass decline), `-u0000262.pt`, `-u0000277.pt`; u143 is the natural one to evaluate on
VAL against DAgger r1 (choice based on TRAIN metrics only).

### Gap 4. Long routes: success falls with the number of rooms to cross (S0, no doors)

Evidence (n1A S0): hops start -> exit 1: 98.6 %, 2: 53.7 %, 3: 31.0 %, 4: 27.5 %,
5: 25.1 %; 6-room maps 27.7 % vs 3-room 51.1 %. gen1: 76 %, 45 %, 25 %, 7 %, 10 %.
Map S0-train-100001 (6 rooms, 4 hops, switch) is 0/144 in n1A and 2/100 in gen1.
Failures here are exploring, not stuck (n1A S0 failures median 34 cells vs 21 for
successes; only 0.4 % of S0 time-outs visit <= 10 cells).
Partly confounded with Gap 1, so split on pad-exit maps only (no USE needed):

| hops (pad maps) | n1A | gen1 |
|---|---|---|
| 1 (100013) | 141/143 (98.6 %) | 76/100 (76.0 %) |
| 2 (100005/6/7/9/17) | 652/719 (90.7 %) | 288/504 (57.1 %) |
| 3 (100008/12) | 181/286 (63.3 %) | 76/201 (37.8 %) |
| 4 (100016) | 79/143 (55.2 %) | 12/100 (12.0 %) |
| 5 (100010/14) | 103/286 (36.0 %) | 25/200 (12.5 %) |

Route length is a clean gap of its own: from 90.7 % at 2 hops to 36.0 % at 5 in n1A,
with no door and no switch involved (few maps per row, so map identity is mixed in).
Likely cause (inference): exploration without memory of where it has been (ADR 0009: GRU
fails at 60-step gaps). Quick test: per-checkpoint success on the two 5-hop pad maps
(100010, 100014) and their cells-visited per decision; any memory/exploration change
should move this row first.

### Gap 5. After all path doors are open, about half the episodes still time out

On-path S1 maps, n1C: 24 episodes passed all path doors, 11 exited, 13 timed out
(failed coverage median 24 cells). n = 24, so the share (46 %) is very uncertain.
Four of these 13 are on switch-exit maps (101001 x3, 101009 x1), where Gap 1 applies; 7 are on
101020 (pad, 1 door).
Cannot tell from logs: when the last door was passed and how far the exit was.

### Gap 6 (design risk, not an observed failure). Door term pays for side doors

`full_level_doors` pays 0.3 for every door in the map, on or off the path. A map with
5 doors (101002) can pay 1.5 > exit 1.0. In n1C, 110 of 567 failed episodes (19.4 %) had a
positive shaped return (max 0.527); gen1 and n1A: 0. No farming is visible: door passing
fell, not rose, and mean door pay per failed S1 episode is 0.074. Coverage pay is small
(mean 0.014-0.10 per episode, coef annealed to 0.05 in gen1/n1A, still 0.18 in n1C);
time penalty is -0.21 per time-out, -0.055 to -0.10 per success. Exit (1.0) dominates
every logged episode (min success return 0.806-0.866). Quick check: count door pay on
side doors once the door index is logged.

### Gap 7. Validation cannot separate any policy yet

| policy | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| random | 3/15 | 0/15 | 0/15 | 0/15 |
| bc_sample (BC 20 epochs) | 2/15 | 0/15 | 0/15 | 0/15 |
| ppo_gen_run1 (gen1 checkpoint) | 5/15 | 0/15 | 0/15 | 0/15 |

All S0 successes are on three maps: 500002 (pad, 3 rooms: 2/3, 2/3, 3/3), 500003
(switch, 3 rooms: only ppo_gen_run1, 2/3) and 500004 (pad: only random, 1/3). The
switch maps 500001 and 500005 are 0/9 for everyone. At 15 episodes per stage the policies
are not separable (Wilson intervals overlap, see the DAgger report). Deaths: 3 / 6 / 10
of 60, all on S3 (monster maps; not analysed further). Quick fix: more val episodes per map
(the val set is for checkpoint comparison; held-out stays untouched).

---

## Per-stage summary (training episodes)

| run | stage | episodes | success | time-out | dead |
|---|---|---|---|---|---|
| gen1 | S0 | 2,007 | 656 (32.7 %) | 1,351 | 0 |
| gen1 | S1 | 2,810 | 147 (5.2 %) | 2,663 | 0 |
| n1A | S0 | 2,869 | 1,244 (43.4 %) | 1,625 | 0 |
| n1C (<= 329) | S0 replay | 140 | 52 (37.1 %) | 88 | 0 |
| n1C (<= 329) | S1 | 520 | 41 (7.9 %) | 479 | 0 |

Maps never solved (>= 5 episodes): gen1 S1: 101001, 101005, 101010, 101011, 101012,
101013, 101017. n1A S0: 100001. n1C S1: 12 of 20 (list in `tables.md` 1c).
Per-map tables with rooms, doors, exit kind, hops: `tables.md` section 1c.

## What the data cannot tell

* Which door was passed and when; time from last door to time-out.
* Where the agent was when it failed (no per-step positions in the logs), so "stuck in a
  corner" vs "looping" is inferred only from `cells_visited` (128-unit cells).
* Whether USE was pressed near the switch/door (no per-step action-position trace);
  action shares are run-wide.
* Whether observation v2 (door channel, key bits) changes Gaps 1-2: no run used v2.
* Anything about combat, keys (S2/S3 never trained on in these runs) or held-out maps.
* Unseen-map performance: all training numbers are on TRAIN maps with sampling.
* Whether post-resume n1C observations are bit-identical to pre-resume (code changed on
  disk between start and resume; needs an engine replay, owner `env-engineer` /
  `trainer-ops`).
