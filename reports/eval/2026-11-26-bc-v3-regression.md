# Why BC on bc_v3 plays worse than BC on bc_v2 (data-only forensics)

Date: 2026-11-26. Author: debugger subagent. Scope: numpy over the saved datasets and the
closed-loop episode logs only (no engine, no training, no GPU; a 32-engine run was live).
No code was changed.

Scripts (all read-only, under `reports/eval/scripts/`):
- `bc_v3_regression.py`: label statistics per stage over all 3,000 + 3,000 episodes, giving
  `reports/eval/2026-11-26-bc-v3-regression.json`
- `bc_v3_regression_obs.py`: observation statistics and the one-step alignment check (300
  episodes per stage), giving `...-regression-obs.json`
- `bc_v3_regression_wall.py`: blocked, still and near-wall states, S0 split by exit kind,
  door channel against depth (400 episodes per stage), giving `...-regression-wall.json`
- `bc_v3_regression_door.py`: door-channel width binned by centre depth, what is visible at
  USE = 1 steps, and kNN turn consistency on lit-door steps (400 episodes per door stage),
  giving `...-regression-door.json`

## Verdict

**The regression is not explained by the teacher data. The cause is not established from
data.**

- The control is S0. It has no doors, and its v2 and v3 datasets are statistically
  identical: every label share, jitter rate, copy rate, state-conditional label
  distribution and observation mean agrees within about 1-2 points.
- Even so, the closed loop on S0 falls from 14/40 to 4/40.
- The fall is on **door-free `exit_pad` maps: 11/22 to 2/22.** On `exit_switch` maps it is
  3/18 to 2/18.

What the data does show is a **new, specific failure mode in the v3 student**: it ends
wedged against a wall.

Five v3 failures and three v2 failures (all S1) are set aside first. In those, strafe
agreement is below 0.5 with `unstick_labels` = 0, so the teacher's strafe labels came from
another branch (`_recover`), not from turn-in-place. Over the rest:
- **105 of 107** v3 failures end flush on a wall line (a final coordinate of 16 mod 32,
  which is the player radius off a 32-unit grid line). For v2 it is **37 of 75**.
- In **99 of 107** v3 failures the student matched the teacher's move label on fewer than
  15% of steps. For v2 it is **2 of 75**.
- Mean agreement on failed S0 episodes: v3 move 0.05, turn 0.02; v2 move 0.53, turn 0.16.

In those episodes unsticks are near zero and strafe agreement is about 1.0 (S0 and S2;
S1 is 0.87 overall because of the set-aside episodes). The teacher's student-driven
unstick fires only when its own label is forward, so it did not fire here. The teacher was
therefore asking for something other than forward (turn in place) while the student did
something else, most likely pushing forward into the wall. That last step is inferred: the
student's executed actions were not saved.

v3 is bimodal. Its successes are mostly short and teacher-like: lengths 52, 55, 60, 74, 74,
91, 352 and 373, with move agreement 0.64-0.91. Its failures are full 1,050-step timeouts
in the wedge. v2 wanders: 31 of its 42 successes take 200-990 steps.

## Ranked causes

| # | Cause | Status | Key numbers |
|---|---|---|---|
| 1 | **The v3 BC model fell into a wall-wedge absorbing state** (pushes forward into a wall, never turns away), with training-run variance deciding whether a model does. One seed per model, so this cannot be told apart from a systematic effect. | consistent, untested | 105/107 failures flush on a wall; 99/107 with move agreement < 0.15. P(label = prev executed, all heads, when the player is not moving) = 0.70 in **both** datasets, so the copy-prev_action shortcut that makes such states absorbing is equally available in both. Blocked states are only about 2% of S0 teacher steps in both, so neither dataset teaches recovery from a wedge. |
| 2 | **Cross-stage effect of the changed S1/S2 labels through shared weights.** The S1/S2 changes could shift the shared trunk's wall behaviour on S0. | consistent, weak, untested | S1/S2 in v3: USE labels 2x (3.4% against 1.8% of steps); presses come during turn-in-place (0.76 of press steps) instead of after standing still (v2: 1.00 still, and 0.79 turn-in-place on the step before). Episodes 15-17% shorter. Fewer blocked steps (0.106 against 0.134). **Fewer** wall-push labels: P(forward & no turn \| centre depth < 0.05) 0.16 against 0.19. The direction of each change argues *against* v3 teaching wall-pushing. |
| 3 | **Observation changed between the two collections** (`env/doom_env.py` and `env/door_cue.py` edited 23:38; bc_v2 finished 23:23, bc_v3 23:47, both evals after 23:55). | files edited; **no measurable effect on the door channel**; not the cause | The raw lit-run width differs (43.7 against 38.6 columns), but that is only because v3 stands further back (near-wall share 0.40 against 0.47). Binned by centre depth, the widths agree: 72.2/71.3, 44.2/44.6, 23.5/23.1, 11.2/11.0 columns for the depth bins <0.02, 0.02-0.05, 0.05-0.1 and 0.1-0.2 (S1; S2 is the same). S0 has no door channel in either. |
| 4 | New unlearnable labels around doors (hidden-geometry "face squarely" turning, pauses, long zero-move runs). | **refuted** as the main cause; a small effect near doors | Zero-move runs are *shorter* in v3 (mean 2.5 against 2.8; steps in runs >= 10: 2.0% against 2.7%). Idle labels 0.09% against 0.02%. kNN turn agreement over all steps is equal or slightly higher in v3 (turn-in-place S2 0.41 against 0.39). **On steps with a lit door channel at short depth it is slightly lower in v3**: 0.451 against 0.470 (S1) and 0.417 against 0.442 (S2), with turn-label entropy +0.1 bits. That is the ALIGN turning, and it is only marginally less predictable. And S0, which has no doors, regresses anyway. |

## Step 1: label statistics per stage (v2 then v3)

| | S0 v2 | S0 v3 | S1 v2 | S1 v3 | S2 v2 | S2 v3 |
|---|---|---|---|---|---|---|
| mean length | 73.9 | 71.9 | 134.5 | 114.6 | 252.8 | 210.7 |
| move fwd / none / back | .715/.285/0 | .723/.277/0 | .710/.290/.001 | .713/.287/0 | .701/.298/.001 | .706/.294/0 |
| strafe nonzero | 0 | 0 | .0014 | 0 | .0013 | 0 |
| turn zero-bin | .193 | .188 | .231 | .209 | .225 | .203 |
| USE label share | .0057 | .0069 | .0178 | .0347 | .0183 | .0341 |
| speed on | .297 | .303 | .102 | .122 | .105 | .123 |
| jitter: any head changes | .623 | .627 | .619 | .638 | .618 | .636 |
| jitter move / turn | .189/.570 | .186/.572 | .197/.582 | .219/.586 | .203/.583 | .222/.586 |
| turn labels without forward (share of turn labels) | .345 | .339 | .353 | .352 | .361 | .358 |
| executed != label (any head) | .328 | .333 | .326 | .331 | .332 | .332 |
| copy prev_action accuracy, move / turn / all | .67/.31/.25 | .67/.30/.24 | .67/.31/.25 | .65/.30/.24 | .66/.30/.25 | .64/.30/.24 |
| turn sign flips (consecutive nonzero) | .126 | .131 | .125 | .137 | .125 | .135 |
| label_unstick share | 0 | 0 | .0014 | 0 | .0013 | 0 |

## Step 2: behaviour around doors and switches in v3

Window around each USE = 1 label, offsets -3, -2, -1, **0**, +1, +2, +3 (S1; S2 is the same
within 0.01):

| | forward | turn in place | standing still |
|---|---|---|---|
| v2 | .590 .383 .099 **0** .748 .785 .795 | .403 .584 .789 **0** .133 .181 .197 | .007 .032 .112 **1.00** .119 .034 .008 |
| v3 | .804 .744 .543 **.016** .583 .744 .765 | .191 .242 .428 **.758** .397 .248 .233 | .005 .013 .029 **.226** .020 .008 .002 |

The old teacher turns in place, stops, then presses. The new teacher presses mid-turn as
soon as the view ray crosses the door face within 58 units, then moves on at once.

How well the observation shows a press (S1 / S2):
- At USE = 1 steps, all 8 centre columns of the door channel are lit in v3 0.874 / 0.870 of
  cases, against v2 0.758 / 0.715. Median centre depth is 0.016 in both.
- P(USE = 1 | centre lit and depth < 0.05): v3 0.147 / 0.147, v2 0.050 / 0.048.

So the v3 USE label sits more cleanly on a visible cue, which matches the higher val USE
precision (0.87 against 0.62).

No long pauses: v3 has fewer and shorter zero-move runs than v2. The "face squarely" ALIGN
phase does not show up as a measurable block of labels: turn-in-place share is +0.6 points
and standing still +0.07 points. Its turn labels are slightly less predictable from
the observation (cause 4).

## Step 3: obvious mistakes checked (all clear)

- Observation layout: identical keys, shapes and dtypes. rays (4, 80), entities (24, 8),
  self (34), prev_action (6), modality (3).
- Frame skip and alignment: the ego turn delta `self[27]` at step t matches the turn of
  `executed_action[t-1]` (frame skip 4) on 97.3-99.2% of steps in both. It matches
  executed t or label t on only 27-29%, so there is no one-step leak. `prev_action[t] ==
  executed_action[t-1]` on 100% of steps in both.
- Label field: BC trains on `action` (`train/bc_spec.py` `LABEL_KEY = "action"`), not
  `executed_action`.
- Episodes: `done` is set only at the last step in 100%; 3,000/3,000 successes in both; no
  early ends.
- Constant fields: attack (0) and `student_driven` (False) in both. **Strafe and move-back
  are constant 0 in v3.** v2 has 0.13-0.14% teacher unstick labels (back + strafe) in
  S1/S2. That is the only categorical label difference, and it concerns being stuck. In
  the S0 closed loop, v2's strafe disagreement is 124 steps against 90 teacher unstick
  labels, so the v2 student strafed on its own at most about 34 times in 40 episodes (v3:
  36 against 36). The v2 student therefore did not escape walls by strafing either. This
  is a minor candidate, not ruled out.
- Confound: `teachers/collect.py`, `navigator.py` and `use_label.py` were edited at
  23:48-23:49, after bc_v3 finished (23:47). The current source may not exactly be what
  produced bc_v3.

## What would settle it, and cheap fixes

These all need training or the engine, so none was run here.
1. **A second and third BC seed on bc_v3** (and one more on bc_v2), with the same
   closed-loop eval: about 12 min of training plus 5 min of eval each, once the live run
   frees memory. If seeds scatter between about 4/40 and 14/40 on S0, the "regression" is
   seed variance, and comparing one model per dataset is not informative.
2. **Cheaper than 1: an open-loop forward pass** of both checkpoints on the same blocked
   and near-wall dataset states, comparing P(forward, no turn). Do not run it while the live
   run holds memory (torch import).
3. **Whatever the seed result, attack the wedge directly with one DAgger round.**
   `teachers/collect.py --student <v3 bc_model.pt> --beta 0.2-0.3` already labels
   student-driven states and saves failed episodes in student mode (line 426,
   `or student is not None`). Add `--max-steps` of about 300, so 1,050-step wedges do not
   swamp the set. Drop the steps produced by the `_recover` branch (its strafe side is a
   coin flip); these are the strafe-label steps with `label_unstick` False. The wedge
   states (teacher label turn-in-place, student pushing forward) are exactly what the BC
   data lacks: blocked states are only about 2% of S0 teacher steps. Aggregate and retrain.
4. Cheap BC-side regulariser against the copy shortcut: drop `prev_action` (replace it with
   its reset value) on about 30% of training steps. P(label == prev | not moving) = 0.70
   in both datasets, which is what lets a wedge persist.

No teacher change is indicated by the data. The v3 USE labelling is cleaner, and the
students' val metrics agree: USE precision 0.87 against 0.62, with turn and move unchanged.
