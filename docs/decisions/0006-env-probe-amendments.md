# ADR 0006: Amendments after the engine probes (ray bands, USE, Windows timer, optional screen, viz_nocheat staging)

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

**Amends ADR 0001 and ADR 0002.** It also adjusts the stage order in ADR 0003 and the default process layout in ADR 0004; those changes are stated openly below. The earlier files are left as written; where they disagree with this one, this one applies.

## Context

`env-engineer` built the single-environment wrapper (`env/config.py`, `env/actions.py`, `env/observation.py`, `env/doom_env.py`) and ran the probes requested by ADRs 0001 and 0002. Sources read for this ADR: `docs/reference/engine-probes.md`, `reports/runs/2026-11-25-env-single-bench.md`, and the code, including `env/bench.py`.

Findings that need a ruling:

1. **Ray rows.** With the status bar rendered the 3D view is rows 0-99, the status bar is rows 100-119, and the horizon is near row 50. ADR 0001's bands (40-55, 56-63, 64-79) assumed a horizon at row 60, so its "horizon" band looks at the floor. The HUD-off horizon (row 60) is inferred, not measured.
2. **USE** is edge-triggered. A press on the first tic of every decision makes a door go 32, 0, 32, 0 (it reverses each time). USE held on all tics, across decisions, is one press: the door opens and stays open. A switch press was not probed.
3. **Windows timer.** Without a 1 ms timer request the step loop is capped near 76 decisions/s. With `timeBeginPeriod(1)` the same single env measured 368, 419 and 703 decisions/s, and one unexplained run at 75.
4. **`screen` cost.** 239 of about 290 microseconds of extraction per decision. The whole decision costs about 1.4 to 2.7 ms, so the engine round trip dominates.
5. **`+viz_nocheat 1`.** Depth buffer, labels buffer and label list are `None`; objects and sectors are empty; POSITION, ANGLE and VELOCITY read 0. Screen, health, armor, ammo, weapons, DAMAGE_TAKEN and the notifications buffer are unchanged. **ADR 0001's premise that depth and labels are "noised" is wrong for 1.3.3: they are absent.**
6. **Memory.** Peak 65 MB for the Python process and 72 MB for `vizdoom.exe`, one instance, MAP01, objects and sectors info off.

Constraints unchanged: one laptop, the user's instruction not to load it until it hangs or shuts down, deadline unknown ("more than 5 days"), organizers' answers on `viz_nocheat` and the harness still unknown. Vectorized throughput has not been measured; the user limited the benchmark to one instance.

## Options considered

**1. Ray bands**
- 1a. Keep ADR 0001 rows. No work. The middle band reads floor; no band sees the horizon when the HUD is on.
- 1b. Hard-code the shifted rows (30-45, 46-53, 54-69). 5 minutes. Wrong again if the HUD is ever off.
- 1c. Define bands as offsets from a horizon row that is derived from `render_hud` (chosen). 30 minutes. No throughput effect in any option.

**2. USE**
- 2a. Keep first-tic-only USE. No work. A policy that outputs USE on consecutive decisions can never open a door; it costs a second engine round trip on every decision where USE is on.
- 2b. First-tic press plus a forced release decision (ADR 0002's own reversal clause). 1 hour. Hides state from the network and still lets a sampled policy reverse doors.
- 2c. USE as plain button state, held for all tics of the decision (chosen). 30 minutes, removes code. One engine round trip per decision always.
- 2d. Env-side cooldown that swallows repeated presses. Rejected: it is a rule about when USE may act.

**3. Windows timer**
- 3a. Off by default. About 76 decisions/s per blocking step loop; since a worker steps its environments in turn, a whole worker would probably be held near that figure (reasoning, not measured).
- 3b. On by default everywhere (current code).
- 3c. On by default for training, off for evaluation and inference, one user-facing switch (chosen). 1 hour.

**4. `screen`**
- 4a. Always emitted (current). Costs 0.24 ms per decision and 14.4 KB per observation through worker pipes (pipe cost unmeasured).
- 4b. Zero-filled when off. Keeps the shape; still ships the bytes.
- 4c. Config switch; when off the key is absent from the observation and the space (chosen). 1 hour in the env; the model must be built from the observation space.

**5. `viz_nocheat` while the answer is unknown**
- 5a. Keep ADR 0003's order (A0, A1, B) and wait. No extra work now; if the answer is "flag on", Stage B and most of Stage A are useless at evaluation.
- 5b. Go pixel-primary now. Abandons the brief's recommended telemetry and puts all training on a CPU-only torch build.
- 5c. Keep A0 as the trainer shake-out, pull the pixel branch and modality dropout ahead of Stage B, and gate Stage B on the organizers' answer (chosen). About 1 day of `worker` time moved earlier, none added.

## Decision

### 1. Ray bands (amends ADR 0001)

Bands are **offsets from the horizon row: [-20, -4), [-4, +4), [+4, +20)**. The horizon row is 50 when `render_hud` is true and 60 when false, computed in one place from `render_hud`, so the two cannot disagree. With the HUD on this gives rows 30-45, 46-53, 54-69, which is `HUD_SHIFTED_RAY_BANDS`. The HUD stays on, because the status-bar key icons are one of the routes to knowing which keys are held. TRAIN and EVAL must keep sharing this one default. The shape stays (3, 80).

### 2. USE (amends ADR 0002)

**The `use` head is the state of the USE button for the whole decision.** If the head says 1, USE is down on all tics; if 0, it is up. The first-tic special case in `env/actions.py` and the split advance in `DoomEnv.step` are removed. This is unconditional pass-through with no dependence on observations or history, and it is how the key behaves for a human on the vanilla engine.

Consequences: a new press happens only on a 0 to 1 change of the head, so two presses need at least one decision with `use=0` between them (about 0.23 s). The network can see its own button state in `prev_action`. ADR 0002's clause "hold USE for the whole decision and force a release tic between decisions" is **withdrawn**; a forced release would recreate the reversal. Reason for the change: under the old encoding the simplest policy, "keep USE on near a door", fails forever; under this one it opens the door once.

### 3. Windows timer (recommendation; the user has the final say)

**Recommend on for training, off for evaluation and inference.** Move it from a constructor argument to one visible switch (a config field plus an environment-variable override) so the user can turn it off without touching code.

Reasoning: the request does not add work per decision; it removes idle waiting between Python and the engine. Total CPU load is set by how many worker processes run, which ADR 0004 caps and the user controls. Turning it off does not make training gentler per decision, it makes it about five times longer. It does raise the timer interrupt rate and lets each worker keep a core busy, so power draw during training is higher than with it off. Real-time play is 8.75 decisions/s, so inference does not need it.

Why the throughput varies: **not determinable from the code.** What the files show: the raw-engine phase alone ranged 583 to 1667 decisions/s across runs of the same code, so extraction is not the cause and something outside our code is (power source or mode, core scheduling, thermal state are candidates, none verified). The full-env loop differs from the raw loop by the split USE advance on about half the steps (a second round trip, removed by ruling 2), episode resets, and 0.29 ms of extraction. Run C (75/s with a fast raw phase in the same process) predates latency logging, so whether its steps were quantized at 15.6 ms is unknown. Any claim about when Windows honours a process's timer request is UNVERIFIED here.

### 4. `screen` is optional (amends ADR 0001)

Add `screen_enabled` to the env config. When false, the `screen` key is **absent** from both the observation dictionary and the observation space, and `modality[1]` is 0 (it is hard-coded to true today). It is a sensor selection, not a mod, so `assert_vanilla` must not reject either value. The value is saved with every checkpoint and evaluation must use the same value the model was trained with. Stage A0 trains with it off. Saving: 0.24 ms of 1.4 to 2.7 ms per decision, plus an unmeasured pipe saving.

Also amended while the entity table is cheap to change: add type id **`hazard`** (explosive barrels currently fall into `decoration`), appended at the end of the table so existing ids do not move. The model must size its type embedding from the table length.

### 5. `viz_nocheat` (amends the hedge in ADR 0001 and the stage order in ADR 0003)

Under the flag the survivors are `screen`, the health, armor, ammo and weapon part of `self`, damage taken, and `prev_action`. Rays, entities and ego-motion are gone. The notifications buffer survives but contains the map name at tic 1, so it stays out of observations. The fallback is therefore a pixel agent with a few scalars, and **Stage B's dead-reckoned map has no input under the flag**, because velocity and angle read 0.

The hedge keeps its shape but gets stronger: because the buffers are cleanly absent, the wrapper's `modality` flags detect the case exactly (already tested), and in training the telemetry is available as training-only labels for the pixel branch.

Plan while the organizers' answer is unknown:
- **Stage A0 unchanged** (telemetry, screen off). It is the quickest proof that the trainer works.
- **Stage A1 follows A0 immediately and now includes modality dropout:** in a share of training episodes the telemetry keys and ego-motion are zeroed and flagged absent, while the true rays and entity presence are kept as auxiliary targets for the pixel branch. This replaces ADR 0001's "switch on at the end of day 2" trigger: dropout is part of A1 regardless of the answer.
- **Stage B is built only after the organizers confirm telemetry is available at evaluation.** If the answer is "flag on", Stage B is dropped in its current form and the remaining time goes to the pixel-primary agent; a map addressed by ego-motion estimated from pixels would need a new ADR. If there is no answer by the time A1 learns, Stage B may be started only if at least 2.5 days remain, and the submitted default is whichever agent scores better on held-out maps with telemetry masked.
- **Every evaluation report carries two columns:** telemetry on, and telemetry masked.

This makes a CUDA build of torch more pressing than ADR 0004 stated, since the pixel branch may have to carry the policy. It remains the user's decision.

### Memory and the worker plan (adjusts the default in ADR 0004)

ADR 0004's 8 GB cap stands and now has wide margin. Extrapolating the single-instance figure (unmeasured: assumes Python memory is per worker and engine memory per environment), 32 environments are roughly 3 GB. **Memory is no longer the limit; CPU and heat are, so the spare memory is not a reason to add workers.** Proposed new default, subject to the user's agreement: **4 worker processes x 8 environments** (still 32 environments and the same rollout shape) with 4 torch threads, about 8 of 24 threads, stepping up to 8 x 4 only if the user is comfortable and the throughput is needed. Environments per worker cost memory, not cores, when stepped in turn. The vectorized benchmark needs the user's go-ahead before it is run.

## Risks

- **USE:** a switch press with held USE is unprobed. A stochastic policy early in training will still reverse doors by toggling the head. Lifts and doors that need a repeat press now cost two decisions.
- **Horizon:** 50 and 60 come from one shift measurement and an inference; vertical look is not in the action set, so the horizon is fixed, but a different screen size setting in a harness would move it.
- **Timer:** the unexplained 75/s run may recur in training and silently cut throughput fivefold. Higher power draw and heat on the user's only machine.
- **Screen off in A0:** the A0 checkpoint cannot be reused for A1 without adding a branch; A1 starts a new run or grafts the encoder.
- **Pixel fallback quality:** 60x80 may be too coarse for a pixel-primary agent (unmeasured; published pixel agents used 108x60 to 128x72). Modality dropout may slow learning of the telemetry path.
- **Rule compliance:** none of the five rulings adds logic between observation and action. Ruling 2 must stay unconditional. The notifications buffer must stay out of observations (map identifier).
- **Label bug to check:** the key-pickup pattern in `env/doom_env.py` (`_KEY_MSG`) looks as if it would also match a locked-door message of the form "you need a blue key", which would set `keys_held` falsely exactly where the keys-held auxiliary label matters most. That message was never probed, so this is unconfirmed.
- **Dead monsters:** whether a corpse keeps category `Monster` is unknown; if it does, corpses look like threats in the entity set.
- **Reset dead time:** turning is ignored for the first 7 tics of an episode (cause unknown). Not ruled on here; it affects at most the first two decisions.

## Validation experiment

All by `env-engineer` unless stated, about half a day in total, results appended to `docs/reference/engine-probes.md` and a new file in `reports/runs/`:

1. **Horizon:** measure the horizon row directly with HUD on and HUD off (row of maximum depth down a long corridor, or the row where floor and ceiling labels meet); confirm 50 and 60.
2. **USE:** with the held encoding, repeat the door probe (expect one open), then `use` 1, 0, 1 (expect a reversal), and press a switch. Locate a locked door on freedoom2 MAP02 or MAP03 and record the notification text with and without the key.
3. **Timer and variance:** re-run the single-env benchmark five times each with the timer on and off, after ruling 2 and with `screen` off and on; always log p50, p90, p99 step latency and the share of steps near 15.6 ms; record power source and Windows power mode for each run.
4. **Vectorized benchmark** (`trainer-ops`, only with the user's go-ahead): 2x8, 4x8, 8x4 workers x environments; decisions/s, total memory, CPU share, desktop responsiveness, temperature if readable; also one worker with the timer off to test the claim in option 3a.
5. **Masked-telemetry evaluation** (`evaluator`, once A1 trains): held-out maps with telemetry on and masked.

Reverse or amend if: held USE fails to press switches or open some door type (then reconsider 2a with the consecutive-press problem documented); the horizon measurement disagrees by more than 2 rows (correct the constant); the timer-on runs still show a 15.6 ms mode in more than a small share of steps (find the cause before any long run); 4 x 8 gives under about 1000 decisions/s (propose 8 x 4 to the user); pixel-only evaluation stays near random after A1 (raise `screen` resolution, which reopens the throughput question).
