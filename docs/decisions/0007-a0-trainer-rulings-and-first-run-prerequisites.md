# ADR 0007: Stage A0 trainer rulings, evaluation map loading, and prerequisites for the first DOOM run

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

**Amends** ADR 0004 (learning-rate schedule, value-target scale, defaults), ADR 0003 (layer norm, auxiliary targets), and ADR 0001/0006 (what the EVAL configuration accepts). Earlier files are left as written; where they disagree with this one, this one applies.

## Context

`worker` built the Stage A0 model and recurrent PPO. Read for this ADR: `agent/` (encoders, model, distributions, inference), `train/` (ppo, trainer, gae, rollout, aux_labels, vec_env, checkpoint, running_stats), `configs/model_config.py`, `configs/train_config.py`, `env/config.py`, and `reports/runs/2026-11-25-a0-toy-memory.md`. Reported by the coordinator: 128 tests pass, 1 skipped; the real observation runs through the real model; 474,474 parameters.

Recorded as known, no ruling needed:
- The env's rays read 0 in the centre columns because the player's weapon sprite has depth 0; `env-engineer` is fixing it. This contradicts section 2 of `docs/reference/engine-probes.md` ("the weapon sprite does not appear in the depth buffer"), which should be corrected when the fix lands.
- **Nothing has been trained on DOOM yet.** Every number below is from a toy task or from reading code.

Toy memory evidence (cue at step 0, recall at step 20, single seed, reduced widths, core 64):

| lr | gamma | Updates | Schedule | Success |
|---|---|---|---|---|
| 1e-3 | 0.99 | 100 | constant | 0.985 (memoryless 0.505) |
| 1e-3 | 0.997 | 100 | constant | 0.83 |
| 3e-4 | 0.997 | 350 | constant | 0.995 |
| 3e-4 | 0.99 | 150 | constant | 0.455 |
| 3e-4 | 0.997 | 300 | decay to 0 | 0.475 |

Constraints unchanged: one laptop, the user's instruction not to load it until it hangs or shuts down, deadline unknown ("more than 5 days"), organizers' answers unknown (viz_nocheat, harness, map delivery). The transcript available to me shows no go-ahead from the user for a multi-instance run.

Two things found in the code that block the first run and are ruled on in item 7:
- `TRAIN_CONFIG` in `env/config.py` trains on freedoom2 MAP01-MAP03, but ADR 0005 reserves Freedoom Phase 2 as never trained on.
- `screen_enabled` defaults to true, while `train/rollout.py` has no `screen` key; a default config would fail in collection.

## Options considered

**1. Learning rate**
- 1a. Linear decay to zero over `total_updates` (current code). No work. Ties learning to a run length that is not known on a stop-and-resume laptop; the toy run under this schedule stayed at chance.
- 1b. Constant 3e-4 (chosen). A one-field change. No throughput effect.
- 1c. Constant 1e-3. Same cost. Three times faster on the toy; riskier for a 474k-parameter recurrent policy on a noisy task; unsupported by anything but one toy seed.
- 1d. Adaptive rate driven by measured KL. Half a day. Deferred.

**2. Provisional values**: accept all as built (no work), or change the ones the code shows to be weak (about 2 hours, no throughput effect). Chosen: accept most, change three.

**3. Layer norm**: keep (no work) or remove (10 minutes, then re-run the encoder tests). Kept.

**4. `value_int`**: keep as an untrained stub (no work, 257 parameters) or remove and add later (breaks checkpoint loading when the intrinsic bonus arrives). Kept.

**5. Time-outs**: keep dropping the time-out transition (loses one transition and its reward each time-out) or bootstrap from the fresh final observation (already coded in the trainer; needs the per-step flag wired through, about 1 hour). The second is chosen, with the first kept as a counted fallback.

**6. EVAL map loading**
- 6a. Keep EVAL as stock IWAD only. Cannot run judge maps delivered as a WAD.
- 6b. Let EVAL take any `scenario_wad`. Removes the guarantee that our training PWADs cannot reach evaluation.
- 6c. A separate harness-owned map source, with the training field still forced empty (chosen). About 2 hours.

**7. First-run prerequisites**: build everything listed in ADR 0004 first (about 1.5 days before any learning signal), or only what keeps the laptop safe and lets the user stop without loss (about 1 day), with measurement extras after (chosen).

## Decision

### 1. Learning rate, Adam epsilon, toy gate

- **Constant learning rate 3e-4** (`lr_final_fraction = 1.0`). Runs on this machine are stopped and resumed at the user's convenience, so a schedule tied to `total_updates` has no defined meaning. Decay is permitted only for a final run of fixed, known length, and then to a floor of 0.1 of the base rate, **never to zero**.
- 1e-3 is not adopted. The steering signal on DOOM is the logged approximate KL: persistently below about 0.003 with no learning argues for doubling the rate; persistently above about 0.03 argues for halving it. These are guidance for the humans choosing the next run, not a mechanism in the code.
- **Adam epsilon 1e-5: accepted.** No source in `docs/research/` supports any value; it is a conventional choice and not worth an experiment.
- **The toy gate counts as met for what it was for: correctness of the recurrent path.** 0.985 against 0.505 for the same network with its state zeroed is far outside chance on 200 episodes, episodes straddle rollouts so state carry-over is exercised, and the first-minibatch ratio deviation is 7e-7. **It is not evidence for any hyperparameter.** One seed cannot rank learning rates or discounts; the table's gamma comparison is confounded with run length; the network was a reduced one; the task has no noise. The decay-to-zero failure shows the run was under-budget at 3e-4 (constant 3e-4 was still at chance at 150 updates and solved by 350), not that decay is harmful. Condition on the gate: repeat the recurrent and memoryless pair on three seeds (minutes of compute, see validation).

### 2. Provisional values

| Item | Ruling |
|---|---|
| Damage head, binary "any damage in next k=8 decisions" | **Accept**, with a fix: see death windows below. |
| Monster-count head, count/24, MSE | **Change** to min(count, 8)/8. About 1.2 labels were visible per state in the benchmark, so count/24 is near 0.05 and gives almost no gradient at weight 0.1. |
| Auxiliary weights 0.1 each | **Accept** as starting values; log each loss; the keys weight is **marked for an experiment** once maps with keys are in training. |
| Value targets divided by running std of returns | **Change the floor**: `return_scale_min` from 0.01 to 1.0. See below. |
| Previous action as six summed 16-d embeddings | **Accept.** |
| Policy-head init gain 0.01 | **Accept.** |
| Token init std 0.02 | **Accept.** |

**Value scale.** `train/running_stats.py` keeps an all-time standard deviation of GAE returns with a floor of 0.01. With sparse rewards the early returns are near zero, the scale sits at the floor, and the first reward of 1.0 becomes a target near 100 in the head's units. Separately, `train/trainer.py` stores values at the old scale and builds targets at the new one without adjusting the head. With a floor of 1.0 the scale can only shrink large returns, never inflate small ones. Watch `return_scale`, value loss, explained variance and pre-clip gradient norm.

**Death windows.** After death the final observation is stale, so `damage_events` marks that event invalid and `damage_window_targets` drops every window that reaches a death. The head therefore never sees its most informative positive. Fix: on an ended episode take the label from the final `info` (death, or damage reported there), which is training-only data; a window ending in death is a valid positive.

**Coupled indices.** `monster_type_id = 1` and `self_damage_index = 3` in `configs/train_config.py` silently depend on the env's tables. Derive them from `env.observation` or add a test that fails when the tables move.

### 3. Layer norm

**Keep** the layer norm after the attention residual in `EntitySetEncoder`. It is applied per token, so permutation invariance and the empty-set behaviour are unaffected (the tests cover both), and entity features arrive on mixed scales. This is recorded as an addition to the Stage A0 description in ADR 0003.

### 4. `value_int`

**Keep as a stub.** ADR 0005 turns on an episodic bonus soon after the trainer runs, and removing the head now would make every A0 checkpoint incompatible then. Requirements: it enters no loss and no logged value until the intrinsic stream exists, and a test asserts it receives no gradient in an update.

### 5. Time-outs and episode ends

The env reports per step, in `info["final_obs_fresh"]`, whether the final observation is the true next state. The vector env reads that key **per step and per env** (today `SyncVecEnv` takes it as a constructor constant) and treats a missing key as stale.

| Case | Next value used for the last transition | Transition kept | Notes |
|---|---|---|---|
| Death or level exit (`terminated`) | 0 | Yes, with its reward | The final observation is never fed to the network, fresh or stale. |
| Time-out, final observation fresh | V(final observation), computed with the hidden state that processed the last real observation, **no reset**, and the episode's telemetry mask | Yes | Advantage chain is cut after it. |
| Time-out, final observation stale | not available | **No**: dropped from all losses; the step before it bootstraps from the stored value of the last real state | Fallback only. Count these per update and log the count; after the env fix it must be zero, and a non-zero count is a bug to report. |
| Rollout ends mid-episode | V(next observation) with the carried hidden state | Yes | Chain continues into the next rollout only through this bootstrap. |

In every case the hidden state is zeroed before the first observation of the next episode. `train/gae.py` and the trainer already implement these four cases; the work is the flag plumbing and the counter. In addition, the auxiliary damage label for an ended episode comes from the final `info` (item 2).

### 6. What EVAL accepts

Keep the guarantee where it belongs: **we add nothing to the game at evaluation.** Map delivery is a harness matter, so it gets harness fields:

- New harness fields `map_wad` (path to a WAD that supplies the maps, default none) and `iwad_file` (path to an IWAD supplied by the harness, default none, in which case the stock name is used). Both join `FIELD_GROUPS`. Under EVAL they are legal if the file exists; they are loaded as given and never edited.
- `scenario_wad` stays a rule field and **must remain empty under EVAL**, as must `extra`, altered engine rewards, and objects/sectors info. Training code never sets `map_wad`; a test asserts that no path under the repo's training-map folders can be passed as `map_wad` in a vanilla config.
- `game_args` stays empty under EVAL with **one exact-match exception: `+viz_nocheat 1`**, which only removes information and is needed for ADR 0006's telemetry-off column. Nothing else is whitelisted.
- Every evaluation report records the SHA-256 of the IWAD and of the map WAD, and the map names.
- We do not assume what lumps judge maps contain and do not reject on content. For **our own** held-out WADs the evaluator additionally lists the lump names and flags any scripting or patch lumps in the report.

Until the organizers answer, the stock-IWAD EVAL remains the default and the path above is the prepared alternative.

### 7. Before the first real DOOM run

The test for "must exist" is: does it keep the laptop safe unattended, or let the user stop at any moment without losing work. In this order:

1. **Multi-process vector env and DOOM training entry point**, layout from config (default 4 workers x 8 envs, 4 torch threads), workers importing no torch. Includes guaranteed shutdown: on normal exit, Ctrl-C, or a crash, every worker and every `vizdoom.exe` is closed, and a start-up check refuses to run if engine processes from an earlier run are still alive. The entry point sets `screen_enabled=False` explicitly for A0.
2. **Below-normal priority** for the main process and workers, with a check (not an assumption) that the engine child processes run at that priority too.
3. **Checkpoint and resume**: save every 10 minutes and on interrupt; resume restores model, optimizer, return scale, update index and RNG state, and refuses a checkpoint whose sensor signature differs.
4. **Resident-memory logging with an enforced cap**: total memory of the process tree logged every update; above 8 GB the run saves a checkpoint and stops.
5. **Cooldown pause** between updates as a config value, default 0. It is a few lines and is the only throttle the user has besides worker count.

Can follow the first run: per-map episode statistics (required before any run with more than one map), and the intrinsic stream.

**First run content.** One bundled navigation scenario from ADR 0005 rung 1, `my_way_home`, loaded as a training scenario with the reward the bundled scenario itself defines; no reward shaping is decided here. **No Freedoom Phase 2 map is used for training**: the `TRAIN_CONFIG` map list of freedoom2 MAP01-MAP03 conflicts with ADR 0005 and must be changed before any run. The first launch is a 10-minute smoke run with the user present, at a reduced layout (2 workers x 4 envs), then the default layout. **Any multi-instance run needs the user's go-ahead first.**

## Risks

- **Learning rate.** 3e-4 constant may be slow on DOOM as it was on the toy; nothing here predicts the right value.
- **Toy gate.** It tests one mechanism. Bugs in telemetry masking, time-out bootstrapping on real episodes, or the auxiliary labels would not show up in it.
- **Value scale.** With a floor of 1.0, tasks whose returns are much smaller than 1 get no normalization; if shaped rewards later make returns large, the unadjusted head will lag each scale change.
- **Stub head.** A stub that silently enters a loss later without a second advantage stream would corrupt training; hence the gradient test.
- **EVAL widening.** `map_wad` is a door that training content could come through by mistake; the folder test and the hashes are the control. The `+viz_nocheat 1` exception must stay an exact match. Rule compliance otherwise unchanged: nothing here touches the path from observation to action.
- **Judge IWAD.** If the judges use an IWAD other than Freedoom, textures and sprites change. That matters for the pixel branch, not for A0.
- **Laptop safety.** Priority inheritance by engine processes is unmeasured; an 8 GB cap protects memory but not temperature, which we cannot read reliably.
- **First-run scenario.** `my_way_home` has no doors, keys or monsters and a fixed layout; success there proves the pipeline, not generalization.
- **Auxiliary heads.** The monster-count head is nearly trivial while telemetry is visible; its value appears only under telemetry masking in A1.

## Validation experiment

1. **Toy, three seeds** (`worker`, about 10 minutes of compute): recurrent and memoryless at lr 1e-3, gamma 0.99; and recurrent at 3e-4, gamma 0.997, constant, 350 updates. Gate holds if every recurrent run is at or above 0.9 and every memoryless run is within 0.1 of 0.5.
2. **Unit tests** (`worker`): `value_int` gets no gradient; a fresh time-out bootstraps from the final observation's value with the un-reset hidden state; a stale time-out is dropped and counted; a missing `final_obs_fresh` key is treated as stale; a damage window ending in death is a valid positive; the coupled indices match the env tables.
3. **EVAL tests** (`env-engineer`): a vanilla config with `map_wad` set passes; with `scenario_wad` set fails; with `game_args` of exactly `+viz_nocheat 1` passes and any other argument fails; a training-folder path as `map_wad` fails.
4. **Smoke run** (`trainer-ops`, with the user's go-ahead): 10 minutes on `my_way_home` at 2 x 4; confirm priority of every process including engines, memory per update, checkpoint written, interrupt and resume reproduce the update index, and no engine process survives exit. Then one hour at the default layout, logging sampling and update rates separately, approximate KL, `return_scale`, explained variance and stale-time-out count.

Reverse or amend if: any seed fails the toy gate (block DOOM training and debug); approximate KL on DOOM sits outside 0.003 to 0.03 for 50 updates (change the rate as in item 1); `return_scale` more than doubles between consecutive updates after the first 50, or explained variance collapses when it moves (freeze the scale or rescale the head, a new ruling); the stale-time-out count is non-zero after the env fix (fix the env before continuing); engine processes do not take the lowered priority (set it on them directly before any long run); the smoke run makes the laptop sluggish at 2 x 4 (stop and return to the user with the measurements).
