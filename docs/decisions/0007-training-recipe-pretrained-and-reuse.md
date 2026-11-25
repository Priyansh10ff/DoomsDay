# ADR 0007: Training recipe, pretrained components, auxiliary losses, open-source reuse, API gotchas

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

**Amends ADR 0004** (recipe values, listed as deltas below) and **ADR 0003** (auxiliary heads, initialisation). Adds rulings that ADR 0006 left open. The earlier files are left as written; where they disagree with this one, this one applies. Nothing here overturns the choice of our own synchronous recurrent PPO.

## Context

The user's question for this round: can the team save time by building on something pretrained or open source instead of writing everything from scratch?

New material read in full: `docs/osc/training-configs.md`, `docs/research/ppo-recipe.md`, `docs/research/pretrained-components.md`, `docs/reference/vizdoom-api.md`, `docs/reference/gymnasium-api.md`, `docs/reference/pytorch-api.md`. Also read: all six ADRs, `docs/reference/engine-probes.md` (including round 2), both benchmark reports, and the code in `env/`, `agent/`, `train/`, `configs/`.

Facts that bound the decision:

- **The trainer already exists.** `worker` has built `train/` (rollout, GAE, PPO update, trainer, checkpoint, toy memory task) and `agent/` (Stage A0 model) to ADR 0004's starting values. The recipe below is therefore written as changes to named config fields, not as a new build. **What does not exist yet:** a subprocess sampler (only the in-process `SyncVecEnv`), a training launcher, and environment configs for the bundled scenarios.
- **Rules.** Rule 4.3: the core agent and training pipeline are written during the event; libraries are allowed, pre-existing agent or training code is not. The brief says an off-the-shelf script scores poorly and bespoke modules are heavily rewarded. Whether pretrained weights count as a "library" is not stated anywhere in `docs/TRACK_BRIEF.md`.
- **Machine.** One Windows 11 laptop, 24 threads, 15.4 GB RAM, RTX 5060 8 GB, torch 2.9.1 CPU-only. The user's instruction not to load it until it hangs or shuts down is a hard constraint. Deadline: "more than 5 days"; no date given.
- **Throughput, latest measurement.** `reports/runs/2026-11-25-env-timer-variance-bench.md` supersedes the 368 to 703 figure: one environment runs at 760 to 936 decisions/s with `screen` on and about 1170 to 1250 with it off (timer on); the 75/s anomaly did not recur in five runs. About 120 to 140 MB per environment. **Vectorised throughput and update cost are still unmeasured.** Any run with more than one engine instance needs the user's go-ahead (ADR 0006); that covers an in-process multi-environment run as well, since each environment starts its own `vizdoom.exe`.
- **Evidence quality.** The implementation-detail studies are feedforward MuJoCo. The reuse and staleness evidence for recurrent PPO is one source (OpenAI Five) at a scale thousands of times ours. No opened source compares BPTT lengths. The depth-auxiliary result (Mirowski et al.) is A3C on DMLab with two unresolved confounds. No frozen-encoder or depth-estimator result is from Doom. R2D2 burn-in is UNVERIFIED.

**ADR 0006 is structurally complete** (all five sections, finished final sentence). Its validation items 1 to 3 have since been run (`engine-probes.md` round 2, the timer-variance report). Its USE, timer and `screen` rulings held. **Its horizon ruling is unresolved, not confirmed:** the round-2 fit gives 51.7 to 52.4 (HUD on), which is 1.7 to 2.4 rows from the constant 50 against a 2-row tolerance, and the docstring in `env/config.py` cites 49.1 / 59.1, which is not what the probe file reports. ADR 0006 also left three things without a ruling, covered under Decision 5: the 7-tic turn dead time after reset, whether a corpse keeps category `Monster`, and (not noticed there) that `TRAIN_CONFIG.maps` lists freedoom2 maps that ADR 0005 reserves as held-out.

## Options considered

**How to use outside work (the user's question)**

- **O1. Import a framework or a trained agent** (Sample Factory, SB3, a Hugging Face Doom checkpoint, fine-tuned or distilled). Build: hours. Throughput: Sample Factory's is the best published, but it has no Windows support. Rejected: breaches rule 4.3, is the exact thing the brief says scores poorly, and the checkpoints are pixel battle agents that have never seen a door, key or exit.
- **O2. Our trainer, with a reference recipe adopted wholesale** (Sample Factory Doom: T 32, 1 epoch, lr 1e-4 constant, clip 0.1, gamma 0.99). Build: under 1 hour (config). Throughput: sampling unchanged; update about a third of today's cost (1 epoch instead of 3). Cost: gamma 0.99 and a 32-step window are tuned for dense-reward combat in 525-decision episodes with 240 to 576 environments; we have 32 environments and a key-then-door task.
- **O3. Our trainer, ADR 0004 values untouched.** Build: none. Throughput: as built. Cost: ignores cheap details with measured support (initialisation, fewer epochs) and leaves the PROVISIONAL fields unanswered.
- **O4 (chosen). Our trainer, ADR 0004 values with targeted changes from the references, plus one early test of the recurrent-specific settings.** Build: about half a day of `worker` time for the recipe changes. Test: one sequential batch of short runs (wall time unknown until the vectorised benchmark exists). Throughput: update cost falls by about a third (2 epochs instead of 3).

**Perception for the pixel branch / `viz_nocheat` fallback**

- **P1. Frozen ImageNet stem** (PIE-G form: first two ResNet18 stages). Build about half a day plus a weights download and a `torchvision` dependency. Throughput: runs on every environment step for every environment; no CPU figure exists; certainly slower than a 3-layer CNN. Rule 4.3 status unknown.
- **P2. Monocular depth estimator** (Depth Anything V2 Small, 24.8M parameters). Build about 1 day. Throughput: 13.3 ms per image on an RTX 4080 at 518x518, no CPU figure; unusable in a loop planned at about 1000 decisions/s.
- **P3 (chosen). Our own small CNN, with engine depth as a training-only prediction target.** Build: already planned in A1 (half a day) plus about 3 hours for the head. Throughput: one extra small head in the update only; nothing at inference.

**Ray / depth auxiliary head in Stage A0**

- **X1. Predict current rays from the GRU state in A0.** Build 2 hours, negligible throughput cost. In A0 the rays are an input, so this is reconstruction.
- **X2. Predict next-step rays from GRU state and chosen action in A0.** Build 3 hours, negligible cost. A real prediction task (after a 40-degree turn the target is geometry that was off-screen), but no opened source measured it.
- **X3 (chosen). No ray head in the first A0 run; mandatory pixel-to-rays head in A1; X2 as the first A0 ablation once A0 learns.** Build: 0 now, 3 hours in A1.

## Decision

### 1. Training recipe (deltas against `configs/train_config.py`, `agent/model.py`, `train/`)

"Own" marks a value chosen by reasoning, not taken from a source. File names in the Source column are under `docs/`.

| Setting (field) | Now | Ruling | Source and reason |
|---|---|---|---|
| Learning rate (`lr`) | 3e-4 | **Keep** | Hide-and-seek and Andrychowicz "safe default" (`ppo-recipe.md`). Sample Factory uses 1e-4, CleanRL 2.5e-4 (`training-configs.md`). |
| Schedule (`lr_final_fraction`, `_set_lr`) | 0.0, over `total_updates` | **Change: floor 0.1, schedule over environment decisions** | Own. Annealing helped in Engstrom, was second-order in Andrychowicz, and Sample Factory Doom uses none. Decay to zero conflicts with ADR 0004's stop-and-resume runs and an unknown deadline. Linear from 3e-4 to 3e-5 over the run's declared budget of **decisions** (not updates, so runs with different rollout lengths follow the same curve), then held at the floor. The position is saved in the checkpoint. A run that only continues an existing checkpoint on a new curriculum rung uses constant 1e-4 (Sample Factory's value). |
| Adam epsilon (`adam_eps`) | 1e-5 | **Keep** | ICLR blog post (secondary, prompt-seeded). Sample Factory uses 1e-6. Betas stay at torch defaults (0.9, 0.999; Sample Factory). |
| Discount (`gamma`) | 0.997 | **Keep, test early** | Own, bracketed by sources. The 0.99 sources are dense-reward combat. The sparse or long-horizon sources sit higher: 0.998 (hide-and-seek), 0.999 (Procgen, RND extrinsic). Sample Factory's 0.995 goes with frame skip 2 and is the same game-time horizon as 0.99 at skip 4, so it is not support for a longer horizon. 0.997 is a horizon of about 333 decisions, 38 s of game time. Andrychowicz: tune per environment. |
| GAE lambda (`gae_lambda`) | 0.95 | **Keep** | Every PPO source in both files. |
| Clip (`clip`) | 0.2 | **Keep, constant** | PPO paper ablation (0.2 slightly best, continuous control), hide-and-seek, Procgen. Sample Factory, CleanRL LSTM and PPO Atari use 0.1. With epochs reduced the wider clip is the smaller risk. Fall back to 0.1 if mean approximate KL stays above 0.02 or clip fraction above 0.3 (thresholds own). |
| Rollout length (`rollout_len`) | 64 | **Keep, test early** | Own. Inside the published range (10 to 128); longer than the PPO-family recurrent systems (10 to 32). Reason to keep: gradient reach of the memory is bounded by this window, and memory is the deliverable. 32 x 64 = 2048 samples per update, the batch Sample Factory uses. |
| BPTT length | = rollout | **Fixed equal to the rollout** | Sample Factory (recurrence = rollout), CleanRL LSTM. No separate field. Each sequence is replayed from the hidden state stored at the rollout start. No burn-in (R2D2 UNVERIFIED). |
| Epochs (`epochs`) | 3 | **Change to 2, test early** | DD-PPO used 2, Sample Factory 1; OpenAI Five measured losses from more reuse. 3 is the feedforward PPO Atari value. Stored hidden states and advantages are stale after the first pass. Evidence is weak for our scale. |
| Minibatches (`num_minibatches`) | 4 | **Keep** | Whole environment sequences, 8 per minibatch, reshuffled each epoch (CleanRL LSTM practice, as described in `training-configs.md` 3.3). 8 gradient steps per rollout. |
| Advantage normalisation | per minibatch, masked | **Keep** | PPO code practice; Andrychowicz: little effect. |
| Entropy (`entropy_coef`) | 0.01 on the summed entropy | **Keep, no schedule in A0** | PPO Atari, hide-and-seek, CleanRL. Sample Factory's `symmetric_kl` 0.001 is a different regulariser and is not adopted. Per-head entropy stays logged; a head pinned at zero or at its maximum for a whole run is the trigger to revisit. |
| Value coefficient (`value_coef`) | 0.5 | **Keep** | Sample Factory, CleanRL. PPO paper used 1. |
| Value-loss clipping | none | **Keep none** | Andrychowicz and Engstrom: hurts or does not help. Sample Factory and CleanRL do clip; the measured evidence says not to. Feedforward MuJoCo evidence only. |
| Gradient clip (`max_grad_norm`) | 1.0 | **Keep** | Own. Sources span 0.5 to 5; second-order. Log the returned norm. |
| Value-target scale (`ReturnScale`) | divide by running std of returns, no mean shift | **Keep this form** | Sample Factory `normalize_returns` (on in one published run, off in the other); Andrychowicz: check, do not assume. The policy loss is already scale-free through advantage normalisation. The statistics carry across curriculum stages and are never reset. `return_scale_min` 1e-2 stays. |
| Reward scale | engine reward, unshaped | **Shaped terms sized so one step is at most about 1** | Sample Factory battle scales (Decision 4). No reward clipping. A scenario whose native rewards are in the hundreds is scaled by 0.01 (Sample Factory does this for `deadly_corridor` and `basic`). |
| Observation normalisation | fixed-range scaling in the env | **Keep; no running normaliser** | Own. Andrychowicz says always normalise; pre-scaled bounded inputs meet that intent without statistics that must ship with the checkpoint and that would shift on unseen maps. `screen` is divided by 255 in the model (Sample Factory `obs_scale`). `env-engineer` adds a test that 10,000 random steps stay inside the declared space bounds. |
| Initialisation | policy heads orthogonal 0.01; all else torch default | **Change** | ICLR blog post, Engstrom, Andrychowicz. Orthogonal everywhere: gain sqrt(2) on Linear and Conv1d layers followed by ReLU; 1.0 on the GRU cell weights, both value heads and the auxiliary heads; 0.01 on the six policy heads (already done); biases 0. Embeddings and the two attention blocks keep torch defaults. |
| Hidden-state reset | zeroed before the first observation of an episode, same in collection and update | **Keep** | CleanRL LSTM practice; already implemented and tested by the ratio check. |
| Bootstrap on timeout | bootstrap on truncation, never on death or exit; stale time-outs dropped | **Keep the rule, fix the freshness flag** | Pardo et al. (ICML 2018). See fix request F1: today every DoomEnv time-out is treated as stale, although the wrapper's own time limit returns a live observation. |
| Remaining-time input | none | **Not added** | Pardo's second case applies only if the judged episodes have a limit; the limit is unknown and a wrong assumed limit is a train/eval mismatch. Question 4 of ADR 0001 stands. |
| Auxiliary weights (`keys_weight`, `damage_weight`, `monsters_weight`) | 0.1 each | **Keep for A0**; keys rises to 0.5 with positive-class weighting when a key stage starts | Own. `damage_horizon_k` 8 (0.9 s) kept, own. |
| Intrinsic stream (off in A0) | second value head reserved | **When ADR 0005's bonus is switched on:** extrinsic 2, intrinsic 1, intrinsic gamma 0.99, intrinsic reward divided by a running std of intrinsic returns | CleanRL RND / RND paper. Departure, own: our bonus is episodic, so intrinsic returns respect episode ends (RND's are non-episodic). |

**Early test (the "schedule" ruling for gamma, rollout length and epochs).** Described under Validation experiment. Until it reports, the table above is the default.

### 2. Pretrained components

**(a) Pretrained Doom agents from Hugging Face: ruled out, with no exception.** Not loaded, fine-tuned, distilled from, cloned from, used to generate data, or used as a teacher or sparring partner. They are pre-existing agents (rule 4.3), they are 128x72 pixel battle agents with a different action set, and Sample Factory does not run on Windows. Their `cfg.json` values are reading material, already captured in `training-configs.md`. No question for the organizers is needed.

**(b) Frozen non-Doom image encoder: not in the default agent; allowed only as a late ablation, behind three conditions.** The pixel branch is our own small CNN. Reasons: the one direct comparison that includes PPO found a from-scratch small CNN comparable (Hansen et al., figure only, manipulation tasks); the one study with a navigation-style task and out-of-distribution tests was negative (Schneider et al., model-based RL); the positive result (PIE-G) is off-policy continuous control. None is Doom. It also weakens the bespoke-architecture case and adds a per-step cost nobody has measured. Conditions for the ablation: the organizers answer the question below with a yes in writing; A1 with our own CNN has already trained; the stem's cost is measured on this machine first.
Question for the organizers: *Do pretrained non-Doom weights (for example an ImageNet ResNet18 from torchvision) used as a frozen feature extractor count as a permitted pre-existing library, or as a pre-existing agent component barred by the Original Work rule?*

**(c) Monocular depth estimator as a `viz_nocheat` fallback sensor: ruled out for training and not adopted as the fallback.** Reasons: its output is relative inverse depth with arbitrary scale and shift per frame, so it cannot stand in for the depth buffer our rays are built from; policies trained on true depth do not transfer to predicted depth without retraining on it; no CPU latency figure exists; it is untested on Doom imagery; and a 24.8M-parameter imported network in front of an agent of under 1M parameters would make most of the perception someone else's. **The fallback is our own learned depth head** (Decision 3): the pixel CNN is trained to predict our rays from the screen, with engine depth as a training-only label. The one door left open: if the organizers confirm the flag is on, answer yes to the question in (b), and our own head's ray error is poor, a depth model may be tried as a measured experiment.
Further question for the organizers: *What hardware does the evaluation run on (is a GPU available), and is there a per-decision time limit?*

### 3. Auxiliary losses

**Stage A0, first run: no ray or depth head.** In Mirowski et al. depth was a target and not an input; in A0 the rays are an input and modality dropout is off, so predicting current rays from the GRU state is reconstruction, which `ppo-recipe.md` records as the weak auxiliary form. A0's job is to prove the trainer, and the three existing heads stay as built.

**Stage A1: a ray-prediction head is mandatory.** Two outputs:
- **D1, from the pixel CNN features:** predict the current rays. The label exists on every training step, including steps where telemetry is dropped from the input. This is the Mirowski setting and it is our `viz_nocheat` depth sensor.
- **D2, from the GRU state:** predict the current rays, with the loss applied **only on steps where telemetry is masked from the input**. Mirowski found the recurrent-state placement at least as good as the feature placement.
- Target: rays reduced to 3 x 16 (nearest value in each group of 5 columns), classified into 8 depth bands with wider bands at distance. Mirowski reported classification ahead of regression on one maze (prompt-seeded, medium confidence); if it costs more than an hour, start with regression.
- Starting weight 0.5 each (own; Mirowski's 1 to 33 are for a different loss scale and do not transfer).
- The entity-presence target that ADRs 0003 and 0006 give the pixel branch **stays** alongside D1, at weight 0.1 (own). It has Doom evidence of its own (Arnold's enemy-present label).

**First A0 ablation after A0 learns (optional, X2):** predict next-step rays from the GRU state and the chosen action. Kept only if it helps on the memory probe of ADR 0003. No source measured this; it is our own idea.

**Interaction with the key-possession head (ADR 0001).**
- The keys head stays, unchanged, reading the GRU state. On A0's bundled scenarios its labels are all zero, so it contributes nothing there; that is expected.
- Each auxiliary loss is a mean over its own valid steps and has its own weight, so the dense ray loss cannot outvote the sparse keys loss by sample count. It can still dominate by gradient size. `worker` logs, every 50 updates, the gradient norm of each loss term at the GRU output. Rule of thumb (own): the ray terms together stay below the policy-plus-value gradient norm; lower the ray weight if not.
- Check: the linear probe for keys held (ADR 0003 validation 3) is run with the ray head on and off. If turning the ray head on lowers probe accuracy, the ray weight is halved.
- No auxiliary output is read by anything that chooses an action. `agent/inference.py` reads only the policy logits today; that must stay true (rule 4.1).

### 4. Open-source reuse

**Principle.** Values and ideas are reused; code is not. No training framework is imported (no Sample Factory, CleanRL, SB3, RLlib in the requirements). The working references are our own notes, which describe the mechanisms in our words. An original source file may be opened to settle an ambiguity; nothing is transcribed from it. Arnold (no licence), DFP (licence "Other") and E3B (CC-BY-NC) are numbers-and-paper only.

| Agent | Keep open | Starting values worth taking |
|---|---|---|
| `worker` | `osc/training-configs.md` sections 1.1, 3, 5; `research/ppo-recipe.md` section 3; `reference/pytorch-api.md` | The recipe table above. Recurrent handling as described in `training-configs.md` 3.3. RND split when curiosity is built: extrinsic 2, intrinsic 1, gamma 0.999 / 0.99 in the source (ours 0.997 / 0.99), predictor trained on 25% of samples, inputs clipped to [-5, 5]. |
| `reward-designer` | `osc/training-configs.md` 1.5; `osc/curiosity.md` section 8; `research/vizdoom-agents.md` (Arnold shaping row) | Sample Factory battle scales as per-step deltas: health +0.005 gained / -0.003 lost per point; armor +0.005 / -0.001; damage dealt +0.003 per point (capped at 200 per step); hit +0.01 (capped at 5); ammo +0.0002 to +0.002; weapon pickup +0.02 to +0.20. Kill +1 and death -0.75 are **our mapping** of the source's FRAGCOUNT and DEATHCOUNT rows, which are multiplayer counters; in single player we apply them to KILLCOUNT and to dying. Not taken: the selected-weapon bonus and weapon-preference multipliers (no weapon head, ADR 0002). Episodic coverage bonus E1: +0.1 for a new cell, cell radius about 75 map units, at most 200 landmarks per episode (Sample Factory's landmark wrapper). Exit, key and door terms have no reference value and belong to the later reward ADR that ADR 0005 defers. Shaping is computed from game-variable deltas in the wrapper, not from the engine's reward knobs, whose signs are untested (`vizdoom-api.md` section 3). |
| `env-engineer` | `reference/vizdoom-api.md`, `reference/gymnasium-api.md`, `reference/engine-probes.md` | Episode timeout 2100 tics for combat scenarios (Sample Factory, DFP). Wrapper order as a reading aid only. |

### 5. API gotchas that change a design decision

1. **Level exit.** Today `DoomEnv.step` infers an exit by elimination (finished, not dead, not timed out), and no probe has ever reached an exit. Decision: **an exit is counted only when two signals agree: the elimination test and an engine reward on the final step of at least `map_exit_reward`** (already 1.0 in the EVAL config). Disagreements are logged and counted as "unknown", not as success. Unknown and to be probed before any exit reward is designed: whether reaching the exit ends the episode at all on a campaign map, or the engine moves to the next level.
2. **Seeding.** `reset` calls `set_seed` on a running game and then `new_episode`; the docs do not say that works. Until a probe shows that the same seed gives the same episode, **no evaluation result may rely on per-episode reproducibility**: `evaluator` reports means with intervals over many episodes. Random spawn position and heading (needed by ADR 0003 against pose leak) cannot come from the seed and must come from the map (`map-forge`).
3. **Gymnasium autoreset.** **We do not use `gymnasium.vector` at all.** `DoomEnv` stays a plain single `gymnasium.Env`; vectorisation is our own `VecEnv` protocol in `train/vec_env.py`, which resets in the same step and hands back the final observation explicitly. NEXT_STEP would cost one dead step per episode and a mask in the rollout. Observations are addressed by key name, never by Dict order.
4. **Fully masked attention row.** It cannot occur: `EntitySetEncoder` prepends a learned empty token that is always unmasked. This is now an invariant of the architecture, not an accident: any change to the encoder must keep one unmasked token, and a test must run forward and backward with zero entities and with telemetry masked and assert finite outputs and gradients.
5. **Worker processes and `vizdoom.exe`.** `AsyncVectorEnv` is not used. The subprocess sampler is still to be built, to our own `VecEnv` protocol. Its workers are started **non-daemon**, with a close message that calls `game.close()`, a `try/finally` and `atexit` in the launcher, and workers that close their games when the parent's pipe closes. Whether `daemon=True` blocks the engine's child process is untested and we do not need to find out. Process counts stay at ADR 0006's 4 x 8 default. `torch.set_num_threads` is called in the main process before any torch operation; workers never import torch.
6. **Turn dead time after reset (left open by ADR 0006).** `episode_start_time` may explain it. Ruling: **leave it at the engine default of 1 in both training and evaluation.** The harness will most likely use the default, and matching it matters more than recovering two decisions per episode.
7. **Checkpoints.** `train/checkpoint.py` loads with `weights_only=False`, which runs pickle. Decision: checkpoints hold `state_dict`s and plain Python numbers, strings, lists and dicts only, and are loaded with `weights_only=True` (fix request F4).

### Fix requests arising from the code read

- **F1, `env/doom_env.py` lines 199-209 and `train/vec_env.py` lines 54-59, 85.** Freshness of the final observation is a constructor constant, and the docstring says DoomEnv finals are never fresh. That is true only when the engine ended the episode. On the wrapper's own time limit (`time_up`) the observation is live. Change: the env reports `info["final_obs_fresh"]` per step (true on `time_up`, false when `finished`), and `SyncVecEnv` reads it, keeping the constructor argument only as an override for fake envs. Effect: time-outs are bootstrapped as ADR 0004 intends, not dropped. Owners: `env-engineer` (env), `worker` (vec env).
- **F2, `env/doom_env.py` line 203.** On a finished episode the previous observation object is returned again. Return a copy (Gymnasium 1.4.0's checker flags reused objects).
- **F3, `env/config.py` line 190, inherited by `configs/a0_env.py` line 15.** `TRAIN_CONFIG.maps` is freedoom2 MAP01-03 and EVAL uses MAP01; ADR 0005 reserves freedoom2 as held-out. The A0 training config therefore points at held-out maps today. A0 must train on the bundled scenarios (ADR 0005 rung 1), which need their own env configs; `map-forge` supplies the later split.
- **F4, `train/checkpoint.py` line 39 and `train/trainer.py`.** Load with `weights_only=True`: store `sensor_signature` as nested lists and compare after converting, so no allow-listing is needed. Also missing: the RNG states (action, minibatch, dropout generators) and the schedule position in decisions are not saved, and `Trainer` has no periodic-save or resume path, which ADR 0004 requires (every 10 minutes and on interrupt).
- **F5, `agent/model.py` lines 53-55.** Only the policy heads are initialised; apply the initialisation row of the recipe table.
- **F6, `env/config.py` lines 41-45.** The `horizon_row` docstring cites 49.1 / 59.1; `engine-probes.md` R2.1 reports 51.7 to 52.4 / 61.7 to 62.4. Reconcile by measurement (Validation 4) and correct whichever is wrong.

## Risks

- **Recipe evidence does not match our setting.** Every measured implementation detail is feedforward MuJoCo. Epochs 2 rests on one source at another scale. Gamma 0.997 and T 64 are our reasoning. The early test uses one seed per arm and can only detect large effects.
- **Learning-rate floor.** If a stage's budget is set far too long, most of the run sits near the top of the schedule; if far too short, it sits at the floor. The budget is a guess until throughput is measured.
- **No running observation normaliser.** If entity deltas or damage spike beyond their nominal range the encoders see outliers. The bounds test covers random play only.
- **No pretrained perception.** If the harness sets `viz_nocheat`, the entry depends on a pixel branch trained from scratch on a CPU-only torch build at 60x80. A pretrained stem might have generalised better across textures (PIE-G), and we will not know unless the ablation's three conditions are met. A CUDA build of torch (ADR 0004, the user's decision) becomes more pressing in that case.
- **Rule 4.3.** Reading references while writing is allowed; near-copies are not. Sample Factory's reward table is taken as numbers; a strict judge could still ask where they came from, so the submission should cite it.
- **Rule 4.1.** The auxiliary heads, the ray head and any predicted rays are learned modules. If a later design feeds predicted rays back into the ray encoder, that is still network-to-network; if anything thresholds them in code, it is a breach.
- **Ray head versus memory.** A dense geometric loss may fill the GRU with near-field geometry at the expense of sparse facts such as keys held. The gradient-norm log and the probe check are the guard.
- **Exit detection.** If a campaign exit does not end the episode, every exit-based metric and reward is wrong until the probe is run. If the harness changes `map_exit_reward`, the second signal is lost at evaluation (it is not needed by the policy, only by our own scoring).
- **Generalisation.** F3 would have trained on held-out maps. Fixed spawns remain until `map-forge` delivers maps with varied starts.
- **Horizon row.** If the true horizon is nearer 52 than 50, the middle ray band sits slightly above it. Small, but it is in every observation.
- **Throughput and sequencing.** Vectorised sampling and update cost are unmeasured, the subprocess sampler is unbuilt, and every multi-environment run waits for the user's go-ahead. Until then only single-environment work can proceed.

## Validation experiment

1. **Toy memory with a delay longer than the short window** (`worker`, minutes, one process, no Doom). Extend `train/toy_memory.py` with a recall delay of 40 steps and run T 32 against T 64 at equal steps. ADR 0004's delay of 20 fits inside both windows and cannot tell them apart. Expected: T 64 solves it, T 32 does not without help. If both solve it, the argument for the longer window is weaker and arm B below decides.
2. **Recipe arms on Doom** (`trainer-ops`). **Prerequisites that do not exist yet:** the subprocess sampler and a launcher (`worker`), env configs for the bundled scenarios (`env-engineer`), ADR 0006's vectorised benchmark, and **the user's go-ahead for multi-environment runs**. Then, sequentially, at the 4 x 8 cap: Stage A0, `screen` off, one sparse navigation scenario (`my_way_home`) and one combat scenario, one seed, equal environment decisions per arm (proposed 2 million; rescale once throughput is known). Every arm uses the same learning-rate curve over decisions.
   - Base: T 64, 2 epochs, gamma 0.997, clip 0.2.
   - B: T 32. C: 1 epoch. D: gamma 0.99. E (if time allows): 3 epochs.
   - Measure: decisions to reach a fixed return, final return, approximate KL, clip fraction, explained variance, sampling and update decisions/s.
   - Reverse a base value only on a large difference: an arm reaches the base's final return in under half the decisions on both scenarios, or the base fails to learn where the arm learns. Small differences keep the base.
3. **Trainer checks after the changes** (`worker`): first-minibatch ratio still 1.0; GAE unit test with a fresh time-out bootstrapping from the final observation; zero-entity and masked-telemetry forward and backward finite; checkpoint save, load with `weights_only=True`, and resume gives identical next-update statistics.
4. **Engine probes** (`env-engineer`, one engine instance, about 4 hours, appended to `docs/reference/engine-probes.md`): (i) reach an exit on a short map and record whether the episode ends, the final-step reward, and each flag; (ii) same seed twice and two different seeds with a fixed action sequence, comparing observation hashes over 200 decisions; (iii) whether a killed monster's label keeps category `Monster`; (iv) 10,000 random steps inside the declared observation bounds; (v) the horizon row re-measured with far-depth pixels only, to settle 50 against 52.
5. **Ray head, in A1** (`worker` and `evaluator`): D1 ray-classification accuracy on held-out maps; A1 with and without the head at equal decisions, with telemetry on and masked; the keys probe with the head on and off.

Reverse or amend if: arm B, C or D wins by the margin above (change that value in a superseding ADR); the exit probe shows the episode does not end on exit (the env needs an explicit exit signal before any exit reward is used); the seed probe fails (evaluation stays statistical and ADR 0005's "fixed generator seeds" must be fixed at map level); the horizon is off by 2 rows or more (correct the constant before any long run, since it changes the observation); the ray head lowers the keys probe at every weight tried (drop D2, keep D1); the organizers allow pretrained weights **and** confirm `viz_nocheat` is on (run the 2(b) ablation before the pixel branch is finalised).
