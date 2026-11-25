# Lesson 06: Collecting experience and GAE (rollouts, values, advantages)

> **Changed since first written (2026-11-25, after ADR 0007).** Factual corrections only; the explanation of rollouts and GAE is unchanged.
> - **Freshness is now per step and per environment.** The "Look out" box about a single `final_obs_fresh` flag is resolved. The environment reports `info["final_obs_fresh"]` on every step, and the vector environment reads it for each ending (3.2, 3.4).
> - **The damage label for an ended episode comes from the final `info`**, so a death is now a valid "damage" event (3.3).
> - **The learning rate is now constant** (3e-4); a decay is allowed only for a final fixed-length run, and never below 10% of the base rate (3.7).
> - **The value-scale floor** changed from 0.01 to 1.0 (`0007-a0-trainer-rulings`), and then to **0.1** (ADR 0008) (3.5).
> - **Worker processes, checkpoint resume, priority, the memory cap and the cooldown now exist** (lesson 08).
> - **The toy memory investigation is finished: no trainer bug was found.** See lesson 09 for why the test failed and how ADR 0008 replaced it (Gate A passed 8 of 8).
> - Line numbers re-checked against the current `train/trainer.py`.
> - **Two ADRs are numbered 0007, and ADR 0008 reconciles them** (`0008-reconcile-0007-files-memory-gate-and-safety-defaults.md`): constant learning rate 3e-4 (any final decay defined over decisions), 3 epochs, value-scale floor 0.1. Line numbers re-checked again after ADR 0008's changes to `train/trainer.py`.

Files taught: `train/vec_env.py`, `train/rollout.py`, `train/trainer.py` (collection, advantages, learning-rate schedule), `train/gae.py`, `train/running_stats.py`, `train/telemetry_mask.py`, `train/aux_labels.py` (how labels are gathered), `train/checkpoint.py`, `configs/train_config.py`, and the tests `tests/test_train_gae.py`, `tests/test_train_ppo.py`, `tests/test_train_aux_and_mask.py`, `tests/test_a0_checkpoint_and_env_config.py`.

Design records: ADR 0004 (our own synchronous recurrent PPO) and ADR 0006 (modality dropout, worker layout). ADR 0007 amends both, and ADR 0008 reconciles the two files numbered 0007. **All ADRs are "Proposed", not ratified.** Nothing has been trained on DOOM yet. The trainer has been run on a fake DOOM-shaped environment in tests, on a toy memory task (lessons 07 and 09), and for one update in a single-environment DOOM smoke run (lesson 08).

Builds on lesson 01 (terminated versus truncated, the fresh or stale final observation) and lesson 05 (the model's `step`, values, hidden-state resets).

---

## 1. What problem this piece solves

The agent learns from its own play. Before it can improve, it must answer a question for every action it took: **was that action better or worse than usual?** Turning left at a junction gave no reward right away, but 300 decisions later the agent reached the exit. How much credit does that left turn deserve?

Two difficulties make this hard in DOOM:

- **Rewards are rare and late.** Under vanilla rules the only reward is +1 at the exit, thousands of decisions after the useful moves.
- **Episodes end in two different ways** (lesson 01). Dying means there is no future. Running out of time means there *was* a future we did not watch. Mixing the two up teaches the agent that the time limit is a kind of death.

This piece plays many games for a short stretch, stores everything, and turns it into a per-decision score called the **advantage**, which lesson 07 uses to update the network.

## 2. The idea in plain words

Think of a **coach reviewing game footage**. For each move, the coach compares two things: what the team expected to happen from that position (their **value** guess), and what actually happened over the next few moments plus their best guess of what would come after. If things went better than expected, the move gets a positive mark; if worse, negative.

- The coach does not wait for the whole season. They watch a **short clip** (64 decisions per game, the **rollout**), and for anything after the clip they use the team's own guess (this is called **bootstrapping**).
- Rewards far in the future count a little less than rewards now (**discounting**).
- If a player was sent off (death), nothing more is counted after it. If the clip just ran out of time (truncation), the coach still adds a guess of what would have followed.

Where the analogy stops being accurate:

- The "expected" side is the network's own value head, which is itself learning. The marks are only as good as that guess, and early in training the guess is poor.
- The coach watches 32 games at once, side by side, each with its own clip boundaries and episode ends.

## 3. A walk through our actual code

### 3.1 Terms first

- **Return**: the total reward from a moment onward. With **discounting**, a reward k decisions in the future is multiplied by γ^k, where **γ (gamma)** is a number just below 1. ADR 0004 sets **γ = 0.997** (`train_config.py:29`). A rough "how far ahead it looks" is 1 / (1 - γ): 333 decisions (about 38 s of game time) for 0.997, against 100 decisions (about 11 s) for the common 0.99. ADR 0004 chose 0.997 because a level takes thousands of decisions; it says this is its own reasoning, not taken from a DOOM source.
- **Value** V(s): the network's guess of the discounted return from situation s (lesson 05).
- **Bootstrapping**: using the value guess in place of the rewards you did not see.
- **TD error** (temporal-difference error), δ_t = r_t + γ V(s_{t+1}) - V(s_t): "the reward I got, plus what I now expect from the next situation, minus what I expected before". Positive means things went better than expected for one step.
- **Advantage** A_t: how much better the action at t turned out than the value expected. We estimate it with GAE.
- **GAE** (generalized advantage estimation): a weighted sum of future TD errors, A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ..., where **λ (lambda)** = 0.95 (`train_config.py:30`). λ near 1 trusts the real rewards over more steps (less bias, more noise); λ near 0 trusts the value guess sooner (less noise, more bias).
- **On-policy**: learning only from experience collected by the current version of the network. PPO is on-policy, so each rollout is used for one update and then thrown away.

### 3.2 Many environments at once: `train/vec_env.py`

A **vectorized environment** runs N environments and steps them together with one batch of N actions. The trainer is written against the `VecEnv` interface (`vec_env.py:37-46`). There are now two implementations: `SyncVecEnv` (`:92-111`), which steps N environments **one after another, in the same process** (used by tests and the toy task), and `SubprocVecEnv`, which runs them in separate worker processes (lesson 08). This module never imports torch (`vec_env.py:1-2`), so worker processes can use it without loading torch.

**Auto-reset** (`vec_env.py:4-10`, `step_autoreset` at `:69-76`): when environment i ends, it is reset immediately. The returned `obs[i]` is then the **first observation of the next episode**. The observation and `info` the ended episode returned go to `final_obs[i]` and `final_infos[i]`. `final_obs_fresh[i]` is read **per step and per environment** from that ending step's `info["final_obs_fresh"]` (`:73`); a missing key counts as stale. For `DoomEnv` it is True at a time-out and False after death or level exit (lesson 01). Both vector environments use the same `step_autoreset`, so they behave identically.

### 3.3 Collecting a rollout: `trainer.py:100-169`

A **rollout** here is T = 64 decisions (`train_config.py:14`) from each of the N environments, stored as (T, N) arrays (`train/rollout.py:31-50`). `collect()`:

- `:104`: store `h0`, the hidden state **before** the first step. The hidden state is **not** reset at the start of a rollout. Episodes run across rollout boundaries, so the memory must too (`tests/test_train_ppo.py:98-112`).
- `:110-159`, for each step t (`:111`: if a stop was requested, the rollout is abandoned; lesson 08):
  - `:113`: build the network's input. `mask_telemetry` may blank the telemetry for this episode (3.6); in A0 it never does.
  - `:115-117`: one `model.step` for all N environments, then sample the six heads.
  - `:118-128`: store the **true** observation, the `episode_start` flags, the action, its log-probability (needed in lesson 07), the value (converted to reward units, 3.5), the keys-held label from `info` and the visible-monster count.
- `:130-137`: step the environments. Store reward, `terminated`, `truncated`. A time-out whose final observation is **not fresh** is marked `valid = False` and counted (3.4).
- `:138-144`: for a time-out whose final observation **is** fresh, compute the bootstrap value: one more model step on that final observation, using the hidden state that just processed step t and the episode's telemetry mask, **without a reset** (`episode_start` zeros at `:143`). That is the value of the true last state, in the same memory context.
- `:145-148`: damage events for the damage label. For an environment that just ended, the event now comes from the final `info` (`train/aux_labels.py:44-68`): a death counts as damage; otherwise `last_step_damage_taken` is used. So a window that ends in death is a valid positive (ADR 0007 ruling 2).
- `:150-155`: log episode returns and lengths.
- `:156-159`: carry the new hidden state; the next `episode_start` is 1 for every environment that just ended (`:158`), so its memory is reset before its new first observation (lesson 05's convention); redraw the telemetry-dropout choice for new episodes.
- `:161-166`: after the loop, finish the damage labels and compute `last_value`, the value of the observation after the last step, used to bootstrap the end of the rollout.
- `:167-168`: store this rollout's count of stale time-outs, and add it to a running total. Both are logged.

### 3.4 GAE with the two kinds of ending: `train/gae.py:18-39`

`compute_gae` walks backwards from the last step (`:31`). For each step t:

- `:32-33`: the "next value" and "next advantage" are normally those of step t+1 (or `last_value` at the end of the rollout).
- `:35`: **choose the next value according to how step t ended:**
  - `terminated[t]` (death or exit): **0**. There is no future.
  - `truncated[t]` (time limit): `bootstrap_value[t]`, the value of the true final observation. The future existed; we guess it.
  - otherwise: `values[t+1]`.
- `:36`: if the episode ended at t, the **chain is cut**: step t does not add any of the next episode's advantage.
- `:37`: δ_t = r_t + γ · next value - V(s_t).
- `:38`: A_t = δ_t + γλ · next advantage, or **0** if step t is not `valid`.
- `:39`: return the advantages and `advantages + values`. The second is the **value target** (also called the return estimate): what the value head will be trained toward in lesson 07.

**The stale final observation** (`trainer.py:4-13`, `gae.py:6-10`). If a time-out's final observation is one decision old (stale), the bootstrap would be computed on the wrong situation. The trainer never does that. Instead:

- the time-out step itself is dropped (`valid[t] = False`): no advantage, and no policy, value or auxiliary loss on it;
- step t-1 bootstraps from `values[t]`, the value of observation t, which is a real state the network saw during collection.

The cost is one transition and its reward per stale time-out. With `DoomEnv` this is now only a fallback: a time-out at the wrapper's own limit is always fresh. Only the engine's backstop limit (lesson 01) could produce a stale one. ADR 0007 ruling 5 says the count must be zero, and a non-zero `stale_timeouts` in the logs is an environment bug to report. The tests check every case against hand-computed numbers: no end, termination, truncation, termination versus truncation, truncation at the last step, and the stale time-out (`tests/test_train_gae.py:26-86`); and in the full trainer with a fake environment (`tests/test_train_ppo.py:63-96`).

> **Resolved (ADR 0007 ruling 5): freshness per step.** The first version of this lesson flagged that `SyncVecEnv` had one freshness flag for all endings, while `DoomEnv` is fresh at time-outs and stale after death. Now the environment reports `info["final_obs_fresh"]` on every step (`env/doom_env.py:189`, `:225`, `:229`), and `step_autoreset` reads it per environment (`vec_env.py:73`). Time-outs bootstrap from the fresh final observation, deaths use 0, and the damage label after death comes from `info` instead of the repeated observation. The test helper `FakeDoomEnv` (`tests/a0_helpers.py:63-103`) now follows the same contract: fresh at time-outs by default, stale after death, with switches to simulate a stale time-out (`stale_timeout=True`) or a missing key (`omit_fresh_key=True`) for the fallback tests (`tests/test_train_ppo.py:132-142`).

### 3.5 Return scaling: `train/running_stats.py` and `trainer.py:185-199`

Rewards in DOOM can be rare (+1) or, with shaping, frequent and large. To keep the value head's numbers in a stable range, ADR 0004 asks for "value targets normalized by a running return scale". The worker implemented this as dividing the targets by the **running standard deviation of all GAE returns seen so far** (`running_stats.py`). Its **floor** (the smallest scale allowed) has changed twice: from 0.01 to 1.0 under `0007-a0-trainer-rulings`, which worried that a first sparse reward of 1.0 could become a target of about 100, and then to **0.1** under ADR 0008 (`train_config.py:32-35`), at most a tenfold amplification. The 1.0 floor was withdrawn because it measured worse on 3 of 3 toy seeds: it weakened the value loss, which is what builds the memory. Lesson 09 explains this in full. The value head therefore predicts in "scaled" units. During collection its outputs are multiplied back by the current scale, so stored values, the bootstrap and `last_value` are all in reward units (`trainer.py:109`, `:124`, `:144`, `:166`). In `update` (`:185-199`) the scale is updated from the new returns (valid steps only), the targets are divided by it, and a diagnostic called **explained variance** is logged: 1 means the values predicted the returns perfectly, 0 means no better than a constant.

### 3.6 Telemetry masking (off in A0): `train/telemetry_mask.py`

ADR 0006 ruling 5 plans **modality dropout** for Stage A1: in a share of training episodes, the rays, entities and ego-motion are hidden from the network and the modality flags set to 0, so it learns to cope if the judges' engine withholds them (lesson 02, `+viz_nocheat`). The mechanism exists already:

- `EpisodeDropout` (`telemetry_mask.py:47-58`) draws, per environment and per episode, whether telemetry is hidden, from a seeded generator.
- `mask_telemetry` (`:16-44`) builds the network's input with those keys zeroed and flagged absent.
- The rollout stores the **true** observation and the dropped flag (`rollout.py:3-6`), and the mask is re-applied both when acting and when recomputing in the update. So hidden inputs still provide true targets for auxiliary losses.

A0 sets the probability to **0.0** (`train_config.py:63-66`).

### 3.7 Checkpoints and the learning-rate schedule

- **Checkpoints** (a saved copy of the training state): `train/checkpoint.py` now saves the whole trainer state and can **resume** from it, refusing a checkpoint whose sensor signature, model config or observation spec differs. Lesson 08 teaches this.
- **Learning rate** (`trainer.py:176-183`, `train_config.py:18-25`): **constant 3e-4** (ADR 0007 ruling 1, confirmed by ADR 0008 point 1; any final decay is now defined over decisions, `lr_decay_decisions`, and its position is saved in the checkpoint). Runs on this laptop are stopped and resumed whenever the user needs the machine, so a schedule tied to a planned number of updates has no clear meaning. A decay is allowed only for a final run of fixed length, and never below 10% of the base rate (`lr_decay_floor_fraction`; `PPOConfig` refuses anything else, `train_config.py:38-43`). The first version decayed to zero.

Built since the first version of this lesson (lesson 08): separate worker processes, resuming a run from a checkpoint, below-normal process priority, the cooldown pause, and process-tree memory logging with a cap.

## 4. A tiny worked example with real numbers

I called `compute_gae` with the same small numbers the tests use: γ = 0.9 and λ = 0.8 (chosen to make hand arithmetic easy, not the training values), three steps of one environment, rewards [1, 2, 3], values [0.5, 1.0, 1.5], and `last_value` = 4. Real output:

```
GAE none adv [5.735840320587158, 6.022000312805176, 5.099999904632568] ret [6.235840320587158, 7.022000312805176, 6.599999904632568]
GAE terminated@1 adv [2.119999885559082, 1.0, 5.099999904632568] ret [2.619999885559082, 2.0, 6.599999904632568]
GAE truncated@1 (boot 7) adv [6.655999660491943, 7.299999237060547, 5.099999904632568] ret [7.155999660491943, 8.299999237060547, 6.599999904632568]
GAE stale truncated@1 (valid False) adv [1.399999976158142, 0.0, 5.099999904632568] ret [1.899999976158142, 1.0, 6.599999904632568]
```

By hand, γλ = 0.72.

**No episode end.**
- Step 2: δ₂ = 3 + 0.9 x 4 - 1.5 = **5.1** = A₂.
- Step 1: δ₁ = 2 + 0.9 x 1.5 - 1 = 2.35; A₁ = 2.35 + 0.72 x 5.1 = **6.022**.
- Step 0: δ₀ = 1 + 0.9 x 1.0 - 0.5 = 1.4; A₀ = 1.4 + 0.72 x 6.022 = **5.736**.
- Value targets are A + V: 5.736 + 0.5 = 6.236, and so on.

**Death at step 1** (`terminated`). The next value is 0 and the chain is cut: A₁ = 2 + 0 - 1 = **1.0**. Step 0 then gets A₀ = 1.4 + 0.72 x 1.0 = **2.12**. Step 2 belongs to the next episode and is unchanged (5.1).

**Time-out at step 1** (`truncated`), with the value of the true final observation = 7. A₁ = 2 + 0.9 x 7 - 1 = **7.3**; A₀ = 1.4 + 0.72 x 7.3 = **6.656**. Same reward, very different credit from the death case: the episode was going well and was only cut off.

**Stale time-out at step 1** (`valid = False`). A₁ = **0** and its target is its own value (1.0), so it teaches nothing. Step 0 bootstraps from V(s₁) = 1.0: A₀ = 1 + 0.9 x 1.0 - 0.5 = **1.4**, with no chain from step 1.

**Return scale** (real output):

```
ReturnScale after [0,1,0,1]: 0.50000001
```

The standard deviation of 0, 1, 0, 1 is 0.5 (plus a tiny ε). Value targets would be divided by 0.5.

## 5. Why we chose this design over the alternatives

From ADR 0004 (**Proposed, not ratified**):

- **A. Stable-Baselines3 / sb3-contrib RecurrentPPO.** Rejected: the brief says an off-the-shelf PPO script scores poorly, and it does not fit dictionary observations with a custom memory without fighting the library.
- **B. Our own synchronous recurrent PPO (chosen).** "Synchronous" means collection pauses while the network updates. That is simpler to reason about: no lag between the policy that acted and the one being trained.
- **C. An asynchronous actor-learner with V-trace (IMPALA style).** "Asynchronous" means the environments keep playing while the network learns; V-trace is a correction for the resulting lag between the policy that acted and the one being trained. Possibly up to twice the throughput, but 3 days or more to build on Windows, with policy-lag debugging.
- **D. Value-based recurrent replay (R2D2 style).** This learns a value for each action from a stored archive of past experience, called a replay buffer. A replay buffer of sequences competes for 15.4 GB of RAM, and factorized actions fit poorly.

Specific rulings the code follows: rollout length 64; store the memory state at the start of each rollout and reset it inside wherever an episode ended; GAE with λ 0.95 and γ 0.997; **"time-limit truncation is not termination: on timeout, bootstrap from the value of the final observation; on death or exit, do not"**; value targets normalized by a running return scale. The stale-observation rule (drop the transition, never bootstrap from a stale observation) is the worker's design for a case the ADR did not foresee. ADR 0004's validation plan asked for GAE unit tests "including a mid-rollout termination and a truncation"; they are in `tests/test_train_gae.py`.

## 6. How it connects

- **Lesson 01** produced `terminated`, `truncated`, the fresh final observation at time-outs and the stale one after death.
- **Lesson 05** produced the values, the log-probabilities and the reset convention used here.
- **Lesson 07 (next)** uses the stored rollout, the advantages and the value targets to update the network with PPO, and shows the toy memory test.
- **Later**: when an exploration reward is added (ADR 0005), a second value head (`value_int`) and a second advantage stream will be needed.

## 7. Common mistakes, and how we would notice them

| Mistake | How it would show up |
|---|---|
| Treating a time-out as a death (next value 0). | Long episodes look like failures at their end. The value head learns "the end of the clock is bad", and explained variance drops near time-outs. `tests/test_train_gae.py:51-56` shows the two must differ. |
| Bootstrapping from a stale final observation. | The credit for the last step is computed from the wrong situation. Hard to see in logs. `tests/test_train_ppo.py:63-80` checks the trainer never does it. |
| Letting the advantage chain run across an episode end. | The next episode's luck leaks into this episode's actions. Checked by the termination tests in `tests/test_train_gae.py`. |
| Resetting the hidden state at every rollout start. | The memory is cut every 64 decisions even mid-episode. `tests/test_train_ppo.py:96-108` checks it is carried. |
| Mixing scaled and unscaled values. | Advantages come out much too large or small, and value loss explodes. Check `return_scale` and `explained_variance` in the update stats. |
| An environment that forgets to report `info["final_obs_fresh"]`. | A missing key counts as stale, so every time-out is dropped: `stale_timeouts` in the logs is non-zero and `valid_frac` drops below 1. ADR 0007 says that count must be zero. |

## 8. Check your understanding

1. Use γ = 0.9, λ = 0.8. One environment, two steps: rewards [0, 1], values [2, 3], `last_value` = 5, and step 1 ends by **death**. Compute A₁ and A₀.
2. Same numbers, but step 1 ends by **time limit**, and the value of the true final observation is 4. Compute A₁ and A₀. Why are the two answers different even though the rewards are the same?
3. Why does the trainer compute the bootstrap value for a fresh time-out with the hidden state that just processed step t and `episode_start = 0`, and not with a fresh (zero) hidden state?

---

## Answers

1. Death: next value is 0 and the chain is cut. A₁ = 1 + 0 - 3 = **-2**. Step 0: δ₀ = 0 + 0.9 x 3 - 2 = 0.7; A₀ = 0.7 + 0.72 x (-2) = 0.7 - 1.44 = **-0.74**. (Worked by hand.)
2. Time limit: A₁ = 1 + 0.9 x 4 - 3 = **1.6**; A₀ = 0.7 + 0.72 x 1.6 = 0.7 + 1.152 = **1.852**. (Worked by hand.) After a death nothing follows, so the value of 3 the agent expected was lost. After a time-out the game would have continued, worth about 4 by the network's own guess, so the step looks good. Treating both the same would punish the agent for running out of time.
3. The final observation is the next situation in the **same** episode, so the network should judge it with the memory it actually had at that point, exactly as it would have if the episode had continued. A zero hidden state would judge it as if the agent had just spawned with no memory, giving a different (wrong) value. Resetting is only for the first observation of a *new* episode (`trainer.py:138-144`).
