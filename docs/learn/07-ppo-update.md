# Lesson 07: The PPO update (clipped objective, value loss, entropy, auxiliary heads, the toy memory test)

> **Changed since first written (2026-11-25, after ADR 0007).** Factual corrections only.
> - **Resolved: the toy memory failure.** Re-run on three seeds after `0007-a0-trainer-rulings`, the recurrent agent scored 0.605, 0.485 and 0.495; the earlier 0.985 was **a single seed that did not hold up**. The `debugger` found **no trainer bug**. ADR 0008 replaced the test with Gate A (passed 8 of 8) and Measurement B. **Lesson 09 explains why it failed and re-teaches the parts of the loss that matter for memory.** This lesson's sections 3.1 to 3.4 still describe `train/ppo.py` correctly; that file did not change.
> - The monster-count target is now min(count, 8) / 8 (was count / 24).
> - The value-scale floor is now **0.1** (ADR 0008; it was 0.01, then 1.0 under `0007-a0-trainer-rulings`).
> - The learning rate is now constant (was decay to zero).
> - Memory use is now logged by the run loop (lesson 08).
> - **Two ADRs are numbered 0007, and ADR 0008 reconciles them**: constant learning rate, 3 epochs (2 epochs is a measured arm), floor 0.1, initialisation as built (orthogonal is a measured arm). See lesson 09.

Files taught: `train/ppo.py` (all of it), `train/trainer.py:185-216` (the update step), `train/aux_labels.py` (the auxiliary labels), `configs/train_config.py`, `train/toy_memory.py`, and the tests `tests/test_train_ppo.py:32-50`, `tests/test_train_aux_and_mask.py:14-67` and `tests/test_toy_memory.py`. Result: `reports/runs/2026-11-25-a0-toy-memory.md`.

Design records: ADR 0004 (PPO settings) and ADR 0003 (auxiliary heads). ADR 0007 amends both. **All seven ADRs are "Proposed", not ratified.** Nothing has been trained on DOOM yet. The only learning results so far are on the toy task (section 4 and lesson 09).

Builds on lesson 05 (log-probabilities, entropy, the six heads, `forward_sequence`) and lesson 06 (the rollout, advantages, value targets, `valid`).

---

## 1. What problem this piece solves

After a rollout we know, for each decision, how much better or worse the chosen action turned out than expected (the advantage, lesson 06). Now we must **change the network's weights** so that good actions become more likely and bad ones less likely, and so that the value guesses get more accurate.

The danger is changing too much at once. One unlucky rollout can contain a few huge advantages. If the network jumps to "always do that", it may destroy behaviour that took hours to learn, and because it then collects its own new experience from the broken behaviour, it may never recover. PPO (proximal policy optimization) is a way to improve the policy **in small, controlled steps** using the same rollout several times.

## 2. The idea in plain words

Think of a **driving instructor adjusting a learner's habits after each lesson**. "That lane change went well, do it a bit more often. That late braking went badly, do it a bit less." But the instructor has a rule: **no habit may change by more than about 20% in one session**, however good or bad one example looked. Big changes wait for more evidence from future lessons.

Alongside, the instructor also trains the learner's **sense of how things are going** (the value), keeps the learner **willing to try different things** (the entropy bonus), and asks side questions ("which keys are you carrying?") that are not graded on driving but make the learner pay attention (the auxiliary losses).

Where the analogy stops being accurate:

- The 20% limit is on the *probability ratio* of each chosen action, not on a habit as a whole, and it only stops the push past the limit; it does not undo changes.
- The "side questions" have known true answers during training only (from `info`). They are never asked during play.

## 3. A walk through our actual code

### 3.1 Terms first

- **Loss**: the number training tries to make smaller. Here it is a weighted sum of several parts.
- **Gradient**: for each weight, the direction and size of change that would increase the loss fastest. Training moves weights against it.
- **Optimizer (Adam)**: the rule that turns gradients into weight changes. ADR 0004 uses Adam with learning rate 3e-4 (`train_config.py:18`; trainer `trainer.py:68`).
- **Learning rate**: how big each weight change is.
- **Epoch**: one pass over the stored rollout. ADR 0004 uses 3 (`train_config.py:16`).
- **Minibatch**: one part of the rollout used for one weight update. Here, 4 minibatches of whole environment sequences (`train_config.py:15`).
- **Probability ratio**: for a stored action, (its probability under the network now) divided by (its probability when it was taken). It is 1 before any update. Computed as exp(new log-probability - old log-probability).
- **Binary cross-entropy (BCE)**: the standard loss for yes/no predictions made as logits; small when the predicted probability is near the true 0 or 1.
- **Mean squared error (MSE)**: the average of (prediction - target)², the standard loss for predicting a number.
- **KL divergence**: a measure of how different two probability distributions are; 0 when identical.

### 3.2 The update loop: `ppo.py:83-110` and `trainer.py:185-199`

`Trainer.update` (`trainer.py:185-199`) computes advantages and value targets (lesson 06), updates and applies the return scale, sets the learning rate, and calls `ppo_update`.

`ppo_update` (`ppo.py:83-110`):

- `:86-88`: split the N environments into `num_minibatches` groups (32 environments / 4 = 8 per group). N must divide evenly.
- `:92-96`: for each of 3 epochs, shuffle the environment order with a seeded generator, and for each group evaluate the loss.
- `:100-103`: compute gradients, **clip the gradient norm** to 1.0 (if the overall size of the gradient is larger than 1, scale it down to 1, which stops one wild minibatch from making a huge step), and take one Adam step.
- `:97-99`: on the very first minibatch of the first epoch, record the largest |ratio - 1|. No weights have changed yet, so it must be about 0. If it is not, collection and recomputation disagree (for example a different reset rule, lesson 05). ADR 0004 asked for this check. `tests/test_train_ppo.py:32-50` checks it is below 1e-5, with and without telemetry masking.
- `:105-110`: average every statistic over all minibatches and return them for logging.

**Why minibatches are whole sequences** (ADR 0004: "never shuffled single steps"). The memory makes each decision depend on all earlier ones. To recompute what the network outputs at step 40, it must replay steps 0 to 39 from the stored starting state `h0`, with the stored resets. So a minibatch is all 64 steps of 8 environments, recomputed by `forward_sequence` (`ppo.py:33-36`). This is called **backpropagation through time**: the gradient flows back through the GRU steps too, which is how the memory learns what to keep.

### 3.3 One minibatch's loss: `ppo.py:31-80`

`masked_mean(x, m)` (`:18-20`) averages only where the mask is true. It is used everywhere so that dropped (stale) steps and invalid labels contribute nothing.

1. **Recompute** (`:33-40`): select the minibatch's environments, re-apply telemetry masking, run `forward_sequence` from the stored `h0` with the stored `episode_start`, build the six-head distribution, and get the new log-probability of each stored action.
2. **Normalize advantages** (`:41-46`): subtract the minibatch's mean advantage and divide by its spread, over valid steps only. This keeps the step size similar whatever the reward scale.
3. **Clipped policy objective** (`:48-51`):

   ```
   ratio = exp(logp_new - logp_old)
   pg    = -min(ratio * A, clip(ratio, 0.8, 1.2) * A)
   ```

   - If A > 0 (the action was better than expected), raising its probability helps, but once the ratio passes 1.2, the clipped term stops growing, and `min` takes it. **No further reward for pushing it higher** in this update.
   - If A < 0 (worse than expected), lowering its probability helps, but below 0.8 the clipped term stops changing. **No further reward for pushing it lower.**
   - The minus sign turns "make the objective large" into "make the loss small".
   - Clip 0.2 is ADR 0004's value (`train_config.py:17`).
4. **Value loss** (`:52-53`): 0.5 x (value - target)² over valid steps, where the targets are the scaled returns from lesson 06. Weight 0.5 in the total (`train_config.py:27`).
5. **Entropy bonus** (`:54-55`): the summed entropy of the six heads (lesson 05), averaged over valid steps. It is **subtracted** from the loss with weight 0.01 (`train_config.py:28`), so higher entropy (more spread-out choices) lowers the loss. This keeps the agent trying different actions instead of settling early. Per-head entropy is also logged (`:78-79`), because one head (say `use`) can collapse while the others still explore.
6. **Auxiliary losses** (`:57-63`), each over the steps where its label is valid:
   - keys held: BCE on the three key logits, against `info["keys_held"]` (`aux_labels.py:16-26`);
   - damage soon: BCE on the damage logit, against "any damage in this decision or the next k-1", with k = 8 (`aux_labels.py:71-93`). For an episode that ended, the last event now comes from the final `info`, so a death counts as damage (lesson 06);
   - visible monsters: MSE against min(number of visible monsters, 8) / 8, from the **true** entity set (`aux_labels.py:29-36`). ADR 0007 changed this from count / 24, because about 1.2 labels were visible per state in the benchmark, so count / 24 was near 0.05 and gave almost no learning signal.
7. **Total** (`:65-67`):

   ```
   loss = pg_loss + 0.5 * v_loss - 0.01 * entropy + 0.1 * keys + 0.1 * damage + 0.1 * monsters
   ```

8. **Diagnostics** (`:68-77`): `approx_kl`, an estimate of how far the policy moved in this minibatch; `clipfrac`, the share of steps whose ratio left [0.8, 1.2]; and `valid_frac`, the share of steps not dropped.

**Values the worker marked PROVISIONAL** when this lesson was first written, and what ADR 0007 ruled on them (`configs/train_config.py`, `configs/model_config.py`):

- the auxiliary loss weights, 0.1 each: **accepted** as starting values (keys weight marked for an experiment);
- the damage window, k = 8, binary "any damage": **accepted**, with deaths now counted as damage;
- the monster target: **changed** to min(count, 8) / 8;
- return scaling by the running standard deviation of returns: floor changed to 1.0 by `0007-a0-trainer-rulings`, then to **0.1** by ADR 0008 (lessons 06 and 09);
- the learning-rate schedule: **changed** to constant 3e-4;
- Adam's ε (1e-5): **accepted**;
- the initialisation sizes (policy-head gain 0.01, learned-token spread 0.02): **accepted**.

**Label validity.** The damage label is marked invalid when the window runs past the rollout before the episode ends, or touches a step whose damage is unknown, for example a time-out whose final observation is stale (`aux_labels.py:44-68`, `:71-93`; tests `tests/test_train_aux_and_mask.py:41-67`). The keys label is valid only when `info` has a `keys_held` entry (`:14-17`). Invalid labels are left out of the loss rather than guessed.

The auxiliary outputs are **losses only**. Nothing reads them to choose actions (lesson 05).

### 3.4 What gets logged: `trainer.py:185-216`

Each update returns: all losses, `entropy` and per-head entropies, `approx_kl`, `clipfrac`, `grad_norm`, `valid_frac`, `first_minibatch_ratio_max_dev`, `explained_variance`, `return_scale`, the learning rate, and the decisions per second for collection (`sample_dps`) and for the update (`update_dps`) separately. Episode returns and lengths are kept in `trainer.log`. The trainer now also reports `stale_timeouts` (which must be 0, lesson 06) and the update index. Memory use of the whole process tree is now logged by the run loop, together with these stats, to `metrics.jsonl` (lesson 08).

## 4. A tiny worked example with real numbers

### 4.1 The clip, one sample at a time

I evaluated the clipped objective, min(ratio x A, clip(ratio, 0.8, 1.2) x A), for a single step with advantage A = +1 and A = -1. Real output:

```
PPO A=+1 ratio=0.7: unclipped +0.70 clipped +0.80 -> min +0.70
PPO A=+1 ratio=0.9: unclipped +0.90 clipped +0.90 -> min +0.90
PPO A=+1 ratio=1.0: unclipped +1.00 clipped +1.00 -> min +1.00
PPO A=+1 ratio=1.1: unclipped +1.10 clipped +1.10 -> min +1.10
PPO A=+1 ratio=1.3: unclipped +1.30 clipped +1.20 -> min +1.20
PPO A=-1 ratio=0.7: unclipped -0.70 clipped -0.80 -> min -0.80
PPO A=-1 ratio=0.9: unclipped -0.90 clipped -0.90 -> min -0.90
PPO A=-1 ratio=1.0: unclipped -1.00 clipped -1.00 -> min -1.00
PPO A=-1 ratio=1.1: unclipped -1.10 clipped -1.10 -> min -1.10
PPO A=-1 ratio=1.3: unclipped -1.30 clipped -1.20 -> min -1.30
```

Reading it (the update tries to make these numbers **large**):

- **A = +1, ratio 1.3**: the good action is already 30% more likely than when it was taken. The objective is capped at 1.20, so it is flat there: the gradient no longer pushes the probability up.
- **A = +1, ratio 0.7**: the good action became *less* likely. The `min` picks the unclipped 0.70, so the gradient still pushes it back up. The clip only stops over-shooting, never correction.
- **A = -1, ratio 0.7**: the bad action is already 30% less likely; the objective is held at -0.80 and stops rewarding further decrease.
- **A = -1, ratio 1.3**: the bad action became *more* likely; `min` picks -1.30, so the gradient pushes it down.

### 4.2 The toy memory test (the worker's results, not my runs)

I did not rerun it. The first numbers below are from `reports/runs/2026-11-25-a0-toy-memory.md` (one seed, before ADR 0007). The three-seed rerun is in `reports/runs/2026-11-25-a0-toy-three-seeds.md` and is shown further down. **Lesson 09 has the investigation and the new results.**

**The task** (`train/toy_memory.py:41-102`). A tiny pure-numpy environment with the same observation keys as DOOM:

- At step 0, one of two cues is shown in `self` (position 0 or 1).
- From step 1 on, nothing shows which cue it was. The rays and entities are random noise, so the set encoder and masks are still exercised.
- At step 20, an "answer now" flag appears, and the first action head must name the cue: reward 1 if right, 0 if wrong. The episode then ends (`terminated`).
- A network without memory cannot know the cue at step 20, so its best possible success rate is **0.5**.
- Episodes last 21 decisions and rollouts 32, so episodes cross rollout boundaries. That tests that the memory is carried across rollouts and reset at the right place.

**The setup**: 16 environments, 100 updates (about 51,000 decisions), the A0 network at reduced widths (memory 64, fusion 64), learning rate **1e-3 constant**, γ 0.99, other settings as ADR 0004. The "memoryless" run is the same network with `recurrent=False`, so its hidden state is zeroed every step (lesson 05).

| Run | Success (last 200 episodes) | Largest abs(ratio - 1) on the first minibatch |
|---|---|---|
| Recurrent | **0.985** | 7e-7 |
| Memoryless | **0.505** | 5e-7 |

In that first run the recurrent agent recalled the cue almost always, and the memoryless one was at chance. Both ratio checks were about 0, so collection and recomputation agreed. Wall time for both runs: 72 s. The test at the time (`tests/test_toy_memory.py`, since rewritten as Gate A; lesson 09) required recurrent ≥ 0.9 and memoryless ≤ 0.65.

**The three-seed rerun FAILED** (after ADR 0007's changes: value-scale floor 1.0, monster target min(count, 8)/8, damage label from the final `info`). Gate: every recurrent run ≥ 0.9, every memoryless run within 0.1 of 0.5. From the report:

| Seed | Recurrent, 1e-3, γ 0.99, 100 updates | Memoryless, same | Recurrent, 3e-4, γ 0.997, constant, 350 updates |
|---|---|---|---|
| 0 | 0.605 | 0.505 | 0.775 |
| 1 | 0.485 | 0.49 | 0.57 |
| 2 | 0.495 | 0.495 | 0.51 |

- **No recurrent run reached 0.9**, so the gate fails on every seed. The memoryless runs are at chance, as expected.
- The first-minibatch ratio check stayed at most 5e-7 in every run, so the recomputation is still consistent with collection.
- The worker's ablation: putting the value-scale floor back to 0.01 gave 0.87 (seed 0) and 0.745 (seed 1).
- With the old floor *and* the old monster target, seed 0 gave exactly 0.985 again. So on seed 0 the two ADR 0007 value changes explain the drop, and the damage-label change has no effect on this task.
- **Seed 1 fails even with the old floor** (0.745), so the earlier single-seed pass "was read too confidently" (the worker's words).
- The worker offers hypotheses, explicitly not established: on this task the value loss itself trains the memory, and the 1.0 floor weakens that signal; learning is slow and very sensitive to the seed.

That block is now resolved. The `debugger` found no trainer bug: a weakened value-loss signal (the 1.0 floor) and a short budget explain the failure. ADR 0008 replaced the test with **Gate A** (gap 4, 8 seeds; passed 8 of 8) and **Measurement B** (gap 20 now 8 of 8 on every arm at 300 updates; gap 60 fails, 0 of 8). See **lesson 09**.

**Earlier diagnostics at ADR 0004's own settings** (recurrent only, seed 0, from the first report, before ADR 0007):

| Learning rate | γ | Updates | Schedule | Success |
|---|---|---|---|---|
| 3e-4 | 0.997 | 100 | decay to 0 | 0.51 |
| 3e-4 | 0.997 | 300 | decay to 0 | 0.475 |
| 3e-4 | 0.99 | 150 | constant | 0.455 |
| 1e-3 | 0.997 | 100 | constant | 0.83 |
| 3e-4 | 0.997 | 350 | constant | 0.995 |

At ADR 0004's learning rate of 3e-4 with linear decay to zero, the task stayed at chance even after 300 updates. With a constant 3e-4 it was solved, but only after about 350 updates. The worker reads this as sensitivity to the learning rate and the training budget, not a trainer bug. The effect of γ is unclear.

**What this does and does not show:**

- The first run showed the recurrent machinery **could** learn to use memory on one seed. The three-seed rerun shows that it does not do so **reliably** with the current settings. The investigation (lesson 09) found the cause was a setting and the budget, not a bug.
- **Seeds.** The first result was one seed, and it did not hold up on three. Never trust a single-seed pass again.
- **It never truncates** (every episode ends by `terminated`), so the time-limit and stale-observation paths of lesson 06 are tested only by the unit tests, not by learning.
- **It has no key labels** (its `info` is empty, so the keys loss is always masked out), and its damage slot is always 0. The auxiliary heads get little real signal here.
- **It says nothing about DOOM.** The report states this too.
- The decay-to-zero schedule has since been replaced by a constant learning rate (ADR 0007 ruling 1).

## 5. Why we chose this design over the alternatives

From ADR 0004 (**Proposed, not ratified**):

- **Our own PPO**, not a library's (option A rejected): the brief says an off-the-shelf PPO script scores poorly, and the rules require the training pipeline to be our own work.
- **Synchronous** (option B) over asynchronous with V-trace (option C): no lag between the acting and the learning policy, so no correction for it is needed, and it is simpler to build and check. V-trace (a correction for the lag between the acting and the learning policy, lesson 06) only matters if sampling is asynchronous.
- **Starting values**: 3 epochs, clip 0.2, Adam 3e-4 with linear decay, gradient-norm clip 1.0, value weight 0.5, entropy weight 0.01 on the summed entropy of the six heads ("reviewed per head once runs exist"), advantage normalization per minibatch, whole-sequence minibatches. ADR 0004 says these come from papers that used PPO; its research agent could not open the PPO paper itself.
- **Auxiliary heads** (ADR 0003): keys held, damage in the next k steps, and number of visible monsters, to make the memory hold useful facts. ADR 0003 cites MERLIN: memory trained only through the reward fails over long delays; a predictive extra loss fixes it.
- **The toy memory task** is ADR 0004's own validation item 2(i). Its rule: "the toy memory task cannot be solved: block all further training until fixed". It passed once on one seed, at a higher learning rate than the ADR's, and **failed on three seeds** after `0007-a0-trainer-rulings`' changes. ADR 0008 then split it into Gate A (passed) and Measurement B (lesson 09).

## 6. How it connects

- **Lessons 04 and 05** built the network this update trains; **lesson 06** built the rollout and advantages it uses.
- **Lesson 08** covers what was built since: the multi-process environment workers, the DOOM training entry point, checkpointing and resuming, and the laptop-protection features. **Lesson 09** covers the toy investigation. **Still to come**: the throughput benchmark, the first real A0 run on `my_way_home`, then the pixel branch and modality dropout (Stage A1), reward shaping, the exploration signal, and the curriculum.

## 7. Common mistakes, and how we would notice them

| Mistake | How it would show up in the logs |
|---|---|
| Collection and recomputation disagree (different resets, masking, or a changed model between them). | `first_minibatch_ratio_max_dev` is clearly above 0 (it should be about 1e-6). |
| Learning rate too high. | `approx_kl` and `clipfrac` jump (for example most steps clipped), entropy collapses fast, returns become erratic. |
| Learning rate too low. | Returns flat for a long time, `approx_kl` near 0. ADR 0007's guidance for DOOM: KL persistently below about 0.003 with no learning argues for doubling the rate; above about 0.03, for halving it. |
| Entropy weight too small. | Per-head entropy falls quickly toward 0 on some head (for example `use` always 0) before any reward is found; the agent stops exploring. |
| Auxiliary weights too large. | The auxiliary losses fall nicely while the policy does not improve; the memory serves the side questions instead of the game. |
| Value targets on a wrong scale. | `v_loss` very large, `explained_variance` near 0 or negative for a long time. Check `return_scale`. |
| Treating a single-seed toy result as evidence. | Exactly what happened here: 0.985 on one seed, then 0.605 / 0.485 / 0.495 on three. Always report several seeds. |

## 8. Check your understanding

1. A stored action had probability 0.20 when it was taken and has 0.26 now; its normalized advantage is +2. What is the ratio, what are the unclipped and clipped terms, and does the policy loss still push this action's probability up?
2. Why is the entropy term *subtracted* in the total loss, and what would you expect to see in the logs if its weight were set to 0?
3. In the first toy run, the memoryless agent scored 0.505 and the recurrent one 0.985. Why is about 0.5 the best a memoryless agent can do, and why does that make this a good test of the trainer?

---

## Answers

1. Ratio = 0.26 / 0.20 = **1.3**. Unclipped term: 1.3 x 2 = 2.6. Clipped term: clip(1.3, 0.8, 1.2) x 2 = 1.2 x 2 = **2.4**. The `min` is 2.4, the clipped one, which does not change when the ratio changes, so **no**: this step no longer pushes the probability up in this update. (Worked by hand.)
2. The loss is minimized, and we want entropy to be *high* (choices kept spread out), so it enters with a minus sign: more entropy, lower loss. With weight 0, nothing resists the policy becoming certain early. In the logs, the per-head entropies (`entropy_head0` to `entropy_head5`) would drop toward 0 quickly, and the agent would repeat a few actions before it has found any reward.
3. At step 20 the memoryless agent's input contains no trace of the cue (only the "answer now" flag and noise), so whatever it answers is right half the time on average: 0.5. Any success clearly above 0.5 must come from carrying information across 20 steps, so it can only be achieved if the memory, its resets, and the sequence recomputation in the update all work. That is exactly the part of a recurrent trainer that "fails silently" (ADR 0004's risk list).
