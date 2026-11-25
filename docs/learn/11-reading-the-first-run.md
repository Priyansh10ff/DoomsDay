# Lesson 11: Reading the first real DOOM run, and its verdict

Files taught (what each logged number means): `train/trainer.py` (`episode_record` `:66-76`, episode bookkeeping `:171-181`, `update` `:211-225`, speed `:236-237`), `train/ppo.py` (`:44-79`, `:92-110`), `train/running_stats.py`, `train/run_loop.py:35`, `:48-56`, `:188-205`, `configs/train_config.py:14-35`.

Evidence:

- the run report `reports/runs/2026-11-25-first-doom-run-a0-4x8.md` (called "the report" below);
- the random baseline `reports/runs/2026-11-25-random-baseline.json` (its `summary` block);
- the run folder `reports/runs/20261125-nogit-7a675759/` (`metrics.jsonl`, `episodes.jsonl`, `stdout.log`).

Decision record: `docs/decisions/0009-toy-results-memory-plan-and-first-doom-run.md` (ADR 0009), Decision 5 ("What counts as learning") and Decision 3 (the toy diagnostic D0 to D4). **ADR 0009 is still "Proposed", not ratified by the user.**

**Where the numbers come from.** Every number here is either quoted from the report or the baseline file, or **recomputed by me from the run's own log files** with a small read-only script (no engine, no training). Recomputed numbers are marked *(recomputed)*. Numbers worked out with pencil arithmetic are marked *(by hand)*. Nothing is made up.

Builds on lesson 06 (returns, values, explained variance, return scale), lesson 07 (PPO, KL, clipping, entropy bonus) and lesson 10 (the run's launch, its files, and the verdict rules fixed in advance).

---

## 1. What problem this piece solves

The first real training run has finished. It ran for 2 h 20 min on `my_way_home`: one maze-like map, no monsters, a goal somewhere in it, and a random start point each episode. It produced 10,060 episodes and 1,814 network updates.

Now someone asks: **"Did the agent learn to play DOOM?"** There are three ways to get this wrong:

- **Praise too early.** The agent reaches the goal 73% of the time at the end. That sounds good. But a player pressing random buttons also reaches it sometimes. How much of the 73% is skill?
- **Miss a warning sign.** Half-way through, the agent got *worse*, nearly back to random, then recovered by itself. If we only look at the final number, we never see that, and the next run might not recover.
- **Claim too much.** Reaching one goal on one map we trained on says nothing about new maps, fighting monsters or using keys. Those are what the judges test.

This lesson teaches how to read the run's files so that none of these happens, and what the honest verdict is.

## 2. The idea in plain words

Think of **a new driver's first month of lessons, always on the same practice course**.

- **The baseline.** Before praising them, ask: how often would someone who steers at random finish the course? On a short, forgiving course, quite often. Skill is only the part *above* that.
- **Averaging over many attempts.** One lap tells you little; luck dominates. You judge on the average of many laps. That is what the 500-episode windows do.
- **The driver's "health signs".** Besides lap results, the instructor watches *how* they drive: still trying things, or doing the same thing every time? Making small adjustments or wild ones? These are entropy, KL and clip fraction.
- **The bad week.** In week two the driver got worse, then better again. The instructor has guesses why, but did not check any of them. Saying "I know why" would be dishonest.
- **What it does not show.** Driving well on the practice course says nothing about driving in an unfamiliar city, at night, in traffic.

Where the analogy stops being accurate:

- A human driver understands the goal. Our agent only has a number: +1 at the goal, a tiny minus per moment. It has no idea it is "driving to a place".
- A human learns between lessons. Our agent learns only when the trainer updates its weights, every 2,048 decisions.
- A driver who does well on one course has probably learned skills that transfer. For our agent this is exactly what is **not** known: it may have learned "this maze", not "how to explore a maze".

### Terms used in this lesson

New terms, defined here and added to `GLOSSARY.md`:

- **Success rate**: the share of episodes that ended at the goal.
- **Time-out rate**: the share of episodes that ended because the time limit ran out.
- **Window (episode window)**: a block of consecutive finished episodes (here 500) whose results are averaged together.
- **Learning curve**: how a result such as success rate changes over the course of training.
- **Standard error**: how much an average would typically change if you repeated the measurement with different random luck; for a success rate p over n episodes it is about √(p(1-p)/n).
- **Plateau**: a stretch of training where the curve stays roughly flat.
- **Monotonic**: always going the same way (here, never dropping).
- **Deterministic policy**: a policy that always picks the same action in the same situation; its entropy is 0.
- **Entropy collapse**: the policy's entropy falling close to 0, so it stops trying different actions.
- **Clip fraction**: the share of stored steps whose probability ratio moved outside PPO's allowed range (here 0.8 to 1.2) during an update.
- **Return scale**: the number the value targets are divided by, the running standard deviation of all returns seen so far, never below 0.1.
- **Gradient norm**: the overall size of the gradient, all weights together; logged before it is clipped to 1.0.
- **Held-out map**: a map kept out of training on purpose, used only to test whether the agent works somewhere new.

## 3. A walk through our actual code: what each logged number means

You cannot read a log correctly unless you know how each number was produced. Here is where each comes from.

### 3.1 `episodes.jsonl`: one line per finished episode

- **Built by** `episode_record` (`trainer.py:66-76`). Fields as in lesson 10. `success` is set equal to `level_exit` (`:75`): the episode ended at the goal.
- **When it is built** (`trainer.py:173-181`): inside the collection loop, at the step where an environment's episode ended. Its `return` is the plain sum of that episode's rewards (`:171`), **not** discounted. Its `length` is the number of decisions.
- **The `decision` stamp** (`:174`) is `decisions_done + (t + 1) * n`: total decisions taken by all 32 engines so far, at the step this episode ended.
- **The `update_idx` stamp** is the trainer's update counter *before* this rollout's update. The counter goes up only at the end of `update` (`:223`). So an episode stamped `update_idx` 0 was collected for the first update, which `metrics.jsonl` calls `update_idx` 1. **The two files are one apart.** I align them this way in section 4.
- **Written to disk** by `drain_episodes` (`run_loop.py:38-45`) after every update (`:188`), and once more at the very end (`:212`).

### 3.2 `metrics.jsonl`: one line per update

Each line is the `stats` dictionary from `Trainer.update` and `train_iteration`, plus memory and recent-episode figures added in `run_loop.py:190-195`.

**First, an important detail.** One update is 3 epochs x 4 minibatches = **12 small weight steps** (`ppo.py:92-107`; `train_config.py:15-16`). Every policy statistic below is computed on each of the 12 and then **averaged** (`ppo.py:105-108`). So `approx_kl` is "the average KL across the 12 steps of this update", not "how far the policy moved by the end of the update".

**`entropy` and `entropy_head0` to `entropy_head5`** (`ppo.py:54-55`, `:78-79`).

- Lesson 05 defined entropy: 0 when one option is certain, largest when all options are equally likely.
- Each head's entropy is computed separately. `entropy` is their **sum**.
- The maximum for a head with k options is ln k:
  - move (head 0, 3 options): ln 3 ≈ 1.10;
  - strafe (head 1, 3 options): 1.10;
  - turn (head 2, 7 options): ln 7 ≈ 1.95;
  - attack, use, speed (heads 3-5, 2 options each): ln 2 ≈ 0.69 each.
  - Sum: 1.10 + 1.10 + 1.95 + 3 x 0.69 ≈ 6.22 *(by hand)*. The very first update logged 6.218 *(recomputed)*: at the start the network is almost perfectly undecided.
- Why entropy appears in the loss at all (`ppo.py:65`): the loss **subtracts** `entropy_coef x entropy`, with `entropy_coef` = 0.01 (`train_config.py:28`). Subtracting it means training gets a small reward for staying undecided. This is the "entropy bonus" of lesson 07. It is the only thing in the loss that pushes back against the policy becoming certain too soon.

**`approx_kl`** (`ppo.py:69`).

- Lesson 07: KL divergence measures how different two probability distributions are; 0 means identical.
- Here it estimates how far the policy moved from the version that collected the rollout, as `(ratio - 1) - log(ratio)`, averaged over valid steps.
- Too small (near 0) for a long time: the policy is barely changing, so it is not learning.
- Too large: updates are big jumps, which tends to make training unstable.
- ADR 0009 calls KL "mostly between 0.003 and 0.03" healthy.

**`clipfrac`** (`ppo.py:70`).

- Lesson 07: PPO clips the probability ratio to [1 - 0.2, 1 + 0.2] (`train_config.py:17`).
- `clipfrac` is the share of steps where the ratio has already left that range: steps where PPO is refusing to push further.
- A low clip fraction means the updates are gentle. ADR 0009's alarm is above 0.5 for 20 updates in a row.

**`grad_norm`** (`ppo.py:102`).

- `clip_grad_norm_` returns the size of the gradient **before** it scales it down to 1.0 (`train_config.py:26`; lesson 07).
- So a logged value of 1.70 is possible. It means that step was scaled down.

**`return_scale`** (`trainer.py:213-214`, `:221`; `running_stats.py`).

- Lessons 06 and 09: value targets are divided by the return scale before the value head learns them. That keeps the value head working on numbers of a similar size whatever the reward size is.
- **How it is computed** (`running_stats.py:18-36`): the standard deviation of **every return seen since the start of the run**, merged batch by batch. It never forgets, so after many updates it moves only slowly.
- It is floored at 0.1 (`train_config.py:35`). So small returns are never amplified more than 10 times.
- It does **not** directly scale the policy's step. Advantages are re-normalised inside every minibatch (`ppo.py:44-46`): their average is subtracted and they are divided by their own spread. The return scale reaches the policy only indirectly, through how well the value head learns.

**`explained_variance`** (`trainer.py:217-220`).

- `1 - var(returns - values) / var(returns)`, over the valid steps of this rollout.
- The values are the ones **stored when the rollout was collected** (`trainer.py:145`), before this update changed anything.
- Lesson 06: 1 is perfect, 0 is no better than always guessing the average, negative is worse than that.

**`sample_dps`, `update_dps`** (`trainer.py:236-237`): decisions per second while collecting and while updating, as in lesson 10.

**`recent_success`, `recent_return`, `recent_length`** (`run_loop.py:35`, `:48-56`): averages over only the **last 20** episodes. Lesson 10 explained why these are too noisy to judge by. The verdict uses `episodes.jsonl` instead.

**`tree_rss_bytes`, `available_bytes`** (`run_loop.py:190-193`): the memory of the whole process tree, and the system's free memory, read once per update.

### 3.3 How the run stopped

`run_loop.py:204-205`: once 8,400 s had passed, the loop asked itself to stop. It finished the current update, wrote a checkpoint, and stopped. `stdout.log` ends with "stopped (max_seconds) 0.03s after the request" and "engine processes alive after shutdown: 0".

## 4. The run, number by number

### 4.1 The random-policy baseline: why we need it, and what ours is

**Why.** On `my_way_home` the start point is random, and a random button-presser sometimes stumbles into the goal. In the baseline file, random successes include episodes of 14, 21, 30 and 34 decisions, which suggests some start points are close to the goal (start-point positions are not logged, so this is an inference). If we did not measure this, any success rate above zero might look like learning.

**How it was measured.** Uniform random actions (`env.action_space.sample()`), the same TRAIN config, one engine, 200 episodes, 96,034 decisions, 451.5 s.

| Baseline metric | Value | Name in ADR 0009 |
|---|---|---|
| Success | 0.140 | p0 |
| Mean return | -0.052 (std 0.393, standard error 0.028) | R0 |
| Mean length | 480.2 decisions (median 525, the time limit) | L0 |
| Time-out rate | 0.860 | |

**Measured after the run, not before.** ADR 0009 Decision 5 lists the baseline under "Before launch". It was in fact measured after the run had started. The report says so ("Order deviation"). The comparison is still valid, because it used the same config. But the point of measuring first is that the bar is fixed before anyone sees the results. Here the bar was fixed in ADR 0009 as "p0 + 0.15", and p0 was then measured honestly. That is the next-best thing, not the same thing.

### 4.2 What one episode's return means *(by hand, from the scenario's rules)*

The scenario's reward is -0.0001 per tic and +1 at the goal. One decision is 4 tics, so -0.0004 per decision.

- **A time-out** (525 decisions, no goal): 525 x -0.0004 = **-0.21**. The baseline file logs -0.2100 for every time-out.
- **A success after L decisions**: about 1 - 0.0004 x L. For example, the baseline episode of 309 decisions: 1 - 0.1236 = 0.876; the file logs 0.8767.

So the mean return mixes two things: **how often** the agent succeeds, and **how fast**. Worked through for the last 500 episodes, with p = 0.726 successes whose mean length is 181.8 decisions *(recomputed)*:

- successes: 0.726 x (1 - 0.0004 x 181.8) = 0.726 x 0.927 ≈ 0.673;
- time-outs: 0.274 x (-0.21) ≈ -0.058;
- total ≈ **0.616**. That matches the report's last-500 mean return of +0.616.

This is why ADR 0009 asks for both success and return to beat random. Return also rewards getting there faster.

### 4.3 Why 500-episode windows

From lesson 10, the standard error of a success rate p over n episodes is about √(p(1-p)/n).

- Last 500 episodes, p = 0.726: √(0.726 x 0.274 / 500) ≈ **0.020** *(by hand; the report says 0.020)*.
- Baseline, 200 episodes, p = 0.14: √(0.14 x 0.86 / 200) ≈ **0.025** *(by hand; the report says "about 0.025")*.
- A 20-episode `recent_success` around p = 0.6: √(0.6 x 0.4 / 20) ≈ 0.11 *(by hand)*. Too noisy to read.

A difference of +0.586 between two numbers that each wobble by about 0.02 to 0.025 is not luck.

**Two reading traps** *(by hand, from how the files are built)*:

- **The first episodes to finish lean towards successes.** A time-out cannot end before decision 525 of its engine. A success can end at decision 14. So the very first finished episodes lean towards the short, lucky ones. This is the likely reason the untrained network's first 100 episodes show success 0.22, above the random 0.14 (not tested separately). The first full window of 500 (0.168) is closer to random.
- **Per-engine numbers are small samples.** The report says success over the last 2,000 episodes ranges from 0.47 to 0.77 across the 32 engines. Each engine finished only 51 to 74 of those episodes *(recomputed)*. At p ≈ 0.6 and n ≈ 62, the standard error is about 0.06 *(by hand)*, so a spread of roughly ±0.12 is what chance alone gives. This range does **not** show that some start points are harder. It might be true, but these numbers cannot show it.

### 4.4 The curve, in 500-episode windows (from the report)

| Episodes | Decisions at end | Success | Mean return | Mean length | Time-out |
|---|---|---|---|---|---|
| 0-500 | 239,776 | 0.168 | -0.018 | 466.2 | 0.832 |
| 1500-2000 | 841,856 | 0.410 | +0.255 | 386.4 | 0.590 |
| 3000-3500 | 1,434,176 | 0.368 | +0.209 | 396.8 | 0.632 |
| 4500-5000 | 2,075,200 | 0.176 | -0.002 | 445.6 | 0.824 |
| 5500-6000 | 2,427,808 | 0.654 | +0.530 | 311.4 | 0.346 |
| 9500-10000 | 3,698,464 | 0.718 | +0.606 | 279.5 | 0.282 |
| last 500 (9560-10060) | | 0.726 | +0.616 | | |

(Selected rows; the full 20-row table is in the report.)

How to read it:

- **Success and time-out add up to 1** in every row, because the scenario has no monsters: every failure is a time-out at 525 decisions. Deaths were 0 *(recomputed: no episode is `dead`, and every failure has length 525)*.
- **Mean length falls as success rises**, because successes end early and time-outs always last 525. That is not extra information, just the same fact seen again.
- **Count by decisions, not episodes.** Episodes get shorter as the agent improves, so later windows of 500 episodes cover fewer decisions. The first window spans 239,776 decisions. The last full one spans 3,698,464 - 3,559,264 = 139,200 *(by hand)*.
- **The curve is not a smooth rise.** It goes: rise to about 0.41, plateau near 0.37, **drop back to 0.176**, sudden jump to 0.65, then a noisy rise to about 0.72. Section 4.6 is about the drop.

### 4.5 The verdict against ADR 0009's rule

ADR 0009 Decision 5 fixed the rule before the run. "Learning" needs all three conditions, over the last 500 finished episodes, plus at least 1,000,000 decisions in total.

| Condition | Required | Measured | Met |
|---|---|---|---|
| Enough decisions | at least 1,000,000 | 3,714,880 (report) | yes |
| No throughput anomaly (see below) | as written: no engine below 50 decisions/s for more than 10 min | about 14 to 26 per engine *(by hand: 457 to 826 / 32)* | **no, by the letter**; yes under the corrected rule |
| Success above random | at least p0 + 0.15 = 0.29 | 0.726 - 0.140 = **+0.586** | yes |
| Return above random | at least R0 + 0.15 = 0.098 | 0.616 - (-0.052) = **+0.668** | yes |
| Success rising: last third of the run above the first third | last third higher | first third 0.335, last third 0.617 *(recomputed, by episode count; 0.330 and 0.613 by decision count)* | yes |

**The throughput condition, carefully.** ADR 0009's second "no verdict" condition is "sampling below 50 decisions/s per engine instance for more than 10 minutes". At 4x8, each worker steps 8 engines one after another (lesson 10 section 3.3), so every engine runs at about 17 to 21 decisions/s on average *(by hand: the report's 540 to 661 total / 32)*. **By the letter, the condition was tripped for the whole run**, by design and not by a fault. Lesson 10 section 3.5 recorded that the coordinator reported an advisor correction: the alarm line becomes **half the benchmarked rate for the chosen layout**, about 743 / 2 ≈ 371 decisions/s in total *(by hand)*. This run's lowest sampling rate was 457, above that line. **That correction is still not written in any file under `docs/decisions/` or `reports/runs/`** (I searched again), and the report's verdict table does not mention this condition at all.

**Verdict: learning, on the training scenario, under the corrected throughput rule.** Under ADR 0009 as written, the run would be "no verdict".

- **"Scenario solved"** needs success at least 0.9. 0.726 is short of that, so it is **not solved**.
- **"Hard task, not a bug"** needs success within 0.1 of p0. It is far above, so this does not apply.
- **"No verdict"** needs fewer than 1,000,000 decisions, or the throughput alarm. The first did not happen; the second depends on the unrecorded correction (above).

Two small differences from the report (listed again in section 7):

- The report checks "rising" with the first 500 against the last 500 episodes (0.168 to 0.726). ADR 0009's rule is thirds. By thirds the condition is still met.
- 3,714,880 is the `decision` stamp of the last *finished episode*. The run trained on 1,814 x 2,048 = **3,715,072** decisions *(by hand)*. The 192 difference is 6 x 32: the last six steps of the final rollout, after the last episode finished. It changes nothing.

### 4.6 The mid-run collapse: what was seen, and what is not known

**What was observed.** Success climbed to about 0.37-0.41 and stayed there, then fell back to 0.176 (episodes 4,500-5,000), close to the random 0.14. Then it jumped to 0.654 within the next 1,000 episodes. Nobody changed anything; the run recovered by itself.

**What the entropy did.** I lined up success with entropy per block of 100 updates, using the one-apart alignment of section 3.1 *(recomputed)*:

| Updates | Episodes | Success | Entropy (sum, max 6.22) | Move head (max 1.10) | Turn head (max 1.95) |
|---|---|---|---|---|---|
| 1-100 | 426 | 0.185 | 5.62 | 0.98 | 1.67 |
| 301-400 | 532 | 0.410 | 5.17 | 0.88 | 1.74 |
| 401-500 | 522 | 0.368 | 5.05 | 0.99 | 1.64 |
| 501-600 | 516 | 0.355 | 3.42 | 0.58 | 1.42 |
| 601-700 | 516 | 0.372 | 1.65 | 0.30 | 0.69 |
| 701-800 | 501 | 0.287 | 1.22 | 0.44 | 0.30 |
| 801-900 | 479 | 0.219 | 1.16 | 0.27 | 0.21 |
| 901-1000 | 462 | 0.180 | 0.52 | 0.11 | 0.02 |
| 1001-1100 | 494 | 0.249 | 1.11 | 0.28 | 0.008 |
| 1101-1200 | 669 | **0.647** | 1.75 | 0.47 | **0.015** |
| 1201-1300 | 615 | 0.585 | 2.15 | 0.58 | 0.053 |
| 1701-1800 | 718 | 0.706 | 2.51 | 0.17 | 0.30 |

The entropy columns for the rows the report also lists match the report's table.

Reading it, carefully:

- **Entropy fell first.** Between updates 500 and 700 the summed entropy dropped from about 5 to 1.65, while success stayed near 0.36-0.37. Success only started to fall after that (updates 701-1000). The report's wording, "nearly deterministic ... while success was dropping", is true for updates 701-1000. But the table shows the fall in entropy *began before* the fall in success.
- **The lowest point.** Summed entropy reached 0.122 at update 953. At that update the turn head's entropy was 0.0019 *(recomputed)*, out of a possible 1.95. The policy was choosing the same turn option almost every time. (The report's "0.02" is the 901-1000 average.)
- **The recovery happened while the turn head was still almost fixed.** In updates 1101-1200, success jumped to 0.647 while the turn head's entropy was still 0.015. So "the turn head locked onto one option" was true both during the worst stretch and during the recovery. That alone cannot explain the drop.

**What this looks like, in plain words.** The policy became very sure of itself, on a task where it had seen the goal only about a third of the time. In RL this is called a **premature entropy collapse**: the agent stops trying alternatives before it has found a good behaviour. Then, without any change from us, it got out of it.

**The cause is not established.** The report lists three candidate causes. **None has been tested:**

1. **The entropy bonus is too small for a sparse reward.** `entropy_coef` is 0.01 (`train_config.py:28`). With a reward that comes only at the goal, the push towards certainty may outweigh this small push towards trying things.
2. **The turn head locked onto one option**, so the agent kept spinning or never turned. The table shows it did lock. Which option it chose, and what the agent was doing, were not logged. The recovery with the turn head still locked (above) is a point against this being the whole story.
3. **The return scale interacting with a falling success rate.** During the drop, the return scale was almost flat: about 0.18-0.19 in every block from update 501 to 1100 *(recomputed)*. That is not what you would see if it were swinging. But it is a running figure over the whole run (section 3.2), so a flat line does not rule this out.

**Why one seed matters here.** Recall from lesson 09 why one seed is not evidence: a single run mixes the method with the luck of that run. We have **one seed**. So we do not know which is typical: the collapse, or the recovery. A second run with a different seed might collapse and never recover. The report says a second seed is needed.

**A gap in the rules.** ADR 0009's trainer-problem list catches "any head's entropy under 10% of its maximum **in the first 50 updates** without success rising". The collapse here came at update about 950. So, by the letter of the rule, it triggered nothing. ADR 0009's "hard task" test does say "no head's entropy collapsed", but it does not say what to do about a collapse late in a run.

### 4.7 The health metrics: how to read each one (from the report)

Means over six equal parts of the run (about 302 updates each):

| Metric | 1 | 2 | 3 | 4 | 5 | 6 | Min | Max |
|---|---|---|---|---|---|---|---|---|
| Entropy | 5.56 | 4.52 | 1.31 | 1.16 | 2.44 | 2.50 | 0.122 | 6.22 |
| Explained variance | 0.790 | 0.827 | 0.806 | 0.831 | 0.719 | 0.728 | -4.07 | 0.999 |
| Approx. KL | 0.0069 | 0.0056 | 0.0030 | 0.0034 | 0.0063 | 0.0061 | 0.0001 | 0.0255 |
| Clip fraction | 0.100 | 0.075 | 0.038 | 0.042 | 0.082 | 0.080 | 0 | 0.317 |
| Gradient norm | 1.10 | 0.82 | 0.63 | 0.68 | 0.92 | 0.94 | 0.16 | 1.70 |
| Return scale | 0.139 | 0.189 | 0.184 | 0.193 | 0.217 | 0.226 | 0.100 | 0.230 |
| Sampling, decisions/s | 661 | 653 | 639 | 643 | 574 | 540 | 457 | 826 |

My recompute from `metrics.jsonl` matches these to the shown digits.

**Entropy** tells you **how undecided the policy is**. Read it per head, not only the sum. The sum hides which choice collapsed. Here the move and turn heads did most of the falling.

**Explained variance** tells you **whether the value head is predicting returns**. 0.72 to 0.83 on average is good: the value head explains most of the variation in returns, so the advantages built from it are meaningful. The minimum, -4.07, is the **very first update** *(recomputed)*, before the value head had learned anything. Only 2 updates in the whole run were below -1, never 2 in a row *(recomputed)*. The ADR 0009 alarm (below -1 for 50 updates in a row) did not come close. The strongly negative values seen in the smoke runs (lesson 10) did not appear here.

**Approximate KL** tells you **how big each update's step was**. All values were between 0.0001 and 0.0255, well under the 0.05 alarm. 77% of updates were inside ADR 0009's "healthy" band of 0.003 to 0.03 *(recomputed)*. Notice that the KL was **lowest in parts 3 and 4** (0.0030, 0.0034), exactly when entropy was lowest. That fits: a nearly certain policy has little room to move.

**Clip fraction** tells you **how often PPO had to hold the policy back**. Average 4% to 10%, maximum 0.317, never near the 0.5 alarm. Like the KL, it was lowest during the collapse.

**Gradient norm** tells you **how strongly the loss was pulling on the weights**. Averages of 0.63 to 1.10 and a maximum of 1.70 mean some steps were scaled down to 1.0 by clipping, but none were wild. No value was non-finite.

**Return scale** tells you **how spread out the returns are**. It sat on the 0.1 floor for the first updates (the first 10 are all 0.100, *recomputed*), then rose slowly to 0.230 as more goal rewards arrived. The ADR 0009 alarm, "more than doubling between consecutive updates after update 50", never came close: the largest jump after update 50 was 1.2% *(recomputed)*.

**Sampling speed** tells you **how fast experience was collected**. It fell from about 660 to about 540 decisions/s. The report offers one guess, not tested: shorter episodes mean more resets, and resets may be slower than steps.

**Other checks in the report:** no non-finite value in any metric; `valid_frac` 1.0 throughout; stale time-outs 0. The auxiliary losses (keys, damage, monster count) went to near zero because this map has no keys, damage or monsters: **they were never exercised**, so this run says nothing about them.

**Running through ADR 0009's trainer-problem list** (the report does not do this item by item; I did, from the logs):

| Trigger | This run |
|---|---|
| Non-finite loss or gradient | none (report) |
| KL above 0.05 or clip fraction above 0.5 for 20 updates | never (max 0.0255 and 0.317) |
| A head's entropy under 10% of max in the first 50 updates | no (it happened around update 950, outside the rule's window) |
| Success falling more than 0.1 below p0 and staying | no (lowest window 0.176, above p0) |
| All entropies at maximum with KL near zero after 100 updates | no |
| First-minibatch ratio deviation above 1e-4 | no (0.0 on every update, *recomputed*) |
| `return_scale` more than doubling after update 50 | no (largest jump 1.2%) |
| Value loss or gradient norm jumping 10x when goal rewards arrive | **not checked** item by item; gradient norm max 1.70 |
| Explained variance below -1 for 50 updates | no (never twice in a row) |
| Process-tree memory rising steadily | no: about 2.55 GiB at the first update and 2.53 GiB at the last *(recomputed)* |

### 4.8 Laptop load (from the report)

- Process-tree memory stayed far under the 8 GiB cap. From `metrics.jsonl` it was 2.47 to 2.62 GiB *(recomputed)*; the report gives "2.59 to 2.82 GiB" (see section 7).
- Free system memory: minimum 3.52 GiB, typically 4.1 to 4.4 GiB.
- All 37 processes ran at below-normal priority (`stdout.log` lists every one as `below_normal=True`).
- CPU was 27% at the one moment it was sampled. It was not logged continuously, so we do not know the peak.

### 4.9 What this run does not show

Say this every time the result is mentioned:

- **One map, trained on.** `my_way_home` has random start points, but it is always the same map. The agent was tested on the map it trained on. The judges use **maps we have never seen**. No held-out map evaluation exists yet.
- **No combat.** The map has no monsters. Deaths were 0. Nothing about aiming, dodging or health management was tested.
- **No keys or doors.** Nothing about long-horizon progress or memory was tested. Lesson 09's limit still stands: no memory beyond 20 decisions has been shown.
- **Not the vanilla EVAL path.** The checkpoint has not been run under EVAL config.
- **One seed.** Both the collapse and the recovery could be luck.
- **No reward shaping and no exploration bonus.** `rewards/` and `intrinsic/` exist but were not used.

What it *does* show: the observation pipeline, the model, our PPO and the run infrastructure work together, for 2 h 20 min, on 32 engines, without crashing, leaking memory or leaving processes behind, and the policy learns to reach the goal on that map far more often than random.

### 4.10 The toy diagnostic D0 to D4: planned, NOT yet run

INDEX listed this for lesson 11. **It has not been run, and there are no results.** Here is what each arm is meant to find out, from ADR 0009 Decision 3.

Recall lesson 09: on the toy memory task, the agent remembered a cue across a gap of 20 steps, but failed at a gap of 60. The failed runs could not say *why*. Three explanations fit:

- **the window**: the 64-step rollout is too short for the gradient to connect the cue to the answer;
- **the budget**: 300 updates were too few;
- **the label**: the dense cue label (the keys head, weight 0.1) was too weak.

The diagnostic runs gap 60, seeds 0 to 3, 300 updates unless stated, and logs the keys head's accuracy at each episode step. Each arm changes one thing:

| Arm | What it is | Question it answers |
|---|---|---|
| D0 | the base, unchanged | The missing baseline: the base was never run at gap 60. |
| D1 | dense cue label at weight 1.0 instead of 0.1 | Is the label too weak? |
| D2 | update-gate bias, rollout 128, 150 updates | Is the window the limit? With 128 steps, the whole episode fits in one window. |
| D3 | dense cue label, 900 updates | Is the budget the limit? |
| D4 | supervised only: dense cue label, windows of 32, no RL at all | Can a dense label carry a fact across a window boundary in this design *at all*? |

D4 matters most. If D4 fails, both the keys-head plan and Stage B's training plan need redesign before either is relied on. ADR 0009 fixed the readings of every outcome in advance, so nobody can choose an interpretation after seeing the results.

## 5. Why we judged it this way instead of the alternatives

From ADR 0009 (Options, "First-run verdict"; **Proposed, not ratified**):

- **V1, judge by eye from the console.** No build needed. But the console shows a 20-episode average with a standard error near 0.1, so it would likely have swung a lot, and it is easy to read into such swings whatever you hope to see.
- **V2 (chosen): a random baseline, per-episode results on disk, and 500-episode windows against thresholds stated in advance.** About an hour of build and 10 minutes of compute.
  - The baseline turns "success 0.73" into "0.59 above random".
  - The thresholds were written before the data existed, so the verdict is not bent to fit the result.
  - Per-episode files made the collapse visible. Without `episodes.jsonl`, the only record would have been the 20-episode averages.

ADR 0009 named a risk: "p0 could be high enough on easy start points that +0.15 is reached by learning those alone." This run cannot rule it out. Per-start-point results are not logged, so it cannot be checked directly. The size of the gain (0.14 to 0.726) suggests more than easy start points alone, but that is a judgement, not a measurement.

## 6. How it connects

- **Lesson 06**: return, value, explained variance and return scale are read here for the first time on DOOM.
- **Lesson 07**: entropy bonus, KL and clipping are what kept the updates gentle, and the entropy bonus is the first suspect for the collapse.
- **Lesson 09**: why one seed is not evidence; the memory limit that this run did not test.
- **Lesson 10**: the files, the launch, and the verdict rules this lesson applies.
- **Next**: `rewards/` and `intrinsic/` (ADR 0010) are built and tested on synthetic data, but not yet wired into the trainer. ADR 0009 says that after a "learning" verdict, the next step is a comparison with the coverage bonus. Lesson 12 will teach the reward terms and the episodic coverage bonus. A second seed, the D0-D4 diagnostic and a held-out evaluation are still to be run.

## 7. Common mistakes, and how we would notice them

| Mistake | How we would notice |
|---|---|
| Reporting "73% success" without the baseline. | Random reaches 14%. Always report the difference (+0.586) and the baseline beside it. |
| Judging on the first 100 episodes. | They lean towards short successes (0.22, above random 0.14) because time-outs cannot finish early. Use full 500-episode windows. |
| Reading only the summed entropy. | The sum hides which head collapsed. Plot `entropy_head0` to `entropy_head5` against their maxima (1.10, 1.10, 1.95, 0.69, 0.69, 0.69). |
| Calling the collapse "explained". | None of the three candidates has been tested. The recovery happened with the turn head still locked. Say "observed, cause unknown". |
| Treating the recovery as reliable. | One seed. Run a second seed before relying on it. |
| Claiming the agent "navigates DOOM". | One trained-on map, no monsters, no keys, no held-out test. Run on a held-out map under EVAL before any such claim. |
| Taking per-engine differences as real. | About 62 episodes per engine give a standard error near 0.06. The 0.47 to 0.77 spread is about what chance gives. |
| Mixing up `update_idx` between the two files. | An episode stamped `update_idx` k belongs to the rollout of `metrics.jsonl` update k+1. Off-by-one joins shift the timeline by one update. |
| Thinking the auxiliary heads were tested. | Their losses went to near zero because there was nothing to predict on this map. |

**Things in the report or code that look wrong or unclear** (only the first affects how the verdict must be worded):

0. **The throughput "no verdict" condition is not addressed.** The report's verdict table checks decisions, success, return and rising, but not ADR 0009's "below 50 decisions/s per engine instance for more than 10 minutes". By the letter that was tripped all run (about 17 to 21 per engine). The run is "learning" only under the advisor's correction (half the benchmarked rate), which is not recorded in any decision file. Who: `advisor`, to record the corrected rule.
1. **"Success rising" uses a different test from ADR 0009's.** Report, verdict table: "0.168 (first 500) to 0.726 (last 500)". ADR 0009 Decision 5 says "success in the last third of the run above the first third". By thirds it is 0.335 to 0.617 *(recomputed)*, still met. Who: whoever writes run reports (coordinator), to use the ADR's test.
2. **Decision total.** Report: "Decisions ... 3,714,880" and "1,814 updates, ... 3,714,880 decisions". That is the last finished episode's stamp. Decisions trained: 1,814 x 2,048 = 3,715,072. Tiny, but the two are different quantities.
3. **Collapse timing.** Report: "the policy became nearly deterministic ... while success was dropping". The entropy fall began around updates 501-700, while success was still on its plateau, and continued while success fell. Also, the report's "turn-head entropy 0.02" is the 901-1000 average; at update 953 itself it was 0.0019.
4. **Memory units.** Report: "Process-tree memory: 2.59 to 2.82 GiB". In `metrics.jsonl` the maximum is 2,814,595,072 bytes, which is 2.81 **GB** (decimal) or 2.62 GiB. The run summary in `stdout.log` gives `max_tree_rss_bytes` 2,815,156,224, also 2.62 GiB. The upper figure seems to be in GB with a GiB label. The cap was never approached either way.
5. **The 500-episode table's last row is not "the last 500".** It covers episodes 9,500-10,000 (0.718). The last 500 are 9,560-10,060 (0.726). Both are correct, but they are different windows.
6. **ADR 0009's entropy-collapse trigger only covers the first 50 updates**, so a collapse at update 950 fires no rule. Who: `advisor`, to decide whether a later collapse should count.
7. **Stale docstring in the code.** `train/running_stats.py:4` says the scale is "floored at `min_scale` (1.0 per ADR 0007 ruling 2)". The config uses 0.1 (`configs/train_config.py:35`, ADR 0008 point 3), and this run's minimum was 0.100. The code is right and the comment is out of date. Who: `worker`.
8. **`config.json` description** says the layout was "recommended by trainer-ops". The report already corrects this: `trainer-ops` recommended 4x2 and the coordinator chose 4x8 (as lesson 10 said).

## 8. Check your understanding

1. The last 500 episodes have a success rate of 0.726. Why is "the agent succeeds 73% of the time" a misleading summary of what it learned, and what should be said instead?
2. Between updates 900 and 1000, summed entropy averaged 0.52 and the turn head's entropy was 0.02 out of 1.95. KL and clip fraction were also at their lowest in this part of the run. Why do these three go down together, and why does the recovery in updates 1101-1200 make "the turn head locked" a weak explanation for the drop in success?
3. A teammate wants to tell the judges "our agent learns to navigate DOOM levels". List three things this run did not test that the judges will.

---

## Answers

1. A random button-presser already succeeds 14% of the time on this map, because some start points are near the goal. The part due to learning is about **0.726 - 0.140 = +0.586**, measured over 500 episodes with a standard error of about 0.02. Also, it is success **on the map it trained on**, one seed, no monsters or keys. A fair summary: "on the training map `my_way_home`, success rose from the random 0.14 to 0.73 over 3.7 million decisions, in one run, with a mid-run collapse we have not explained."
2. When the policy is nearly certain, the probability of the action it picks is already close to 1. Each update can change it very little, so the probability ratio stays close to 1. A ratio close to 1 means a small KL and few steps clipped. So low entropy brings low KL and low clip fraction with it. The turn head stayed almost fixed (0.015) during updates 1101-1200, yet success jumped to 0.647 then. So "the turn head was locked" was true both when the agent was doing badly and when it was doing well. That alone cannot explain why success dropped. Something else changed, and we have not found what. The cause is still not established.
3. Any three of:
   - **unseen maps**: it was tested only on the map it trained on, and no held-out evaluation exists;
   - **combat and survival**: there were no monsters, and deaths were 0;
   - **keys, doors and long-horizon memory**: none on this map, and the agent has shown no memory beyond 20 decisions (lesson 09);
   - **the vanilla EVAL path**: the checkpoint has not been run under EVAL config;
   - **reliability**: one seed only.
