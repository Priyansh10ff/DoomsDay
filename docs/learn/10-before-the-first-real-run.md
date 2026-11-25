# Lesson 10: Before the first real run

Files taught: `configs/ops_launch.py`, `configs/ops/*.json` (especially `a0_4x8_main.json`), `train/trainer.py:55-76` (per-episode records), `train/run_loop.py:30-56` (`episodes.jsonl` and the recent-episode statistics), `train/doom_factory.py:17-22` (the trainer's refusal of `phase2_engine_test`), and `tests/test_episode_log.py`.

Evidence:

- `reports/runs/2026-11-25-part1-single-instance-checks.md` (Part 1);
- `reports/runs/2026-11-25-part2-layout-benchmark.md` (Part 2);
- `reports/runs/INDEX.md` (one line per run).

Decision record: `docs/decisions/0009-toy-results-memory-plan-and-first-doom-run.md` (ADR 0009). Its header says "Date: 2026-11-26"; the advisor states that is an erratum for **2026-11-25**.

**All ADRs, 0009 included, are "Proposed", not ratified by the user.**

**State of things, plainly.** The first real DOOM training run **started at 08:04:45 on 2026-11-25 and is still running** while this lesson is written. **Nothing has been learned from it yet, and this lesson reports none of its results.** A run that is not finished must not be read for conclusions. Section 4 explains how to read its files when it is done.

Builds on lesson 08 (workers, the layout guard, the laptop protections, checkpoints) and lesson 09 (the memory test, Gate A and Measurement B).

---

## 1. What problem this piece solves

Starting a long training run on the user's only laptop is expensive. It takes hours of a machine someone needs, and a bad run wastes all of that. Before pressing "go", the team had to answer five questions:

1. **What should we expect the agent to remember?** (The memory results of lesson 09.)
2. **Do the safety mechanisms work with real engines,** not only in toy tests?
3. **How many engines, and arranged how?**
4. **What exactly is being run?**
5. **How will we know afterwards whether it learned, failed for a boring reason, or is broken?**

This lesson goes through each, from the records the team left.

## 2. The idea in plain words

Think of a **first long test drive of a new car**.

- You do not expect it to cross a desert; you decide in advance what this drive can and cannot show (the memory ruling).
- You test the brakes, the emergency stop and the restart in the car park first (the single-instance checks).
- You pick how many passengers and how much luggage, by weighing the car, not by guessing (the layout benchmark).
- You write down what "a good drive" means **before** you set off, so you cannot talk yourself into it afterwards (the success criteria).

Where the analogy stops being accurate:

- A test drive gives an answer as it goes. A training run's early numbers move a lot by chance, so reading them half-way is likely to mislead.
- A car's brakes are tested with the real car. Here one check, the real console Ctrl-C, could not be done (section 3.2).

### Terms used in this lesson

- **Baseline**: the result of a simple reference, here a random policy, that "learning" must beat.
- **Throughput**: decisions processed per second. **End-to-end throughput** counts both collecting experience and updating the network.
- **Per-engine rate**: decisions per second for one engine instance.
- **Launcher**: a small program that starts another with prepared settings.
- **Void comparison**: a comparison whose setup makes it unable to show the thing it was meant to test.

## 3. A walk through what was decided and checked

### 3.1 The long-horizon memory ruling (ADR 0009 Decisions 1 to 3)

Lesson 09 ended with: the agent remembered over 20 steps, but failed at 60, and the key-and-door task needs minutes. ADR 0009 is the advisor's ruling. In plain words:

**Do not rely on the GRU to remember across long delays.**

- Key-to-door delays are minutes: thousands of decisions against a 64-decision rollout window. The agent has already failed at 60.
- So, in ADR 0009's words, "the plan must not rest on unlabelled recall through the GRU". This holds whatever the toy's fairness: the toy is harsher than DOOM in some ways (the cue is visible for one step among noise; nothing in between) and easier in others (two cues, no distractions).

**Instead, observe it or store it in structure** (Decision 3):

- **Key possession is to be read from the screen.** The status bar shows the held keys, so with the `screen` sensor on, "I have the blue key" is in the observation at every step. No memory is needed, and it survives `+viz_nocheat`. That is the job of the pixel stage (Stage A1). Whether the icons are legible at 60x80 is so far only a judgement from looking at images. The planned test: a small classifier from `screen` to the three key bits, trained on frames from some maps and tested on others, needing at least 0.95 accuracy per bit. If it fails, the full-resolution status-bar crop that ADR 0001 reserved is added.
- **Door location belongs to the map memory (Stage B)**, with dense training-only labels from maps whose door positions are known.
- **A longer rollout window is not the fix.** Rollout 64 stays: even 128 or more would not reach minutes, since a minute is about 525 decisions.
- **The stage order A0, then A1, then B stands.** Stage B now has a three-part gate: a toy diagnostic must pass; the organizers must confirm telemetry is available (or at least 2.5 days remain without an answer); and A1 must meet ADR 0003's exit-rate condition.

**What is not yet known, and the diagnostic for it.** At gap 60 the toy failed, but the runs could not say *why*:

- the window was too short;
- the budget was too small;
- the label was too weak.

ADR 0009 fixes, in advance, a 45-minute diagnostic with five arms (D0 to D4) and its readings. The most important is **D4**: plain supervised training with a dense cue label and windows of 32 steps. It tests whether a dense label can carry a fact across a window boundary *at all*. If D4 fails, both the keys-head plan and Stage B's training plan need redesign. **The diagnostic has not been reported yet.**

**None of the variants was adopted** (Decisions 1 and 2):

- B1, the update-gate bias;
- B2, the dense cue label;
- E, 2 epochs;
- I, orthogonal initialisation.

**Why a tie against a base that already scores 8 of 8 proves nothing.** At gap 20 the base scored 8/8. No variant can score more than 8/8, so no result could have shown it to be *better*. And a variant could only have shown harm by dropping seeds, which none did. The comparison could only ever end in a tie. That is the ceiling effect of lesson 09, and ADR 0009 now names it: such a comparison is **void**, "not adopted, not refuted".

**The one adoption rule** (Decision 1), which replaces ADR 0008's three conflicting sentences:

- A change replaces the base only if, at a setting declared before the runs, it scores **at least 2 seeds out of 8 more** than the base. Anything less keeps the base. The rule is the same for every arm.
- If the base already scores 7 or 8 of 8 at that setting, the comparison is **void**.
- A change wanted because it is *cheaper*, not better, is adopted only on a DOOM measurement:
  - the update takes more than 30% of wall time;
  - the change cuts that by at least a quarter;
  - return over equal decisions is not lower.

  Two epochs might come back this way.
- For future comparisons, the declared sensitive setting is **gap 20 at 100 updates** (Decision 4), where the base has not yet been measured.

Measurement B as a whole is still **failing**, because its gap-60 leg fails. ADR 0008's blocks stay: no key-and-door curriculum stage, and no claim of long-horizon memory. ADR 0009 adds one possible route around the first block, the keys-icon read test for an agent with the screen on, but that route needs the user's explicit agreement.

### 3.2 The three single-instance checks (Part 1)

`trainer-ops` ran these one engine at a time, 2 torch threads, below-normal priority. Each protects against a specific way a long run could go wrong.

**a. A real time-out takes the fresh-observation path.**

- *What it protects against:* the learner mishandling the end of a time-limited episode (lessons 01 and 06). A time-out should bootstrap from the value of the true final observation. If instead the engine's own backstop limit fired, the observation would be stale and the transition dropped.
- *How:* the episode limit was cut to 200 tics (50 decisions), so time-outs happen every few seconds (`configs/ops/timeout_smoke.json`, run `20261125-nogit-2d4e952a`). A probe inspected two 128-step rollouts. Every episode ended at exactly 50 decisions, marked `truncated` and `valid`, with a bootstrap value computed. The engine's backstop never fired.
- *Result:* over 250 updates (16,000 decisions), **stale time-outs 0 on every update**, `valid_frac` 1.0 throughout. This replaces the earlier "stale count 0" readings, which were vacuous because no time-out had happened.

**b. Hard-kill the trainer and count survivors, for both worker designs.**

- *What it protects against:* orphaned `vizdoom.exe` processes eating memory after a crash (lesson 08).
- *How:* start a 1x1 run, wait for updates, record the main process, the worker and the engine, then kill **only the main process** hard. Count the recorded processes still alive at 1, 3, 5, 10, 15 and 20 s, and count `vizdoom.exe` system-wide.
- *Result:* **0 survivors with non-daemon workers** (`20261125-nogit-a9f0b2fb`) **and 0 with daemon workers** (`20261125-nogit-7cfb625c`). The workers' parent-polling closed the game within a second. Per ADR 0008, the non-daemon design stays the default.

**c. Interrupt, then resume with a new session seed.**

- *What it protects against:* losing work when the user stops the run, and replaying the same episodes after every resume (lesson 08).
- *How:* the real training loop at 1x1 was interrupted at a random moment (19.20 s) by calling `stop.on_sigint()`, the same function the Ctrl-C handler calls. The interrupt arrived during update 41; the update finished, a checkpoint was written 0.11 s after the request, and it loaded with `weights_only=True` at update 41. Then the real launcher resumed from it: `session 1, env seed 3964924996` (a new seed, not the run seed 0), and it ran on to update 107 (run `20261125-nogit-8653f5c1`).

**Untested, plainly: a real console Ctrl-C.** `trainer-ops` tried to start the trainer in its own console and send it a real Ctrl-C. A security hook blocked that helper, and `trainer-ops` stopped instead of working around it. Two things therefore still need a person at a terminal:

1. that a real console Ctrl-C reaches the trainer's handler;
2. whether it also reaches the engine processes directly.

Because of this, the main run is bounded by a time limit (`--max-seconds`), not by someone pressing Ctrl-C.

### 3.3 The layout benchmark (Part 2)

**What "4 workers x 8 envs" means.** Four worker processes, each running eight DOOM engines: **32 engine instances** in total. The main process runs the network for all 32 at once and does the update (lesson 08).

**The measurements.** 90 seconds per layout, one at a time, 4 torch threads, rollout 64 (`configs/ops/bench_*.json`). From the report:

| Layout | Engines | Sampling decisions/s (median) | End-to-end decisions/s | Per-engine sampling | Process tree peak | Lowest system free memory | CPU (of 24 threads) |
|---|---|---|---|---|---|---|---|
| 1x1 (Part 1) | 1 | 177 | about 133 | 177 | 409 MiB | 5.44 GiB | not measured |
| 2x4 | 8 | 401 | 257 | 50 | 926 MiB | 5.07 GiB | 25% |
| 4x2 | 8 | 574 | 338 | 72 | 1011 MiB | 5.07 GiB | 26% |
| 4x4 | 16 | 620 | 391 | 39 | 1554 MiB | 4.81 GiB | 27% |
| 4x8 | 32 | 743 | 520 | 23 | 2653 MiB | 4.27 GiB | 30% |

All runs: every process below normal, stale time-outs 0, no NaN, 0 engines left after.

**Why per-engine speed falls as environments per worker rise.** Inside a worker, the environments are stepped **one after another** (`subproc_vec_env.py`, the `step` command loops over its environments; lesson 08). A worker with 8 engines must wait for each in turn before it can reply, so each of its engines advances at about an eighth of the worker's pace. The workers themselves run in parallel. So:

- **total speed** grows mainly with the number of **workers**: 177 → 401 → 620 sampling decisions/s for 1 → 2 → 4 workers;
- doubling environments per worker at 4 workers added only about 20% (620 → 743);
- **per-engine speed** falls with environments per worker: 72/s at 4x2, 23/s at 4x8.

By hand, per worker: 743 / 4 ≈ 186 decisions/s at 4x8, close to the single engine's 177. Each worker is running at about one engine's pace, shared among its eight engines.

**The trade-off.** More engines per worker gives more total experience per second, and more different episodes in each update (32 instead of 8), at the cost of more memory and slower progress for each individual game. Per-engine speed matters because each game advances more slowly in wall-clock time. It also matters because ADR 0009 used it as an alarm: "sampling below 50 decisions/s per engine instance for more than 10 minutes" means "no verdict, stop and hand to `debugger`". That line was written to catch the unexplained 17/s smoke run. 4x4 and 4x8 fall under it **by design**, not because anything is wrong.

### 3.4 How the run is launched: `configs/ops_launch.py` and `configs/ops/*.json`

`train.run_doom` has no flags for the episode limit or the worker design, so `trainer-ops` added a **launcher** (`ops_launch.py:1-17`):

- **The JSON file** (one per run in `configs/ops/`) holds `argv` (the `run_doom` arguments), `env_overrides` and `safety_overrides`.
- **`launch()`** (`:27-64`):
  - replaces the two names `run_doom` reads (`a0_train_env_config` and `SafetyConfig`) for this process only (`:39-48`);
  - asserts that every override took effect (`:50-54`);
  - prints the effective values;
  - copies the JSON into the run folder;
  - calls `run_doom.main()`.

  The defaults in `configs/*.py` are untouched.
- **Only one extra argument is allowed**, `--resume <checkpoint>` (`:67-74`). Every setting lives in the JSON.
- **Run names** are `20261125-nogit-<first 8 hex of the SHA-256 of the JSON>`. `nogit` because the repository has no commits yet, so no run can be tied to an exact code version (`reports/runs/INDEX.md`).

**Two small code changes for this run:**

- **Per-episode results on disk** (ADR 0009 Decision 5 item 2: "a 2.5-hour run whose per-episode results exist only in memory should not be launched"). `episode_record` (`train/trainer.py:66-76`) builds one record per finished episode. `drain_episodes` (`train/run_loop.py:38-45`) appends them to `episodes.jsonl` and flushes after every update, so the file survives an interrupt. Episodes from an abandoned rollout are written too. `tests/test_episode_log.py:45-75` checks one line per episode, the right fields, and that a resume appends rather than overwrites.
- **The trainer refuses `phase2_engine_test`.** `DoomEnvFactory.__post_init__` (`train/doom_factory.py:17-22`) raises if the environment config has it set. This closes the gap flagged in lesson 09; test `tests/test_episode_log.py:78-89`.

### 3.5 The run now in progress

- **Run** `20261125-nogit-7a675759`, from `configs/ops/a0_4x8_main.json`, launched by the **coordinator** at **08:04:45 on 2026-11-25**.
- **Layout:** 4 workers x 8 envs = **32 engine instances**, 4 torch threads, `--allow-multi-instance 32`.
- **Protections:** below-normal priority, the 8 GiB memory cap (and the 1.5 GiB free-memory stop), a checkpoint every 600 s, non-daemon workers.
- **Length:** stops itself after **8400 s** (2 h 20 min), so by about 10:25 (by hand: 08:04:45 + 2:20:00 = 10:24:45), inside the user's approval window that ends at 11:00.
- **What it trains on:** `my_way_home`, with the scenario's own sparse reward: -0.0001 per tic, +1.0 at the goal, 2100-tic limit (525 decisions). **No exploration bonus.**
- **Model and recipe:** the A0 network with the base recipe (constant learning rate 3e-4, 3 epochs, floor 0.1, initialisation as built), screen off.
- **Its own header** (the only part of its log I read): "planned layout: 32 engine instance(s) in 4 worker process(es), 4 torch thread(s), estimated memory 2894 MiB".

**Whose decision this was.**

- **`trainer-ops` recommended 4x2** (8 engines). It meets every ADR 0009 rule as written: 72 decisions/s per engine, at least 2 GB free, low CPU. It offered 4x8 only as an alternative, "if `advisor` rules that the per-instance rule should be per worker".
- **The coordinator chose 4x8.** It reports that the advisor corrected the rule, replacing the fixed "50 decisions/s per engine" line with **half the benchmarked rate for the chosen layout**. Under that reading, 4x8 does not trip the alarm by construction.
- The coordinator's reasons: **520 against 338 decisions/s end to end**, with **4.27 GiB of memory still free** at 4x8 in the benchmark. That is well above ADR 0009's 2 GB margin. ADR 0009 itself had *estimated* that 4x8 would not fit; the benchmark measured that it does.
- **This was the coordinator's call, against `trainer-ops`' recommendation.** I could not find the advisor's correction in any file under `docs/decisions/` or `reports/runs/`; it is known to me only from the coordinator's message.

**What was not done before launch.** ADR 0009 Decision 5 lists, under "Before launch", a **random baseline**: 200 episodes of uniform random actions, giving the success rate p0, mean return R0 and mean episode length L0. **That baseline has not been measured yet.** The learning criteria below depend on it, so no verdict can be given until it exists.

## 4. Reading the run's files (no results here)

The run writes three files in `reports/runs/20261125-nogit-7a675759/`. You may read them; do not write there while the run is going.

**`episodes.jsonl`**: one line per finished episode (`trainer.py:66-76`):

| Field | Meaning |
|---|---|
| `update_idx` | The update during whose rollout the episode ended. |
| `decision` | Total decisions processed so far, across all 32 engines, when it ended. |
| `env_index` | Which of the 32 engines (0-31). |
| `return` | Total reward of the episode. |
| `length` | Decisions in the episode. |
| `terminated`, `truncated` | How it ended, as in lesson 01. |
| `level_exit`, `dead`, `timeout` | The ending, from the environment's end-of-episode `info`. |
| `success` | Equal to `level_exit`: reached the goal. |

**`metrics.jsonl`**: one line per update, with every statistic of lessons 06 to 08:

- losses: `loss`, `pg_loss`, `v_loss`, and the three auxiliary losses;
- policy health: `entropy` and `entropy_head0` to `entropy_head5`, `approx_kl`, `clipfrac`, `grad_norm`, `first_minibatch_ratio_max_dev`;
- value health: `explained_variance`, `return_scale`;
- bookkeeping: `valid_frac`, `stale_timeouts`, `lr`;
- speed: `sample_dps`, `update_dps`;
- memory: `tree_rss_bytes`, `available_bytes`;
- episodes: `episodes_done`, plus `recent_return`, `recent_success` and `recent_length` over the **last 20 episodes** (`run_loop.py:35`, `:48-56`).

**`stdout.log`**: the console output: the layout plan, the priority check, one line per update, and the stop summary.

**How to read them properly** (ADR 0009 Decision 5):

- **Judge on windows of the last 500 finished episodes from `episodes.jsonl`, not on `recent_success`.** A 20-episode average jumps around a lot: ADR 0009 puts its standard error near 0.1. For example, if the true success rate were 0.3, two different sets of 20 episodes could easily read 0.15 and 0.45. Over 500 episodes the standard error is about 0.02 (by hand: √(0.3 x 0.7 / 500) ≈ 0.02).
- **Count decisions, not minutes**, using the `decision` field.
- **Wait for the run to finish and for the random baseline.**

### What would count as what (fixed in advance by ADR 0009)

With p0, R0, L0 from the random baseline:

- **Learning:** over the last 500 episodes, success at least **p0 + 0.15**, mean return at least **R0 + 0.15**, **and** success in the last third of the run higher than in the first third.
- **Scenario solved:** success at least 0.9, and the mean length of successful episodes under half of L0.
- **No verdict:**
  - fewer than **1,000,000 decisions** in total. A learning curve on a sparse-reward map can stay flat for a long time before it rises, so too few decisions cannot tell "not yet" from "never";
  - or the throughput anomaly (as corrected, see 3.5) for more than 10 minutes.
- **A hard task, not a bug:** success stays within 0.1 of p0 **while the trainer is healthy**:
  - approximate KL mostly between 0.003 and 0.03;
  - no head's entropy collapsed;
  - losses finite;
  - memory flat.

  The reward is sparse and there is no exploration bonus, so a flat curve is a known outcome for plain actor-critic on this kind of map. The response is ADR 0005's exploration bonus, not debugging.
- **A trainer problem** (then `debugger`): any of these:
  - a non-finite loss or gradient;
  - KL above 0.05 or clip fraction above 0.5 for 20 updates in a row;
  - a head's entropy under 10% of its maximum in the first 50 updates without success rising;
  - all entropies at maximum with KL near zero after 100 updates;
  - the first-minibatch ratio deviation above 1e-4;
  - `return_scale` more than doubling between updates after update 50;
  - value loss or gradient norm jumping more than ten times and not settling;
  - **explained variance below -1 for 50 consecutive updates**;
  - memory rising steadily.

**By hand, for scale only** (not a prediction). If the benchmark's 520 decisions/s held for the whole run, 8400 s would give about 4.4 million decisions, so the 1-million line would be passed after about 32 minutes (1,000,000 / 520 ≈ 1,923 s). An episode that times out returns about -0.21 (2100 tics x -0.0001). One that reaches the goal after L decisions returns about 1 - 0.0004 x L. Whether the real run sustains that speed is one of the things its files will show.

### Two things `trainer-ops` flagged that nobody has explained yet

- **Explained variance strongly negative.** In the time-out smoke it was mostly strongly negative (last value -43.7). The resume leg at the normal 2100-tic limit logged -36.1 at update 107. Explained variance compares the value guesses with the actual returns: 1 is perfect, 0 is no better than predicting the average, and strongly negative means the guesses are much *worse* than the average. One candidate cause the report offers is very low variance of the returns, which makes the ratio unstable; it is **not established**. In the 4x8 benchmark it rose to 0.57. ADR 0009 counts "below -1 for 50 consecutive updates" as a trainer problem, so this is one to watch, and it is routed to `worker`.
- **One fast entropy drop.** In the time-out smoke run, the summed entropy fell from 6.22 to 2.90 (minimum 2.00) within 250 updates. The report calls it "a fast early decline, but not a collapse". Nobody has explained it. Recall from lesson 05 that 6.22 is the maximum: an undecided policy. A fast fall means the policy became confident quickly, which is worrying on a task where it has hardly been rewarded.

## 5. Why these choices over the alternatives

From ADR 0009 (**Proposed, not ratified**):

- **Adoption after a tie:**
  - T1, adopt on a tie: lets any change that does nothing into the recipe.
  - T2, a tie keeps the base.
  - **T3 (chosen)**: T2, plus voiding comparisons where the base is at the ceiling, plus a cost clause.
- **What to do about gap 60:**
  - P1, a longer window: cannot reach minutes, and its cost doubles per doubling.
  - P2, build a new memory now: 1.5 to 2 days on a mechanism that just failed to show itself.
  - P3, stronger auxiliary labels now: the same unverified mechanism.
  - **P4 (chosen)**: a 45-minute diagnostic with readings fixed in advance, while designing so that long-lived facts are observed or held in structure.
- **The first run's verdict:**
  - V1, judge by eye from the console: a 20-episode mean has a standard error near 0.1.
  - **V2 (chosen)**: a random baseline, per-episode results on disk, and 500-episode windows against thresholds stated in advance. The baseline part of V2 has not yet been done.
- **The layout:** ADR 0009 said to size it by memory, not by ADR 0006's 4x8, and that `trainer-ops`' benchmark decides. The benchmark measured enough memory for 4x8. The per-engine rule was the disagreement (3.5).

## 6. How it connects

- **Lesson 08**: every protection exercised here (layout guard, priority, watcher, boundary checkpoints, parent polling, session seeds).
- **Lesson 09**: the memory results this ruling interprets.
- **Next**: the run's report, with the random baseline and one of the four verdicts. The toy diagnostic D0-D4. And, depending on both, the exploration bonus (ADR 0005) or the pixel stage (Stage A1) with its keys-icon read test.

## 7. Common mistakes, and how we would notice them

| Mistake | How we would notice |
|---|---|
| Reading the run's numbers while it is still going and calling it. | Early success rates swing widely; the console's 20-episode `recent_success` has a standard error near 0.1. Wait for the end, and use 500-episode windows. |
| Declaring "learning" without the random baseline. | One untrained smoke episode already reached the goal (return 0.954), so the random success rate is not zero. Without p0, "+0.15 above random" cannot be checked. |
| Treating the 4x8 per-engine rate (23/s) as the 17/s anomaly. | It is by construction: each worker steps 8 engines in turn. Compare per-worker rates (about 186/s), not per-engine. |
| Adopting a variant because it tied a base at 8/8. | ADR 0009 calls that comparison void. Use the sensitive setting (gap 20, 100 updates) and the "2 more seeds out of 8" rule. |
| Expecting the A0 agent to remember a key over minutes. | It has not shown memory beyond 20 steps on the toy. Key possession is to be *seen* on the status bar (Stage A1). |
| Assuming a real Ctrl-C stops the run cleanly. | Untested from a real console; rely on `--max-seconds`. |
| Ignoring strongly negative explained variance. | ADR 0009's trigger: below -1 for 50 consecutive updates. Plot `explained_variance` from `metrics.jsonl`. |

## 8. Check your understanding

1. Arm B1 scored 8/8 at gap 20, the same as the base. Why does ADR 0009 call this comparison void rather than "B1 is as good as the base"? What would a fair comparison look like?
2. At 4x8 the per-engine sampling rate is 23 decisions/s, but at 1x1 it is 177. Why, and does that mean 4x8 is slower overall?
3. The run finishes. Its last 500 episodes show a success rate of 0.20. What else do you need before saying "it learned", and what would make you say "hard task, not a bug" instead?

---

## Answers

1. The base was already at 8/8, the maximum. No arm could score higher, so no result could show an improvement, and the comparison was bound to end in a tie whatever the arm did. That is the ceiling effect. A fair comparison uses a setting declared in advance where the base scores well below 7/8. ADR 0009 declares gap 20 at 100 updates. There, an arm must beat the base by at least 2 seeds out of 8 to be adopted.
2. Each worker steps its environments one after another, so its eight engines share one worker's pace: about 743 / 4 ≈ 186 decisions/s per worker, split eight ways ≈ 23 per engine (by hand). Overall, 4x8 is **faster**: 743 decisions/s sampling and 520 end to end, against 177 and about 133 at 1x1. It produces more experience per second, though each individual game advances more slowly.
3. You need the random baseline p0 and R0 (not yet measured): learning requires success at least p0 + 0.15 **and** mean return at least R0 + 0.15, **and** success in the last third of the run above the first third. You also need at least 1,000,000 decisions in total, or there is no verdict. If success is within 0.1 of p0 while the trainer is healthy, it is a hard task, not a bug, and the response is an exploration bonus:
   - KL mostly between 0.003 and 0.03;
   - no entropy collapse;
   - finite losses;
   - flat memory.
