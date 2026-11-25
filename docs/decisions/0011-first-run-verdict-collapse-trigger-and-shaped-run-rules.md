# ADR 0011: First-run verdict, throughput rule, late entropy collapse, the shaped run, and the baseline rule

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor). Number allocated by the coordinator.

**Amends**, without editing any earlier file:

- `0009-toy-results-memory-plan-and-first-doom-run.md`, Decision 5: the sentence "or sampling below 50 decisions/s per engine instance for more than 10 minutes. The second condition is the 17-per-second anomaly"; the trigger "Any head's entropy under 10% of its maximum in the first 50 updates without success rising"; and the "Before launch" baseline item.
- `0005-exploration-and-curriculum-maps.md`: "it has its own value head and discount (ADR 0004)". `0004-rl-algorithm-and-training-setup.md`: "Two value heads and two advantage streams when an intrinsic reward is on". Both are time-boxed, not withdrawn (Decision 3).
- `0007-training-recipe-pretrained-and-reuse.md`: the "Intrinsic stream" row of the recipe table and three values in its `reward-designer` reuse row (Decision 3).

Tags: **[M]** measured, **[C]** read from code or reports, **[J]** judgement, **[H]** arithmetic by hand from [M] figures, shown in place.

## Context

**The run.** `reports/runs/20261125-nogit-7a675759/`: Stage A0, `my_way_home`, 4 workers x 8 envs, engine reward only, seed 0, 8400 s. 1,814 updates, 10,060 episodes, 3,715,072 decisions. Last 500 episodes: success 0.726, return +0.616. Random policy: 0.140 / -0.052 (200 episodes). Every learning condition of ADR 0009 is met; by thirds success went 0.335 to 0.617. [M] (`reports/runs/2026-11-25-first-doom-run-a0-4x8.md`, `docs/learn/11-reading-the-first-run.md`)

**Throughput.** Sampling 540 to 661 decisions/s by sixths of the run, single-update range 457 to 826; that is 17 to 21 per engine instance. ADR 0009's line is 50 per instance, so the run is "no verdict" as written. The 90 s benchmark of the same layout gave sampling median 743 and end-to-end 520. `trainer-ops` reported before launch that 4 x 4 and 4 x 8 "would trip that rule by construction" and routed it to this role. [M] (`reports/runs/2026-11-25-part2-layout-benchmark.md`)

- Two different corrected lines are in circulation. The run report compares sampling (540 to 660) with half of the end-to-end figure (260). Lesson 11 uses half of the sampling median (about 371). They are different quantities. [C]
- End-to-end rate of the run: 08:04:45 to 10:25:02 is 8,417 s; 3,715,072 / 8,417 = 441 decisions/s; 1,814 / (8,417 / 60) = 12.9 updates per minute. [H]

**The collapse.** Success plateaued near 0.37 to 0.41, fell to 0.176 (episodes 4,500 to 5,000), and recovered to 0.654 without intervention. Summed entropy (maximum 6.22) fell from 5.05 (updates 401-500) to a minimum of 0.122 at update 953, where the turn head was at 0.0019 of a possible 1.95. Entropy fell before success did; the move head fell first (0.58 at updates 501-600 while the turn head was still 1.42). Success recovered to 0.647 (updates 1101-1200) while the turn head was still at 0.015. KL, clip fraction, gradient norm and explained variance stayed inside every ADR 0009 alarm; `return_scale` was 0.18 to 0.19 throughout the collapse. [M]

- Cause not established. Three candidates in the report, none tested: entropy coefficient 0.01 too low for a sparse reward; the turn head locking onto one bin; return-scale interaction. A fourth, mine and equally untested: per-minibatch advantage normalisation turning small value errors into full-size policy gradients while most advantages are near zero. [J]
- What cannot be recovered: only `checkpoint.pt` and `checkpoint.pt.prev` survive, so the collapsed policy cannot be inspected, and no log says which turn bin was chosen. [C]
- Sources in `docs/research/` use entropy coefficients of 0.01 or lower (PPO Atari, Procgen, hide-and-seek, Episodic Curiosity on ViZDoom 0.01; Sample Factory Doom 0.003; RND 0.001). None supports a value above 0.01. [C]

**Reward code.** `rewards/` and `intrinsic/` are wired behind `--shaping my_way_home` (`train/run_doom.py:76`). ADR 0010 departs from standing ADRs in two places and asks three questions. Two wiring facts from `worker`, both confirmed in code [C]:

- On `--resume` with `--shaping`, the anneal starts at `decisions_done // num_envs` (`train/run_doom.py:120-124`, `rewards/factory.py:23-25`). For the first run's checkpoint: 3,715,072 / 32 = 116,096 per-env decisions, above `anneal_decisions` 60,000, and `LinearAnneal` clamps (`intrinsic/schedule.py:22-24`). The coefficient would be 0.05 from the first step and never 0.25. [H]
- With shaping on, `return` in `episodes.jsonl` and `recent_return` in `metrics.jsonl` are the shaped return; the engine return is the separate field `engine_return` (`train/trainer.py:74-88`).

**What exists for experiments.** `--seed` and `--shaping` are flags today. There is no flag or launcher field for the entropy coefficient (`train/run_doom.py:57-78`, `configs/ops_launch.py`), so that arm needs a build. [C]

**Constraints.**

- One laptop is the training box and must stay usable. Measured for 4 x 8 over 2 h 20 min: process tree 2.47 to 2.62 GiB, minimum free memory 3.52 GiB, CPU 27% at the one moment sampled (25 to 31% in the 90 s benchmarks). Temperature and continuous CPU were not measured. [M]
- The approval quoted in the benchmark report covered runs until 11:00 on 2026-11-25. It has expired, and a quotation in a file is not consent. **Every run proposed here needs the user's go-ahead.**
- Deadline unknown ("more than 5 days"). No generated or held-out map exists yet; `reports/eval/` is empty. [C]
- All ADRs remain Proposed.

## Options considered

**1. Throughput "no verdict" line**

| Option | Build | Effect |
|---|---|---|
| T1. Keep 50 decisions/s per engine instance | none | Fails 4 x 4 (39) and 4 x 8 (23) by construction. Forces 4 x 2 at 338 decisions/s end to end against 520: about a third less experience per hour. |
| T2. 50 decisions/s per worker process (`trainer-ops`' proposal) | none | Passes 4 x 8 (186). An absolute number still depends on envs per worker and the timer; no measurement says 50 is the right figure for any other layout. |
| T3 (chosen). Half of the benchmarked rate of the same layout, like for like, checked on both sampling and end-to-end | none; a 90 s benchmark per new layout | Works for any layout. Needs a benchmark to exist before a long run. |

**2. Late collapse: the trigger**

| Option | Build | Effect |
|---|---|---|
| C1. Entropy only: any head under 10% of its maximum at any time, stop | 1 h | Backtest: fires through updates 1101-1300 (turn head 0.015 to 0.053, success 0.59 to 0.65). Would have killed a run that ended at 0.726. |
| C2. Success only: 500-episode success falls 0.15 below its run best | 1 h | Cannot tell a collapse from a harder stretch or noise; gives no warning before success moves. |
| C3 (chosen). Entropy and success together, in three levels (watch, event, stop); flag and continue, stop only when the event persists | 1 to 2 h in `train/run_loop.py`; no throughput cost | Backtest in Decision 2. |

**2b. Late collapse: what to run** (each long run is 2 h 20 min of the laptop)

| Option | Laptop time | What it buys |
|---|---|---|
| X1. Nothing; move on to maps | 0 | The trainer has collapsed once in one run. The same trainer will be used on every later map, where a collapse costs more. |
| X2. Entropy coefficient up now, in the next run | 0 extra | Two changes at once if the next run is the shaped one; no source supports a value above 0.01; one seed cannot separate the effect from luck. |
| X3. Full grid: seeds x entropy x shaping | 14 h or more | Not affordable. |
| X4 (chosen). Log forensics first (no engine), then two runs that are needed anyway and each differ from the first run in one thing, then one conditional entropy run | 4 h 40 min certain, 7 h at most | Decision 2. |

**3a. Bonus summed into one reward stream**

| Option | Build | Effect |
|---|---|---|
| S1. Accept permanently | none | Leaves ADR 0004/0005 contradicted, and a learned bonus (E2) has no bound to rely on. |
| S2. Reject; build the two-stream trainer before any shaped run | about half a day of `worker`, new tests; update cost up slightly (second GAE and value loss, unmeasured) | Changes the trainer and the reward in the same run, on a trainer whose one collapse is unexplained. |
| S3 (chosen). Time-box to named events | none now | Decision 3. |

**3b. Three rejected reward values**: follow 0007-recipe's values (no build; each is farmable by arithmetic), accept ADR 0010's values as settled (no build; none is measured), or accept the rejections and hold the replacements as provisional (chosen).

**4. Next shaped run**: resume the first run's checkpoint (saves the 2.4M decisions it took to reach 0.65, but the bonus would sit at 0.05 on a policy that already succeeds, so it tests nothing ADR 0005's ablation asks) or a fresh start (chosen; 2 h 20 min).

**5. Baseline**: re-measure before every run (451.5 s each time, no new information on an unchanged env), or measure once per env configuration, before launch, and reuse (chosen).

## Decision

### 1. Throughput rule and the verdict for `20261125-nogit-7a675759`

**This replaces the second "no verdict" condition of ADR 0009.** The first (fewer than 1,000,000 decisions) stands.

- **Reference.** Before any long run, a 90 s benchmark of the **same ops config** (layout, torch threads, shaping preset) on the same code gives two figures: `B_sample`, the median of `sample_dps` over updates 3 onward, and `B_e2e`, decisions / (sampling time + update time). No benchmark, no long run.
- **No verdict (throughput anomaly)** when either holds:
  - the median of `sample_dps` over all updates in any 10-minute span of the run is below 0.5 x `B_sample`; or
  - decisions trained / wall seconds over the whole run is below 0.5 x `B_e2e`.
- **On the first condition:** stop, record power source, Windows power mode and step latency, and hand to `debugger`. No learning conclusion is drawn.
- **The "17 per second" anomaly in this form:** 1 x 1 benchmarked at a sampling median of 177, so its line is 88.5; 17 to 18 is far below it. The rule catches the original case and does not depend on layout. [H]
- **A watch, not a verdict:** sampling fell from 661 to 540 across this run (18%), unexplained. A fall of more than 30% from the first 10 minutes' median is reported by `trainer-ops`; it changes no verdict. [J]
- The per-instance figure is retired. T2 is not adopted.

**Applied to the run.** [H]

| Line | Value | Run | Result |
|---|---|---|---|
| Sampling: 0.5 x 743 | 371.5 | lowest single update 457, so no 10-minute median can be below it | clear |
| End to end: 0.5 x 520 | 260 | 441 | clear |

**Verdict: learning, on the training scenario, unstable** (one collapse event that recovered, Decision 2). Not "scenario solved" (0.726 against 0.9). One seed, one map that it trained on, no enemies, doors or keys, no held-out or vanilla-EVAL evaluation. It says nothing about unseen maps.

This correction was written after the run. The defence is that the benchmark report identified the defect in the rule before launch, that the rule as written measured the layout and not a fault, and that the run clears both candidate lines, so the verdict does not depend on which was chosen. The run report's throughput row should cite this ADR and compare like with like (coordinator).

### 2. Late entropy collapse

**Trigger. This extends ADR 0009's first-50-updates clause, which stands, to the whole run.** "Success" is over the trailing 500 finished episodes; "run best" is the highest such value so far; a head's maximum is ln of its option count (1.10, 1.10, 1.95, 0.69, 0.69, 0.69). Thresholds are [J].

| Level | Condition | Action |
|---|---|---|
| Watch | Summed entropy, mean of the last 100 updates, below half its maximum (3.11) while success is below 0.5 | Logged. No action. |
| Entropy warning | Any head's entropy, mean of the last 50 updates, below 10% of its maximum | Logged. Not a fault alone: a converged policy is narrow. The run report must show that head's action histogram. |
| **Collapse event** | Entropy warning **and** (success at least 0.15 below the run best, **or** success below p0 + 0.10) | Logged with update index. The run continues. The verdict carries "unstable". |
| **Stop** | A collapse event still active 600 updates after it began (about 46 minutes at 12.9 updates/min) | Save and stop. Verdict: trainer problem. |

A collapse event that is still active when the run ends is also a trainer problem. The monitor runs in the training loop only; it reads logs and can only request a stop. It is not in the path from observation to action.

**Backtest on the first run** (lesson 11's per-100-update table; block means, so boundaries are approximate) [H]:

| Updates | Turn head (10% line 0.195) | Success | Level |
|---|---|---|---|
| 601-700 | 0.69; sum 1.65 | 0.372 | Watch (sum below 3.11, success below 0.5) |
| 801-900 | 0.21 | 0.219 | Borderline |
| 901-1000 | 0.02 | 0.180 | Collapse event (below 0.41 - 0.15 = 0.26 and below 0.14 + 0.10 = 0.24) |
| 1001-1100 | 0.008 | 0.249 | Collapse event |
| 1101-1200 | 0.015 | 0.647 | Entropy warning only: event over |
| 1201-1300 | 0.053 | 0.585 | Entropy warning only (run best 0.654; line 0.504) |

The event lasts roughly 200 to 300 updates, so the stop would not have fired and the run would have finished as it did. The watch fires about 300 updates before the trough. `evaluator` computes the exact indices (Validation 1).

**Fix requests before the next long run** (`worker`; none changes the training mathematics; throughput cost negligible, to be confirmed in the 90 s benchmark):

- F1. `train/trainer.py` (collection loop near `:151-157`) and `train/run_loop.py:190-195`: per update, write the share of each option chosen for each of the six heads to `metrics.jsonl`. Candidate 2 cannot be tested without it.
- F2. `train/ppo.py:44-46`: log the mean standard deviation of advantages **before** normalisation. Tests the fourth candidate.
- F3. `train/checkpoint.py:58-61`: besides `.prev`, keep every periodic checkpoint under a numbered name (14 were written in the first run). Report the file size; if a run's set exceeds 1 GB, keep every second one.
- F4. `train/run_loop.py`: the monitor above, writing `health_level` and the event's start update to `metrics.jsonl`, and requesting a stop through the existing stop controller on the Stop level.
- F5. `train/run_doom.py:57-78` and `:92`: add `--entropy-coef` (default `PPOConfig.entropy_coef`), printed at start and stored with the run.

**Experiments, in order.** One run at a time, never two at once. Same layout as the first run (4 x 8, 4 torch threads, 8400 s), because it is the only layout whose 2 h 20 min load has been measured. Each run needs the user's go-ahead and a continuous CPU log.

| # | What | Differs from `7a675759` in | Cost | Build needed |
|---|---|---|---|---|
| E0 | Forensics on the existing `metrics.jsonl` and `episodes.jsonl` | nothing | under 1 h, no engine | none |
| E1 | **Shaped, fresh start, seed 0** (`--shaping my_way_home`) | reward only | 2 h 20 min | F1 to F4; one-instance smoke (ADR 0010 validation 2); 90 s benchmark |
| E2 | **Unshaped, seed 1** | seed only | 2 h 20 min | none beyond E1's |
| E3 | Unshaped, seed 0, entropy coefficient 0.03. **Only if a collapse event occurs in E1 or E2** | entropy coefficient only | 2 h 20 min | F5 |

E1 goes first because it is on the delivery path and exercises new wiring; E2 always follows, because without a second unshaped seed no difference in E1 can be told from seed variance. Certain cost 4 h 40 min; at most 7 h (3 x 8,400 s = 25,200 s). After that, no further run is spent on this question without a new ADR.

**E0 outputs** (descriptive, no pass or fail): exact trigger indices; which head fell first, per update; value loss, explained variance, gradient norm and `return_scale` per update over updates 400 to 1200, including the one ADR 0009 trigger lesson 11 left unchecked (a ten-fold jump); success per env and per worker during the collapse; length distribution of successful episodes before, during and after.

**E2 readings** (all in engine return, 500-episode windows):

- Learning with no collapse event: the collapse is 1 unshaped seed of 2. No recipe change; the monitor stays.
- Collapse event that recovers: the instability reproduces. Run E3.
- Stop level, or event active at the end: trainer problem. E3 is mandatory before any other long run.
- Success within 0.1 of p0 with no entropy warning: ADR 0009's "hard task" reading; the first run's learning was seed-dependent and the bonus matters more.

**E3 pass or fail** (0.03 is [J]; three times the current value, above every sourced value):

- Pass: no collapse event **and** last-500 success at least 0.626 (the first run's 0.726 - 0.10). Then 0.03 becomes the provisional base. This is one seed and does not meet ADR 0009 Decision 1's seed-count rule, which was written for toy arms; it is recorded as provisional for that reason.
- Fail, collapse: an event fires. The coefficient is not the lever at this size. No more runs; a new ADR chooses between per-head coefficients and the advantage-normalisation candidate, from F1, F2 and E0.
- Fail, slow: no event but last-500 success below 0.626. Revert to 0.01.

### 3. ADR 0010's two departures and its three questions

**(a) Bonus summed into the single reward stream: accepted, time-boxed to events.**

- **Allowed:** the bounded episodic coverage bonus (E1 of ADR 0005; at most 0.25 per episode against a goal of 1, annealed) on `my_way_home` and on generated multi-room maps **without keys**.
- **Expires at the first of:** (i) the start of the learned-novelty build (E2 of ADR 0005); (ii) the first training run of a key-and-door stage; (iii) run E1 above showing mean explained variance below 0.5 in any sixth of the run (the first run: 0.719 to 0.831). From that point the two-stream trainer of ADR 0004/0005 (change 4 in ADR 0010's list for `worker`) must exist first.
- **ADR 0010's question 2: yes.** For the coverage bonus, bounded-by-construction scaling replaces the running standard deviation of 0007-recipe, in one stream or two: it is stateless and identical across workers and resumes. 0007-recipe's "extrinsic 2, intrinsic 1" is not applied to it; the coefficient alone sets its size. Intrinsic gamma 0.99 stays the starting value for the second stream. The running standard deviation is required for any learned bonus.

**(b) Three values of 0007-recipe rejected as farmable: the rejections are accepted; the replacements are provisional.**

- The arithmetic is correct [H]: +0.005 gained against -0.003 lost pays 0.002 per point for a hurt-then-heal loop; kill +1 equals the exit; 200 landmarks x 0.1 = 20 per episode against an exit of 1. 0007-recipe itself marks kill +1 as "our mapping", not a sourced value.
- The replacements (health +0.002, kill 0.02 under a 0.4 cap, coverage bounded at 0.25) are **unmeasured**. Only coverage is live on `my_way_home`; every other scale in `MY_WAY_HOME_REWARD` is 0. They stand as starting values and are not ratified as tuned.
- **ADR 0010's question 1.** "The exit must out-earn everything else in an episode" is a **hard rule for every configuration whose episodes contain an exit**, including the last curriculum rung, and the config check that enforces it stays. A combat-only pretraining scenario with no exit cannot satisfy it and gets its own preset, in which no repeatable positive term is uncapped; that preset, and any raise of the kill scale or the 0.4 cap, needs a measured entry in a later ADR.
- **ADR 0010's question 3 stands as written:** `FULL_LEVEL_REWARD` is not used in a real run until `env-engineer` has probed whether a campaign exit ends the episode.

### 4. The next shaped run, and which return the verdict uses

- **Fresh start.** Run E1 starts from initial weights, seed 0, with no `--resume`. Resuming `7a675759` with `--shaping` is not to be used for the ablation: the bonus would be at 0.05 throughout (Context).
- **Resume rule (fix request F6, `worker`).** `train/checkpoint.py:44` stores the shaping preset name (or none) in the checkpoint; `train/run_doom.py:120-124` refuses a `--resume` whose `--shaping` differs from the stored one, in either direction. A deliberate change of reward between curriculum stages is made with an explicit flag that states the anneal start in per-env decisions, so it is never implied by an unrelated run's counter.
- **The verdict uses engine return.** For every rule of ADR 0009 Decision 5 and of this ADR: `engine_return` from `episodes.jsonl` when present, otherwise `return`; and `success`, which shaping does not change. The run report states which field was read. `recent_return` is a 20-episode console figure and is never a verdict input in either mode.
- Fix request F7 (`worker`), `train/run_loop.py:48-56`: when `engine_return` is present, also write `recent_engine_return`, so the console figure is not mistaken for the engine return.
- **E1 readings.** Arms are compared at equal decisions, on engine return and success:
  - Wiring: no episode's `engine_return` above 1.0 (a value near 2 is the double-pay bug; stop).
  - Bonus helps: trailing-500 success reaches 0.5 by 1,213,904 decisions (half of 2,427,808, the stamp of the first run's first window above 0.5), **or** last-500 success is at least 0.826; with no collapse event; **and** the margin holds against the better of the two unshaped runs once E2 exists.
  - Bonus hurts: last-500 success at or below 0.626, or time-out share above 0.374 (the first run's 0.274 + 0.10). Then resize `budget_cells` and `coef_start` from the measured `cells_visited` before concluding, as ADR 0010 says.
  - Anything between: not shown at one seed. The bonus is kept for multi-room maps, where the sparse reward is expected to be harder, and resized from `cells_visited`.
- A known confound, stated now: the coefficient falls from 0.25 to 0.05 over the first 1,920,000 decisions (60,000 x 32), which covers the span in which the first run collapsed (about 1.0M to 2.1M). A collapse in E1 cannot be assigned to the reward or to the trainer by E1 alone.

### 5. Random-policy baseline

- **Standing baseline for the A0 TRAIN config on `my_way_home`:** p0 0.140, R0 -0.052 (standard error 0.028), L0 480.2 decisions, 200 episodes (`reports/runs/2026-11-25-random-baseline.json`). [M] Thresholds: success 0.29, return 0.098.
- It is a property of the env configuration, not of a run. It is reused, not re-measured, while scenario, episode limit, frame skip, action space and engine version are unchanged. It is in engine return, so it applies to shaped runs unchanged.
- **For any new scenario or map set the baseline is measured before launch:** 200 episodes of uniform random actions on the exact TRAIN config, reported per map when there is more than one. The ops config names the baseline file, and `trainer-ops` does not launch a long run whose config names none.
- This time it was measured after launch. The verdict stands because the thresholds were fixed in ADR 0009 beforehand and the measurement used the same config. That is recorded as a deviation, not as a precedent.

## Risks

- **Seven hours on one map.** `my_way_home` is one layout that the agent trains on. Nothing here measures generalisation, and no generated or held-out map exists. If the deadline is nearer than assumed, E2 and E3 are the first things to cut, and the collapse stays unexplained. `map-forge`'s generator probe (ADR 0005) needs no long run and must not wait for these.
- **One seed per arm.** Every reading above detects only large effects. Two unshaped seeds is still two.
- **The trigger is fitted to one run.** Its thresholds reproduce one collapse. It may miss a slower one or flag a healthy narrow policy on a map where success is naturally volatile.
- **Entropy 0.03 has no source** and may slow or prevent learning on a sparse reward.
- **A narrow turn head may be a bad habit.** Success of 0.6 with the turn head near zero entropy suggests one turning behaviour repeated. If F1 shows a single non-zero turn bin dominating, the policy may be circling this maze, which would not transfer. Unmeasured.
- **Single stream with a decaying bonus** makes value targets non-stationary; expiry condition (iii) is a coarse guard and is confounded with the collapse question.
- **Post-hoc rule.** The throughput line was corrected after the run it judges.
- **Laptop.** Memory is measured; temperature and responsiveness over hours are not. The watcher protects memory only. Whether to run unattended is the user's call.
- **Rule compliance.** Nothing here touches the path from observation to action: the monitor, histograms and checkpoints are training-side logs; the entropy coefficient is a loss term; position reaches the reward only. If a later change fed `health_level`, a histogram or `info["shaping"]` into an observation or an action choice, it would breach `docs/RULES.md`. Shaping stays refused on a vanilla-eval config.

## Validation experiment

1. **E0, log forensics** (`evaluator`, under 1 hour, no engine): the outputs listed in Decision 2, written to `reports/eval/`. Confirms or corrects the backtest table.
2. **Builds F1 to F7** (`worker`), with tests: histogram shares sum to 1 per head; the monitor reproduces the backtest levels when fed the first run's `metrics.jsonl` and `episodes.jsonl`; a mismatched `--resume` / `--shaping` is refused; numbered checkpoints load with `weights_only=True`.
3. **Smoke and benchmark** (`trainer-ops`): ADR 0010's one-instance shaped smoke, then a 90 s benchmark of the E1 config giving `B_sample` and `B_e2e`.
4. **Runs E1, E2 and, if triggered, E3** (`trainer-ops`), each with the user's go-ahead, a continuous CPU log, and a report that states the verdict under this ADR, the field read for return, every health level reached, and the action histograms at the lowest-entropy point.

Reverse or amend if:

- E0 shows the backtest is wrong by more than 100 updates on the event's start or end, or that the stop would have fired: re-set the thresholds before F4 is relied on.
- The shaped 90 s benchmark is below 0.9 x 520 end to end: the wrapper costs throughput and ADR 0010's integration is reviewed before E1.
- E2 learns with no event **and** E1 shows none: the collapse is 1 run of 3; Decision 2's experiments end there and the monitor is all that remains.
- E3 passes: 0.03 becomes the provisional base in a superseding entry; E3 fails by collapse: the entropy-coefficient candidate is closed.
- E1 trips expiry condition (iii): the two-stream trainer is built before any further shaped run.
- The user gives a deadline that leaves fewer than 3 days: E2 and E3 are dropped and the time goes to maps and the pixel branch.
