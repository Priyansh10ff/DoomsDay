# ADR 0009: Reading the ADR 0008 toy results, the memory plan, and criteria for the first DOOM run

Date: 2026-11-26. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor). Number allocated by the coordinator.

**Amends** `0008-reconcile-0007-files-memory-gate-and-safety-defaults.md` (its arm-adoption rule, the scope of Measurement B, and its reversal clause about Stage B) and restates the Stage B gate of `0006-env-probe-amendments.md`. No earlier file is edited.

Tags: **[M]** measured, **[C]** read from code or reports, **[J]** judgement.

## Context

Results of ADR 0008's toy runs (`reports/runs/2026-11-25-adr0008-toy-gate-and-memory.md`; toy network core 64, 16 envs, rollout 32, lr 1e-3, gamma 0.99, floor 0.1, 3 epochs, built init), 8 seeds per arm unless stated:

- **Gate A** (gap 4, 100 updates): recurrent 8/8 at or above 0.995; memoryless 0.45 to 0.56. Passed. The block on single-instance DOOM runs is lifted. [M]
- **Gap 20, 300 updates:** B0 base 8/8, B1 update-gate bias 8/8, B2 bias plus dense cue label 8/8, E (2 epochs) 8/8, I (orthogonal init) 8/8. Every arm is at the ceiling. [M]
- **Gap 60, 300 updates:** B1 0/8 (0.46 to 0.55); B2 0/4 (0.445 to 0.52). B0, E and I were not run at gap 60; B2 seeds 4 to 7 were not run. [M]
- The update-gate bias raises the cue's trace in the hidden state at step 20, at initialization, from 1.3e-6 to 4.6e-4. [M]
- The floor comparison was cut by the compute budget: only floor 0.01 on two seeds ran. Floor 0.1 against 1.0 at equal settings is still unmeasured.
- The worker changed no defaults; in `configs/model_config.py` both arm switches are off. [C]

The worker found a contradiction in ADR 0008: one sentence adopts E and I when their count is "not lower than B0's", the next says "Ties keep the base", and the reversal clause says "beats B0 by 2 seeds or more".

ADR 0008 also said: if B2 does not beat B1, the keys head is not established as a memory mechanism and Stage B's priority must be weighed against the `viz_nocheat` hedge in a new ADR. That outcome occurred. **This is that ADR.**

Facts bearing on the first DOOM run (`reports/runs/2026-11-25-a0-single-instance-smoke.md`, `docs/reference/engine-probes.md`, `train/run_loop.py`, `env/config.py`):

- `my_way_home`: -0.0001 per tic (-0.0004 per decision), +1 at the goal, 2100-tic limit (525 decisions), 17 random start points with random heading. A full time-out returns about -0.21. One untrained smoke episode reached the goal (return 0.954), so the random baseline is not zero. [M]
- Single-instance sampling has measured 119 to 187 decisions/s, about 200 in a quiet run, and **17 to 18 decisions/s in the latest smoke run, cause undetermined.** The bundled TRAIN config does set the timer request. [M], [C]
- At 1 x 1, system available memory was 3.3 to 3.45 GB; the run stops itself below 1.5 GB. [M]
- **No real DOOM time-out has yet fired** through the trainer; every stale count of zero so far is vacuous. [C]
- The value scale sat at its 0.1 floor in the smoke runs. The first +1 reward in a real run is the case the unfinished floor toy was meant to cover. [M]
- The run loop writes a 20-episode `recent_return` per update; per-episode results are held in memory only. [C]

Standing constraints: one laptop, the user's load instruction, deadline unknown ("more than 5 days"), organizers' answers unknown. The user's standing approval to run without asking lasts until 11:00 today.

## Options considered

**Adoption rule after a tie**
- T1. Adopt on a tie. No build. Lets any no-effect change into the recipe.
- T2. Tie keeps the base, always. No build.
- T3 (chosen). T2, plus: a comparison in which the base is already at the ceiling is void, and a change wanted for cost reasons needs a pre-declared cost measurement.

**What to do about gap 60**
- P1. Lengthen the backpropagation window (rollout 128 or more). Config only; update cost and rollout memory roughly double per doubling. Cannot reach minutes: a minute is about 525 decisions.
- P2. Build a new memory mechanism now (slot memory, or Stage B first). 1.5 to 2 days. Built on a mechanism (dense labels crossing rollout boundaries) that just failed to show itself, and Stage B has no input under `viz_nocheat`.
- P3. Stronger auxiliary objective immediately (keys weight up, more labels). Hours. Same unverified mechanism.
- P4 (chosen). A 45-minute toy diagnostic that separates window, budget and label weight, with readings fixed in advance; meanwhile design the agent so that facts needed over minutes are **observed or held in structure**, not recalled through gradient; stage order unchanged. No effect on training throughput.

**First-run verdict**
- V1. Judge by eye from the console. No build; a 20-episode mean has a standard error near 0.1.
- V2 (chosen). Measure a random baseline first, write per-episode results to disk, and judge on windows of 500 episodes against thresholds stated here. About 1 hour of build and 10 minutes of single-instance compute.

## Decision

### 1. One adoption rule [J]

This replaces three sentences of ADR 0008: "An arm's change is adopted ... only if its count at gap 20 is not lower than B0's (for E and I) or is higher (for B1, B2)", "Ties keep the base", and "arm E or I beats B0 by 2 seeds or more (adopt it)".

**A change replaces the base only if, at a setting declared before the runs, it scores at least 2 seeds out of 8 more than the base. Anything less keeps the base. The rule is the same for every arm.** If the base scores 7 or 8 of 8 at that setting, the comparison is **void**: the arm is "not adopted, not refuted". A change wanted because it is cheaper, not better, is adopted only on a DOOM measurement in which the update takes more than 30% of wall time, the change cuts that by at least a quarter, and return over equal decisions is not lower.

Applied: **E (2 epochs) and I (orthogonal init) are not adopted.** Both comparisons are void. Fix request F5 of `0007-training-recipe-pretrained-and-reuse.md` is closed as not adopted. 2 epochs may return under the cost clause once multi-instance update time is known.

### 2. B1 and B2 [M] for the scores, [J] for the ruling

- **B1 (update-gate bias): not adopted.** Void comparison at gap 20; no effect at gap 60. It does what it was designed to do at initialization (trace up about 340 times) and that did not change any outcome measured. The switch stays in the code, off.
- **B2 (dense cue label):** there is no default to adopt; the keys head already exists. What matters is the finding: **a dense label at weight 0.1 did not produce a memory across the rollout boundary in 300 updates. The keys head is not established as a memory mechanism.** Its weight stays 0.1 until the diagnostic in Decision 3 reports.

### 3. What gap 60 means, and what happens next

**What was measured:** at gap 60 the agent is at chance within 300 updates for B1 and B2. **What was not:** why. Gap 60 gives about a third as many episodes per update as gap 20; the base arm was not run; whether the keys head could decode the cue past step 32 was not logged. So the result does not yet distinguish "the window is the limit", "the budget was too small" and "the label was too weak". [M] for the scores, [C] for the gaps

**Is the toy a fair model of key-then-door?** Partly. It is harsher than DOOM in three ways (the cue is visible for one step among larger noise; there is no intermediate signal of any kind; DOOM lets the agent see a door again when it returns near it) and easier in others (two cues, no distractors, fixed gap). **The planning conclusion does not depend on that fairness:** key-to-door delays are minutes, which is thousands of decisions against a 64-decision window, and the agent has now failed at 60. The plan must not rest on unlabelled recall through the GRU. [J]

**Design consequence: observe it or store it in structure.**
- **Key possession is to be observed, not remembered.** The status bar shows held keys, so with `screen` on the fact is in the observation at every step, and it survives `viz_nocheat`. This is Stage A1's job. Its legibility at 60 x 80 is so far a judgement from looking at images. **Test:** a small supervised classifier from `screen` to the three keys-held bits, trained on frames from some maps and tested on others, at least 0.95 accuracy per bit; if it fails at 60 x 80, add the full-resolution status-bar crop that ADR 0001 reserved.
- **Door location belongs to the map memory of Stage B**, with dense training-only labels from maps whose door positions we know.
- **No longer window as a memory fix.** Rollout 64 stays.

**Toy diagnostic, fixed now** (`worker`, about 45 minutes, one process, 4 threads; gap 60, seeds 0 to 3, 300 updates unless stated; every run logs keys-head accuracy by episode step):

| Arm | What it is | What it separates |
|---|---|---|
| D0 | Base | The missing baseline |
| D1 | B2 with keys weight 1.0 | Label weight |
| D2 | B1 with rollout 128, 150 updates | Window: the whole episode fits in one window |
| D3 | B2 with 900 updates | Budget |
| D4 | Supervised only: dense cue label, truncated windows of 32, no RL | Whether a dense label can carry a fact across a window boundary at all |

Readings, declared in advance:
- D4 fails: dense labels do not cross boundaries in this design. The keys-head plan and Stage B's auxiliary plan both need redesign before either is relied on.
- D4 passes and the keys head decodes the cue past step 32 in D1 or D3, but the answer is at chance: the memory is there and the policy does not read it; that is a budget or exploration matter.
- D4 passes and the keys head does not decode past step 32 in B2-like arms: weight or interference; D1 against the earlier B2 shows which.
- Only D2 passes: memory is limited to the window, which confirms the design consequence above.
- D3 passes: 300 updates was too few; Measurement B's budget is revised in a later ADR, not here.

**Weighing Stage B against `viz_nocheat`** (the weighing ADR 0008 called for). [J]
- For moving B earlier: A0 has no demonstrated memory beyond 20 decisions, and door location has no other home.
- Against: under `viz_nocheat` Stage B has no input at all; its auxiliary supervision relies on the same mechanism that B2 just failed to show, so building it before D4 reports would be building on an unverified base; and with the screen on, the key half of the problem disappears.
- **Ruling: the order A0, A1, B stands.** Stage B's gate, restated in full: (i) D4 passes; (ii) the organizers confirm telemetry at evaluation, or there is no answer and at least 2.5 days remain; (iii) ADR 0003's exit-rate condition on A1.
- If the organizers confirm `viz_nocheat` is on, the route to a map memory is ego-motion estimated by the network from pixels and previous actions, supervised by true ego-motion as a training-only label. That is a new ADR, not decided here.

### 4. Measurement B's criterion [J]

- **The pass rule stands (7 of 8 seeds at or above 0.9), and Measurement B as a whole is still failing**, because its gap-60 leg fails. Gap 20 passing on every arm at 300 updates is recorded as passed and as no longer able to separate arms.
- **For future arm comparisons** the declared sensitive setting is gap 20 at **100** updates, 8 seeds, count at or above 0.9, used with the rule of Decision 1. The base has not been measured there at floor 0.1; that run comes first.
- **The blocks from ADR 0008 stay:** no key-and-door curriculum stage and no claim of long-horizon memory. They are lifted only by a named measurement: gap 60 passing the rule on some arm; **or**, for an agent with `screen` on, the keys-held read test of Decision 3 passing, in which case a key-and-door stage may start for that agent while every claim about remembering door locations stays barred until ADR 0003's memory probe is run on DOOM maps. This second route narrows ADR 0008's block and needs the user's explicit agreement.
- The floor stays 0.1. The base passed 8/8 with it at gap 20 [M]; 0.1 against 1.0 remains unmeasured and is low priority.

### 5. The first DOOM training run

**Before launch**
1. **Random baseline** (`env-engineer`, one instance, about 10 minutes): 200 episodes of uniform random actions on the TRAIN config. Record success rate p0, mean return R0, mean length L0.
2. **Per-episode results on disk** (`worker`, about 1 hour): one line per finished episode with return, length, how it ended (goal, time-out, death) and update index. **A 2.5-hour run whose per-episode results exist only in memory should not be launched.**
3. **Layout sized to memory, not to ADR 0006's 4 x 8.** The layout's estimated memory must leave at least 2 GB of system available memory at launch (the 1.5 GB stop plus margin). With about 3.4 GB available, 4 x 8 (about 2.6 GB by ADR 0006's figures) does not fit; 2 x 8 (about 1.3 GB) is at the edge. `trainer-ops`' benchmark decides, and closing other applications changes the answer.
4. **Approval window.** A 2.5-hour run started this morning may run past 11:00. The coordinator either sets the run's time limit to end by 11:00 or gets the user's agreement to continue.

**What counts as learning** (windows of the last 500 finished episodes; success means the episode ended at the goal)
- **Learning:** success at least p0 + 0.15, mean return at least R0 + 0.15, and success in the last third of the run above the first third.
- **Scenario solved:** success at least 0.9 and mean length of successful episodes under half of L0.
- **No verdict:** fewer than 1,000,000 decisions in total (about 110 decisions/s sustained over 2.5 hours), or sampling below 50 decisions/s per engine instance for more than 10 minutes. The second condition is the 17-per-second anomaly; stop, record machine state, and hand to `debugger`.

**A hard task, not a trainer problem:** success stays within 0.1 of p0 while the trainer is healthy: approximate KL mostly between 0.003 and 0.03, no head's entropy collapsed, losses finite, memory flat. The reward is sparse and there is no exploration bonus; a flat curve here is a known outcome for plain actor-critic on sparse-start variants of this map (`docs/research/exploration.md`). The response is ADR 0005's episodic coverage bonus, not debugging.

**A trainer problem:** any of the following.
- Non-finite loss or gradient norm.
- Approximate KL above 0.05, or clip fraction above 0.5, for 20 consecutive updates.
- Any head's entropy under 10% of its maximum in the first 50 updates without success rising; or success falling more than 0.1 below p0 and staying there.
- All entropies at maximum with KL near zero after 100 updates.
- First-minibatch ratio deviation above 1e-4.
- `return_scale` more than doubling between consecutive updates after update 50; value loss or pre-clip gradient norm jumping by more than ten times when the first goal rewards arrive and not settling within 20 updates.
- Explained variance below -1 for 50 consecutive updates.
- Process-tree memory rising steadily across the run.

**Reading the stale time-out count.** Most early episodes will time out, so the first ten minutes are the first real test of that path. Zero with time-outs occurring is the expected result. Non-zero means the engine's own limit fired before the wrapper's; the fallback is correct but drops one transition per time-out. Do not abort: record it and send it to `env-engineer` after the run.

**Watch during the run:** success, return and length over 500 episodes; share of episodes ending by time-out; sampling and update decisions/s; approximate KL; clip fraction; entropy per head; explained variance; `return_scale`; value loss; gradient norm; stale time-outs; process-tree and system available memory.

**After the run:** learning: keep the checkpoint and move to the baseline comparison with the coverage bonus. Flat and healthy: build the coverage bonus and rerun. Trainer problem: `debugger`. ADR 0004's expectation of "learning within the first hour" is withdrawn; nothing measured supports it.

## Risks

- **The diagnostic may be inconclusive at 4 seeds.** Arms near chance or near ceiling read clearly; anything between needs more seeds.
- **Observed-not-remembered depends on pixels.** If the status bar is not rendered by the judges' harness, or the icons are illegible, key possession falls back on the unproven keys head.
- **Stage B stays third** while A0 has no long memory. If telemetry turns out to be allowed, a day was spent on a hedge; if it is not, Stage B as designed was never usable. The ruling accepts the first cost to avoid the second.
- **The key-and-door route through the read test** allows a stage to start without any demonstrated door memory; the agent may only learn doors that are visible from the key.
- **First-run thresholds are set before any DOOM data exists** and may be mis-sized; p0 could be high enough on easy start points that +0.15 is reached by learning those alone. Per-start-point statistics are not available.
- **`my_way_home` is one layout.** Success there says the pipeline works. It says nothing about unseen maps (rule: never validate on a single map), and it is evaluated on the map it trained on.
- **Laptop.** A multi-instance run for hours is the heaviest load so far. The memory rule above protects memory, not temperature.
- **Rule compliance.** Nothing here adds logic between observation and action. The keys-held read test and any door labels are training-only; a classifier output must never be fed to code that chooses actions.

## Validation experiment

1. **Toy diagnostic D0 to D4** as specified in Decision 3, results as per-seed tables with keys-head accuracy by step, in `reports/runs/`.
2. **Base at the sensitive setting** (gap 20, 100 updates, 8 seeds), about 5 minutes, so later arm comparisons have a baseline.
3. **Random baseline and the first run** as specified in Decision 5, with a run report giving p0, R0, L0, the 500-episode curves, the health metrics, the stale count, throughput and memory, and which of the four verdicts applies.
4. **Keys-held read test** when Stage A1's pixel branch exists.

Reverse or amend if: D4 fails (the keys head and Stage B's auxiliary supervision are redesigned before use, new ADR); D2 alone passes and nothing else does (consider rollout 128 only as a measured DOOM arm, with its update cost); D0 passes at gap 60 (the bias or label hurt, and Decision 2 becomes a finding against them); the base scores 7 or more of 8 at the sensitive setting (pick a harder one before comparing arms); the first run is "no verdict" on throughput (no learning conclusion is drawn and the anomaly is fixed first); the random baseline p0 is above 0.5 (the thresholds here are replaced by ones on mean length to goal); the organizers answer on `viz_nocheat` (Stage B's gate is settled either way).
