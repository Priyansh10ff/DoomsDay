# ADR 0008: Reconciling the two 0007 files, the toy memory gate, GRU memory, and safety defaults

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

**This ADR supersedes both of the following files on every point listed in Decision 1, and nowhere else:**

- `docs/decisions/0007-training-recipe-pretrained-and-reuse.md` (called **0007-recipe** below)
- `docs/decisions/0007-a0-trainer-rulings-and-first-run-prerequisites.md` (called **0007-rulings** below)

It also changes one default from ADR 0006 (the 4 x 8 layout is no longer a default) and replaces the toy-gate wording of ADR 0004 and of both 0007 files. No existing file is edited. **The next free ADR number is 0009.**

Each ruling is tagged with what it rests on: **[M]** measured on this machine, **[C]** reading the code or reference notes, **[J]** judgement.

## Context

**The collision.** Two advisor runs each wrote an ADR numbered 0007. 0007-rulings (mine) was written without re-listing the folder first; that is how the duplicate got through. The two files disagree on several trainer settings, and the code follows neither on two of them.

**The gate.** ADR 0004 required a toy memory task that "only a correct recurrent implementation solves". 0007-rulings accepted a single-seed pass (0.985) and asked for three seeds. Measured since (`reports/runs/2026-11-25-a0-toy-three-seeds.md`, `reports/runs/2026-11-25-toy-memory-root-cause.md`; toy network, core 64, lr 1e-3, gamma 0.99, 3 epochs, torch-default init):

| Configuration (cue-to-answer gap 20 unless stated) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Floor 1.0, value_coef 0.5, 100 updates (0007-rulings) | 0.605 | 0.485 | 0.495 |
| Floor 0.01, value_coef 0.5, 100 updates (before 0007-rulings; 0007-recipe's value) | 0.870 | 0.745 | 0.690 |
| Floor 1.0, value_coef 2.0, 100 updates | 0.960 | 0.645 | 0.555 |
| Floor 0.01, value head detached from the GRU | 0.510 | 0.470 | 0.495 |
| Floor 1.0, 300 updates | 0.935 | 0.970 | 0.495 |
| **Gap 4**, floor 1.0, 100 updates | 1.000 | 0.995 | 1.000 |

Further measurements by the debugger: at initialization the GRU's update gate is about 0.5, so the cue's trace in the hidden state halves every step (2.5e-2 at step 0, 1.5e-6 at step 20). Supervised training of the same network with a loss only at the answer step stays at chance on three seeds; with a loss at every step it reaches 1.00 on three seeds by step 200. Auxiliary losses off made no difference, but the toy's auxiliary targets are near-constant, so that arm tested nothing about auxiliary supervision.

What this establishes: **no trainer bug was found; the earlier 0.985 was a favourable seed; the floor of 1.0 that 0007-rulings introduced measured worse on all three seeds; and in this network a long-gap memory forms only through a signal that is present at every step (here, the value loss), with an onset that depends on the seed.** One discrepancy between the two reports is unresolved: the return scale on the toy is given as about 0.29 in one and about 0.5 in the other.

**Safety facts read from the code.** `RunConfig()` defaults to 4 workers x 8 environments and `train/run_doom.py` takes its defaults from it, so `python -m train.run_doom` with no flags starts 32 engine instances. The memory cap is checked once per update (`train/run_loop.py` 68-83). An interrupt during an update saves whatever the weights are at that moment (`train/run_loop.py` 97-99). `train/subproc_vec_env.py` line 92 starts daemon workers; the subprocess tests do not appear to start a real engine. `train/checkpoint.py` line 42 loads with `weights_only=False`. `TRAIN_CONFIG` now points at the bundled `my_way_home` scenario, so the held-out conflict flagged in both 0007 files is fixed.

**Standing constraints.** One laptop; the user's instruction not to load it until it hangs or shuts down; **no go-ahead for any multi-instance run appears in what I have been shown**; deadline unknown ("more than 5 days"); organizers' answers unknown.

## Options considered

**Reconciling the recipe**
- R1. Take 0007-recipe wholesale. An hour of config work. Changes epochs and initialization at the same moment the gate is failing, so no measurement so far would describe the resulting trainer.
- R2. Take 0007-rulings wholesale. No work. Keeps a floor that measured worse and ignores cheap, sourced details.
- R3 (chosen). One base recipe equal to what the toy measurements were taken on, with each disputed change admitted only as a single-change arm in the gate runs. About 2 hours of toy compute on 4 threads, one process, no engine.

**The gate**
- G1. Keep "gap 20, 100 updates, every seed at 0.9". It fails for reasons that are not trainer bugs, so it cannot serve as a correctness test, and with three seeds it cannot tell a 60% reliable learner from a 100% one.
- G2. Replace it with gap 4. Passes today. Alone, it would hide the memory weakness.
- G3 (chosen). Two tests with different jobs: a correctness gate at a gap the network can learn, and a memory-capability measurement at gaps of 20 and beyond the rollout, with criteria fixed here before any arm is run. Throughput effect on training: none.

**Making memory not depend on the value loss alone**
- M1. Do nothing; rely on value-loss strength (floor or coefficient). No build. Leaves the seed-dependent onset.
- M2. Initialize the GRU's update gate toward keeping state. About 1 hour with its test. No throughput cost.
- M3. Dense auxiliary labels for facts that must be remembered. Hours per label; needs the label to exist in training data. Negligible throughput cost.
- M4. Rescale or normalize the fused input so a small cue is not drowned. Half a day; changes every encoder's contribution. Deferred.
- Chosen: M2 and M3, each as a measured arm; M1's floor question settled separately; M4 deferred.

**Safety default**: leave 4 x 8 as default with a warning (no work, one typo from 32 engines), or default to one instance and require an explicit, counted flag for more (about 1 hour; chosen).

## Decision

### 1. Reconciliation: one ruling per point

Base recipe for all gate runs and for the first DOOM runs: **what the toy numbers were measured on**, with the floor corrected (point 3).

| # | Point | 0007-recipe | 0007-rulings | Code now | Ruling | Basis |
|---|---|---|---|---|---|---|
| 1 | Learning-rate schedule | Linear 3e-4 to 3e-5 over a declared budget of decisions; continuation runs at constant 1e-4 | Constant 3e-4; decay only in a final fixed-length run, floor 0.1 | Constant | **Constant 3e-4.** The budget a schedule would be defined over is unknown until throughput is measured. When a final fixed-length run uses decay, it is defined over decisions, not updates, and its position is saved in the checkpoint (taken from 0007-recipe). The 1e-4 continuation value is not adopted as a rule. | [J]; toy runs are budget-sensitive [M] |
| 2 | Epochs | 2 | silent (3) | 3 | **3 stays in the base**; 2 is a single-change arm. Every measurement so far is at 3. | [J] |
| 3 | Value-scale floor | 0.01 | 1.0 | 1.0 | **0.1.** The 1.0 ruling is withdrawn: it measured worse on 3 of 3 seeds. Its motive (a first sparse reward becoming a target of 100) came from reading code and was never measured. 0.1 keeps amplification of small returns but bounds it at ten times. On the toy 0.1 and 0.01 should behave the same, since the measured scale is above both; that is to be confirmed. | withdrawal [M]; the value 0.1 [J] |
| 4 | Value coefficient | 0.5 | 0.5 | 0.5 | **0.5.** Raising it to 2.0 at floor 1.0 helped one seed of three. | [M], weak |
| 5 | Initialization | Orthogonal everywhere (sqrt 2 hidden, 1.0 GRU, value and auxiliary heads, 0.01 policy heads), biases 0 | As built: policy heads 0.01, the rest torch default | As built | **As built in the base; 0007-recipe's orthogonal scheme is a single-change arm.** Whichever is adopted, the GRU update-gate bias of Decision 3 overrides "biases 0". | [J]; sources are feedforward MuJoCo |
| 6 | Worker processes | Non-daemon, with close message, `try/finally`, `atexit`, workers that exit when the parent's pipe closes | Silent; requires guaranteed shutdown | Daemon, `atexit` close, kill of recorded descendants | **Decided by a test, not by recall:** at 1 x 1 with a real engine, kill the parent process hard and count surviving `vizdoom.exe` after 15 s. Target design is non-daemon workers that also poll for a dead parent and then close their games. The daemon form stays only if it leaves zero survivors and the non-daemon form does not. | [C]; neither file measured it |
| 7 | Checkpoint loading | `weights_only=True`, plain types only | Silent | `weights_only=False` | **`weights_only=True`.** The submitted checkpoint will be loaded by other people. A save-and-load round-trip test decides which stored types pass; anything that does not (tuples, NumPy RNG state) is converted to lists, numbers and strings. No broad allow-listing and no return to `False`. | [C] |
| 8 | Keys auxiliary weight | 0.1 in A0; 0.5 with positive-class weighting when a key stage starts | 0.1, marked for an experiment | 0.1 | **0.1 in A0. 0.5 with class weighting is the starting arm of that experiment**, not a settled value. | [J] |
| 9 | Monster-count target | As built (count/24) | min(count, 8)/8 | Capped at 8 | **Capped at 8 stays.** The single toy seed that moved when the cap changed is noise-level evidence in either direction. | [J] |
| 10 | Toy test design | Gap 40, rollout 32 against 64 | Three seeds at gap 20 | Three seeds at gap 20 | **Both replaced by Decision 2.** | [M] |
| 11 | Default layout | 4 x 8 | 4 x 8; first launch 2 x 4 | 4 x 8 | **1 x 1.** See Decision 5. | [J], user's instruction |
| 12 | First DOOM runs | Five recipe arms on two scenarios, 2M decisions each | Smoke run, then one hour on `my_way_home` | none run | **In sequence:** gate A, single-instance smoke, then (with the user's go-ahead) the multi-instance smoke, then 0007-recipe's arms. | [J] |

Points where only one file ruled, and which **stand unchanged**: from 0007-recipe, the pretrained-component rulings, the ray-head plan for A1, exit detection by two agreeing signals, no use of `gymnasium.vector`, the turn dead-time ruling, reward-scale guidance, no value-loss clipping, and its fix requests; from 0007-rulings, layer norm kept, `value_int` kept as a stub, the time-out table, the damage death-window fix, the coupled-index test, and the EVAL fields with the single `+viz_nocheat 1` exception. Points on which the files already agree (Adam epsilon 1e-5, freshness flag per step, no training on Freedoom Phase 2) need no ruling. 0007-recipe's note that the horizon row may be 52 and not 50 remains open with `env-engineer`.

**Preventing number collisions.** (a) ADR numbers are allocated by one party: the coordinator names the number and filename in the request, as it did for this file. (b) An advisor run lists `docs/decisions/` immediately before and immediately after writing, and reports a duplicate instead of continuing. (c) A test fails when two files in `docs/decisions/` share a four-digit prefix; the two existing 0007 files are listed as the one known exception. (d) ADRs are cited by full filename whenever a number is ambiguous. Only one advisor run should be writing decisions at a time; (c) is the backstop if that fails.

### 2. The gate

Two tests. Lag, budget, seeds and pass rule are fixed here, before any arm runs. Toy settings otherwise as in the reports (16 envs, rollout 32, lr 1e-3, gamma 0.99, core 64). Seeds 0 to 7.

**Gate A: trainer correctness. Blocks all DOOM training.**
- Gap 4, 100 updates, base recipe (floor 0.1, 3 epochs, built init), 8 seeds.
- Pass: every recurrent seed at or above 0.95; every memoryless seed within 0.1 of 0.5; first-minibatch ratio deviation below 1e-5 in every run.
- About 10 minutes of compute.
- **What it is evidence of:** hidden state carried across rollouts, reset masks, sequence replay, GAE and the update are consistent, on a task the network can represent. It passed 3 of 3 at floor 1.0 [M] and must be rerun on the base recipe. **What it is not evidence of:** any ability to remember over long gaps, the truncation path (covered by unit tests and the smoke run in Decision 4), or any hyperparameter choice for DOOM.
- This is a different question from the old gate, not a lower bar for the same one; the old question is kept below, where it currently fails.

**Measurement B: memory capability. Does not block the first pipeline runs; blocks the start of any key-and-door curriculum stage and any claim about long-horizon memory.**
- Gaps **20** (inside the 32-step rollout) and **60** (longer than the rollout, so cue and answer never share a backpropagation window, as with a key found hundreds of decisions before its door).
- 300 updates, 8 seeds per arm.
- Pass rule per arm and gap: **at least 7 of 8 seeds at or above 0.9.** For calibration: a learner that truly succeeds on 60% of seeds passes this rule about 1 time in 10; one that succeeds on 90% passes about 8 times in 10. Eight seeds is still coarse and the result is reported as counts, not as a certificate.
- Arms, one change each from the base: **B0** base; **B1** update-gate bias (Decision 3); **B2** B1 plus a dense cue label (Decision 3); **E** 2 epochs; **I** orthogonal initialization. Gap 20 for all five; gap 60 for B1 and B2. About 80 minutes of compute.
- **Current status: failing.** The nearest measured configuration (floor 1.0, 300 updates) scored 2 of 3.
- **What it is evidence of:** whether this network and recipe can form a memory over the stated gap from a sparse answer-time reward, with or without a dense label. A toy with two cues and no distractor events is a lower bound on difficulty, not a model of DOOM.
- An arm's change is adopted into the recipe only if its count at gap 20 is not lower than B0's (for E and I) or is higher (for B1, B2). Ties keep the base.

**A third, small measurement for the floor:** gap 4 with rewards multiplied by 0.05, floors 0.01, 0.1 and 1.0, 3 seeds each (about 6 minutes). This is the only case where the three floors differ on the toy, and it is the case the 1.0 ruling worried about.

**The block from 0007-rulings** ("any seed fails, block DOOM training") is replaced: gate A passing on the base recipe lifts it for single-instance runs. Multi-instance runs additionally need the user's go-ahead (Decision 5).

### 3. Memory that does not depend on the value loss alone

- **Update-gate bias (arm B1).** Initialize the GRU cell's update-gate bias so that units start with retention times spread between about 2 and 256 decisions, instead of all near 2. This is an initialization only; every parameter stays trained. `worker` must establish the direction and the position of the update gate in the bias vector **by test, not from documentation recall**: the test is the debugger's own metric, the cue's trace in the hidden state at step 20 at initialization, which is 1.5e-6 today and must rise by orders of magnitude. The idea is from memory of the recurrent-initialization literature and is not in `docs/research/`; treat it as our own until someone sources it. [J], diagnosis [M]
- **Dense labels for facts that must persist (arm B2).** The supervised result (loss at every step learns; loss at the answer step does not) is the strongest measurement in the report, and the keys-held head is exactly such a label: present at every step after pickup. The toy gets a "cue identity" auxiliary head read from the GRU state at every step, to measure whether a dense label builds the memory inside PPO, including across rollout boundaries. If B2 passes where B1 does not, the keys head is confirmed as the mechanism for key memory and its weight experiment (point 8) becomes urgent. [M] for the supervised result; its transfer to PPO is what B2 measures
- **Not changed:** the value head keeps its gradient into the GRU (detaching it removed all learning [M]); input scaling is deferred.

**What this implies for Stage A0 and Stage B.** [J], from a toy
- A0 should be expected to learn short-reaction behaviour and, with the keys label, to hold "I have the blue key". It should **not** be expected to remember where a locked door was seen. No dense label for door location exists: the probes never located a locked door, because the engine API does not expose line specials. Without one, the only route is the value loss after reward already correlates with the memory, which is the seed-dependent onset seen in the toy.
- Stage B is therefore needed **for door-location memory specifically**, and before any curriculum stage in which the door is seen before the key. Its design gains two requirements: unvisited cells persist unchanged by construction, and its write gate starts biased toward keeping, for the same reason as B1.
- **ADR 0006's stage order (A0, then A1, then B) is not changed.** Stage B still has no input under `viz_nocheat`, so the pixel branch remains the more urgent hedge. With the screen on, the status-bar key icons make key possession visible without memory, which narrows the memory problem to door location in that configuration too.
- A dense "locked door seen" label requires maps whose door positions and colours we know. That raises the priority of `map-forge`'s own generator probe (ADR 0005), since generated maps can carry that metadata as training-only labels.

### 4. Worker's questions

- **Resume re-seeding the environments with the same seed: not acceptable as the default.** On a machine where runs are stopped and resumed often, every session would begin with the same episodes. On resume, environments are reset with a seed derived from the run seed and a session counter saved in the checkpoint; the first session is unchanged. Whether `set_seed` on a running game affects the next episode at all is unprobed (`docs/reference/vizdoom-api.md`); `env-engineer`'s same-seed probe decides whether this ruling has any effect, and it is harmless if not. [C], [J]
- **A smoke run to exercise a real DOOM time-out: yes, and it is exempt from the gate block**, because it tests plumbing and trains nothing of value. One engine instance, 2 torch threads, at most 5 minutes, with a short episode limit so several time-outs occur. It must record which time-out path fired (the wrapper's own limit, which gives a fresh final observation, or the engine's, which gives a stale one), the stale counter, and that a bootstrap value was computed. [J]

### 5. Safety defaults

- **Default layout is one engine instance:** `RunConfig` defaults to 1 worker x 1 environment and 2 torch threads. ADR 0006's 4 x 8 becomes the recommended layout to ask the user for, not a default. [J], user's instruction
- **More than one engine instance requires an explicit flag that states the count**, `--allow-multi-instance N`, where N must equal workers x environments; otherwise the run refuses and prints the planned instance count, thread count and estimated memory. The check counts **engine instances**, so it also covers several environments in one process. It lives where the vector env is built, not only in the command-line parser.
- **The flag prevents accidents; it is not consent.** Only `trainer-ops` passes it, and only for a layout no larger than one the user has approved; the run report quotes the user's approval and its date. A larger layout needs a new approval.
- **Memory is checked on a clock, not per update:** every 5 seconds, by a watcher that only sets a stop flag. Stop conditions: the process tree above 8 GB, **or system-wide available memory below 1.5 GB** whatever our own usage. Below 0.75 GB available, engines are closed immediately without waiting for a safe point. [C] for the gap; thresholds [J]
- **Checkpoints are written only on update boundaries.** Ctrl-C sets a flag; a rollout in progress is abandoned, an update in progress is finished, then the checkpoint is written. Time to stop is therefore bounded by one update, which is logged. A second Ctrl-C stops at once without saving, leaving the previous checkpoint intact. The previous checkpoint file is kept as a second generation. The smoke run also checks whether Ctrl-C reaches the engine processes directly. [C]

### 6. Held-out maps folder

- **`heldout_maps/` at the repository root**, with `generated/` (our generator, reserved seeds) and `external/` (hand-made or third-party maps), and a `MANIFEST.json` listing each file's SHA-256, source, generator seed and date.
- **The split is enforced both ways:** EVAL already refuses `maps/` and `curriculum/`; any non-vanilla (training) config must refuse a path under `heldout_maps/`. A training config with a Freedoom Phase 2 IWAD and no scenario WAD, that is, the Phase 2 campaign maps by name, is also refused.
- WAD files there stay out of git (ADR 0005, licence question open); the manifest is tracked. The ignore rule is assigned to `map-forge`, since this role cannot edit it. [J]

## Risks

- **Reconciliation by arms costs time.** About 2 hours of toy compute before the recipe is settled, and the toy may not predict DOOM in either direction.
- **Gate A can pass while DOOM-specific paths are broken** (telemetry masking, real time-outs, auxiliary labels). The smoke run covers one of these.
- **Measurement B may fail on every arm.** Then A0 has no demonstrated long-gap memory, the key-and-door stage stays blocked, and the remaining routes are dense labels on real maps and Stage B, both of which depend on other open items (map generation, `viz_nocheat`).
- **The update-gate bias may slow reactive control** by making part of the state sluggish; hence the spread of time scales and the requirement that it beat B0.
- **Floor 0.1 is a compromise without a measurement** until the scaled-reward toy runs.
- **Dense labels are training-only**, which is rule-safe, but an agent that relies on them generalizes only as far as the labelled facts do. Any label built from map metadata must stay out of observations (rule compliance: nothing in this ADR adds logic between observation and action).
- **Safety.** A flag can be passed by mistake by an agent; the approval quote in the report is a paper control, not a technical one. The 5-second watcher cannot see temperature. A hard parent kill may still orphan engines until the test in point 6 is run.
- **Process.** A test for duplicate ADR numbers detects a collision after the fact; only single allocation prevents it.
- **Generalization.** `my_way_home` remains a single fixed layout; nothing here improves that.

## Validation experiment

In this order. Items 1 to 3 are toy-only, one process, 4 threads.

1. **Gate A** (`worker`, about 10 minutes) on the base recipe. If it fails on any seed: stop, hand to `debugger`.
2. **Bias test and floor toy** (`worker`, about 15 minutes): the cue trace at step 20 at initialization with and without the update-gate bias; the scaled-reward floor comparison.
3. **Measurement B** (`worker`, about 80 minutes), all arms, results as per-seed tables in `reports/runs/`, including the arms that fail.
4. **Checkpoint round trip** with `weights_only=True`; **duplicate-prefix test**; **layout guard tests** (no flag gives one instance; a wrong N refuses; an in-process multi-env build refuses); **train-config refusal** of `heldout_maps/` and of Phase 2 campaign maps.
5. **Single-instance engine checks** (`trainer-ops`, one instance, no approval needed beyond what single-instance probes already had): the time-out smoke run of Decision 4; the hard-kill survivor count for daemon and non-daemon workers; Ctrl-C at a random moment followed by resume, confirming the update index and that the weights equal those at the last boundary.
6. **Only with the user's go-ahead:** the multi-instance smoke run, then 0007-recipe's arms.

Reverse or amend if: gate A fails (the block stays and this ADR's base recipe is wrong somewhere); floor 0.1 behaves differently from 0.01 at gap 20, or the scaled-reward toy shows 0.01 or 1.0 clearly better (take the better one); B1 lowers the count against B0 (drop the bias); B2 does not beat B1 (the keys head is not established as a memory mechanism and Stage B's priority rises further, to be weighed against the `viz_nocheat` hedge in a new ADR); arm E or I beats B0 by 2 seeds or more (adopt it); the hard-kill test shows survivors under both worker designs (add a start-up and periodic sweep that kills engines whose parent is gone before any multi-instance run); `weights_only=True` cannot load a required field after conversion (report it; do not fall back).
