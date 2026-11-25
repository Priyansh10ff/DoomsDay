# Lesson 09: Why the memory test failed, and what that taught us

Files taught: `train/ppo.py` (the loss, re-taught here as far as this story needs it), `train/trainer.py:185-199` (the update and the value scale), `train/running_stats.py`, `train/toy_memory.py`, `train/toy_experiments.py`, `agent/init_schemes.py`, `agent/model.py:46-64` (the GRU and its initialisation), `configs/model_config.py:41-51`, `configs/train_config.py:32-35`, and the tests `tests/test_toy_memory.py`, `tests/test_agent_init_schemes.py`, `tests/test_adr_numbering.py`.

Evidence:

- `reports/runs/2026-11-25-a0-toy-three-seeds.md` (the failure);
- `reports/runs/2026-11-25-toy-memory-root-cause.md` (the `debugger`'s investigation);
- `reports/runs/2026-11-25-adr0008-toy-gate-and-memory.md` (the `worker`'s runs after the new decision).

Decision record: `docs/decisions/0008-reconcile-0007-files-memory-gate-and-safety-defaults.md` (ADR 0008).

**Two ADRs share the number 0007:** `0007-training-recipe-pretrained-and-reuse.md` and `0007-a0-trainer-rulings-and-first-run-prerequisites.md`. They were written by two advisor runs that did not see each other, and they disagree on several trainer settings. **ADR 0008 reconciles them point by point** and supersedes both on the points it lists. A new test, `tests/test_adr_numbering.py`, fails if two decision files share a number again; this pair is its one recorded exception. **All ADRs, 0008 included, are "Proposed", not ratified by the user. Nothing has been trained on DOOM yet.**

This lesson is the one held back in lessons 06 and 07. It builds on lesson 05 (the GRU and its hidden state), lesson 06 (returns, values, advantages) and lesson 07 (the update and the toy task).

---

## 1. What problem this piece solves

The whole point of a recurrent agent is to remember. A DOOM agent sees a blue key in one room and the blue door several rooms later. It must carry "I have the key" and "the door was over there" across hundreds of decisions in which neither is visible.

Our test for that ability was the **toy memory task** (lesson 07): a cue at step 0, an answer demanded at step 20. It passed once, at 0.985. Run again on three seeds it failed: 0.605, 0.485, 0.495, where 0.5 is guessing. The first question was **"is the training code broken?"**, because a silent bug in recurrent PPO would poison every later result. The second question, once that was answered, was **"how far can this agent remember at all?"**. The answer matters directly for the keycard-and-door part of the brief.

## 2. The idea in plain words

Imagine teaching someone to remember a colour shown to them at the start of a long, noisy phone call, and to say it at the end. You only tell them "right" or "wrong" at the very end.

- If the call is short (4 sentences), they quickly learn to hold on to the colour.
- If the call is long (20 sentences) and their memory naturally fades by half with every sentence, then by the end almost nothing is left. Your "right/wrong" at the end gives them almost no clue that the colour at the start was what mattered.
- But suppose that, all through the call, they also try to guess "how well will this call go?". If they have even a slight habit of answering one colour, that guess depends on the colour at every sentence, so they get a small lesson **every sentence**, not only at the end. That steady stream of small lessons is what builds the memory.

That is what the investigation found. The "right/wrong at the end" is the policy loss; the "how well will this go" guess is the value loss; the fading is the GRU's update gate at its starting value.

Where the analogy stops being accurate:

- Our network's "memory fading by half" is a property of its **starting** weights. Training can change it; the problem is that training needs a signal to change it, and the signal is what fades.
- A person can decide to pay attention. The network cannot: it only changes where a gradient tells it to.

### Terms used in this lesson

- **Gap** (or lag): the number of steps between the cue and the moment it must be used.
- **Trace**: how much a past input still changes the hidden state now. The `debugger`'s metric is the size of the difference between the hidden states of two episodes that differ only in the cue (`train/toy_memory.py:159-180`).
- **Gradient flow**: how a loss at one step changes weights that acted at earlier steps, through backpropagation through time (lesson 07). If the trace is tiny, the gradient back to the cue is tiny too.
- **Dense signal**: a training signal present at every step. A **sparse** one appears only at some steps (here, only at the answer).
- **Detach**: cut the gradient path at a point, so a loss cannot change anything before it.
- **Positive control**: a version of the experiment that *should* succeed if the machinery works, used to separate "the code is broken" from "the task is hard".
- **Ceiling effect**: when every option scores near the maximum, so the test cannot tell them apart.

## 3. A walk through our actual code, and the evidence

### 3.1 The loss, re-taught where it matters: `train/ppo.py:31-80`

`evaluate_minibatch` recomputes a stretch of the rollout and builds one loss from five parts (`:65-67`):

```
loss = pg_loss + 0.5 * v_loss - 0.01 * entropy + 0.1 * (keys + damage + monsters)
```

Two parts matter here.

1. **The policy loss** (`:48-51`): for each stored action, the clipped ratio times the **advantage** (lesson 07, section 3.3). In the toy, every action before the answer step has no effect on the reward, so its advantage is noise around 0. Only the answer action carries real information. So for memory, **the policy loss is a sparse signal at one step**, the answer.
2. **The value loss** (`:52-53`): 0.5 x (value - target)² at **every** valid step, where the target is the GAE return divided by the return scale (lesson 06; scale in `trainer.py:187-188`). In the toy the only reward is at the answer, so the return at step t is about γ^(20-t) x (reward at the answer). Now suppose the policy has even a slight bias, say it answers "cue 0" a bit more often. Then the expected return at step 5 is higher when the cue was 0 than when it was 1. **The value target at every step depends on the cue**, so the value loss at every step pushes the hidden state to keep the cue. That is a dense signal. Better still, at step 1 it only has to reach back 1 step, at step 2 two steps, and so on, so the memory can be built up gradually.

Both losses reach the GRU through **backpropagation through time**: `forward_sequence` replays the stored 32-step stretch from the stored starting state `h0` (lesson 05), so the gradient flows back through every GRU step **inside that stretch**, and stops at its start.

### 3.2 Step 1: is the code broken? The positive control

The `debugger` changed one thing: the gap, from 20 to 4. Same code, same settings, same floor (`root-cause.md`, section 3):

| Seed | 0 | 1 | 2 |
|---|---|---|---|
| Success, gap 4 | 1.000 | 0.995 | 1.000 |

Episodes still crossed rollout boundaries, so carrying the hidden state across rollouts, the reset masks, GAE and the sequence replay were all exercised. The first-minibatch ratio check stayed below 1e-6. **No trainer bug was found.** The code can learn a memory the network is able to form; gap 20 is harder for another reason.

### 3.3 Step 2: why gap 20 is hard. The fading trace

A **GRU** updates its hidden state as (`agent/init_schemes.py:4-7`):

```
h_new = (1 - z) * n + z * h
```

- `n` is the new candidate content;
- `z` is the **update gate**, a number between 0 and 1 for each of the hidden units: the fraction of the old value each unit keeps.

At initialisation, the weights are small and random and the biases near 0, so `z` is about **0.5**: each unit keeps about half of its old value per step. Whatever the cue wrote into the state at step 0 fades by roughly half every step.

The `debugger` measured it (`root-cause.md`, section 1), and I re-measured it (section 4): the cue's trace is 2.5e-2 at step 0 and about **1.3e-6 at step 20**. The gradient from the answer back to the cue is correspondingly tiny, about 1e-5 of its size at gap 0.

**The supervised check** settles that this is about the *signal*, not about PPO. The `debugger` trained the same network directly on the right answer, with clean labels (`root-cause.md`, section 2):

| Loss placed | Gap | Seeds | Final accuracy |
|---|---|---|---|
| only at the answer step | 20 | 0, 1, 2 | 0.48, 0.41, 0.50 (chance) |
| at every step (dense) | 20 | 0, 1, 2 | 1.00, 1.00, 1.00 (all by step 200) |

Even with perfect labels, a loss only at the end could not teach a 20-step memory within 500 steps. A loss at every step taught it on every seed.

### 3.4 Step 3: memory forms through the value loss

If the value loss is the dense signal, cutting it should stop learning. The `debugger` **detached** the value head from the GRU: the value head still learned, but its gradient could no longer reach the hidden state (`root-cause.md`, section 4):

| Configuration (gap 20) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Floor 0.01, value head attached | 0.870 | 0.745 | 0.690 |
| Floor 0.01, value head **detached** from the GRU | 0.510 | 0.470 | 0.495 |

With the value path cut, every seed sat at chance, even when the policy had become strongly biased. The policy gradient alone did not build the memory. Probes every 25 updates showed the cue's effect on the value rising **before or together with** its effect on the answer.

The onset depended on the seed: it waited for a chance bias in the answer to make the value target depend on the cue. With 300 updates at the old settings, seed 0 crossed 0.9 at update 125, seed 1 at 250, and seed 2 never (`root-cause.md`, section 5).

### 3.5 Step 4: why the value-scale floor mattered

Recall from lesson 06: value targets are divided by the **return scale**, the running standard deviation of returns, but never by less than a **floor** (`train/running_stats.py`; `configs/train_config.py:32-35`). The value head learns in these scaled units.

On the toy, returns are between 0 and 1 and their spread is about **0.29** (`debugger`; the `worker` reports mean final scales of 0.25 to 0.33; ADR 0008 notes one report gave about 0.5, a discrepancy it leaves open). So:

- **Floor 0.01** (the original value): the scale is 0.29, so targets are multiplied by about 1 / 0.29 ≈ 3.4 (by hand).
- **Floor 1.0** (introduced by `0007-a0-trainer-rulings`): the scale is held at 1.0, so targets are **not** amplified. The value loss, and its gradient into the GRU, are about 3 to 4 times weaker. That is the dense signal the memory depends on.

That explains most of the drop from 0.870 / 0.745 / 0.690 to 0.605 / 0.485 / 0.495. The `worker` also reproduced the original 0.985 exactly on seed 0, by restoring the old floor *and* the old monster target (`three-seeds.md`).

**What ADR 0008 did** (Decision 1, point 3): it withdrew the 1.0 floor, because it measured worse on 3 of 3 seeds. Its motive, that a first sparse reward could become a target of 100, had come from reading code and was never measured. ADR 0008 set **0.1**: at most a tenfold amplification. On the toy, 0.1 and 0.01 **should** behave the same, because the measured scale (about 0.25 to 0.33) is above both. ADR 0008 lists that as "to be confirmed". The planned comparison of the three floors with small rewards was mostly not run for lack of time; only floor 0.01 on two seeds ran (`adr0008-toy-gate-and-memory.md`). So **0.1 is still a judgement without a measurement**, as ADR 0008 says.

### 3.6 Step 5: splitting the test in two (ADR 0008 Decision 2)

The old test asked two questions at once: "does the trainer work?" and "can it remember 20 steps?". Its failure could not say which. ADR 0008 replaced it with two tests whose pass rules were fixed **before** any run, with 8 seeds (0 to 7) each:

**Gate A: does the trainer work?** It **blocks all DOOM training**.

- Gap 4, 100 updates, base recipe (floor 0.1, 3 epochs, initialisation as built).
- Pass: every recurrent seed at least 0.95, every memoryless seed within 0.1 of 0.5, and the first-minibatch ratio deviation below 1e-5.

Result (`adr0008-toy-gate-and-memory.md`): **passed, 8 of 8.**

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Recurrent | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Memoryless | 0.500 | 0.515 | 0.560 | 0.470 | 0.450 | 0.520 | 0.490 | 0.520 |

The ratio deviation was at most 9.5e-7 in every run. `tests/test_toy_memory.py` runs a subset in the test suite (recurrent seeds 0 and 1, memoryless seed 0), with the definition checked at `:28-31`. The full 8 seeds take about 8 minutes, too slow for every test run.

**Measurement B: how far does memory reach?** It does not block the first pipeline runs; it **blocks any key-and-door curriculum stage and any claim about long-horizon memory**.

- 300 updates, 8 seeds per arm. Pass per arm and gap: **at least 7 of 8 seeds at or above 0.9**.
- Each **arm** changes one thing from the base:
  - B0: the base;
  - B1: the update-gate bias (3.7);
  - B2: B1 plus a dense cue label (3.8);
  - E: 2 epochs instead of 3 (from `0007-recipe`);
  - I: orthogonal initialisation (from `0007-recipe`): starting weight matrices whose rows are at right angles to each other and of a set length, a common alternative to PyTorch's default random start (`agent/init_schemes.py:42-63`).

Results:

| Gap | Arm | Seeds at or above 0.9 |
|---|---|---|
| 20 | B0, B1, B2, E, I | **8/8 each** |
| 60 | B1 | **0/8** (all between 0.46 and 0.55) |
| 60 | B2 | **0/4** (seeds 4-7 not run for lack of time) |

- **Gap 20 now passes on every arm.** Compared with the failing runs, two things changed: the floor (1.0 to 0.1) and the budget (100 to 300 updates).
- At gap 20 every arm is at the ceiling (lowest per-arm scores 0.945 to 0.995), so this measurement **cannot tell the arms apart**.
- **Gap 60 fails on every arm that was run.** B0, E and I were not run at gap 60.

Why 60 is special: the rollout is 32 steps, so at gap 60 the cue and the answer **never fall in the same replayed stretch**. Gradient flows back only to the start of a stretch (3.1); across a rollout boundary, the stored `h0` is treated as a fixed input. Nothing can send a gradient from the answer back to the step where the cue was written. That is my reading of why it fails; the report measured the failure, not the mechanism. In DOOM, a key found hundreds of decisions before its door is the gap-60 situation, many times over.

**Adoption.** No default was changed. ADR 0008 adopts B1 and B2 only if they score *higher* than B0, and 8/8 against 8/8 is a tie. For E and I, ADR 0008's wording conflicted: "not lower than B0's" said adopt, while "ties keep the base" said don't. ADR 0009 has since replaced it with one rule for every arm: adopt only if it beats the base by at least 2 seeds out of 8, at a setting declared in advance. Any comparison where the base already scores 7 or 8 of 8 is **void**. So none of B1, B2, E or I was adopted (lesson 10).

### 3.7 The update-gate bias (arm B1): `agent/init_schemes.py`, `agent/model.py:63-64`

**The idea.** If the memory fades because `z` starts at 0.5, start some units with `z` near 1 instead. A unit that keeps a fraction `z` per step has a **retention time** τ = 1 / (1 - z) steps: z = 0.5 gives τ = 2, z = 0.99 gives τ = 100.

- `retention_biases` (`init_schemes.py:24-31`) spreads τ **log-uniformly** (evenly on a logarithmic scale) from 2 to 256 decisions across the hidden units (`model_config.py:49-51`). It computes the bias that gives z = 1 - 1/τ when the input is 0: bias = log(z / (1 - z)).
- `init_gru_update_bias` (`:34-39`) writes these into the GRU cell's bias for the update gate.
- It is applied **after** any other initialisation (`model.py:63-64`), and only if `gru_update_bias_init` is on. It is **off** in the base recipe.
- It is **initialisation only**: every parameter stays trained.

**Found by test, not by recall.** PyTorch stores the GRU's three gate biases side by side in one vector. ADR 0008 told the `worker` to establish which third is the update gate **by measurement**. `tests/test_agent_init_schemes.py:37-46` puts the bias on each third in turn and requires the step-20 cue trace to rise at least 100 times on the chosen one, and at least 10 times more than on the others. The report's numbers at step 20: base 1.34e-6; bias on chunk 0, 1.20e-5; on chunk 1, **4.59e-4**; on chunk 2, 2.07e-6. So chunk 1 (`GRU_UPDATE_GATE_CHUNK`, `init_schemes.py:21`) is the update gate.

**What it changed:** the cue trace at step 20 at initialisation rose from **1.3e-6 to 4.6e-4**, about 340 times. **What it did not change:** the gap-20 result (already 8/8 without it) and the gap-60 result (0/8 with it).

ADR 0008 notes the idea comes from memory of the recurrent-initialisation literature and is not in `docs/research/`, so the team should treat it as its own until someone finds a source. It also names a risk: units that are slow to change may make reactions sluggish, which is why it must beat the base to be adopted.

### 3.8 The dense cue label (arm B2): the keys head as a memory teacher

The supervised check (3.3) said a dense label teaches memory. Our model already has a head that gets a dense label: the **keys-held** auxiliary head (lesson 05), whose label is present at every step after a pickup. For arm B2, `ToyConfig.dense_cue_label` (`train/toy_memory.py:38`, `:79-83`) makes the toy report the cue as `info["keys_held"]` at every step. The existing keys head then learns it at weight 0.1.

At gap 20, B2 tied at 8/8. At gap 60, it was at chance on the 4 seeds run. So on this evidence **the dense label did not rescue memory beyond the rollout length either**.

### 3.9 Why one seed is not evidence

The first result, 0.985, was one seed. It was a lucky one. With the same settings, seed 1 failed even with the old floor (0.745). The `worker`'s own words: the single-seed pass "was read too confidently".

ADR 0008's pass rule, "7 of 8 seeds at 0.9", was chosen with numbers in mind. Worked by hand from the binomial formula, which gives the chance of exactly k successes in n independent tries, each with the same success chance p:

- A learner that truly succeeds on 60% of seeds passes a **one-seed** test 60% of the time, and the 7-of-8 rule about **11%** of the time: 8 x 0.6⁷ x 0.4 + 0.6⁸ ≈ 0.090 + 0.017 = 0.106.
- A learner that succeeds on 90% of seeds passes the 7-of-8 rule about **81%** of the time: 8 x 0.9⁷ x 0.1 + 0.9⁸ ≈ 0.383 + 0.430 = 0.813.

Eight seeds are still coarse, so results are reported as counts, "8/8" or "0/8", not as certificates.

## 4. A tiny worked example with real numbers

I ran the `debugger`'s metric myself: `train.toy_memory.cue_trace` on the toy network (seed 0, at initialisation, no training), torch on 2 threads, no ViZDoom, 1.4 s. Real output:

```
gap  0: cue trace base 0.0253   with update-gate bias 0.00956
gap  4: cue trace base 0.00225   with update-gate bias 0.00279
gap  8: cue trace base 0.000308   with update-gate bias 0.00125
gap 20: cue trace base 1.34e-06   with update-gate bias 0.000459
gap 60: cue trace base 4.89e-08   with update-gate bias 0.000171
base: update gate z at init (zero input): min 0.442 mean 0.498 max 0.5574
bias: update gate z at init (zero input): min 0.500 mean 0.895 max 0.9961
retention bias: first/middle/last bias [0.0, 3.114, 5.541]  keep z [0.5, 0.9575, 0.9961]  tau [2.0, 23.5, 256.0]
```

Reading it:

- **The base fades fast.** From gap 4 to gap 8 the trace drops from 2.25e-3 to 3.08e-4: a factor of about 0.61 per step (by hand: (3.08e-4 / 2.25e-3)^(1/4) ≈ 0.61). By gap 20 it is 1.34e-6, matching the reports' 1.34e-6 / 1.5e-6. "Roughly halves every step" is the right picture.
- **The update gate starts near 0.5** in the base (mean 0.498, range 0.44 to 0.56).
- **With the bias**, the gate ranges from 0.5 (τ = 2) to 0.996 (τ = 256), with a middle unit at τ ≈ 23.5. The trace at step 20 is 4.59e-4, about 340 times the base, as the report says. At step 60 it is still 1.7e-4.
- **The trade-off is visible at gap 0**: with the bias, the cue's immediate effect is smaller (0.0096 vs 0.0253), because units that keep more of their old state take in less of the new input.

By hand, for one unit: with z = 0.5, what survives 20 steps is 0.5²⁰ ≈ 9.5e-7. With z = 0.9961 (τ = 256), it is 0.9961²⁰ ≈ 0.925.

**And yet, gap 60 still failed with the bias.** A strong trace at initialisation is not enough: something must also *teach* the network to write the cue and read it 60 steps later, and with 32-step rollouts no gradient connects the two (3.6).

## 5. Why ADR 0008 chose this over the alternatives

From ADR 0008 (**Proposed, not ratified**):

- **Reconciling the two 0007 files** (R1 to R3). Taking one file wholesale would either change several settings at once while the gate was failing (R1), or keep a floor that measured worse (R2). R3 was chosen: a base recipe equal to what the toy numbers were measured on, with each disputed change admitted only as a single-change arm. The rulings, point by point (Decision 1):
  - constant learning rate 3e-4;
  - 3 epochs (2 is an arm);
  - floor 0.1;
  - value coefficient 0.5;
  - initialisation as built (orthogonal is an arm);
  - workers decided by a hard-kill test;
  - checkpoints loaded with `weights_only=True`;
  - keys weight 0.1;
  - monster cap 8;
  - default layout 1 x 1.
- **The gate** (G1 to G3):
  - G1, keep "gap 20, 100 updates, every seed 0.9": it fails for reasons that are not bugs and cannot tell a 60% learner from a 100% one with three seeds.
  - G2, switch to gap 4: passes, but hides the memory weakness.
  - **G3 (chosen)**: two tests with different jobs and pass rules fixed in advance.
- **Making memory not depend on the value loss alone** (M1 to M4):
  - M1, do nothing and rely on value-loss strength: leaves the seed-dependent onset.
  - **M2, the update-gate bias, and M3, dense labels (chosen, each as a measured arm).**
  - M4, rescaling the fused input so a small cue is not drowned: deferred. The `debugger` measured that flipping the cue changes the fused input by 0.09, while the noise changes it by 1.63 per step.
- **The value head keeps its gradient into the GRU**: detaching it removed all learning.

## 6. How it connects, and what it means for DOOM

- **Lessons 05 to 07**: this lesson explains *why* the GRU's memory depends on the value loss and on the return scale, and why the "memoryless" control and the ratio check are what proved the code correct.
- **Lesson 08**: Gate A passing lifts the block on **single-instance** DOOM runs. Multi-instance runs still need the user's go-ahead.

**What this means for DOOM, plainly:**

- Measurement B says this network, trained this way, has **not** shown memory over 60 steps. At frame skip 4, 60 decisions are 60 / 8.75 ≈ **6.9 seconds of game time** (by hand). Our rollout of 64 decisions is about 7.3 seconds.
- The keycard-then-door task needs memory over far longer: finding a key and walking back to its door easily takes tens of seconds to minutes, which is hundreds to thousands of decisions. For example, 30 s is about 262 decisions (by hand).
- ADR 0008's reading (its Decision 3, a judgement from a toy):
  - A0 can be expected to learn short reactions and, through the dense keys label, to hold "I have the blue key".
  - It should **not** be expected to remember where a locked door was seen; no dense label for door locations exists.
  - Stage B (the ego-centric map) is needed for that, and its write gate should also start biased toward keeping.
  - With the screen on, the status-bar key icons show key possession without any memory.
- **The advisor has since ruled** (ADR 0009, `0009-toy-results-memory-plan-and-first-doom-run.md`; lesson 10): do not rely on the GRU to remember across long delays. Key possession is to be read from the status bar on screen (the pixel stage), and door location belongs to the map-memory stage. Why the gap-60 runs failed (window, budget or label weight) is still unknown; a fixed diagnostic is planned. The toy has two cues and no distractions; it is not a model of DOOM.

## 7. Common mistakes, and how we would notice them

| Mistake | How we would notice |
|---|---|
| Trusting a single seed. | Exactly what happened: 0.985, then 0.605 / 0.485 / 0.495. Always report counts over several seeds, with the pass rule written down first. |
| Reading a Gate A pass as "the agent can remember". | Gate A uses a 4-step gap. It shows the trainer is consistent, nothing about long memory. Measurement B is the memory test, and it fails at 60. |
| Cutting the value head's gradient into the GRU (for example to "stabilise" the value). | Memory stops forming: the toy falls to chance on every seed (3.4). |
| Raising the value-scale floor so small returns are not amplified. | The dense memory signal weakens; on the toy, the floor of 1.0 cost about a third of the success rate. Watch `return_scale` in the logs: if it sits exactly at the floor, the floor is doing the work. |
| Comparing arms that are all at the ceiling. | Gap 20 now gives 8/8 for everything; differences between arms there mean nothing. A harder setting is needed to separate them. |
| Assuming a gradient crosses rollout boundaries. | It does not: each replay starts from a stored `h0`. Memory longer than the rollout must come from somewhere else (a dense label, an explicit memory such as Stage B, or longer rollouts, each with costs). |
| Writing a new ADR with an already used number. | `tests/test_adr_numbering.py` fails. ADR 0008 also asks that the coordinator allocate numbers. |

## 8. Check your understanding

1. In the toy at gap 20, why can the value loss build the memory when the policy loss, on its own, cannot? Use the words "dense" and "sparse".
2. A learner truly succeeds on 60% of seeds. How likely is it to pass a single-seed test, and the 7-of-8 rule? What does that say about the first 0.985?
3. A hidden unit has update gate z = 0.5; another has z = 0.9961. Roughly what fraction of the cue survives in each after 20 steps? With the bias, the trace at step 20 rose 340 times, yet gap 60 still failed. Why was a stronger trace not enough?

---

## Answers

1. The policy loss carries information only at the answer step, because earlier actions do not affect the reward. That is a **sparse** signal, and its gradient must travel back 20 GRU steps, through a trace that halves each step, to reach the cue. Almost nothing arrives. The value loss acts at **every** step. As soon as the policy has any bias in its answer, the expected return at every step depends on the cue, so the value target does too. That **dense** signal teaches the memory a short reach at a time: step 1 needs only one step back, and so on. The detach experiment confirms it: cutting the value path into the GRU leaves every seed at chance.
2. A single-seed test: **60%**. The 7-of-8 rule: about **11%** (8 x 0.6⁷ x 0.4 + 0.6⁸ ≈ 0.106, by hand). So one passing seed says little: a learner that fails 40% of the time passes it more often than not. The 0.985 was that kind of lucky draw; seed 1 later failed even under the same old settings.
3. 0.5²⁰ ≈ 9.5e-7 (essentially nothing), and 0.9961²⁰ ≈ 0.925 (most of it), both by hand. But a trace that survives only means the *information* is still there at initialisation. Learning still needs a gradient that ties the answer to the cue so the network learns to write and read it. At gap 60 the cue and the answer never share a 32-step replayed stretch, so no gradient connects them, and the arm stayed at chance (0/8).
