# ADR 0004: RL algorithm and training setup (own synchronous recurrent PPO, 8 env workers, capped load)

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

## Context

The training pipeline must be our own (rule 4.3; a default Stable-Baselines3 PPO script is rejected by the brief). The agent is recurrent (ADR 0003) with factorized heads (ADR 0002).

Machine: one Windows 11 laptop, 24 CPU threads, 15.4 GB RAM, RTX 5060 laptop GPU with 8 GB. torch 2.9.1 is the CPU-only build, so the GPU is currently unused. **The user's instruction "don't use my lap's component too much that my lap starts hanging or shuts down" is treated as a hard constraint.** Deadline unknown ("more than 5 days").

Evidence weighed (`docs/research/vizdoom-agents.md`, `docs/osc/vizdoom-and-rl-references.md`):

- Published Doom results used 5x10^7 (DFP) to 4x10^9 (Sample Factory) frames. Headline budgets are out of reach here.
- Asynchronous sampling roughly doubled FPS over a synchronous framework on one server, but that is a cross-codebase comparison, not a clean algorithm comparison. V-trace matters only if sampling is asynchronous.
- Sample Factory has no Windows support and could not be used even if the rules allowed it.
- The PPO paper itself could not be opened by the research agent; PPO settings come from papers that used PPO.
- **No throughput or per-process memory figure exists for this machine.** Everything numeric below is a starting value to be measured.

Windows uses the `spawn` start method: each worker process re-imports its modules, and each ViZDoom instance is a separate engine process.

## Options considered

**A. Stable-Baselines3 / sb3-contrib RecurrentPPO.** Build 2 hours. Rejected: scores poorly by the brief, and does not fit dictionary observations with a custom memory state without fighting the library.

**B. Our own synchronous recurrent PPO with subprocess env workers and central batched inference (chosen).** Build about 1.5 days including tests. Throughput: sampling pauses during the update; unmeasured. Simple to reason about: no policy lag, no off-policy correction.

**C. Our own asynchronous actor-learner with V-trace (IMPALA/APPO style).** Build 3 days or more with shared-memory queues on Windows `spawn`, plus policy-lag debugging. Possibly up to twice the throughput. Too much of a 5-day budget before anything learns.

**D. Value-based recurrent replay (DRQN / R2D2 style).** Build about 2 days. A replay buffer of sequences competes for 15.4 GB RAM, and factorized actions fit poorly. Rejected.

## Decision

**Option B: a synchronous, recurrent PPO written by us.** Libraries only for plumbing (torch, numpy, gymnasium interface, multiprocessing).

### Process layout and load caps (the worker count assumed)

- **8 environment worker processes**, each stepping **4** ViZDoom instances in turn: **32 environments**. Workers import numpy and vizdoom only, **never torch**, so `spawn` does not multiply torch's memory.
- **One main process** holds the model, does batched inference for all 32 environments each step, and runs the update. `torch.set_num_threads(4)`.
- Intended CPU load: about 8 threads for simulation plus 4 for torch, **12 of 24 threads, about half the machine**.
- **RAM budget: at most 8 GB for the whole training job**, leaving about 7 GB for the system. Per-instance memory is unmeasured; if 32 environments exceed the budget, drop to 2 per worker (16 environments) before anything else.
- All training processes run at below-normal priority. Worker count, envs per worker and torch threads are config values with these as defaults; **raising them needs the user's agreement**.
- Checkpoint every 10 minutes and on interrupt, resumable, so the user can stop training whenever the laptop is needed or runs hot. Optional cooldown pause between updates as a config value.
- If a full-speed run still makes the laptop sluggish or hot, the first step down is 6 workers and 3 torch threads.

### Rollout and update (starting values)

- Rollout length **T = 64** decisions per environment (about 7 s of game time at frame skip 4); 32 x 64 = 2048 samples per update.
- Recurrent handling: store the GRU (and later map) state at the start of each rollout; recompute through the whole 64-step sequence in the update; reset state inside the sequence wherever an episode ended. **Minibatches are whole environment sequences** (4 minibatches of 8 environments), never shuffled single steps.
- 3 epochs per rollout. Clip 0.2. Adam, learning rate 3e-4 with linear decay. Gradient-norm clip 1.0. Value coefficient 0.5. Entropy coefficient 0.01 on the summed entropy of the six heads, reviewed per head once runs exist.
- GAE with lambda 0.95 and **gamma 0.997** (not 0.99): a level takes thousands of decisions, and 0.99 has a horizon of about 100 decisions, roughly 11 s of game time. This value is our reasoning, not taken from a Doom source.
- **Time-limit truncation is not termination:** on timeout, bootstrap from the value of the final observation; on death or exit, do not.
- Advantage normalization per minibatch; value targets normalized by a running return scale.
- Two value heads and two advantage streams when an intrinsic reward is on (ADR 0005).
- Auxiliary losses from ADR 0003 are added to the PPO loss with small fixed weights.
- Logged every update: decisions per second for sampling and for the update separately, resident memory, per-head entropy, approximate KL, explained variance, episode return and length per map.

### GPU

**Recommend that the user install a CUDA build of torch, but not as a blocker for Stage A0.**
- Stage A0 is under 1M parameters on vectors; CPU is expected to be adequate (unmeasured) and the GPU would add little.
- The pixel branch (Stage A1), the `viz_nocheat` fallback and the neural map (Stage B) are where the GPU pays. It also moves learner load off the CPU, which helps the "don't hang the laptop" constraint, at the cost of GPU heat.
- This changes the user's Python environment, so it is **the user's decision**. The right build for an RTX 5060 and Python 3.12 must be looked up with the official PyTorch install selector at install time, not taken from memory, and installed in a separate virtual environment so the working CPU setup is not lost. Device selection must be a config value with CPU fallback so both paths keep working.

### Budget

Unmeasured, so stated as a plan to be corrected: aim for a sampler that sustains on the order of 10^3 decisions per second under the caps above, which would give roughly 5x10^7 to 10^8 decisions over the days available. If the benchmark shows far less, the curriculum (ADR 0005) must shrink before the architecture grows.

## Risks

- **Throughput may be far below published figures.** A from-scratch synchronous sampler on Windows with Python pipes is the likely bottleneck; no datapoint exists.
- **RAM.** 32 engine processes plus workers may exceed the 8 GB budget.
- **Thermal and usability.** Sustained load at half the threads for days may still throttle or annoy the user; the caps are a guess until tried.
- **Correctness traps** in recurrent PPO: stale hidden states after several epochs, wrong reset masking, bootstrapping across episode ends, treating timeouts as deaths. These fail silently.
- **Windows specifics:** `spawn` requires import-safe modules and a guarded entry point; orphaned engine processes after a crash will keep eating RAM, so workers must close their games on exit and the launcher must clean up.
- **Sample efficiency.** On-policy PPO discards data; with sparse key/door rewards this may be too slow whatever the throughput.
- **Originality.** CleanRL's recurrent PPO is reading material only; copying it would breach rule 4.3.
- **CUDA install** may fail or destabilize the environment on a new GPU generation; hence the separate environment.

## Validation experiment

1. **Benchmark before any training** (`trainer-ops`, about 3 hours once the wrapper exists): random actions, frame skip 4, ADR 0001 observation; sweep (workers x envs per worker) over 4x2, 8x2, 8x4, 12x4; for each record decisions per second, total resident memory, CPU utilization, and whether the desktop stays responsive. Written to `reports/runs/`. The chosen default is the largest setting within 8 GB and about 50% CPU that the user finds comfortable.
2. **Trainer correctness** (`worker`, half a day): (i) a memory toy task in a tiny pure-Python env (cue at step 0, reward for recalling it at step 20) that only a correct recurrent implementation solves; (ii) unit tests for GAE against hand-computed values including a mid-rollout termination and a truncation; (iii) a check that recomputed log-probabilities in the first epoch equal the stored ones (ratio 1.0).
3. **First real run:** Stage A0 on a bundled scenario; learning within the first hour, with update time and sampling time logged separately.

Reverse or amend if: the sampler is idle more than about 40% of wall time during updates **and** total throughput is below the plan (then consider asynchronous collection, a superseding ADR); the toy memory task cannot be solved (block all further training until fixed); RAM exceeds budget at 16 environments (then reduce observation cost or resolution before reducing environments further); gamma 0.997 gives unstable value learning (fall back to 0.995, then 0.99).
