# DoomsDay: Project Context for Claude Code

This file is loaded at the start of every Claude Code session in this repo. It holds the hackathon brief, the rules we must never break, and the current state of our decisions. Keep it up to date: when the team makes a decision listed under "Open decisions", move it to "Decisions made" with the date.

---

## 1. What we are building

A hackathon entry for **Track 1: "Can It Play DOOM?"**
Domain: Deep Reinforcement Learning, Spatial Intelligence and Autonomous Control.
Environment: DOOM via the **ViZDoom** engine.

**Mission:** design and train an autonomous agent that plays DOOM levels **zero-shot**: surviving demon ambushes, navigating multi-room architectures, handling doors and keycards, and reaching the level exit.

- No static dataset is provided. The agent learns only by interacting with the game engine.
- Standard RL bots fail when a wall moves or an enemy spawns somewhere new, because they memorize button sequences. Our agent must generalize to **unseen maps and layouts**, not overfit to one scenario.

### Compute and sensory guidance (from the organizers)
- Training on raw RGB pixels alone is expensive: heavy GPU use, with most training time spent learning basic edge detection.
- The organizers **strongly recommend ego-centric spatial data** extracted from the engine (sector geometry, depth, visible entity telemetry). This runs at thousands of ticks per second on a CPU.
- A **hybrid** of pixel buffers plus spatial telemetry is explicitly allowed if it helps.

---

## 2. Core challenge objectives

The agent must handle all three:

1. **Partial observability.** Navigate 3D space with no god-view radar, using only what is in the player's field of view and local sensory range.
2. **Long-horizon progression.** Solve multi-step room dependencies: explore branches to find a keycard, remember where a locked door was seen earlier, and backtrack to unlock the exit.
3. **Tactical combat and survival.** Manage health, ammo and movement momentum, dodge projectiles, and prioritize among multiple threats.

---

## 3. Deliverables

The final submission must include the **trained agent, the training pipeline, and the inference setup**, covering:

1. **Sensory and state representation.** The observation pipeline from the engine (ego-centric rays, entity vectors, depth buffers, or hybrid vision plus spatial input), designed so the agent **cannot simply memorize map coordinates**.
2. **Model architecture and memory.** A custom network topology, specifically showing how the agent keeps **spatial memory across rooms** and handles a **variable number of enemies**.
3. **Training and exploration pipeline.** The RL implementation, **custom reward shaping**, and **exploration mechanisms** that push the agent into unfamiliar rooms instead of getting stuck in corners.

### How we are scored (innovation signal)
- Innovative, custom-engineered architectures are **heavily rewarded**: bespoke neural modules, custom spatial memory systems, intrinsic curiosity mechanisms, hierarchical control policies.
- An off-the-shelf **Stable-Baselines3 PPO script with default hyperparameters will score poorly.** Do not propose that as the solution. Libraries may be used for plumbing, but the core agent and training loop must be our own design.

---

## 4. HARD CONSTRAINTS (never break these)

These come straight from the ground rules. Breaking any of them can disqualify the entry. If a request would break one, stop and say so instead of implementing it.

### 4.1 Zero hardcoded scripts
- Every action the final agent takes must come from **trained neural network parameters**.
- **Forbidden in the decision path:** hand-written `if/else` rule trees that choose actions, classical pathfinders (**A\***, **Dijkstra**, BFS-to-goal planners, etc.), and trigonometry aimbots (e.g. `atan2` to compute and snap aim at an enemy).
- No "scripted fallback" or "safety override" that picks actions when the network is unsure. No action masking that encodes strategy.
- *Our working interpretation (confirm with organizers if it becomes important):* geometry inside the **observation pipeline** (transforming entities into the player's local frame, raycasting, line-of-sight filtering) is a sensory encoding, which the brief itself recommends. The line we never cross is code that turns observations into actions. Observation features must describe the world; they must not pre-compute the answer (e.g. a "turn left N degrees to aim" feature is effectively an aimbot).

### 4.2 Engine modifications are for training only
- **Allowed during training:** modifying the engine, map scripts (ACS), or game variables for pretraining and curriculum learning, e.g. auto-opening doors, infinite ammo, isolating combat vs navigation phases.
- **Final evaluation runs on the standard, unmodified DOOM engine with vanilla rules. No mods.**
- Therefore: the inference setup must run on stock ViZDoom with vanilla rules. Any observation the agent depends on at inference must be obtainable from the unmodified engine. Reward shaping, cheats and curriculum hacks must be switchable off and must never be required at inference.
- Always keep a "vanilla eval" path in the code and test against it regularly.

### 4.3 Original work
- The repository must be **created during the hackathon**.
- Open-source libraries (vizdoom, gymnasium, torch, jax, etc.) are allowed, but the **core agent architecture and training pipeline must be built during the event**.
- Do not paste in agent or training code from existing repos, papers' reference implementations, or prior projects. Write it here.

### 4.4 Zero-shot live evaluation on unseen maps
- Judging tests the trained model **live on evaluation maps we have never seen**.
- Never tune against a single map. Never feed absolute map coordinates or map IDs into the policy. Train and validate on varied / randomized / held-out maps.

### Quick compliance check before merging agent or env code
- [ ] Is every action produced by the network's forward pass?
- [ ] Does the observation avoid absolute coordinates and map-specific identifiers?
- [ ] Does inference run on unmodified ViZDoom with vanilla rules?
- [ ] Is the agent / training code our own, written during the event?
- [ ] Have we checked on a map the model did not train on?

---

## 5. Recommended libraries and concepts (from the brief)

**Core libraries**
- `vizdoom`: official Python API for the DOOM engine (`pip install vizdoom`). Supports raw screen buffers and internal state extraction (objects, sectors). Useful engine outputs to check in the installed version: depth buffer, labels buffer, objects info, sectors info, game variables. Note that sector info covers the whole map, so it must be filtered to what the player can actually perceive (range / line of sight) to respect partial observability. Do not use a fully revealed automap at inference, as it is a god-view.
- `gymnasium`: standard wrapper for step / action / observation loops.
- `torch` or `jax` + `flax`: for custom spatial encoders, attention heads, recurrent memory cells and custom RL losses written from scratch.

**Concepts to research**
- Ego-centric spatial encoding: raycasting, local coordinate frame transformations, line-of-sight filtering.
- Spatial memory and exploration: recurrent policies (LSTM / GRU), differentiable occupancy grids, intrinsic curiosity rewards.
- Generalization in RL: curriculum learning, domain randomization, hierarchical sub-goal policies.

---

## 6. Open decisions (not yet made by the team)

Do not assume answers to these. If a task depends on one, ask the team, or present options with a recommendation and wait for a choice.

| Topic | Status |
|---|---|
| Observation design (pure spatial telemetry vs hybrid with pixels / depth) | OPEN |
| RL algorithm (custom PPO variant, IMPALA-style, other) | OPEN |
| Model architecture (encoders, entity attention / set encoder, memory module) | OPEN |
| Spatial memory approach (recurrent state, occupancy grid, external memory) | OPEN |
| Exploration mechanism (curiosity, count-based novelty, other) | OPEN |
| Hierarchical control (yes / no, and what the sub-goals are) | OPEN |
| Reward shaping terms | OPEN |
| Curriculum and training maps / map generation | OPEN |
| Action space (which buttons, discrete vs combined) | OPEN |
| Eval harness details from organizers (allowed buffers, action set, time limits, submission format) | OPEN, find out |
| Hardware available for training (CPU cores, GPU) | OPEN |
| Repo layout, Python version, environment manager | OPEN |
| Team members and roles | OPEN |
| Deadline and milestones | OPEN |

## 7. Decisions made

_Add entries as: `YYYY-MM-DD: decision (who decided, why)`._

- 2026-11-25: Deep learning framework is **PyTorch**, not JAX + Flax (decided by Priyansh when defining the `worker` subagent, which builds the model and training loop on PyTorch; no further rationale recorded).

- 2026-11-25 (evening): **Plan change** (decided by Priyansh with the team, from `DOOM_RL_Hackathon_Brief.md` by Raghav Sharma; reason: time is shorter and the first generated-maps run stalled on doors at about 5%). (a) Build a **scripted teacher** (pathfinding and aim allowed in `teachers/` only, training-time, never imported by the inference path), behaviour-clone it, then fine-tune with PPO. (b) The trainer stays **our own recurrent PPO**; Sample Factory is not used (no Windows support). (c) **No LLM planner.** Still undecided: pretrained weights as a frozen combat skill (conflicts with ADR 0007(a); needs an explicit yes), and whether the organizers accept scripted teachers.

- 2026-11-26 (03:00): **Working assumption: evaluation does NOT set `+viz_nocheat`**, so the depth buffer, labels buffer and sector info are available at inference (decided by Priyansh; reason: the brief strongly recommends engine telemetry and asks for our own inference setup; "unmodified engine" is read as no engine mods / vanilla rules, not as buffers off). No pixel fallback is built unless the organizers say otherwise. The question is still worth sending to the organizers.

- 2026-11-26 (03:10): **Option B** (decided by Priyansh): launch the night-3 run now; build these gaps alongside it and apply them in the day run on top of the night's model (no restart from scratch):
  1. **Locked-door map channel**: egomap channel 5 set when USE gives the "needs key" message at the door under the view ray; the model grows a 6th egomap input with zero-initialised weights so it starts out identical; re-collect teacher data with it.
  2. **Longer episodes** (about 4 minutes of game time instead of 2) and **mixed skill levels 1-4** per episode.
  3. **Lifts and stairs** in the generator: measure on Freedoom Phase 1 first; likely too big for the deadline.

## 8. Current state of the repo

- 2026-11-24: Repository started. Only this file exists. No code yet.
- 2026-11-25: Subagents defined in `.claude/agents/`. Installed into the system Python 3.12: `vizdoom` 1.3.3, `gymnasium` 1.4.0 (no virtual environment; `torch` 2.9.1 is the CPU-only build).
- 2026-11-25: Research and open-source scouting reports written (`docs/research/`, `docs/osc/`). ADRs 0001-0006 written by `advisor` in `docs/decisions/`; all are **Proposed, not yet ratified by the team**, so section 6 is unchanged.
- 2026-11-25: `env/` built: single-environment Gymnasium wrapper over ViZDoom with TRAIN and EVAL configs, ego-centric observations (rays, entity set, self vector, optional screen), six-head action space. Engine probes in `docs/reference/engine-probes.md`. No vectorized multi-process environment yet. Known open bug at time of writing: centre rays read 0 because the player's weapon sprite has depth 0 (fix in progress).
- 2026-11-25: `agent/` (Stage A0 model: ray encoder, entity attention encoder, GRU, six heads, auxiliary heads; 474,474 parameters), `train/` (our own recurrent PPO, GAE, rollout storage, checkpoints) and `configs/` built. Tested on a synthetic memory task only. **Nothing has been trained on DOOM yet**; there is no DOOM training entry point, resume, or process-priority control.
- 2026-11-25: Tests in `tests/`: 128 passed, 1 skipped (map-translation test, needs a custom map). Lessons 01-07 in `docs/learn/`.
- 2026-11-25 (later): ADRs 0007 (two files share this number), 0008, 0009 and 0010 added; all ADRs remain **Proposed**. The centre-ray bug is fixed (rays and entity depth ignore the player's weapon, firing overlay and bullet puffs). TRAIN is the bundled `my_way_home` scenario; training configs refuse `heldout_maps/` and Freedoom Phase 2 maps. `train/` now has a multi-process vector env, the entry point `python -m train.run_doom` (bare = 1 worker x 1 env; more needs `--allow-multi-instance N`), below-normal priority, checkpoint/resume, an 8 GiB memory cap with a 5 s watcher, and per-episode logging (`episodes.jsonl`). Launch configs for runs live in `configs/ops/` and are started with `python -m configs.ops_launch`.
- Toy memory results: Gate A (4-step gap) passes 8 of 8 seeds; a 20-step gap passes 8 of 8 at 300 updates; a 60-step gap fails on every variant tried. ADR 0009: do not rely on the GRU for long delays.
- `rewards/` and `intrinsic/` built (reward terms, episodic coverage bonus, exploit guards), tested on synthetic data only and **not yet wired into the trainer**.
- First DOOM training run started 2026-11-25 08:04: `reports/runs/20261125-nogit-7a675759/` (4 workers x 8 envs, `my_way_home`, sparse built-in reward, no exploration bonus). See `reports/runs/INDEX.md` for its outcome. No random-policy baseline had been measured when it started.
- First run result (2026-11-25, report `reports/runs/2026-11-25-first-doom-run-a0-4x8.md`): finished cleanly, 1,814 updates, 10,060 episodes, 3.71M decisions. Last 500 episodes: success 0.726, return +0.616; random policy (200 episodes, measured after the run): success 0.140, return -0.052. The learning conditions of ADR 0009 are met on the training scenario, but by ADR 0009 as written the run is **"no verdict"** (17-21 decisions/s per engine, under its 50/s line); it counts as "learning" only under advisor's corrected throughput rule, which is not yet in any ADR; one seed, no held-out evaluation. Mid-run collapse (success back to 0.176, entropy minimum 0.122) then recovery, cause not established. Checkpoint: `reports/runs/20261125-nogit-7a675759/checkpoint.pt`.
- 2026-11-25 (after the run): reward shaping and the coverage bonus are wired into `train.run_doom` behind `--shaping my_way_home` (off by default; default path unchanged). With it on, `episodes.jsonl` also records `engine_return` (unshaped) and `shaping_episode`; `return` and `recent_return` are then the shaped values. Checkpoint save no longer leaves a moment with no `checkpoint.pt`; load falls back to `.prev`. Only a 40 s 1x1 smoke has used the flag; **no shaped training run yet**. Tests: 257 passed, 1 skipped.
- ADR 0011 (Proposed, 2026-11-25): throughput "no verdict" line becomes half the same-layout benchmark; under it the first run reads "learning on the training scenario, unstable". Adds a late-collapse monitor, requires the shaped run to be a fresh start, and says verdicts use `engine_return`, never the shaped `recent_return`. Its monitor and extra logging (0011-F1 to F7) are **not built yet**. Lessons 11 and 12 written.
- 2026-11-25 (maps): `maps/` holds our own seeded map generator (writes vanilla-format PWADs; the engine builds nodes itself, no external tool), QA, loader and `splits.json`; `curriculum/` holds a scheduler; `heldout_maps/` holds 20 generated held-out maps (WADs git-ignored, rebuilt byte-identically by `python -m maps.build`). Batch 1: 80 train / 20 val / 20 held-out, evenly over S0 navigation, S1 doors, S2 key + locked door, S3 monsters and pickups; all 120 passed validation (solvability search + engine checks; none walked start-to-exit in the engine). The loader refuses held-out maps in training. **Not wired into `train/`**: the vec env resets without a map option, so the curriculum is bypassed. Nothing has been trained on these maps. Report: `reports/runs/2026-11-25-maps-batch1.md`.
- `env/` `info` now carries `health`, `armor`, `ammo` (10 slots) and `damage_taken_total` for rewards; observation unchanged. CUDA PyTorch is **not installed**: Prismor blocked `torch==2.9.1+cu128` (known CVEs); torch stays 2.9.1+cpu. Tests at map-forge's last full run: 307 passed, 1 skipped.
- 2026-11-25 (afternoon): `python -m train.run_doom --maps generated --shaping full_level` trains on the TRAIN split of `maps/` with a map chosen per episode by `curriculum/` (state checkpointed; stage and map id logged only, never in the model input). Only a 70 s 1x1 smoke has run on it; **no real training on generated maps yet**. ADR 0011 monitor and logging (0011-F1 to F7), `--entropy-coef`, numbered checkpoints, `--shaping passthrough` and `--device cpu|cuda|auto|hybrid` are built. torch is now **2.9.1+cu128** (Prismor was uninstalled by Priyansh; CUDA works on the RTX 5060). Measured on 8x6 with 8 torch threads: CPU 811 decisions/s end to end, full-GPU 847; `hybrid` unmeasured at that layout. Layout benchmark: 8 workers x 6 envs, 8 torch threads = 818 decisions/s (4x8 = 459); 12x4 tripped the memory guard. Rules audit 01 (`reports/audit/`): 7 PASS, 1 FAIL (single-map training, now addressed by the wiring but not yet by a run), 5 UNCLEAR. Tests: 365 passed, 2 skipped. Known gap: the health monitor's low-success line is `my_way_home`'s, so it can raise a false collapse/stop on generated maps.
- 2026-11-25 13:57: first run on the generated maps started, `reports/runs/20261125-nogit-45275c79/` (`configs/ops/gen_run1_4x8.json`: `--maps generated --shaping full_level`, 4x8, health monitor flag-only), set to stop itself about 16:43 because Priyansh shuts the laptop down at 17:00 and restarts about 21:00; resume config `configs/ops/gen_run1_4x8_resume_night1.json`. See `reports/runs/INDEX.md` for the outcome. Before launch: `env/` refuses training maps that are not in the configured PWAD (audit E1: the engine otherwise silently loads the IWAD map of that name), drops entity rows for labels with no visible pixel of their own (R9; about 3% of rows, so the `my_way_home` checkpoint's inputs differ slightly), and gives each engine its own config file. `evaluate/` (`python -m evaluate.run --checkpoint ... --split val|heldout|my_way_home`) is the vanilla evaluation entry point; the `my_way_home` checkpoint scored 14/20 there and 0/4 on two unseen val maps. Tests: 385 passed, 2 skipped. Memory note: free RAM, not CPU, limits the layout; Chrome's background processes alone held about 6 GiB.
- 2026-11-25 (evening): generated-maps run 1 finished (4.63M decisions): S0 reached 0.62 then promoted; S1 flat at 0.05; val S0 5/15, rest 0. Built: `teachers/` (scripted route-following teacher, training only, isolation test; 100% on S0-S2 train maps), `train/bc.py` (imitation, multi-dataset, DAgger via `teachers.collect --student`), `run_doom --init-from` and `--value-warmup-updates`, and the training-only door-passed reward (`--shaping full_level_doors`, `maps/doors.py`, `rewards/doors.py`). Imitation + DAgger did **not** beat PPO: train S0 18/40, doors about 0 (`reports/eval/2026-11-25-bc-dagger.md`). Both methods stall at the first door; a closed door is not marked in the observation (rays see a flat wall), cause not yet proven. Overnight run started 21:21: `reports/runs/20261125-night1-A-dagger-doors/` (stops about 07:21). Two concurrent training runs are not possible (leftover-engine guard).
- Tests: 193 passed, 1 skipped before the run; lessons 01-10 in `docs/learn/`.
- Probe finding that constrains the design: `+viz_nocheat 1` removes the depth and labels buffers, objects, sectors and position variables entirely; only the screen and health/armor/ammo/weapons survive. Whether the judges set it is an open question for the organizers.

_Update this section as major pieces land (env wrapper, model, trainer, eval script)._

---

## 9. Subagents

Rules for every subagent working in this repo:

1. **Read first.** Before doing anything else, read `docs/TRACK_BRIEF.md` (the verbatim hackathon brief) and `docs/RULES.md` (the hard-constraint checklist).
2. **Write detailed output to files**, not into the reply:

   | Folder | Contents |
   |---|---|
   | `docs/research/` | Research notes, literature surveys, technique comparisons |
   | `docs/reference/` | API notes, engine facts, lookup documentation |
   | `docs/decisions/` | Decision records (date, options, rationale) |
   | `docs/osc/` | OSC documents |
   | `reports/eval/` | Evaluation results (vanilla engine, held-out maps) |
   | `reports/audit/` | Compliance audits against `docs/RULES.md` |
   | `reports/security/` | Security reviews of dependencies, downloads, code |
   | `reports/runs/` | Training-run summaries (config, metrics, observations) |
   | `third_party/downloads/` | Downloaded third-party files: **untrusted, never executed in place** |

3. **Return a summary of at most 15 lines**, including the paths of the files written.
4. **Never `git commit`.** End any code change with a single one-line commit message for a human to use.

### Teach once a day

**Once a day** (changed 2026-11-25 by Priyansh, to save time; previously after every task): at the end of the day's work, invoke the `teacher` subagent on all files changed that day by `worker`, `env-engineer`, `map-forge` or `reward-designer` to write the next lesson(s). Implementation tasks no longer wait for a lesson. Lessons go to `docs/learn/`.
