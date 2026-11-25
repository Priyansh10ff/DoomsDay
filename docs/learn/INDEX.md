# DoomsDay course outline

One lesson per piece of the project, in the order the pieces are built. Each lesson teaches the code as it is written in this repository. New terms are collected in `GLOSSARY.md`.

Status: **done** = written; **next** = the next lesson to write once its code or result exists; no status = planned on paper (an ADR exists) but not built yet, so not taught yet.

| # | Lesson | Status |
|---|---|---|
| 01 | [The environment wrapper (`DoomEnv`): Gymnasium loop, frame skip, terminated vs truncated, config field groups, TRAIN (`my_way_home`) and EVAL, harness map fields, the timer switch, `obs` vs `info`, `final_obs_fresh`](01-environment-wrapper.md) | done |
| 02 | [The observation: rays from depth (horizon bands, `NO_READING`, the weapon fix), the entity set (14 types), the `self` vector, the optional screen, modality flags, no absolute coordinates](02-observation.md) | done |
| 03 | [The action space: six categorical heads, turn bins, frame skip, held USE and why first-tic USE was dropped](03-action-space.md) | done |
| 04 | [The encoders: rays, the entity set (attention, why order must not matter) and `self` into one feature vector](04-encoders.md) | done |
| 05 | [Memory and the action heads: the GRU, hidden state and resets, six categorical heads, the inference path](05-memory-and-heads.md) | done |
| 06 | [Collecting experience and GAE: rollouts, values, advantages, terminated vs truncated, per-step freshness](06-rollouts-and-gae.md) | done |
| 07 | [The PPO update: clipped objective, value loss, entropy, auxiliary heads, the first toy memory runs](07-ppo-update.md) | done |
| 08 | [The training run on DOOM: worker processes, the entry point and layout guard, laptop protection (priority, leftover check, memory watcher, cooldown), checkpoint and resume, the single-instance smoke runs](08-training-run.md) (rewritten after ADR 0008) | done |
| 09 | [Why the memory test failed, and what that taught us: no PPO bug, the fading trace, memory through the value loss, the value-scale floor, Gate A and Measurement B, the update-gate bias, why one seed is not evidence, what it means for DOOM](09-why-the-memory-test-failed.md) | done |
| 10 | [Before the first real run: the long-horizon memory ruling (ADR 0009), the three single-instance checks, the layout benchmark, the 4x8 run now in progress and how its results will be judged](10-before-the-first-real-run.md) | done |
| 11 | [Reading the first real DOOM run: what each logged metric means, the random baseline, 500-episode windows, the verdict ("learning, on the training scenario", under the corrected throughput rule), the mid-run entropy collapse and recovery (cause not established), health metrics, what the run does not show; the toy diagnostic D0-D4 (planned, not yet run)](11-reading-the-first-run.md) | done |
| 12 | [Reward shaping and the episodic coverage bonus (ADR 0010): the nine terms and their guards, setups refused before start, the coverage cells, the anneal and its per-env clock, the wrapper and factory, the `--shaping my_way_home` flag (off by default), `engine_return` vs shaped `return`, resume offsets, the `.prev` checkpoint copy](12-rewards-and-coverage-bonus.md) | done |
| 13 | Exploration, second stage: the bonus ablation of ADR 0005 (with and without coverage, once run) and the two-stream trainer that gives the bonus its own value head (ADR 0010 change 4, not built); learned novelty E2 is a placeholder only | next |
| 14 | Pixel branch and modality dropout (ADR 0003 Stage A1, ADR 0006 ruling 5) | |
| 15 | Spatial memory, second stage: the ego-centric neural map (ADR 0003 Stage B; ADR 0008 says it is needed for door-location memory) | |
| 16 | Curriculum and training maps, including `heldout_maps/` (ADR 0005, ADR 0008 Decision 6, `map-forge`) | |
| 17 | Results of the toy diagnostic D0-D4 and a second seed of the first run (once run) | |

Lessons 13 to 17 are a provisional order, named from the ADRs only. Lesson 12 taught the reward terms and the episodic coverage bonus together, because they are one piece of code (`rewards/`, `intrinsic/`, ADR 0010), now wired into `train/run_doom.py` behind `--shaping my_way_home`, off by default. As of 2026-11-25 no training run has used the shaped reward (only a 40 s 1x1 smoke, reported to the teacher but not recorded in `reports/runs/`), and on `my_way_home` all nine terms have scale 0, so a shaped run pays the scenario's own reward plus the coverage bonus only. Lesson 11 cites `train/trainer.py` line numbers from before the shaping change; `episode_record` is now at `:70-89`.

All ADRs 0001 to 0010 are marked **Proposed, pending ratification by the user** as of 2026-11-25. Two files share the number 0007; ADR 0008 (`0008-reconcile-0007-files-memory-gate-and-safety-defaults.md`) reconciles them, and `tests/test_adr_numbering.py` now prevents a repeat. The first DOOM training run (4x8, `my_way_home`, run `20261125-nogit-7a675759`) ran from 08:04:45 to 10:25:02 on 2026-11-25 and is taught in lesson 11. Its verdict is "learning, on the training scenario" under the advisor's corrected throughput rule, which is not yet recorded in a decision file; under ADR 0009 as written it would be "no verdict" (one seed, the map it trained on, no held-out evaluation).
