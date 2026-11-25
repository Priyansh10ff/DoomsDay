# Single-env throughput: does the ray invalid-depth mask explain 877 to ~370 decisions/s?

Date: 2026-11-25. Run by env-engineer. One engine at a time, never more; no training run.

## Setup

- `python -m env.bench --per-key` (EVAL sensors: freedoom2 MAP01, skill 3, 160x120, HUD on, frame skip 4, **screen on**, 1 ms timer request, random actions, 25 s cap). Single env, 1 process, no vectorization (vectorized bench not run: one-engine constraint).
- Power: AC, battery 49%, scheme "Turbo". 24 logical CPUs. `vizdoom.exe` count from other processes checked before each run (map-forge may run one engine): 0 for every run except B1 (1, probably the previous run's engine still exiting during the 3 s gap). CPU load before runs 1-4% (A1: 28%).
- **A** = code as it was. **B** = same code with `env.observation.invalid_depth_mask` monkeypatched to return `None` from a scratchpad wrapper (no repo edit, nothing to remove). B therefore also skips the `np.where` in `rays()`; the entity valid-depth change from the same round-3 fix stays in, so B is not the full round-2 code.
- Runs interleaved A1 B1 A2 B2 A3 B3.

## Results: A vs B (before the info-field change)

| Run | Variant | Decisions/s | Step ms p50 / p90 / p99 | Python CPU s (25 s) |
|---|---|---|---|---|
| A1 | mask on | 425.5 | 1.39 / 4.12 / 12.23 | 16.56 |
| B1 | mask bypassed | 596.4 | 1.15 / 3.54 / 7.59 | 18.83 |
| A2 | mask on | 483.2 | 1.35 / 3.99 / 10.32 | 17.11 |
| B2 | mask bypassed | 458.6 | 1.24 / 4.04 / 12.23 | 16.78 |
| A3 | mask on | 349.9 | 1.95 / 4.28 / 12.81 | 15.14 |
| B3 | mask bypassed | 389.6 | 1.47 / 4.20 / 12.22 | 15.34 |

A: median 425, range 350-483. B: median 459, range 390-596. The ranges overlap; B2 < A2. B1 is the run that lifts B's range, and it is the one run with another `vizdoom.exe` present (cause unverified: a lingering engine or map-forge).

**Microbenchmark on 300 real EVAL states** (best of 5): mask alone 23.0 us, rays without mask 42.0 us, mask + rays 70.8 us, entities 14.9 us. **The mask costs about 29 us per decision.** The drop being explained (877 to ~370/s) is about 1.5 ms per decision, about 50 times larger.

## Where the step time goes (scratchpad profiler, 15 s each)

| Notifications buffer | Decisions/s (runs) | `advance_action` ms | all Python observe ms |
|---|---|---|---|
| on, size 4 (as shipped) | 366, 329, 421, 368 | 2.0-2.6 | 0.37-0.39 |
| on, size 1 | 340, 450 | 1.8-2.5 | 0.40 |
| off | 447, 487, 488, 568 | 1.36-1.83 | 0.36-0.39 |

- About 85% of a step is the engine's `advance_action`; all Python observation work (mask, rays, entities, self, screen) is ~0.39 ms.
- The notifications buffer off was the fastest setting in all 4 of its runs, which suggests about 0.5 ms per decision, but this is **suggestive, not measured**: the later `env.bench` runs with it on (533-634/s, below) beat these profiler runs with it off, so machine drift is as large as the effect. It feeds `info["keys_held"]` only. Removing it is a design question for `advisor`, not done. Even with it off the env stays well below 877/s.

## After adding the reward fields to `info` (AMMO0-9 appended to the game variables)

| Run | Decisions/s | Step ms p50 / p90 / p99 |
|---|---|---|
| post-1 | 532.8 | 1.47 / 2.71 / 8.47 |
| post-2 | 578.4 | 1.44 / 2.27 / 7.16 |
| post-3 | 633.7 | 1.39 / 1.89 / 6.96 |

Faster than the A runs, which can only be machine state (the change adds 10 game variables and work). No measurable cost; the noise (350-634/s for near-identical code within one hour) swamps it. Observations are bit-identical: SHA-256 of every obs array over a seeded 300-step EVAL rollout is `4b9d16c1...` before and after.

## Reading

- **The mask is not the cause** of the drop from 877/s. It costs ~29 us/decision (~1-2%). Not optimised: the saving would be within noise, so it is not worth touching a tested function.
- **The drop is not attributed.** Candidates, none proven: (1) the notifications buffer (about 0.5 ms; whether it was enabled or sized 1 in the 877/s round could not be checked, as there is no git history); (2) machine state: wall time per step varies 2x between back-to-back runs of identical code, and engine time dominates.
- Conservative single-env figure: **~350-630 decisions/s** (screen on, frame skip 4, so ~1,400-2,500 tics/s).

Raw JSON: session scratchpad (`ab_pre_*.json`, `post_*.json`), not copied into the repo.
