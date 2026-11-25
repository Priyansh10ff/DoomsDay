# ADR 0010: Reward shaping terms and the episodic coverage bonus (first version)

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: `reward-designer`. Number allocated by the coordinator.

This is the reward ADR that `0005-exploration-and-curriculum-maps.md` deferred. It builds that ADR's exploration option E1 and nothing of E2. It edits no earlier ADR. It **departs from two standing rulings**, listed under "Conflicts with standing ADRs", which the `advisor` should confirm or overrule.

Tags: **[M]** measured, **[C]** read from code or reference notes, **[J]** judgement. **No scale in this ADR has been measured in training.** Every number is [J] unless tagged.

## Context

- The first DOOM run (bundled `my_way_home`, sparse built-in reward, no bonus) is in progress. ADR 0009 Decision 5 says that a flat curve with a healthy trainer is answered by ADR 0005's coverage bonus. This is what the next run uses.
- `env.DoomEnv` returns the engine reward unshaped and a privileged `info` with: absolute pose, `killcount`, `itemcount`, `secretcount`, `keys_held`, `engine_reward`, and on ending steps `dead`, `timeout`, `level_exit`, `last_step_damage_taken`, `final_obs_fresh`. **It does not carry health, armor, ammo or a cumulative damage counter.** [C]
- After death or level exit the ending step's `info` is a copy of the previous step's. Pose and counters do not move on that step. [C]
- `my_way_home` pays -0.0001 per tic through `living_reward` and +1 at the goal through its map script, then exits. Adding `map_exit_reward` paid the goal twice (probe R3.4). The goal object is a GreenArmor pickup. [M]
- The trainer has one value stream. `value_int` in `agent/model.py` is a stub and a test asserts it gets no gradient. [C]
- Env worker processes must never import torch, and nothing from the trainer reaches a worker except actions. [C]
- Rules: rewards may use privileged information; observations may not. Nothing in `rewards/` or `intrinsic/` may run at inference.

## Options considered

**Where the reward is computed**
- I1. In the trainer, from `infos`. Needs edits in `train/trainer.py` (forbidden while the run is live) and makes the trainer handle the stale ending `info`.
- I2 (chosen). A `gymnasium` wrapper around the env inside its worker, built by a factory that wraps `DoomEnvFactory`. One call site changes in `train/run_doom.py`. Cost: the anneal clock must be per-env and resumes need an offset.
- I3. Engine reward knobs (`kill_reward` etc.). Their signs are untested (`docs/reference/vizdoom-api.md`), they cannot express guards, and they are how the goal was paid twice.

**Coverage bonus shape**
- C1. Landmark set with pose including heading, +0.1 each (Sample Factory's wrapper, cited in 0007-recipe). Heading in the key pays for turning; up to 200 x 0.1 = 20 per episode against an exit of 1.
- C2. Count-based, bonus shrinking as 1/sqrt(n). The sum over revisits diverges, so pacing between two cells pays for ever.
- C3 (chosen). First visit to an x/y grid cell pays 1/budget, for at most `budget_cells` cells per episode. Bounded in [0, 1] per episode by construction.

**Normalisation**
- N1. Running std of intrinsic returns (RND recipe, 0007-recipe). State lives per worker process, differs between workers and restarts on resume.
- N2 (chosen). Bounded by construction (C3); the coefficient alone sets the size. Stateless.

## Decision

### 1. Integration

`rewards.factory.ShapedEnvFactory(inner_factory, RewardConfig, CoverageConfig, start_decisions)` returns `rewards.wrapper.ShapedRewardEnv(inner())`.

- The wrapper returns the env's observation object untouched, never reads the action, and never removes or changes an `info` key. It adds `info["shaping"]` (every step: raw and paid value per term, engine, extrinsic, intrinsic raw / coefficient / scaled, total) and `info["shaping_episode"]` (ending step: episode sums, cells visited, guard counters, unscaled engine return).
- The factory forwards `is_engine_instance`, so the layout guard still counts every engine, and keeps the inner factory, so its `phase2_engine_test` refusal still runs. It refuses a vanilla-eval config. It validates the whole reward setup in the main process before any engine starts.
- Evaluation and inference build `env.DoomEnv` directly. A test asserts that no file under `agent/` or `env/` imports `rewards`, `intrinsic` or `configs.reward_config`, and that importing them loads neither torch nor vizdoom.
- **v1 sums the bonus into the single reward** (`sum_intrinsic_into_reward=True`). This is temporary and is forced by the no-edit limit on `train/`; see "Conflicts".

### 2. Extrinsic terms

All scales are named fields of `configs/reward_config.py::RewardConfig`. Each is a non-negative magnitude; the sign is in the raw value. Paid value = scale x raw. Total = sum of paid values + `engine_reward_scale` x engine reward. Defaults are the `FULL_LEVEL_REWARD` preset.

| Term | Raw value per decision | Scale | Reason for the scale |
|---|---|---|---|
| `level_exit` | 1 on an exit | 1.0 | The anchor. Equals `my_way_home`'s goal, so return scales carry over between scenarios, and meets 0007-recipe's "one step at most about 1". |
| `keycard` | new key colours held | 0.1 | Progress signal for the long-horizon objective. Three keys total 0.3, well under the exit. No reference value exists (0007-recipe). |
| `kill` | new kills | 0.02 | Small: kills are a means. 20 kills fill the whole positive cap. See conflict 1. |
| `damage_taken` | minus health points lost | 0.003 | 0007-recipe's Sample Factory value for health lost. A full health bar costs 0.3. |
| `health` | health points gained | 0.002 | Must not exceed `damage_taken`, or a hurt-then-heal loop pays. A medkit (25) pays 0.05. |
| `ammo` | ammo gained, in units of 10, at most 5 units per decision | 0.005 | Inside 0007-recipe's 0.0002 to 0.002 per raw unit (this is 0.0005). Spending ammo is not penalised, so the agent is not taught to stop shooting. |
| `armor` | armor points gained | 0.001 | Must not exceed `damage_taken`. Green armor (100) pays 0.1. |
| `death` | minus 1 on death | 0.75 | 0007-recipe's mapping. Below the exit, so an attempt with an even chance of success beats hiding. |
| `time` | minus 1 | 0.0002 | Half of `my_way_home`'s -0.0004 per decision. Must satisfy `time x episode limit < death`: with 0.75 the limit is 3,750 decisions. |

Guard parameters: `positive_shaping_cap` 0.4 (episode cap on the summed positive pay of kill, health, ammo and armor; chosen so that cap + 3 keys + starting coverage coefficient = 0.95 < 1.0); `ammo_unit` 10 (one clip); `ammo_step_clip_units` 5 (a backpack cannot dominate a step); `exit_confirm_engine_reward` 0.5 (0007-recipe's two-signal exit rule: the exit pays only if the final step's engine reward is at least half of `map_exit_reward` 1.0, leaving room for per-tic costs; disagreements are counted and logged, not paid).

Health, armor and ammo pay **gains only**. A loss of health is paid once, by `damage_taken`; paying it again as a negative health delta would count it twice.

### 3. No double counting with a scenario's own reward

- `engine_reward_scale` is 0.0 for full levels: the engine reward is ignored and only our terms pay.
- For a bundled scenario it is 1.0 and the scenario's reward passes through. `rewards.validate.check_reward_setup` then refuses any non-zero scale on a term the scenario already pays. What the scenario pays is derived from its non-zero engine knobs (`KNOB_PAYS` maps every knob to our terms; an unclassified knob is refused) plus `scenario_script_pays`, which declares payouts made by a map script that no knob reveals.
- `MY_WAY_HOME_REWARD`: pass-through on, script pays `level_exit` (value 1.0), `living_reward` pays `time`; all nine of our scales are 0. The other seven are off because the map has no monsters or keys, `info` lacks their fields, and an armor term would pay the goal armour a second time. **The next run's reward is therefore the scenario's own reward plus the coverage bonus, and nothing else.**
- `PASSTHROUGH_REWARD` with `NO_COVERAGE` returns exactly the engine reward (the "no bonus" arm of ADR 0005's ablation).

### 4. Episodic coverage bonus (E1)

`intrinsic/coverage.py`, settings in `CoverageConfig`.

| Field | Default | Reason |
|---|---|---|
| `cell_size` | 128 map units | About one doorway to a small room; a corridor is one cell wide, so a cell is a place and not a step. 0007-recipe cites a 75-unit radius; 128 is coarser to make edge effects rarer. |
| `budget_cells` | 64 | A new cell pays 1/64; the episode's raw bonus is at most 1. The reachable cell count of `my_way_home` is not measured; read `cells_visited` from the first run and resize. |
| `coef_start` | 0.25 | One new cell pays 0.0039, about ten decisions of `my_way_home`'s time cost and about four decisions of walking, so entering a new cell is net positive. A whole episode's bonus stays under a quarter of the goal. |
| `coef_end` | 0.05 | Not zero: later curriculum maps are new places. Small enough that the goal dominates. |
| `hold_decisions` | 0 | No reason to delay. |
| `anneal_decisions` | 60,000 per env | About 1.9M decisions at 32 envs, the size of one 2.5-hour run (ADR 0009). |

- The cell set is emptied at every reset. The start cell is marked visited and pays nothing.
- Heading and height are not part of the cell.
- The coefficient is linear from `coef_start` to `coef_end`, then flat (`intrinsic/schedule.py`). **Its clock is per-env decisions**, counted by the wrapper. Envs step in lockstep, so this is total decisions / env count. On resume the factory must be given `start_decisions`, or the schedule restarts at the top.
- Intrinsic and extrinsic reward are logged separately at every step and per episode.
- `intrinsic/learned_novelty.py` is a marked placeholder for E2 that raises `NotImplementedError`. E2 needs torch and policy features, so it belongs on the trainer side, not in this wrapper. **ICM is not built.**

### 5. Exploits and guards

Every guard is part of the reward computation. None constrains or overrides an action. Each row has a test in `tests/test_reward_exploits.py` unless another file is named.

| # | Exploit | Guard | Test |
|---|---|---|---|
| E1 | Spinning on the spot for coverage | Heading is not in the cell key | `test_spinning_on_the_spot_earns_no_coverage` |
| E2 | Jittering across a cell edge or corner | A cell pays once per episode; per-episode budget bounds any map | `test_jittering_across_a_cell_edge_pays_once`, `test_coverage_is_bounded_on_an_endless_map` |
| E3 | Standing still to avoid the death penalty | Start cell unpaid; every idle step is strictly negative; no term pays for staying alive | `test_standing_still_earns_nothing_and_loses_to_reaching_the_exit`, `test_no_term_pays_per_step_for_staying_alive` |
| E4 | Taking damage and healing in a loop | Gains only; refusal of `health` or `armor` above `damage_taken`; episode cap if the damage counter misses a loss | `test_hurt_then_heal_loop_never_nets_positive`, `test_heal_farming_stays_non_positive_even_if_damage_went_unreported` |
| E5 | Farming kills or pickups (respawning monsters, spawned Lost Souls, big ammo jumps) | Episode cap 0.4 on repeatable positive pay; kill counter as a high-water mark; ammo step clip | `test_farming_pickups_and_kills_is_capped_per_episode`, `test_kill_counter_that_drops_and_returns_is_not_paid_twice`, `test_ammo_weapon_switch_and_big_jumps` |
| E6 | Key paid again (flickering report, key handed out at start) | Once per colour per episode; keys held at reset never pay | `test_key_is_paid_once_per_colour_and_not_for_keys_held_at_start` |
| E7 | Weapon switch read as an ammo gain | Per-slot ammo totals, not the selected weapon's ammo | `test_ammo_weapon_switch_and_big_jumps` |
| E8 | Firing for reward (the ICM failure) | No term reads the action; spending ammo pays nothing; firing cannot change a cell | `test_firing_until_empty_earns_nothing` |
| E9 | Door toggling | No term reads the action or any door state; the far side of a doorway pays once | `test_door_toggling_earns_nothing_because_rewards_never_see_the_action` |
| E10 | Delaying the exit to collect more | Config refused unless exit > cap + 3 x keycard + `coef_start`; no positive per-step term | `test_no_episode_without_an_exit_can_out_earn_the_exit`, `test_setups_where_delay_could_beat_the_exit_are_refused` |
| E11 | Dying early to stop the time penalty | Config refused unless `death > time x episode limit` | `test_dying_early_is_worse_than_waiting_out_the_clock` |
| E12 | Goal or term paid twice | Section 3 | `tests/test_rewards_and_coverage.py::test_my_way_home_goal_is_paid_exactly_once`, `test_double_pay_setups_are_refused` |
| E13 | A reward carried across episodes (health 30 to 100 at respawn) | Every baseline is taken from the reset `info` | `tests/test_rewards_and_coverage.py::test_deltas_start_from_the_reset_info_and_never_cross_episodes` |
| E14 | Damage paid twice on the ending step | One damage source per step | `tests/test_rewards_and_coverage.py::test_damage_has_one_source_per_step` |
| E15 | A term silently paying zero | Non-zero scale with an absent `info` field raises at reset | `tests/test_rewards_and_coverage.py::test_missing_info_field_raises_instead_of_paying_zero` |

**Not guarded, stated plainly:**
- **Hiding versus trying (E3).** Idling to the time limit costs about `time / (1 - gamma)` in bootstrapped value (0.067 at gamma 0.997), far less than death at 0.75. Nothing in a reward can make hiding worse than dying without making suicide attractive. The check is behavioural: watch the share of episodes ending by time-out and `cells_visited`.
- **Coverage pulls the agent away from a visible exit** for at most `coef x (remaining cells / 64)`. This is the intended exploration pressure and it anneals.
- **Events in the final decision are lost.** A kill, key or pickup in the same decision as death or exit is not paid, because the ending `info` is stale.
- **`KILLCOUNT` semantics** (infighting kills, resurrected monsters) are unprobed.
- **Ammo slots that share a pool** (pistol and chaingun) would count one pickup in two slots: bounded at twice a small bonus.
- **The cap ends kill and pickup reward after 0.4**, which may be too tight for monster-heavy levels.
- **A non-Markov reward.** Coverage and the cap depend on episode history the policy must track in its recurrent state.

## Conflicts with standing ADRs (for `advisor`)

1. **0007-recipe's reward-designer row is not followed on three values.** Health +0.005 gained / -0.003 lost pays 0.002 per point for a hurt-then-heal loop; this ADR uses +0.002 / -0.003. Kill +1 equals the exit, so farming kills would beat finishing the level; this ADR uses 0.02 under a 0.4 cap. Coverage +0.1 per cell for up to 200 landmarks is 20 per episode; this ADR bounds it at 0.25. **Question: is "the exit must out-earn everything else in an episode" the right hard rule, or should combat stages relax it?**
2. **ADR 0005 and 0004 require the bonus to have its own value head and discount; 0007-recipe adds extrinsic 2, intrinsic 1, intrinsic gamma 0.99, running-std normalisation.** v1 sums into one stream, with stateless normalisation. The single stream is temporary. **Question: is the bounded-by-construction normalisation accepted in place of the running std once the second stream exists?**
3. **0007-recipe's two-signal exit rule** is implemented in the reward, but whether a campaign exit ends the episode is still unprobed. Until it is, `FULL_LEVEL_REWARD` must not be used for a real run.

## Changes needed elsewhere

**`worker` (`train/`), for the next run:**
1. `train/run_doom.py`: replace `factory = DoomEnvFactory(env_cfg)` with `ShapedEnvFactory(DoomEnvFactory(env_cfg), MY_WAY_HOME_REWARD, MY_WAY_HOME_COVERAGE, start_decisions=per_env_decisions(decisions_done, run.num_envs))`, behind a flag (for example `--reward my_way_home|passthrough`). On `--resume`, read `decisions_done` from the checkpoint **before** the vec env is built; today the env is built first.
2. `train/trainer.py::episode_record`: copy `final_info["shaping_episode"]` into `episodes.jsonl` (at least `engine_return_unscaled`, `intrinsic`, `intrinsic_raw`, `cells_visited`, each `paid/<term>`). **`return` will otherwise be the shaped return and must not be read against ADR 0009's R0 thresholds, which are in engine reward.**
3. Per update, log the means of `info["shaping"]` fields (`extrinsic`, `intrinsic`, `intrinsic_coef`) to `metrics.jsonl`. For an env that ended, the step's values are in `final_infos[i]`, not `infos[i]`.
4. Later, for the two-stream trainer: set `sum_intrinsic_into_reward=False`, store `info["shaping"]["intrinsic"]` as a second (T, N) reward, add a second GAE with its own gamma, train `value_int`, and update the test that asserts it gets no gradient.

**`env-engineer` (`env/`), before any non-zero health, armor, ammo or damage scale:** add to every per-step `info`: `health` (HEALTH), `armor` (ARMOR), `ammo` (tuple of AMMO0 to AMMO9, per slot, **not** SELECTED_WEAPON_AMMO), `damage_taken_total` (cumulative DAMAGE_TAKEN). Info path only. Also: probe whether a campaign exit ends the episode, what `KILLCOUNT` counts, and whether DAMAGE_TAKEN includes damage absorbed by armor and damage from floors.

**`evaluator`:** the vanilla-reward-only evaluation of ADR 0005 uses `env.DoomEnv` unwrapped.

## Risks

- **Unmeasured scales.** Every number is a judgement; the first run with the bonus is the first evidence.
- **Cell budget mis-sized.** If `my_way_home` has far fewer than 64 reachable cells the bonus is smaller than intended; if a later map has far more, exploration pay stops early.
- **One value stream with a decaying bonus** makes the value target non-stationary.
- **The coverage bonus may not help key-then-door progression** (ADR 0005's caveat stands), and `my_way_home` is one layout: success there says nothing about unseen maps.
- **Rule compliance.** Position reaches the reward only. If any later change routed `info["shaping"]` into an observation it would breach the rules; the import test covers `agent/` and `env/`, not a future trainer change.

## Validation experiment

1. **Unit tests** (done): `tests/test_rewards_and_coverage.py`, `tests/test_reward_exploits.py`, 55 tests, synthetic `info` and the toy env, no engine.
2. **One-instance smoke run** after `worker`'s change 1 and 2 (`trainer-ops`, a few minutes): `engine_return_unscaled` of a goal episode is about 0.95 to 1.0 and never near 2; `cells_visited` distribution recorded; throughput unchanged within noise.
3. **ADR 0005's ablation** (`evaluator`): `my_way_home`, `PASSTHROUGH_REWARD` + `NO_COVERAGE` against `MY_WAY_HOME_REWARD` + `MY_WAY_HOME_COVERAGE` at equal decisions. Measure exit rate on **engine return**, `cells_visited`, time-out share, shots per episode.

Reverse or amend if: the bonus arm does not beat the no-bonus arm on exit rate (resize `budget_cells` and `coef_start` from the measured cell counts before concluding); time-out share rises with the bonus on (the agent is wandering, lower `coef_end` or shorten the anneal); shots per episode rise with the bonus (should be impossible here; treat as a bug); the smoke run shows a goal return near 2 (double pay; stop).
