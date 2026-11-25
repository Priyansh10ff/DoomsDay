# Lesson 12: Reward shaping, the episodic coverage bonus, and how they were wired into the trainer

Files taught:

- `configs/reward_config.py` (term list, presets, coverage settings);
- `rewards/terms.py` (the nine reward terms and their guards), `rewards/validate.py` (setups refused before any engine starts), `rewards/wrapper.py` (`ShapedRewardEnv`), `rewards/factory.py` (`ShapedEnvFactory`, `per_env_decisions`);
- `intrinsic/coverage.py` (the coverage bonus), `intrinsic/schedule.py` (its shrinking coefficient);
- the wiring: `train/run_doom.py` (the `--shaping` flag, `build_env_factory`, reading the checkpoint before the workers start), `train/trainer.py` (`episode_record`), `train/checkpoint.py` (the `.prev` copy and fallback);
- tests: `tests/test_rewards_and_coverage.py`, `tests/test_reward_exploits.py`, `tests/test_shaping_integration.py`.

Decision record: `docs/decisions/0010-reward-shaping-and-episodic-coverage-bonus.md` (ADR 0010).

Builds on lesson 01 (reward, `info`, privileged information), lesson 06 (rewards become advantages through GAE), lesson 08 (worker processes, factories, checkpoints), lesson 09 (what the GRU can and cannot remember) and lesson 11 (the first run's numbers).

**Facts to keep in mind for the whole lesson:**

1. **Every reward scale is a judgement, not a measurement.** ADR 0010 says so itself: "No scale in this ADR has been measured in training."
2. **Shaping is training-only.** Nothing in `rewards/` or `intrinsic/` runs at inference, and the trained agent must never need it.
3. **No training run has used the shaped reward yet.** The only shaped run is a 40-second, 1 worker x 1 env smoke with the flag on: 7 episodes. It is not recorded in `reports/runs/`; its files exist only in the session's temporary scratchpad (`smoke_shaping/`). Section 4 part E quotes its 7 episodes, which show the wiring works and nothing about learning.
4. **ADR 0010 is Proposed, not ratified** by the user.
5. **The bonus is added into the single reward**, which departs from ADR 0005 (which wants a separate value head for it). No one has ruled on this yet.
6. **Resuming a checkpoint that was trained without shaping, with the flag on, starts the bonus part-way down its schedule** (section 3.9 shows how far: often all the way down).
7. **With the flag on, `return` and `recent_return` are shaped values.** They cannot be compared with lesson 11's numbers. Use `engine_return` (section 3.8).

---

## 1. What problem this piece solves

In the first run (lesson 11) the only reward was the one built into `my_way_home`: +1 for reaching the goal, -0.0001 per tic. Every step before the goal is a tiny minus. At the start the agent presses random buttons, so it reaches the goal only by luck (14% of episodes for a random policy, lesson 11). On `my_way_home` that luck was enough. On a large level with keys and doors it will not be: a random agent may never see the exit, so there is nothing to learn from. It will also tend to stay in the room where it started, or bump into one corner, because nothing tells it that a new room is worth anything.

This piece gives the agent two kinds of extra reward during training:

- **reward terms** for things that happen in the game (picking up a key, losing health, dying, reaching the exit), so that progress pays before the exit is ever reached;
- a **coverage bonus**: a small payment the first time the agent stands in a part of the map it has not been in during this episode, so it is pushed into new rooms instead of staying in a corner.

It also has to do this **without creating new ways to cheat the reward**, and without anything leaking into the observation or into inference.

## 2. The idea in plain words

**Analogy: a museum treasure hunt for children.**

- The main prize is at the exit of the museum. That is the level exit (worth 1).
- Each child gets a stamp card with 64 boxes. The first time they enter a new room, they get a stamp. A full card is worth a quarter of the main prize. Going back into a room you already stamped gets nothing. That is the coverage bonus.
- Over the weeks, the organisers make stamps worth less (from a quarter of the prize down to a twentieth), because the children have learned to explore and should now go for the exit. That is the **anneal** (the coefficient shrinks over training).
- The rules are written so that no trick beats finding the exit: no stamp for spinning on the spot, no stamp for walking back and forth through one doorway, and the most you can get without the exit is always less than the exit. Those are the **guards**.

Where the analogy stops being accurate:

- A child can see the stamp card. **Our agent cannot.** It never sees which cells it has visited; it only feels the reward. It must remember, in its GRU memory, where it has been. Lesson 09 showed our GRU can hold a cue across a 20-decision gap but failed at 60, while a `my_way_home` episode lasts up to 525 decisions, so the agent may not "know" it already visited a place.
- Rooms in a museum are real rooms. Our "rooms" are squares of a fixed grid, 128 map units wide, laid over the map. A square can cut a real room in half, or a corridor can run through several squares.
- Children all get the same card at the same time. Our stamps lose value per environment, counted by each environment itself (section 3.6), which causes surprises on resume.

### Terms used in this lesson

New terms, defined here and added to `GLOSSARY.md`:

- **Extrinsic reward**: reward for something that happens in the game itself (an exit, a key, damage).
- **Intrinsic reward**: reward the agent gets for its own behaviour, such as exploring, not for a game event.
- **Coverage bonus (episodic)**: our intrinsic reward; a payment the first time in an episode that the player stands in a grid cell.
- **Cell**: one square of the fixed 128 x 128 map-unit grid the coverage bonus uses.
- **Coefficient**: the number the raw bonus is multiplied by before it is added to the reward.
- **Anneal**: shrink a number gradually during training, here the bonus coefficient from 0.25 to 0.05.
- **Reward hacking (exploit)**: the agent finding a way to collect reward without doing what we wanted.
- **Guard**: a rule inside the reward computation that stops one exploit; it never touches the action.
- **Double pay**: the same game event paid twice, once by the scenario's own reward and once by ours.
- **Pass-through**: letting the scenario's own engine reward flow into the training reward unchanged.
- **Preset**: a named, ready-made set of reward settings, such as `MY_WAY_HOME_REWARD`.
- **Gymnasium wrapper**: an object that sits around an environment, passes calls through to it and may change what comes back.
- **Environment factory**: a small object that builds a new environment when called; it is sent to a worker process, which calls it there.
- **High-water mark**: the largest value a counter has reached so far; pay only when the counter goes above it.
- **Non-Markov reward**: a reward that depends on the past of the episode, not only on the current situation.
- **Ablation**: an experiment that switches one piece off to measure what that piece contributes.
- **Atomic replace**: replacing a file in one operating-system step, so other readers see either the whole old file or the whole new one, never half of each.

## 3. A walk through our actual code

### 3.1 First, what is actually switched on today

This matters more than anything else in the lesson. There are three presets in `configs/reward_config.py`:

| Preset | Lines | What it pays |
|---|---|---|
| `FULL_LEVEL_REWARD` + `FULL_LEVEL_COVERAGE` | `:108-109` | all nine of our terms, engine reward ignored, plus coverage. **Cannot be used today** (section 3.4). |
| `MY_WAY_HOME_REWARD` + `MY_WAY_HOME_COVERAGE` | `:115-119` | **all nine of our scales are 0**. The scenario's own reward passes through, plus the coverage bonus. |
| `PASSTHROUGH_REWARD` + `NO_COVERAGE` | `:123-127` | exactly the engine reward, nothing added. |

The trainer's flag offers only one choice, `my_way_home` (`train/run_doom.py:36-38`). So **the reward a shaped run would get today is: the scenario's own reward (+1 goal, -0.0004 per decision) plus the coverage bonus. Nothing else.** The nine terms are built and tested, but none of them pays in any run we can launch now. Section 3.3 teaches them because they are the design for full levels, not because they are live.

Why are they all off on `my_way_home`? `reward_config.py:111-114` explains: the map script already pays the goal, `living_reward` already pays the time cost, the map has no monsters or keys, and `info` does not yet carry health, armor or ammo. An armor term would also pay the goal a second time, because the goal object is a GreenArmor pickup.

### 3.2 The settings: `configs/reward_config.py`

- `:18-19` `TERMS`: the nine term names, in logging order: `level_exit`, `keycard`, `kill`, `damage_taken`, `health`, `ammo`, `armor`, `death`, `time`.
- `:21` `CAPPED_TERMS`: the four positive terms an agent could repeat (kill, health, ammo, armor). Their total per episode is capped.
- `:26-40` `KNOB_PAYS`: for each of ViZDoom's built-in reward settings ("knobs", such as `living_reward`), which of our terms it already pays. Used to refuse double pay.
- `:43-84` `RewardConfig`. Read the sign rule at `:9-11`: **every scale is a positive size; the sign lives in the raw value.** Damage, death and time have negative raw values. So a term's pay is always `scale x raw`.
  - `:50` `engine_reward_scale`: 0.0 means "ignore the scenario's own reward", 1.0 means "pass it through".
  - `:52`, `:55` `scenario_script_pays`, `scenario_exit_value`: what the map script pays that no knob reveals (the `my_way_home` goal is paid by its script).
  - `:57-65` the nine scales. `:68-73` the guard settings. `:75` `require_exit_dominance`.
  - `:81` `sum_intrinsic_into_reward = True`: **the bonus is added into the one reward the trainer sees.** This is the departure from ADR 0005 (fact 5).
- `:87-104` `CoverageConfig`: `cell_size` 128, `budget_cells` 64, `coef_start` 0.25, `coef_end` 0.05, `hold_decisions` 0, `anneal_decisions` 60,000. `episode_max` (`:101-104`) is the largest raw bonus one episode can earn: 1.0.

The file is imported by worker processes, so it imports neither torch nor vizdoom (`:3`). A test checks this (`test_reward_code_is_training_only_and_torch_free`, `tests/test_rewards_and_coverage.py:354-371`).

### 3.3 The nine terms: `rewards/terms.py`

The most important design choice is at `:3-4` and `:106`: **`step(info, ended)` takes no action and no observation.** A reward that cannot see the action cannot pay for pressing a button. So there is no way to earn reward by firing or by pressing USE on a door again and again. The tests `test_firing_until_empty_earns_nothing` and `test_door_toggling_earns_nothing_because_rewards_never_see_the_action` check this, and `test_step_signatures_take_no_action_or_observation` checks the function signature itself.

Rewards read the `info` dictionary (lesson 01), which carries privileged facts such as absolute position. This is allowed: rewards may use privileged information, observations may not.

- `:31-34` the `info` field names. The first group exists in `env.DoomEnv` today. The second group (`health`, `armor`, `ammo`, `damage_taken_total`) **does not exist yet**; ADR 0010 asks `env-engineer` to add it.
- `:38-40` which field each term needs.
- `:43-44`, `:78-83` `MissingInfoField`: if a term has a non-zero scale and its field is missing, the code **raises an error** instead of silently paying 0. A silent 0 would look like "the agent never picks up health" when really the term was never measured.
- `:86-95` `reset`: every starting value is taken from the **first `info` of the episode**. So if the previous episode ended at 30 health and the new one starts at 100, that jump never pays. Keys held at the start never pay (`:93`).
- `:106-167` `step`, one decision:
  - `:116` `time` is always -1 raw (multiplied by its scale).
  - `:117-120` `keycard`: pays only for key colours not paid before in this episode.
  - `:121-122` `kill`: new kills since the last step, never negative.
  - `:123-126` `health`, `armor`: **gains only**. Losing health is paid once, by `damage_taken`. Paying it again as a negative health change would count it twice.
  - `:127-130` `ammo`: gains per ammo slot, in units of 10, at most 5 units per decision. Per slot, not "current weapon's ammo", so switching weapons is not mistaken for a pickup. If the slot count changes, nothing pays.
  - `:131-138` `damage_taken`: one source per step. After death or a level exit the ending `info` is a copy of the previous step's (ADR 0010 "Context"; the same reason `final_obs_fresh` is False then, lesson 01), so the cumulative counter would be stale. On every ending step (time-outs included) the env's `last_step_damage_taken` is used instead.
  - `:140-149` on an ending step: `death` pays -1 raw; `level_exit` pays +1 raw only if the final engine reward agrees (`exit_confirm_engine_reward`), otherwise a disagreement is counted. This is the "two-signal exit rule": two independent signs that the level was really finished.
  - `:151` pay = scale x raw.
  - `:152-162` **the episode cap**: once kill, health, ammo and armor together have paid 0.4 in this episode, they stop paying. Whatever was cut off is recorded as `capped_away`.
  - `:164-166` the kill counter is kept as a **high-water mark**, so a count that drops (a monster resurrected) and comes back is not paid twice.

### 3.4 Refusing bad setups before anything starts: `rewards/validate.py`

`check_reward_setup` (`:48-118`) runs in the main process when the factory is built, before any engine exists. It raises on the first problem.

- `:63-72` **double pay**: if a term is paid by the scenario (`engine_paid_terms`, `:22-33`) and also has a non-zero scale, refuse. An unknown knob is refused too (`:29-31`): every knob must be classified first.
- `:74-82` **the exit must be payable**: with the two-signal rule on, the env's `map_exit_reward` must be at least the threshold, or the exit could never pay.
- `:84-89` **no heal farming**: health or armor may not pay more per point than damage costs, or "get hurt, heal, repeat" would earn reward.
- `:91-98` coverage settings must make sense (cell size positive, coefficient only goes down).
- `:100-107` **the exit must win**: everything an episode can earn without exiting (`max_non_exit_positive`, `:42-45`: cap 0.4 + 3 keys x 0.1 + coverage 0.25 x 1 = 0.95) must be less than the exit (1.0).
- `:109-118` **no suicide shortcut**: dying (0.75) must cost more than the time penalty over a full episode. With `time` 0.0002 per decision the episode limit can be at most 3,750 decisions.

Why `FULL_LEVEL_REWARD` cannot be used today: the TRAIN config (`configs/a0_env.py`) sets only `living_reward` -0.0001, so `map_exit_reward` is 0, and the check at `:74-82` refuses. I ran this (section 4, part D). ADR 0010 adds a second reason: it is not yet probed whether a campaign exit even ends the episode.

### 3.5 The coverage bonus: `intrinsic/coverage.py`

- `:36-41` `_cell`: the cell is `(floor(x / 128), floor(y / 128))` from `info["pos_x"]`, `info["pos_y"]`. **Heading and height are not part of it**, so turning on the spot never enters a new cell.
- `:48-51` `reset`: the visited set starts with the start cell, which pays nothing. So standing still earns 0.
- `:53-64` `step`: a cell not yet visited this episode pays `1 / budget_cells` = 1/64, for at most 64 cells. A visited cell pays 0. So the raw bonus of one episode is between 0 and 1, **whatever the map's size**. That is why no running statistics are needed (ADR "N2"): the size is fixed by construction.
- The set is emptied at every reset: the bonus is about this episode, not about what the agent saw in earlier episodes. That is what "episodic" means here.

### 3.6 The shrinking coefficient: `intrinsic/schedule.py`

`LinearAnneal.__call__` (`:20-24`): hold at `start` for `hold_decisions`, then fall in a straight line to `end` over `anneal_decisions`, then stay at `end`. With our settings: 0.25 at 0, 0.05 from 60,000 on.

**Whose decisions?** The docstring (`:3-6`) explains: the wrapper lives inside a worker process and has no channel from the trainer, so it counts **its own environment's decisions**. All environments step together, so this equals total decisions divided by the number of environments. With 32 environments, 60,000 per environment is about 1.92 million total decisions. ADR 0010 sized this as "one 2.5-hour run", but the first run did 3.71 million decisions in 2 h 20 min, so at 4 x 8 the bonus would reach its floor about half-way through a run like that. At 1 x 1 it reaches the floor after only 60,000 decisions in total.

### 3.7 The wrapper and the factory: `rewards/wrapper.py`, `rewards/factory.py`

`ShapedRewardEnv` is a Gymnasium wrapper around one `DoomEnv`, inside its worker.

- `:32-41` it builds the term calculator, the coverage tracker and the schedule, and sets its clock `self.decisions` to `start_decisions` (the resume offset).
- `:51-57` `reset` passes through, then takes the starting values from the reset `info`.
- `:59-95` `step`:
  - `:60` calls the real env with the action. The action goes nowhere else.
  - `:62` copies `info`, so the env's own dictionary is never changed.
  - `:65-71` extrinsic = `engine_reward_scale` x engine reward + the terms' total; intrinsic = coefficient x raw coverage; total = extrinsic + intrinsic (because `sum_intrinsic_into_reward` is True).
  - `:72` the clock goes up by one.
  - `:74-78` `info["shaping"]` on every step: raw and paid per term, engine, extrinsic, intrinsic (raw, coefficient, scaled), total.
  - `:79-94` running episode sums, and on the ending step `info["shaping_episode"]`, which includes `engine_return_unscaled` (the plain engine return), `cells_visited` and `intrinsic_coef`.
  - `:95` returns the **same observation object** the env returned, and the shaped reward.

`ShapedEnvFactory` (`rewards/factory.py:28-55`):

- `:37-47` `__post_init__` runs when the factory is created, in the main process. It reads the inner factory's env config, works out the episode limit in decisions (`:41-43`, 2100 tics / 4 = 525 for `my_way_home`), **refuses a vanilla-eval config** (`:44-46`), and calls `check_reward_setup` (`:47`).
- `:49-51` it forwards `is_engine_instance`, so the layout guard of lesson 08 still counts each engine.
- `:53-55` calling it builds the inner env and wraps it. This happens in the worker.
- `:23-25` `per_env_decisions(decisions_done, num_envs)` = `decisions_done // num_envs`: the resume offset.

Evaluation and inference never use this factory. They build `env.DoomEnv` directly. A test checks that nothing under `agent/` or `env/` imports `rewards`, `intrinsic` or `configs.reward_config` (`tests/test_rewards_and_coverage.py:354-371`).

### 3.8 Wiring into the trainer

**`train/run_doom.py`**

- `:36-38` `SHAPING_PRESETS`: the only choice is `my_way_home`. The names are looked up only when the flag is used (the reward modules are imported inside the function, `:50-51`), so the default path imports no reward code. A test runs `import train.run_doom` in a fresh Python and checks that neither torch nor any reward module was loaded (`test_entry_point_imports_no_reward_code_or_torch_by_default`).
- `:41-54` `build_env_factory`: with `shaping=None` it returns exactly `DoomEnvFactory(env_cfg)`, the same object as before this change (`test_default_factory_is_exactly_todays_doom_factory`). Otherwise it wraps it in `ShapedEnvFactory` with the resume offset.
- `:76-77` `--shaping`, default `None`: **off unless you ask for it.** Runs started through `configs/ops_launch.py` can pass it in the JSON `argv` list, which the launcher adds to `run_doom`'s arguments (`configs/ops_launch.py:32`); no launch config uses it yet.
- `:119-123` **the order fix.** On `--resume` with shaping on, the checkpoint's `decisions_done` is read **before** the workers are built, because the factory (with its offset) is copied into each worker when the vector env is created (`:131`). Before this change the vector env was built first and the trainer state was loaded afterwards, too late for the offset. `test_resume_reads_decisions_before_the_vec_env_is_built` checks this by replacing `SubprocVecEnv` with a fake that records the factory and stops.
- `:125-127` prints the offset, so a human can see where the anneal starts.

**`train/trainer.py`**

- `:64-67` the trainer knows the wrapper's keys by name only, so `train/` never imports `rewards/`.
- `:70-89` `episode_record` (lesson 11 taught it at older line numbers, `:66-76`; the lines have moved). New at `:85-88`: if the ending `info` has `shaping_episode`, the record gets `engine_return` (the unshaped engine return) and a copy of the whole `shaping_episode`. Without shaping, the record is unchanged (`test_unshaped_record_is_unchanged`).
- `:184` `self._ep_ret += st.reward`: the reward here is whatever the env returned. With shaping on, that is the shaped total. So **`return` in `episodes.jsonl`, and `recent_return` in `metrics.jsonl` (`train/run_loop.py:53`), are shaped values.** Only `engine_return` in `episodes.jsonl` is comparable with lesson 11's run. `success` is unaffected: it is `level_exit`, a game fact.

**`train/checkpoint.py`** (short, ties to lesson 08)

- `:57-62` `save_checkpoint` writes the new checkpoint to `.tmp`, **copies** the current main file to `.prev` (through its own temp file), and only then replaces the main file with an **atomic replace**. The main file is never renamed away, so a crash at any moment leaves a loadable `checkpoint.pt`.
- `:65-73` `checkpoint_file_to_read`: use the main file; if it is missing, use `.prev`; **never** `.tmp`, which may be half-written.
- `:76-84` `load_checkpoint` records which file it actually read in `loaded_from`; `run_doom.py:143` prints it.
- `:87-90` `checkpoint_decisions_done`: reads only `decisions_done`, without building a trainer. This is what `run_doom.py:123` calls.

### 3.9 What happens on resume

The offset is `decisions_done // num_envs` of the **new** run (`run_doom.py:124`, `factory.py:23-25`). Two consequences, both visible in section 4 part C:

- **Resuming an unshaped checkpoint with `--shaping`** gives the bonus the clock of a run that never had it. The first run (3.71 million decisions at 32 envs) would start at 115,937 per-env decisions, past the 60,000 end of the anneal: **the bonus starts, and stays, at its floor 0.05.** Any 32-env checkpoint past about 1.92 million decisions does the same.
- **Resuming with a different layout** gives a wrong clock: the same 3.71 million decisions resumed at 1 x 1 gives an offset of 3.71 million.

Nothing in the checkpoint records whether shaping was on (`RunConfig`, `configs/train_config.py:96-107`, has no shaping field), so resume cannot warn about either case.

## 4. A tiny worked example with real numbers

Parts A to D come from one demo script I ran (in the session scratchpad, deleted afterwards). It imports our code and the test helper `ScriptEnv` from `tests/reward_test_helpers.py`, which replays hand-written `info` dictionaries. **No game engine and no training ran in my demo.** Part E is different: it reads the records of a run someone else made earlier.

**Part A: one short episode through the `my_way_home` preset.** The positions and engine rewards are **a made-up script**: the player walks 100 map units per decision along x (far faster than real movement), the engine pays -0.0004 per decision, and decision 5 reaches the goal (+1 - 0.0004 = 0.9996, as in probe R3.4). The printed values are **real output** of `ShapedRewardEnv`:

```
step 1 x=100 engine=-0.0004 intrinsic_raw=0.000000 coef=0.2500 intrinsic=0.000000 reward=-0.000400
step 2 x=200 engine=-0.0004 intrinsic_raw=0.015625 coef=0.2500 intrinsic=0.003906 reward=+0.003506
step 3 x=300 engine=-0.0004 intrinsic_raw=0.015625 coef=0.2500 intrinsic=0.003906 reward=+0.003506
step 4 x=400 engine=-0.0004 intrinsic_raw=0.015625 coef=0.2500 intrinsic=0.003906 reward=+0.003506
step 5 x=400 engine=+0.9996 intrinsic_raw=0.000000 coef=0.2500 intrinsic=0.000000 reward=+0.999600
episode: total=1.009718 engine_return_unscaled=0.998000 intrinsic=0.011718 cells_visited=4 length=5
```

Reading it:

- Step 1: x = 100 is still cell 0 (floor(100/128) = 0), the start cell. No bonus; only the time cost.
- Steps 2-4: x = 200, 300, 400 are cells 1, 2, 3. Each pays raw 1/64 = 0.015625, times 0.25 = 0.003906. Net per step: 0.003906 - 0.0004 = **+0.003506**. Entering a new cell is worth about ten decisions of time cost, as ADR 0010 intended.
- Step 5: the goal pays, through the scenario's own reward. Our `level_exit` term is 0 in this preset, so the goal is paid **once**.
- The episode: `total` 1.0097 is what goes into `return`; `engine_return_unscaled` 0.998 is what goes into `engine_return`. The difference is exactly the bonus, 0.0117.

**Part B: the coefficient over per-env decisions** (real output):

```
0 0.25
15000 0.2
30000 0.15
60000 0.05
200000 0.05
```

**Part C: the resume offsets** (real output):

```
decisions_done=3710000 num_envs=32 -> start_decisions=115937 coef=0.0500
decisions_done=3710000 num_envs=1 -> start_decisions=3710000 coef=0.0500
decisions_done=64000 num_envs=32 -> start_decisions=2000 coef=0.2433
```

The first line is the first run's checkpoint resumed at the same layout: the bonus would start at its floor.

**Part D: validation** (real output):

```
refused: exit_confirm_engine_reward = 0.5 but the env's map_exit_reward is 0.0: the exit term could never pay. Set the knob (env-engineer) or set exit_confirm_engine_reward to None
my_way_home preset: accepted
```

**Part E: the shaped smoke run** (real records, read from `episodes.jsonl` in the scratchpad folder `smoke_shaping/`, which I take to be the reported 40-second 1 x 1 smoke: every record has `env_index` 0, and `metrics.jsonl` has 59 updates; 7 episodes; a wiring check, not evidence of learning):

| length | success | `return` (shaped) | `engine_return` | `cells_visited` | bonus | `intrinsic_coef` |
|---|---|---|---|---|---|---|
| 184 | True | 0.9499 | 0.9265 | 7 | 0.0234 | 0.2494 |
| 525 | False (time-out) | -0.1673 | -0.21 | 12 | 0.0427 | 0.2476 |
| 525 | False (time-out) | -0.1791 | -0.21 | 9 | 0.0309 | 0.2459 |
| 525 | False (time-out) | -0.1640 | -0.21 | 13 | 0.0460 | 0.2441 |
| 525 | False (time-out) | -0.1833 | -0.21 | 8 | 0.0267 | 0.2424 |
| 525 | False (time-out) | -0.1760 | -0.21 | 10 | 0.0340 | 0.2406 |
| 525 | False (time-out) | -0.1837 | -0.21 | 8 | 0.0263 | 0.2389 |

What it shows: the goal episode's `engine_return` is 0.93, not near 2, so the goal was paid once (ADR 0010's stop condition is not met). It is just below ADR 0010's expected "about 0.95 to 1.0" only because that episode spent 184 decisions x 0.0004 = 0.074 on the time cost: the ADR's range is too narrow, not a bug. Every time-out has `engine_return` exactly -0.21, the 525-decision time cost. `return` minus `engine_return` equals the bonus in every row. The coefficient falls a little each episode, as the per-env clock runs. And `cells_visited` was 7 to 13 out of a budget of 64: if that holds over a real run, only 6 to 12 cells were paid (the start cell pays nothing), so the bonus is a tenth to a fifth of its intended size (common mistake 4). Seven episodes from an untrained network say nothing about whether the bonus helps.

**By hand, two more numbers worth knowing:**

- An agent that stands still for a whole `my_way_home` episode: 2,100 tics x -0.0001 = **-0.21** engine return, and 0 bonus (the start cell pays nothing). Shaped return: -0.21.
- The most bonus one episode can ever give: 0.25 x 1 = **0.25** at the start of training, 0.05 x 1 = 0.05 after the anneal. Always below the goal's 1.

## 5. Why we chose this design over the alternatives

From ADR 0010, "Options considered":

**Where the reward is computed.**
- *In the trainer, from `infos` (I1).* Rejected: needed edits to `train/trainer.py` while the first run was live, and the trainer would have to handle the stale ending `info` itself.
- *Engine reward knobs such as `kill_reward` (I3).* Rejected: their signs are untested, they cannot express guards, and they are how the `my_way_home` goal was once paid twice.
- *A wrapper inside each worker (I2), chosen.* Only one line in `run_doom.py` changes, and the reward code sits beside the env. Cost: the anneal clock must be per environment, and resume needs an offset (sections 3.6, 3.9).

**Shape of the coverage bonus.**
- *Landmarks with heading, +0.1 each (C1, from Sample Factory, an open-source ViZDoom training library cited in ADR 0007-recipe).* Rejected: heading in the key pays for turning, and up to 200 landmarks is 20 per episode against an exit worth 1.
- *Count-based, shrinking as 1/sqrt(visits) (C2).* Rejected: the total over revisits never stops growing, so pacing between two cells pays for ever.
- *First visit per cell, 1/64, at most 64 cells (C3), chosen.* Bounded between 0 and 1 per episode by construction.

**Normalisation.**
- *Running standard deviation of intrinsic returns (N1, the recipe of RND, "random network distillation", a published curiosity method).* Rejected: the statistic would live in each worker process, differ between workers, and restart on resume.
- *Bounded by construction (N2), chosen.* The coefficient alone sets the size; nothing to store.

**One reward stream for now.** ADR 0005 and 0004 want the bonus to have its own value head and its own discount. v1 adds it into the single reward because `train/` could not be edited while the first run was live. ADR 0010 calls this temporary and lists the trainer changes needed to split it (its "Changes needed elsewhere", item 4). Not yet ruled on.

**Three values differ from ADR 0007-recipe** (health +0.005, kill +1, coverage +0.1 per landmark), because each would let something beat the exit. ADR 0010 asks the `advisor` whether "the exit must out-earn everything else in an episode" is the right hard rule.

## 6. How it connects

- **Lesson 01**: `info` is the privileged side channel. This lesson is the first code that reads absolute position from it, and it does so for the reward only. If any later change put `info["shaping"]` or `cells_visited` into an observation, that would break the rules (ADR 0010, "Risks").
- **Lesson 06**: the shaped reward is what goes into `ro.rewards` and then into GAE. A +0.0035 step instead of -0.0004 raises the TD error by 0.0039 when a new cell is entered, so the advantage of "walk somewhere new" goes up.
- **Lesson 08**: the factory is pickled and sent to workers; that is why the wrapper cannot hear from the trainer and counts its own clock. The `.prev` copy is a fix to lesson 08's checkpoint.
- **Lesson 09**: the coverage reward is **non-Markov**: whether a cell pays depends on the episode's past. To predict it, the value head needs to remember where the agent has been. Our GRU held a cue across 20 decisions but not 60 (lesson 09), so the value guesses for "is this cell new?" will be noisy. ADR 0010 lists this as a risk.
- **Lesson 11**: lesson 11's numbers (random 0.140 success, -0.052 return; last-500 0.726 success, +0.616 return) are engine returns. With the flag on, compare them with `engine_return` and `success`, never with `return`.
- **Next**: the ablation of ADR 0005 (same decisions, with and without the bonus), and later the two-stream trainer that gives the bonus its own value head (`value_int`, which exists in `agent/model.py` but is not trained yet). A learned novelty bonus (`intrinsic/learned_novelty.py`) is only a placeholder that raises `NotImplementedError`.

## 7. Common mistakes, and how we would notice them in the logs

1. **Comparing shaped returns with the first run.** With the flag on, `recent_return` in `metrics.jsonl` includes up to 0.25 of bonus per episode. It will look better than lesson 11's +0.616 without any extra skill. *Notice:* read `engine_return` and `success` in `episodes.jsonl` instead. `metrics.jsonl` has **no** unshaped return at all (see "Things that look wrong", item 1, at the end of section 7).
2. **The goal paid twice.** *Notice:* `engine_return` near 2, or `shaping_episode["paid/level_exit"]` non-zero on `my_way_home`. ADR 0010 says: stop the run.
3. **The bonus too large: the agent wanders instead of finishing.** *Notice:* the time-out rate (`timeout` in `episodes.jsonl`) rises while `cells_visited` rises and success stays flat.
4. **The cell budget too big for the map.** If `my_way_home` has only, say, 15 reachable cells, the bonus can never be more than 15/64 of its intended size. *Notice:* look at the distribution of `shaping_episode["cells_visited"]`; it has never been measured.
5. **Resuming an unshaped checkpoint and expecting a full-strength bonus.** *Notice:* the start-up line "reward shaping: my_way_home, anneal start at 115937 per-env decisions", and `shaping_episode["intrinsic_coef"]` equal to 0.05 from the very first episode.
6. **Resuming with a different number of environments.** *Notice:* the same start-up line shows an offset that is not the original run's per-env count.
7. **Forgetting the flag.** *Notice:* records in `episodes.jsonl` have no `engine_return` and no `shaping_episode` fields.
8. **Turning on a term whose `info` field does not exist yet** (health, armor, ammo, damage). *Notice:* the run fails at the first reset with `MissingInfoField`. That is intended: better a crash than a term silently paying 0.
9. **Firing or door-toggling for reward** should be impossible. *Notice:* if shots per episode rise when the bonus is on, ADR 0010 says treat it as a bug.

**Things that look wrong or unclear:**

1. **ADR 0010's change 3 is not built.** Nothing writes per-update means of `info["shaping"]` (`extrinsic`, `intrinsic`, `intrinsic_coef`) to `metrics.jsonl`. With the flag on, `recent_return` (`train/run_loop.py:53`) is shaped and has no unshaped counterpart there. Who: `worker`.
2. **The checkpoint does not record whether shaping was on, and the resume offset divides by the new run's environment count** (`train/run_doom.py:124`, `rewards/factory.py:23-25`). Resuming an unshaped checkpoint, or resuming at a different layout, gives a bonus clock that does not match the run's history, with no warning. The integration test even builds its checkpoint from a 2-env trainer and resumes it at 1 x 1 (`tests/test_shaping_integration.py:74-93`). Who: `worker` (record the preset and per-env clock in the checkpoint), `advisor` (what the clock should mean across layouts).
3. **A test comment gives the wrong reason.** `tests/test_rewards_and_coverage.py:391` says the full-level preset "would pay time twice here". But `FULL_LEVEL_REWARD` has `engine_reward_scale` 0, so nothing is double-paid; the real refusal is the `map_exit_reward` check (`rewards/validate.py:74-82`), as part D shows. The test passes for a different reason than its comment says. Who: `reward-designer`.
4. **The ADR's ablation cannot measure `cells_visited` on the no-bonus arm as wired.** With the flag off there is no wrapper, so no `shaping_episode`; with `PASSTHROUGH_REWARD` + `NO_COVERAGE` the wrapper sets `cells_visited` to `None` (`rewards/wrapper.py:91`). ADR 0010's validation step 3 asks for `cells_visited` on both arms. Also, `--shaping` offers no `passthrough` choice. Who: `advisor`, then `worker`.
5. **Smaller differences from the ADR.** The ADR suggested `--reward my_way_home|passthrough`; the code has `--shaping my_way_home`. The exploit numbers (E1-E15) in the ADR table do not match the numbering in the test file's section headers; refer to tests by name. The 40-second shaped smoke run has no record in `reports/runs/`; its files are only in the session scratchpad, which can be cleared. Who: coordinator / `trainer-ops` for the record.
6. **The anneal length rests on a wrong run-size estimate.** ADR 0010 sizes `anneal_decisions` 60,000 per env as "about 1.9M decisions at 32 envs, the size of one 2.5-hour run". The first run did 3.71 million in 2 h 20 min, so the bonus would sit at its floor for about the second half of such a run; at 1 x 1 the anneal ends after 60,000 decisions in total. ADR 0010's smoke expectation (goal `engine_return` "about 0.95 to 1.0") is also slightly too narrow: the smoke's goal episode gave 0.9265 with no double pay. Who: `reward-designer`, `advisor`.

## 8. Check your understanding

1. Using the `my_way_home` preset at the start of training, what is the shaped return of an episode in which the agent stands still until the time limit, and what is the shaped return of an episode that walks through 10 new cells and then times out? Which of the two does the bonus prefer, and by how much?
2. A teammate resumes the first run's checkpoint (3.71 million decisions, 4 x 8) with `--shaping my_way_home` and says "now the agent gets the full exploration bonus". Why is that wrong, and why could the wrapper not simply ask the trainer how far training has gone?
3. Another teammate suggests adding `cells_visited` to the `self` vector, "so the agent knows how much it has explored and does not have to remember it". Give two reasons from this project why that must not be done.

---

## Answers

1. **Standing still:** 2,100 tics x -0.0001 = -0.21 from the engine, and 0 bonus, because the start cell pays nothing. Shaped return **-0.21**. **Ten new cells, then time-out:** still -0.21 from the engine (the time cost is the same), plus 10 x 0.25/64 = 10 x 0.003906 = 0.039 of bonus. Shaped return about **-0.171**. The bonus prefers exploring by 0.039 per episode. Both are far below reaching the goal (about +1), so the goal still dominates.
2. The offset is `3,710,000 // 32 = 115,937` per-env decisions, past the 60,000 where the anneal ends. So the coefficient is already at its floor of **0.05**, a fifth of the starting 0.25, from the first decision. The wrapper cannot ask the trainer because it runs inside a worker process, and there is no channel from the trainer to a worker for the training clock: only actions (and reset seeds) travel there (lesson 08, ADR 0010 "Context"). That is why it counts its own decisions and must be given `start_decisions` when it is built, before the workers start.
3. Any two of:
   - `cells_visited` is computed from **absolute position**, which is privileged `info`. Observations must not depend on it (rules: no absolute coordinates, no god-view). ADR 0010 "Risks" names exactly this as a breach.
   - The wrapper **does not exist at inference**: evaluation builds `env.DoomEnv` directly. The feature would be missing when the judges run the agent, so the network would see something it never saw in training.
   - The rules require that nothing reward-related is needed at inference (ADR 0010, `docs/RULES.md` "Engine and evaluation"). A reward helper inside the observation breaks that separation.
