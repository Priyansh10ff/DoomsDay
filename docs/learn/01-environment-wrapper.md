# Lesson 01: The environment wrapper (`DoomEnv`)

> **Changed since first written (2026-11-25).**
>
> *First revision (ADR 0006):*
> - Config fields sorted into four groups, plus `assert_same_sensors`.
> - The Windows timer request became a config switch.
> - USE is held for the whole decision.
> - The wrapper stops time-limit episodes itself, so the final observation at a time-out is fresh.
> - The key-message pattern was fixed.
> - The screen became optional.
>
> *Second revision (ADR 0007):*
> - **Two new `info` keys:**
>   - `final_obs_fresh` (every step) says whether the returned observation is the true current state.
>   - `last_step_damage_taken` (at episode end) reports damage in the final decision, even after death.
> - **TRAIN is now the bundled `my_way_home` scenario:** -0.0001 per tic, +1.0 at the goal, which ends the level. The default map-exit reward was removed for training, because it paid the goal twice.
> - **EVAL gained two harness fields,** `map_wad` and `iwad_file`, so judges' maps delivered as a WAD can be loaded.
> - **EVAL now allows exactly one extra console argument,** `+viz_nocheat 1`.
> - The "EVAL map is also a training map" warning is resolved: no Freedoom map is trained on any more.
> - All line numbers were re-checked.
>
> *Third revision (ADR 0008):* training configs now **refuse held-out maps**: anything under `heldout_maps/`, and the Freedoom Phase 2 campaign maps (see the new paragraph after the "Resolved" box in 3.2). Line numbers in `env/config.py` re-checked again.

Files taught: `env/__init__.py`, `env/config.py`, `env/wintimer.py`, `env/doom_env.py`, and the tests that guard them (`tests/test_config_and_source.py`, `tests/test_env_engine.py`, `tests/test_env_episode.py`). The supporting tools `env/bench.py` and `env/probes/run_probes.py` are described in section 3.7.

Design records: ADRs 0001, 0002, 0006 and 0007 (`docs/decisions/0007-a0-trainer-rulings-and-first-run-prerequisites.md`). **All seven ADRs (0001 to 0007) are marked "Proposed, pending ratification by the user".** The code follows them, but they can still change.

---

## 1. What problem this piece solves

Our network has to learn to play DOOM by trying things, and the only teacher is the game itself. To make that possible, three things must be true.

1. The learning code needs **one fixed way to talk to the game**: "here is what you can see", "I press these buttons", "here is how well that went", "the game is over". ViZDoom's own API is lower level than that.
2. We need **two kinds of game**. In training we want to cheat to make learning easier: kill the player on command, spawn a key, use special training maps with their own scripts and rewards. In the final judging, the game must be stock DOOM with vanilla rules. If a training cheat leaked into evaluation, we could be disqualified (`docs/RULES.md`, "Engine and evaluation").
3. The game knows things the agent must never see, such as the player's exact map position. Training code still needs those things. So we need a place for **secret, training-only facts** that is kept apart from what the network sees.

`DoomEnv` is the piece that does all three.

## 2. The idea in plain words

Think of a **remote-controlled robot in a building**, with a person operating it from another room.

- The operator only sees what the robot's camera and sensors send back: the **observation**.
- Every so often the operator moves the joystick and presses buttons: the **action**.
- A judge calls out points ("+1, you reached the goal"): the **reward**.
- When the robot is destroyed, reaches the goal, or the time runs out, the round ends: the end of an **episode**.
- The judge also keeps a private notebook (where the robot really is, how many items it has picked up, whether the last picture sent was up to date). The operator never sees it: the **`info` dictionary**.

The wrapper is the radio link between the operator's room and the robot. It defines exactly what travels each way.

Where the analogy stops being accurate:

- A real robot keeps moving while the operator thinks. DOOM is **frozen** between our decisions and only moves forward when we send an action.
- Our "operator" (the network) gets a specific bundle of numbers, described in lesson 02.
- In real life nobody hands out points. Here the reward is a number the code computes.

### Terms used in this lesson

- **Agent**: the thing that chooses actions. For us, the neural network plus the small amount of code that picks a button choice from its output.
- **Policy**: the rule the agent uses to choose an action from an observation. For us, the neural network. "Policy path" means "the data route that ends at the network's input".
- **Environment**: everything the agent interacts with; here, DOOM seen through our wrapper.
- **Observation**, **action**, **reward**, **episode**: as in the analogy above.
- **Learner**: the training code that takes collected experience and updates the network's weights (lessons 06 and 07).
- **Gymnasium**: a Python library that defines a standard shape for environments: a class with `reset()` and `step()` methods.
- **Tic**: one tick of DOOM's internal clock. The game runs at 35 tics per second of game time.
- **Mod**: any change to the game made for training: cheat commands, map scripts, changed or extra engine rewards, or extra data from the engine. Allowed in training, forbidden in final evaluation.
- **God-view**: data that covers the whole level, not only what the player can see right now.

## 3. A walk through our actual code

### 3.1 The package front door: `env/__init__.py`

`env/__init__.py:2-5` exports `DoomEnv`, `EnvConfig`, `EVAL_CONFIG` and `TRAIN_CONFIG`.

### 3.2 The settings: `env/config.py`

**One settings object for everything.** `config.py:76-109` defines `EnvConfig`, a frozen dataclass (its fields cannot be changed after creation; `.replace(...)` at `:115-116` returns a modified copy).

**Four groups of fields.** The tuples at `config.py:235-241` sort every field into exactly one group; a test fails if a field is added without a group (`tests/test_config_and_source.py:62`).

| Group | Fields | What it means | Allowed at EVAL |
|---|---|---|---|
| **Vanilla rules** (`RULE_FIELDS`, `:235-236`) | `iwad`, `scenario_wad`, `game_args`, `objects_info`, `sectors_info`, `engine_rewards`, `extra` (`:102-109`) | What "unmodified engine, vanilla rules, no mods" means. | Only the vanilla value; `game_args` may also be exactly `+viz_nocheat 1`. |
| **Harness** (`HARNESS_FIELDS`, `:237`) | `maps`, `skill`, `episode_timeout_tics`, **`map_wad`**, **`iwad_file`** (`:89-95`) | Things the judges' evaluation program decides. Not mods. | Any legal value. |
| **Sensor / action plumbing** (`SENSOR_FIELDS`, `:238`) | `frame_skip`, `render_hud`, `turn_bins`, `max_entities`, `screen_enabled` (`:96-101`) | The agent's own sensors and controls. | Any value, but **the same as in training**. |
| **Host plumbing** (`HOST_FIELDS`, `:239`) | `name`, `vanilla_eval`, `win_timer_ms`, `phase2_engine_test` (`:78-88`) | Settings of our own program that do not touch the game. | Any value. |

Some terms from the table:

- An **IWAD** is the main DOOM data file (maps, textures, monsters). We use Freedoom, bundled with `vizdoom`; `config.py:62` and `:69-73` allow only `freedoom1` and `freedoom2` as stock names.
- A **PWAD** (`scenario_wad`) is an add-on map file loaded on top of the IWAD. It can contain **ACS**, DOOM's built-in map scripting language.
- `game_args` are extra console commands, such as `+sv_cheats 1`.
- `objects_info` and `sectors_info` are whole-level god-views; they may only feed `info`.
- **Engine rewards**: rewards ViZDoom computes inside the engine. `config.py:55-59` lists every such knob; `config.py:60` says the vanilla setting is `map_exit_reward = 1.0`.

**New harness fields for judges' maps (ADR 0007 ruling 6).** We do not know how the judges will deliver their maps. If they come as a WAD file, EVAL must be able to load it, but our training PWADs must never be able to sneak in the same way.

- `map_wad` is a WAD that supplies the maps, and `iwad_file` is an IWAD path supplied by the harness (`config.py:93-95`). Both are loaded **as given** (`game_path` and `pwad_path`, `:198-203`; used at `doom_env.py:136-139`).
- Under EVAL they are legal if the file exists (`config.py:158-162`).
- A `map_wad` under the repository's training-map folders `maps/` or `curriculum/` is rejected (`:163-165`, `:210-233`).
- `scenario_wad` (the training PWAD) must still be empty under EVAL, and `validate()` refuses `map_wad` and `scenario_wad` together (`:175-176`).
- Tests: `tests/test_config_and_source.py:302-343`.

**The vanilla check.** `assert_vanilla()` (`config.py:129-167`):

- `:133-150`: every rule field must have its vanilla value. `game_args` must be `()` or **exactly** `("+viz_nocheat 1",)` (`:140-142`, list at `:208`). That one argument only *removes* information (lesson 02) and is needed to test the agent with telemetry missing (ADR 0006 ruling 5). Nothing else is allowed (`tests/test_config_and_source.py:325`).
- `:151-165`: the harness fields only have to be legal: skill 1 to 5, at least one map name, a positive time limit, and existing files.

`validate()` (`config.py:169-180`) also checks that every engine-reward name is real and that `frame_skip >= 1`, and calls `assert_vanilla()` when `vanilla_eval` is true. `DoomEnv.__init__` calls `validate()` before it starts the engine (`doom_env.py:105`), so **you cannot even start an EVAL environment with a mod in it**. The tests try ten mods on EVAL and check each fails (`tests/test_config_and_source.py:35-53`), and check that changing a harness, sensor or host field does *not* fail (`:69-75`).

**Same sensors in training and evaluation.** A network learns to read its inputs in one exact format. `sensor_signature()` (`config.py:118-120`) lists every sensor field plus the ray bands; `assert_same_sensors(trained, evaluated)` (`:244-250`) raises if any differ (test `tests/test_config_and_source.py:85-91`). Lessons 06 and 08 show that every checkpoint stores this signature and that resuming refuses a mismatch.

**The two named configs.**

- `EVAL_CONFIG` (`config.py:253`): every default. That means freedoom2 MAP01, skill 3, frame skip 4, timeout 126000 tics (one hour of game time), status bar on, screen on, no mods, and the timer request off.
- `TRAIN_CONFIG` (`config.py:297`) is now built from the **bundled ViZDoom scenario `my_way_home`** by `bundled_scenario_config` (`:271-292`). That function reads the scenario's `.cfg` file (`_parse_vizdoom_cfg`, `:257-268`) and takes from it:
  - the scenario WAD, the map, the skill and the time limit;
  - every reward knob the `.cfg` sets.

  It does **not** take the `.cfg`'s sensor settings (status bar, resolution, buttons), so training sees exactly the same sensors as EVAL. It keeps the timer request on (`win_timer_ms=1`).

What `my_way_home` is (`engine-probes.md`, R3.4):

- A small maze. A script (ACS) puts the player at one of 17 random start points, facing a random direction.
- Reward: **-0.0001 per tic** (the `living_reward` knob). Picking up the green armour runs the script line `reward = 1.0; Exit_Normal(0);`, which pays **+1.0** and ends the level.
- Skill 5, time limit **2100 tics** (60 s of game time, 525 decisions). Its map is named `MAP01` inside `my_way_home.wad`; it is not Freedoom's MAP01.
- **No map-exit reward.** The default `map_exit_reward = 1.0` was deliberately *not* added. The probe measured that adding it pays the goal twice: a total of 1.9992 instead of 0.9992. The code comment at `config.py:273-277` says so.

Why the change: ADR 0005 reserves Freedoom Phase 2 (freedoom2) maps as **never trained on**, and the first `TRAIN_CONFIG` trained on freedoom2 MAP01-MAP03. ADR 0007 ruling 7 picked `my_way_home` for the first run. It has no doors, keys or monsters and a fixed layout, so success there proves the pipeline works, not that the agent generalises (ADR 0007, risks). Test: `tests/test_config_and_source.py:344`, and on the real engine `tests/test_env_episode.py:301` checks the goal pays once and ends as a level exit.

> **Resolved: "EVAL means vanilla rules, not unseen map".** The first revision warned that EVAL's MAP01 was also a training map. TRAIN no longer uses any freedoom2 map, so MAP01 is now unseen by training. That is still **one** map, though; a proper held-out map set is `map-forge`'s job and does not exist yet. Some engine *tests* still use freedoom2 MAP01 (`tests/env_test_helpers.py`), for its probed door geometry. The file states it is never a training config.

**Training configs refuse held-out maps (ADR 0008 Decision 6).** The split between training and evaluation maps is now enforced from both sides:

- EVAL already refuses a `map_wad` from `maps/` or `curriculum/` (above).
- Every **non-vanilla** (training) config now runs `assert_trainable()` inside `validate()` (`config.py:169-180`, `:182-196`), before any engine starts. It refuses:
  - any `scenario_wad`, `map_wad` or `iwad_file` under the new folder **`heldout_maps/`** (`HELDOUT_MAP_DIR`, `:216`), with no exemption;
  - the **Freedoom Phase 2 campaign maps**: a freedoom2 IWAD with no scenario or map WAD, that is, freedoom2's own levels by name.
- The engine tests that need freedoom2 MAP01's probed geometry set a host field, `phase2_engine_test=True` (`:84-88`, listed in `HOST_FIELDS` at `:239`), which lifts only the Phase 2 refusal, never the `heldout_maps/` one. Its comment says training code never sets it and trainers should refuse a config that has it set. The trainer now does: `DoomEnvFactory` raises if the field is set (`train/doom_factory.py:17-22`; lesson 10).
- Tests: `tests/test_config_and_source.py:356-390` (refusals, EVAL still allowed to load from `heldout_maps/`, refusal before the engine starts).

ADR 0008 plans `heldout_maps/generated/` (our generator's reserved seeds) and `heldout_maps/external/` (hand-made or third-party maps), with a tracked manifest of file hashes.

### 3.3 The Windows timer request: `config.py:64-66`, `:122-127` and `env/wintimer.py`

**The problem.** Each `step` waits for the separate `vizdoom.exe` program. On Windows, that wait is often rounded up to the system clock tick, 15.6 ms. With one environment, the bench measured a median of **83 decisions per second** with the default clock, against **877** with a 1 ms clock request (5 runs each, `reports/runs/2026-11-25-env-timer-variance-bench.md`).

**The switch.** ADR 0006 ruling 3: **on for training, off for evaluation and inference** (real-time play is only 8.75 decisions per second).

- `EnvConfig.win_timer_ms` (`config.py:81-83`): 1 for TRAIN, `None` for EVAL.
- `effective_timer_ms()` (`config.py:122-127`) lets the environment variable `DOOMSDAY_WIN_TIMER_MS` override it (`0` turns it off).
- `env/wintimer.py:17-31`: `TimerRequest` calls `timeBeginPeriod` at most once and the matching `timeEndPeriod` at most once, only if the first call succeeded. Tests: `tests/test_config_and_source.py:132-147`, `tests/test_env_episode.py:230`.
- In `DoomEnv.__init__` the setting is read before the engine starts (`doom_env.py:128`) and the request is made only after it started (`:130`).

The trade-off: the request removes idle waiting, but each environment process then keeps a CPU core busier, so the laptop draws more power during training.

### 3.4 Building the engine: `doom_env.py:133-169`

`_make_game()` turns an `EnvConfig` into a running `vizdoom.DoomGame`:

- `:136-139`: load the IWAD (stock, or the harness's `iwad_file`), then the one PWAD if any: the training scenario, or the harness's `map_wad`.
- `:140-146`: map, difficulty, 160x120 screen as three colour planes, status bar on or off, no window, no sound.
- `:147-148`: turn on the **depth buffer** and the **labels buffer** (image-sized arrays the engine fills each frame; lesson 02).
- `:149-150`: god-view object and sector lists, off unless a training config turns them on.
- `:151-152`: the **notifications buffer** (the game's text messages), sized to `frame_skip` tics. It is used only for `info["keys_held"]`.
- `:153-154`: which buttons and which **game variables** (numbers the engine reports each frame, such as HEALTH) exist.
- `:155-158`: the engine's own time limit is set **one decision later** than ours, as a backstop.
- `:159-163`: mode, engine rewards, extra console arguments.
- `:164-168`: start the engine (`vizdoom.exe`); if starting fails, close the half-made game.

### 3.5 The Gymnasium loop: `reset()` and `step()`

```
obs, info = env.reset()
while True:
    action = agent(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

- `self.observation_space` (`doom_env.py:109-120`) is a **Dict space**: named arrays with fixed shapes. `screen` is added only when `screen_enabled` is true (`:117-118`).
- `self.action_space` (`:121`) is a **MultiDiscrete** space, `[3, 3, 7, 2, 2, 2]` (lesson 03).

**`reset()` (`doom_env.py:172-190`).**

- `:173`: Gymnasium's base class sets up `self.np_random`. Passing `seed=` makes the run repeatable (a **seed** fixes a random generator's sequence).
- `:175`: choose the map.
- `:176-179`: draw a per-episode engine seed, load the map, start the episode.
- `:180-186`: clear the wrapper's memory: entity tracker, previous heading, previous damage, previous action (zeros), running reward, keys held, step counter.
- `:187-190`: build `(obs, info)` from the first frame; `info["final_obs_fresh"] = True`.

**`step(action)` (`doom_env.py:192-235`).**

- `:195-196`: **frame skip**. Turn the six head indices into one button list (`encode`, lesson 03) and hold it for `frame_skip` = 4 tics with one engine call.
- `:197-198`: if the engine ended the episode during those tics, there is no new frame.
- `:199-203`: update the keys-held record from the game's text (info only).
- `:204-206`: the **reward** for this step: the change in the engine's running total, so all four tics count (`tests/test_env_episode.py:21-33`).
- `:207-208`: remember this action (the next observation's `prev_action`) and count the step.

**Terminated versus truncated (`doom_env.py:210-214`).**

- **terminated**: the game itself ended (death, or level exit; in `my_way_home`, reaching the goal is a level exit). There is no "after".
- **truncated**: we stopped the clock (time limit), but the game could have gone on.

`time_up` (`:212`) is the wrapper's own limit, reached while the episode is still running; `engine_timeout` (`:211`) is the backstop. `terminated = finished and (dead or not engine_timeout)`; `truncated = time_up or (finished and engine_timeout and not dead)`. Lesson 06 shows why the learner must treat them differently.

**What `step` returns at the end of an episode, and the two new `info` keys** (`doom_env.py:215-233`; documented at `:9-18`):

| How it ended | Observation returned | `final_obs_fresh` | `last_step_damage_taken` |
|---|---|---|---|
| Not ended | the new frame | True (`:229`) | absent |
| **Time limit** (`time_up`, `:230-233`) | the **new frame**: the wrapper stopped the clock while the state could still be read | True | damage in the final decision, from the observation path |
| **Death or level exit**, or the engine backstop (`:215-225`) | the **previous step's observation, repeated**: the engine gives no state after it ends an episode (`engine-probes.md`, R2.5) | **False** (`:225`) | read straight from the engine's `DAMAGE_TAKEN` variable, which stays readable after the end (`:222-224`; R3.3) |

Both keys are privileged (`info` only) and are for the trainer:

- **`final_obs_fresh`** tells the learner whether it may use the returned observation to estimate the future (lesson 06). Before this key existed, the trainer had to guess with one setting for all endings.
- **`last_step_damage_taken`** gives the damage label for the last decision even after death, when the repeated observation cannot show it (lesson 06). One quirk from the probe: the console cheat `kill` does not add to `DAMAGE_TAKEN`, so it reports 0.0 there, while a real lethal barrel reported 79 (R3.3).

Tests:

- `tests/test_env_episode.py:51` checks the flag on every step, and that it is False after death.
- `:76` checks the damage on a lethal barrel.
- `:140` checks the fresh time-limit observation.
- `:36` checks that a time-out gives `truncated`, `:98` that death gives `terminated`, and `:259-269` that time-outs mid-decision with USE held are still `truncated`.

`info["engine_reward"]` (`:234`) repeats the step reward, so later **reward shaping** code (extra training-only reward terms we design) can log the engine's part separately.

### 3.6 Two paths: what the agent sees versus `info`

The docstring states it (`doom_env.py:3-7`):

- **Game variables are read in one list, one order** (`doom_env.py:38-44`): the agent's body (0-14), velocity and heading (15-18), absolute position, counts and dead (19-25).
- **`split_vars` (`:69-74`)** cuts it into `ObsVars` (`:47-55`, policy path) and `InfoVars` (`:58-66`, privileged path).
- **`_observe` (`:267-272`)** calls `_build_obs` and `_build_info` separately; `_build_obs` never receives `InfoVars`.
- **`_build_info` (`:310-318`)** holds absolute x, y, z and angle; kill, item and secret counts; the map name; the episode seed; the config name; keys held; the tic; the step count. `step` adds the end-of-episode keys above.

**Privileged information** means facts from the engine's god-view or that would let a network memorize a map. They may help training, but never as a network input.

One honest detail: `ObsVars.heading` (`:55`) *is* the absolute heading. It is used only inside `observation.ego_motion` to rotate the agent's own velocity and compute its turn; the heading itself never reaches the observation (lesson 02; ADR 0001 ruling (d)).

**Keys held, from the game's messages (`doom_env.py:77-91`, `:262-265`).** The engine has no "keys held" variable, so the wrapper reads pickup messages: Freedoom's "Blue passcard secured" / "Red skeleton key secured" (seen on the real engine), and DOOM's "Picked up a blue keycard" (from DOOM's text table, untested). Locked-door refusals ("Yellow key needed for this door.") must not match; tests use captured engine text (`tests/test_config_and_source.py:256`, `:261`) and a real locked door (`tests/test_env_episode.py:210`). A key picked up in a decision that ends in death or exit is still not reported (`doom_env.py:199-201`).

How the tests guard the split:

- `tests/test_config_and_source.py:191` reads the source of `_build_obs` and fails if it mentions `InfoVars`, `pos_x`, `keys_held`, `notifications`, `objects`, `sectors` and similar words, or if `atan2` appears in `doom_env.py`.
- `tests/test_env_engine.py:65-70`: no observation key shares a name with an `info` key.
- `tests/test_env_engine.py:79-112`: a **smoke test** (a quick check that catches large mistakes, not a proof) that searches observation numbers for privileged values.
- `tests/test_env_engine.py:57-59`: EVAL's engine really has no extra arguments and no objects, sectors or automap output.

### 3.7 Housekeeping and supporting tools

- `render()` (`doom_env.py:237-241`) returns the full colour frame for humans.
- `close()` (`:243-253`) shuts down the engine and releases the timer request; safe to call twice. `__del__` (`:255-259`) is a safety net. Always close environments yourself. Lesson 08 shows how the training run makes sure no `vizdoom.exe` is left behind.
- `env/bench.py` measures one environment's speed (`--timer`, `--screen` switches). `env/probes/run_probes.py` holds the engine experiments, including round 3 (`engine-probes.md`). Neither is imported by `doom_env.py`.
- `tests/conftest.py` adds the repository root to Python's import path.

## 4. A tiny worked example with real numbers

From the first revision's run of `DoomEnv(EVAL_CONFIG)`, `reset(seed=0)`, seven steps (real output, trimmed):

```
timer active: False  action_space MultiDiscrete([3 3 7 2 2 2])
rays         shape=(3, 80) dtype=float32
...
screen       shape=(3, 60, 80) dtype=uint8
modality     shape=(3,) dtype=float32
info pos/angle/tic: -192.0 -192.0 0.0 1
step 1 a=[1, 0, 3, 0, 0, 0] r=0.0 term=False trunc=False tic=5 angle=0.00 ego=[0. 0. 0. 0.]
step 3 a=[0, 0, 6, 0, 0, 0] r=0.0 term=False trunc=False tic=13 angle=320.01 ego=[ 0.037  0.031  0.    -0.222]
step 5 a=[1, 1, 3, 1, 1, 0] r=0.0 term=False trunc=False tic=21 angle=280.02 ego=[0.127 0.139 0.    0.   ]
step 7 a=[0, 0, 3, 0, 0, 0] r=0.0 term=False trunc=False tic=29 angle=280.02 ego=[0.058 0.063 0.    0.   ]
```

And from this revision's run of the same steps on the current code (real output):

```
TRAIN: train ('MAP01',) skill 5 timeout 2100 rewards (('living_reward', -0.0001),) wad my_way_home.wad timer 1
reset final_obs_fresh: True ...
step 1: fresh=True ...
step 7: fresh=True ...
```

How to read it:

- **Timer off under EVAL, on under TRAIN** (`timer 1`), as ADR 0006 asks.
- **The two paths.** The start position (-192, -192) is in `info` only.
- **Frame skip.** The tic goes 1, 5, ..., 29: exactly 4 tics per decision, 35 / 4 = 8.75 decisions per second of game time.
- **USE held.** Step 5 holds USE and still advances exactly 4 tics.
- **Reward.** Under EVAL the only reward is +1 at a level exit, so every step gives 0.0.
- **Turning.** Turn bin 6 for 4 tics: 0 → 320.01, 40 degrees clockwise.
- **`final_obs_fresh`** is True on reset and on every step, because no episode ended.
- **TRAIN** really is `my_way_home`: one map, skill 5, 2100 tics, and only the -0.0001 living reward among the engine knobs (the +1 goal comes from the map's script).

By hand: in `my_way_home`, a decision costs 4 x (-0.0001) = -0.0004; a full 2100-tic time-out costs 2100 x (-0.0001) = **-0.21**; reaching the goal quickly gives close to +1. The probe measured 0.9992 for an episode that ended right after the goal.

## 5. Why we chose this design over the alternatives

From ADRs 0001, 0002, 0006 and 0007, all **Proposed, not yet ratified**:

- **Our own `gymnasium.Env` on top of `DoomGame`**, because the bundled `VizdoomEnv` hides the label objects, defaults to frame skip 1, and does not have our action encoding (ADR 0001).
- **One guarded EVAL configuration**, built up in steps. First a `vanilla_eval` switch with `assert_vanilla()`. Then the four field groups (ADR 0006). Then the harness fields for maps (ADR 0007 ruling 6). For map loading, ADR 0007 considered:
  - 6a: keep EVAL stock-IWAD only; it cannot run judge maps delivered as a WAD.
  - 6b: let EVAL take any `scenario_wad`; that removes the guarantee that our training PWADs cannot reach evaluation.
  - 6c (chosen): a separate harness-owned map source, with the training field still forced empty.
- **`+viz_nocheat 1` as the only allowed EVAL argument**, an exact match, because it only removes information.
- **Privileged data in `info` only** (ADR 0001), including the two new trainer keys.
- **Frame skip 4** (ADR 0002), and **the timer on for training, off for evaluation** (ADR 0006).
- **`my_way_home` with its own reward as the first training map** (ADR 0007 ruling 7): no Freedoom Phase 2 map is trained on (ADR 0005), and no reward shaping is decided yet.

## 6. How it connects

- **Lesson 02**: what is inside `obs`. **Lesson 03**: the action encoding.
- **Lessons 04 and 05**: the model reads the observation and sizes itself from the spaces.
- **Lessons 06 and 07**: the learner uses `terminated`, `truncated`, `final_obs_fresh`, `last_step_damage_taken` and `keys_held`.
- **Lesson 08**: the training run creates many `DoomEnv`s in worker processes, each from `TRAIN_CONFIG` with the screen off, and stores the sensor signature in every checkpoint.

## 7. Common mistakes, and how we would notice them

| Mistake | How it would show up |
|---|---|
| Feeding something from `info` into the network. | Training reward climbs fast on the training map and collapses on held-out maps. The source and engine tests should fail. |
| Treating `truncated` like `terminated`. | Time-outs look like failures to the learner. In `my_way_home`, every episode that does not find the goal ends by time-out, so this would be very visible. |
| Ignoring `final_obs_fresh` and always using the returned observation. | After a death, the learner would judge the previous frame as if it were the final one. Lesson 06 logs a "stale time-outs" count, which must stay 0. |
| Adding `map_exit_reward` back to the `my_way_home` config. | The goal pays about 2 instead of 1 (probe R3.4). Episode returns near 2.0 in the logs mean this happened. |
| Evaluating with different sensor settings from training. | `assert_same_sensors` raises; resume refuses the checkpoint (lesson 08). |
| Passing a file from `maps/` or `curriculum/` as an EVAL `map_wad`. | `assert_vanilla` raises before the engine starts. |
| Forgetting `env.close()`. | `vizdoom.exe` processes remain in Task Manager. The training entry point refuses to start while any is alive (lesson 08). |
| Training with the timer request off by accident. | About 80 decisions per second instead of several hundred per environment. |

## 8. Check your understanding

1. A teammate wants to give the agent the number of kills so far. Where does that number live, what would have to change for the network to see it, and which test would object?
2. In one step the agent presses USE. With frame skip 4, how many engine calls does `step` make, how many tics pass, and if the engine reward is -0.0001 per tic, what reward does `step` return?
3. In `my_way_home`, one episode ends by time limit and another ends when the agent touches the green armour. For each: `terminated`, `truncated`, `info["final_obs_fresh"]`, and is the returned observation the final frame or the previous one?

---

## Answers

1. In `info["killcount"]`: read into `InfoVars` (`doom_env.py:63`, filled at `:73`) and written by `_build_info` (`:314`). To reach the network you would have to move it into `ObsVars` and add it to `_build_obs`. `tests/test_config_and_source.py::test_policy_path_receives_no_info_vars` checks the source of `_build_obs` for `killcount` and `InfoVars`. Whether a kill count is harmful is a design question (a whole-level count, not something the player perceives); ask `advisor` first.
2. One engine call, `advance_action(4, True)`, with USE held for all 4 tics (`doom_env.py:195-196`). Four tics pass. The reward is 4 x (-0.0001) = **-0.0004**. (Worked by hand.)
3. **Time limit**: `terminated = False`, `truncated = True`, `final_obs_fresh = True`, and the observation is the **final frame** (the wrapper stopped the clock while the state was readable, `doom_env.py:226-233`). **Goal reached**: the script ends the level, which is a level exit, so `terminated = True`, `truncated = False`, `final_obs_fresh = False`, and the observation is the **previous step's**, repeated (`:215-225`). The probe confirmed the goal ends "as a level exit with `final_obs_fresh = False`" (R3.4).
