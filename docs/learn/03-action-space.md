# Lesson 03: The action space (six heads, turn bins, held USE)

> **Changed since first written (2026-11-25):** line numbers into `env/doom_env.py` and `tests/test_env_episode.py` re-checked after ADR 0007's changes to those files; the content is unchanged.

Files taught: `env/actions.py` (all of it), `env/config.py:32-34` (turn bins), `env/doom_env.py:115`, `:121` and `:192-208` (where actions enter the game), and the tests `tests/test_config_and_source.py:166-177` and `tests/test_env_episode.py:192-207`, `:292-298`.

Design records: ADR 0002 (action space and frame skip) and ADR 0006 ruling 2 (USE), which amends it. **Both are marked "Proposed, pending ratification by the user."** ADR 0002 itself still describes "USE on the first tic only". ADR 0006 overrides that, and the code follows ADR 0006. Measurements: `docs/reference/engine-probes.md`, sections 5 and 6 and R2.2.

Builds on lesson 01 (the `step` loop, frame skip, tics) and lesson 02 (`prev_action` is part of the observation).

---

## 1. What problem this piece solves

In DOOM a player does several things at once: run forward, sidestep a fireball, turn to aim, fire, and press USE to open a door. Our network must output all of that every decision, as numbers, and the wrapper must turn those numbers into ViZDoom button presses.

Two constraints shape how:

- **Every action must come from the network** (rule 4.1). The translation from network output to buttons must be fixed plumbing. No "if a door is ahead, press USE", and no hiding choices from the network ("masking") to steer it.
- **Doors must be openable by a policy that is still learning.** A badly chosen encoding can make the most obvious behaviour, "keep USE pressed when facing a door", fail forever. That happened to our first version (section 3.4).

## 2. The idea in plain words

Think of a **game controller with six separate controls**: a forward/back stick, a left/right sidestep stick, a turning dial with seven click positions, and three on/off buttons (fire, use, run). Every 4 tics the network sets all six controls at once, and the game runs with them held in place for those 4 tics.

Each control is a **head**: one separate choice the network makes, picked from a fixed short list of options. A choice from a fixed list is called **categorical** (as opposed to a continuous number like 3.71).

Where the analogy stops being accurate:

- A human moves the controls continuously. Ours only change every 4 tics, and the turning dial has only seven settings.
- A human coordinates the controls ("strafe left *because* I am turning right"). In our design the six choices are made side by side by one network, but each head's choice does not look at what the other heads picked in the same step (ADR 0002, risks). The network can still learn sensible combinations, because all six heads read the same internal state (lesson 05).

## 3. A walk through our actual code

### 3.1 The buttons: `env/actions.py:13-18`

`BUTTONS` (`actions.py:14-17`) is the list of ViZDoom buttons the engine is told about (`doom_env.py:153`), in a fixed order:

```
MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT_RIGHT_DELTA, ATTACK, USE, SPEED
```

`actions.py:18` gives each position a name (`IDX_FWD = 0`, ..., `IDX_SPEED = 7`). Seven of these are on/off buttons. `TURN_LEFT_RIGHT_DELTA` is a **delta button**: instead of on/off, it takes a number, the turn amount per tic.

### 3.2 The six heads: `actions.py:20-24`

`HEAD_NAMES` (`:20`) and `head_sizes` (`:23-24`):

| Head | Size | Options (index: meaning) | Button(s) driven |
|---|---|---|---|
| `move` | 3 | 0 none, 1 forward, 2 backward | MOVE_FORWARD, MOVE_BACKWARD |
| `strafe` | 3 | 0 none, 1 left, 2 right | MOVE_LEFT, MOVE_RIGHT |
| `turn` | 7 | index into the turn bins (3.3) | TURN_LEFT_RIGHT_DELTA |
| `attack` | 2 | 0 off, 1 on | ATTACK |
| `use` | 2 | 0 up, 1 held down | USE |
| `speed` | 2 | 0 walk, 1 run | SPEED |

The joint number of possible actions is 3 x 3 x 7 x 2 x 2 x 2 = **504**. The network never lists all 504; it makes six small choices.

The environment tells the outside world this shape through `self.action_space = MultiDiscrete([3, 3, 7, 2, 2, 2])` (`doom_env.py:121`). The model reads the head sizes from it (lesson 04), so changing the turn bins changes the model automatically.

There is **no weapon-selection head**. ADR 0002 relies on the game's own vanilla behaviour (switching weapons on pickup and when ammo runs out), because random weapon switching early in training would harm combat learning. A head can be added later.

### 3.3 Turning: `config.py:32-34`

`TURN_BINS` = (-10, -4, -1, 0, +1, +4, +10), in **degrees per tic**. The `turn` head's index picks one of these. The probe measured (`engine-probes.md`, section 5):

- A value of +1 for 4 tics changed the heading by exactly -4.00 degrees; +4 gave -16.00; +10 gave -39.99.
- **A positive value turns right (clockwise).** DOOM's angle counts counter-clockwise, so it goes *down*.
- Turning is symmetric, fractions work, and SPEED has no effect on delta turning.
- **Turning is ignored for the first 7 tics after a new episode** (cause unknown). With frame skip 4, that affects at most the first two decisions. ADR 0006 notes it and makes no ruling.

With frame skip 4, the bins give these turns per decision:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Degrees per tic | -10 | -4 | -1 | 0 | +1 | +4 | +10 |
| Degrees per decision | 40 left | 16 left | 4 left | none | 4 right | 16 right | 40 right |

So the finest aim step is 4 degrees per decision. ADR 0002 lists this as a risk: 4 degrees may be too coarse for distant targets.

`turn_bins` is a **sensor field** (lesson 01), so it must be the same in training and evaluation; `assert_same_sensors` checks it.

### 3.4 Encoding a decision: `actions.py:27-39`

`encode(action, turn_bins)` turns the six indices into the 8-number button list:

- `:29`: unpack the six indices.
- `:30-34`: forward/backward from `move`, left/right from `strafe`. Only one of each pair can be on, because each head picks one option.
- `:35`: the turn value is looked up in the bins.
- `:36-38`: attack, use and speed are copied as 0.0 or 1.0.

`as_index_array` (`:42-43`) reshapes any action into six whole numbers.

In `DoomEnv.step` (`doom_env.py:193-196`), the result is applied with **one** engine call: `set_action(encode(...))`, then `advance_action(frame_skip, True)`. All buttons, USE included, stay exactly as set for all 4 tics. Nothing in this path looks at the observation or at earlier actions; the module docstring calls it "uniform plumbing only: no masking, no observation- or history-dependent logic" (`actions.py:4-6`).

### 3.5 USE: why it is held, not pressed on the first tic

This is the one rule that changed after the probes.

**DOOM's USE is edge-triggered.** An **edge-triggered** button acts only at the moment it goes from up to down, not while it is held. The round-1 probe confirmed this on freedoom2 MAP01's door (sector 27), recording the door's opening (ceiling minus floor) after each decision (`engine-probes.md`, section 6 and R2.2):

| How USE was sent | Door opening after each decision | What happened |
|---|---|---|
| **First tic only** on every decision (ADR 0002's original rule) | 32, 0, 32, 0, 32, 0 ... | Every decision was a *new* press, and pressing USE on a moving door reverses it. The door shakes forever. |
| **Held** on every decision (USE = 1 on 12 decisions in a row) | 32, 64, 92, 92, ... | One press. The door opens and stays open. |
| Held, pattern 1 0 0 0 1 0 0 ... | 32, 64, 92, 92, 60, 28, 0, ... | The second press (after a release) closes the door again, as normal DOOM doors do. |
| Held, pattern 1 0 1 0 0 ... | 32, 64, 32, 0, ... | Releasing and pressing again while the door is still opening reverses it. |

ADR 0002's original idea was: "USE might be edge-triggered, so press it on the first tic of each decision and release it for the rest, to make sure each decision fires". The probe showed that this makes the simplest possible behaviour, "keep USE on while facing a door", **never open the door**. ADR 0006 ruling 2 therefore made the `use` head **the plain state of the USE button for the whole decision**. A new press happens only when the head goes from 0 to 1, so two presses need at least one decision with `use = 0` in between (about 0.23 s). The network can see its own last button state in `prev_action` (lesson 02).

Held USE was also checked on **switches**: standing 36 units in front of six switch lines on MAP02 and MAP03 and holding USE for 3 decisions moved the tagged door or lift every time (`engine-probes.md`, R2.2).

The change also removed code. The old `encode` returned two button lists, and `step` made two engine calls when USE was on. That second round trip on about half of all decisions was one of the code changes made between the slower round-1 benchmark and the faster round-2 one. The bench report says the two changes were not measured separately, so their individual effect is unknown.

The tests:

- `tests/test_config_and_source.py:172-177`: USE in the button list equals the head, and `doom_env.py` contains exactly one `advance_action(` call.
- `tests/test_env_episode.py:192-207`: on the real engine, holding USE for 10 decisions in front of the MAP01 door opens it (opening at least 90) and it stays open.

### 3.6 `prev_action` in the observation: `doom_env.py:115`, `:183`, `:207`

The observation key `prev_action` is the six indices of the last decision, all zeros after a reset (`doom_env.py:183`, `:207`; test `tests/test_env_episode.py:292-298`). Its space (`doom_env.py:115`) is a Box from 0 to `max(head_sizes) - 1` = 6 for all six entries. That range is looser than the truth (`move` can never be 6). That is why the model reads the real head sizes from `action_space`, not from this Box (`agent/obs_spec.py:36-37`, lesson 04).

## 4. A tiny worked example with real numbers

From the demo run for this lesson set (one environment, closed afterwards). Real output:

```
encode use=1: [1.0, 0.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0]
```

That is `encode([1, 0, 5, 1, 1, 0], TURN_BINS)`. Read position by position against `BUTTONS`:

- `move` = 1: MOVE_FORWARD = 1.0, MOVE_BACKWARD = 0.0.
- `strafe` = 0: MOVE_LEFT = MOVE_RIGHT = 0.0.
- `turn` = 5: TURN_LEFT_RIGHT_DELTA = `TURN_BINS[5]` = **4.0** degrees per tic to the right, so 16 degrees right over the decision.
- `attack` = 1, `use` = 1, `speed` = 0.

These 8 numbers are held for all 4 tics. In the first version of this code, the same call returned two lists, one with USE = 1 (first tic) and one with USE = 0 (the other three).

From lesson 01's run on the current code (real output):

```
step 3 a=[0, 0, 6, 0, 0, 0] ... tic=13 angle=320.01
step 4 a=[0, 0, 6, 0, 0, 0] ... tic=17 angle=280.02
step 5 a=[1, 1, 3, 1, 1, 0] ... tic=21 angle=280.02
```

- Turn index 6 = +10 degrees per tic, 4 tics: the angle went 0 → 320.01 → 280.02, about 40 degrees clockwise each decision.
- Step 5 holds USE, and still advances exactly 4 tics (17 → 21) in one engine call.

## 5. Why we chose this design over the alternatives

From ADR 0002 (action space) and ADR 0006 (USE). **Both Proposed, not ratified.**

Action space options (ADR 0002):

- **A. A small hand-picked list of 9 to 12 combinations** (IMPALA style). Fast to build, but *we* would decide which combinations exist, turning is coarse, and adding USE and run doubles the list each time.
- **B. One flat list of all 504 combinations.** One big choice; slow to explore, and each button's meaning must be relearned inside every combination.
- **C. Six independent categorical heads with binned turning (chosen).** Same environment cost, a small policy head, and the joint log-probability and entropy are sums over heads (lesson 05). Sample Factory, a published DOOM agent, also split its actions into heads.
- **D. A continuous turn head plus discrete heads.** Finer aim, but mixing continuous and discrete distributions complicates PPO. Not justified before a baseline learns.

Frame skip options: 2 (finer aim, twice the decisions per game second), **4 (chosen; four published sources)**, or a learned repeat (deferred).

USE options (ADR 0006 ruling 2):

- **2a. Keep first-tic USE.** A policy that outputs USE on consecutive decisions can never open a door, and it costs a second engine call.
- **2b. First-tic press plus a forced release decision.** Hides state from the network and still lets a sampling policy reverse doors.
- **2c. Held USE, plain button state (chosen).** Removes code, one engine call per decision, and it is how the key behaves for a human on the vanilla engine.
- **2d. An environment-side cooldown that ignores repeated presses. Rejected:** it would be a rule about *when* USE may act, which is strategy in the action path (rule 4.1).

## 6. How it connects

- **Lesson 04** shows the `PrevActionEncoder`, which turns `prev_action` into a small vector for the model.
- **Lesson 05** shows the six output layers that produce one categorical choice per head, how an action is sampled, and why the inference path is only "forward pass, then sample or take the most likely option".
- **Lesson 07** shows the entropy bonus, computed per head and summed. Per-head entropy is logged, so we can see, for example, whether the `use` head stopped exploring.

## 7. Common mistakes, and how we would notice them

| Mistake | How it would show up |
|---|---|
| A policy that toggles `use` 1, 0, 1 near a door. | Doors reverse while opening. Early in training this is expected; ADR 0006 lists it as a risk. In logs: the rate of `use = 1` decisions, and episodes that stand at a door for many decisions. |
| Adding "helpful" logic to the action path (cooldowns, auto-USE near doors, forced release). | It breaks rule 4.1. `tests/test_config_and_source.py:172-177` only checks the USE value and the single engine call; review any change to `env/actions.py` or `DoomEnv.step` with `rules-auditor`. |
| Getting the turn direction backwards when reading logs. | Positive turn bins turn right and make `info["angle"]` go **down**. If a plot shows the opposite, the plot is wrong, not the agent. |
| Reading the first two decisions of an episode as "the agent does not turn". | Turning is ignored for the first 7 tics after `new_episode`. Look at decisions 3 onward. |
| Changing `turn_bins` or `frame_skip` between training and evaluation. | `assert_same_sensors` raises. If skipped, every turn would be a different size from what the network learned. |
| The judges' harness fixes a different button set (for example binary TURN_LEFT/TURN_RIGHT only). | Our model could not run as is. ADR 0002 asks the organizers; the answer is still unknown. |

## 8. Check your understanding

1. At a closed door, the network outputs `use = 1` on five decisions in a row. How many USE presses does the engine see, and what does the door do? What would have happened under the first version's "USE on the first tic only" rule?
2. The `turn` head picks index 1 for one decision. How many degrees does the player turn, in which direction, and which way does `info["angle"]` move?
3. Someone proposes adding to `env/actions.py`: "if `use` was 1 on the previous decision, send 0 this time, so every press is clean". Which ADR option is that close to, and why was it rejected?

---

## Answers

1. **One press.** USE goes from up to down once, at the first of the five decisions, and is then held. The door opens and stays open (the probe's held sequence: 32, 64, 92, 92, ...). Under the first-tic rule, each decision was a new press, so the door would go 32, 0, 32, 0, ...: opening, reversing, opening again.
2. Index 1 is -4 degrees per tic; over 4 tics that is **16 degrees to the left**. A negative turn value turns left (counter-clockwise), so DOOM's angle goes **up** by 16.
3. It is close to option **2b** ("a forced release decision") and also to **2d** (an environment rule about when USE may act). ADR 0006 rejected both. A forced release recreates the reversal problem: the next `use = 1` is a fresh press on a moving door. And it makes the action path depend on history, which is a hand-written rule about when the agent may use a button (rule 4.1). The ruling says the USE pass-through must stay unconditional.
