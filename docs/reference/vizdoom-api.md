# ViZDoom API reference (condensed for the DoomsDay agent)

Fetched 2026-11-25 by web (doc fetcher). Installed: **vizdoom 1.3.3** (Windows 11, Python 3.12.10).

## Sources and version

- DoomGame API: https://vizdoom.farama.org/api/python/doom_game/
- Enums: https://vizdoom.farama.org/api/python/enums/
- GameState / Label / Object / Sector: https://vizdoom.farama.org/api/python/game_state/
- Config files: https://vizdoom.farama.org/api/configuration_files/
- FAQ: https://vizdoom.farama.org/faq/
- Release notes: https://vizdoom.farama.org/release_notes/
- Original levels: https://vizdoom.farama.org/environments/original_doom_levels/
- Custom scenarios: https://vizdoom.farama.org/environments/creating_custom/
- Repo, **`master` branch on 2026-11-25 (not the 1.3.3 tag)**:
  - https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/examples/python/multiple_instances.py
  - https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/src/vizdoom/src/viz_main.cpp

**Version caveat.** The doc site shows no version number. The release notes list **1.3.3 (2026-11-22)** as the newest release, so the site broadly matches the installed version. But the game_state page says `Sector.floor_height` was "fixed in 1.4.0", while the release notes put that fix in **1.3.1**. So the site may be built from a branch ahead of 1.3.3. Treat `floor_height` sign on 1.3.3 as **unverified**: measure it.

**Probes win.** Where this note conflicts with `docs/reference/engine-probes.md` (measured on 1.3.3), trust the probes. Each conflict is flagged below.

---

## 1. Modes and timing

| Mode | Who controls | Timing (per docs) |
|---|---|---|
| `PLAYER` (default) | the agent (code) | **Synchronous**: the game waits for the agent's action before advancing. No frame-rate cap is documented. |
| `SPECTATOR` | a human; the agent observes | Synchronous: the game waits for the agent. |
| `ASYNC_PLAYER` | the agent | Runs at a **constant speed** without waiting; default **35 tics/s**, set by `set_ticrate` / config `ticrate`. |
| `ASYNC_SPECTATOR` | a human; the agent observes | Constant speed, does not wait. |

- `set_ticrate` only matters in ASYNC modes. It has no effect once the game is running.
- One tic is 1/35 s of game time (`GameState.tic` docs).
- Source code (`viz_main.cpp`, master): when async is off, ViZDoom replaces the engine's time and wait-for-tic functions with its own stepped clock. That is why sync mode is not wall-clock limited.
- The docs say nothing about wall-clock throughput on Windows. The 15.6 ms timer-tick cap and the `timeBeginPeriod(1)` fix are **our own measurement** (engine-probes section 7), not documented.

## 2. Stepping methods (DoomGame)

| Method | Signature | Notes |
|---|---|---|
| `init` | `init() -> None` as typed | The text says it returns True/False for success, while the stub says `None`. This is a docs inconsistency. Starts the game and the first episode. Many setters do nothing after this. |
| `new_episode` | `new_episode(recording_file_path: str = '') -> None` | Full reset (variables, rewards). A non-empty path records a demo. **There is no seed argument.** |
| `set_seed` | `set_seed(seed: int) -> None` | Seeds the RNG that **generates per-episode seeds**. Config key `seed`. Default is random. 1.3.3 fixed Windows seeds above 2^31-1 in dict configs. The docs **do not say** whether calling it on a running game affects the next `new_episode`. **Test this** before relying on `set_seed(s); new_episode()` for reproducible episodes. |
| `make_action` | `make_action(action, tics: int = 1) -> float` | Equals `set_action` + `advance_action(tics)` + `get_last_reward`. Repeats the action every tic. Returns the reward summed over those tics. |
| `set_action` | `set_action(action) -> None` | The action is held until changed. Values are ordered like `available_buttons`. |
| `advance_action` | `advance_action(tics: int = 1, update_state: bool = True) -> None` | Repeats the last set action. With `update_state=False` the state and reward are not refreshed. Read them later with `get_state` and `get_last_reward`. |
| `get_state` | `get_state() -> GameState or None` | Returns **None once the episode is finished**. |
| `is_episode_finished` | `() -> bool` | Terminal. Later `make_action` / `advance_action` calls do nothing until `new_episode`. |
| `is_player_dead` | `() -> bool` | In single player, death ends the episode. |
| `is_episode_timeout_reached` | `() -> bool` | True if the episode ended by `episode_timeout`. Added 1.3.0.dev1. Use it for terminated vs truncated. |
| `is_new_episode` | `() -> bool` | True before any action. In 1.3.3 it honours `episode_start_time`. |
| `get_episode_time` | `() -> int` | Current episode tic. |
| `get_last_reward` / `get_total_reward` | `() -> float` | The reward since the last state update; the episode sum. |
| `get_last_action` | `() -> list` | Mostly for SPECTATOR. |
| `get_game_variable` | `(GameVariable) -> float` | Works even if the variable is **not** in `available_game_variables`. Returns 0 if not found. |
| `send_game_command` | `(cmd: str) -> None` | Console command. Raises if over 1023 bytes. Some commands are blocked in some modes. |
| `load_config` | `(path) -> bool` | Repeated calls overwrite only overlapping keys. Relative paths resolve from the config file. |
| `set_config` | `(config: str or dict) -> bool` | Added 1.3.0. Relative paths resolve from the **cwd**. |
| `get_instance_id` | `() -> str` | Added 1.3.1. |
| `save` / `load` | `(file_path)` | ZDoom save states. On `load`, combat counters and the total reward keep accumulating. |
| `close` | `() -> None` | Also called by the destructor. The game can be re-`init`ed. |

- Frame skip = the `tics` argument. `set_render_all_frames(True)` only matters with a visible window.
- The FAQ says that in multiplayer PLAYER mode, frame skip cannot be used because each agent must step one tic at a time.

## 3. Configuration-file keys

**Syntax** (configuration_files page):
- One `key = value` per line. Keys and values are case-insensitive, and **underscores are ignored**, so `episode_timeout` = `episodeTimeout`.
- `#` starts a comment. Do not quote strings. Relative paths resolve from the config file's directory. Later lines override earlier ones.
- A bad line is skipped with a warning on stderr.
- Lists use `key = { A B C }`, which replaces the list. `key += { D }` appends. Lists may span lines.

Defaults below come from the DoomGame setter docs. "n/s" means the docs do not state it.

| Key | Accepted values | Default | Notes |
|---|---|---|---|
| `doom_game_path` | path to IWAD | "" | Empty: looks for doom2.wad, then freedoom2.wad. |
| `doom_scenario_path` | path to PWAD | "" | Empty: default maps of the IWAD. |
| `doom_map` | map name, case-insensitive | `map01` | |
| `doom_skill` | 1 (very easy) to 5 (nightmare) | 3 | Takes effect on the next episode. |
| `doom_config_path` | path | "" (`_vizdoom.ini`) | |
| `episode_timeout` | int tics, 0 = none | 0 | The config page calls it "actions (tics)". It is tics, not decisions. |
| `episode_start_time` | int tics | 1 | Delay before the agent acts. The config example uses 14 with the comment "after the gun is unholstered". **Untested lead:** this may explain the probes' unexplained 7-tic turn dead time after `new_episode` (engine-probes section 5). |
| `living_reward` | float | 0 | Per tic. |
| `death_penalty` | float | 0 | Applied once per death (changed in 1.3.3). |
| `death_reward`, `kill_reward`, `item_reward`, `secret_reward`, `frag_reward`, **`map_exit_reward`** | float | 0 | The release notes say these were added in 1.3.0. |
| `hit_reward`, `hit_taken_reward`, `hit_taken_penalty`, `damage_made_reward`, `damage_taken_reward`, `damage_taken_penalty`, `health_reward`, `armor_reward` | float | 0 | Version added: n/s. |
| `screen_resolution` | `RES_WxH` (list below) | `RES_320X240` | Also sets the depth, labels and automap size. |
| `screen_format` | a ScreenFormat (below) | `CRCGCB` | Affects the screen and automap buffers. |
| `depth_buffer_enabled`, `labels_buffer_enabled`, `automap_buffer_enabled`, `audio_buffer_enabled`, `notifications_buffer_enabled` | bool | false | Audio needs OpenAL Soft. `init` throws `ViZDoomNoOpenALSoundException` without it. |
| `notifications_buffer_size`, `audio_buffer_size` | int tics | notif: 1; audio: n/s | |
| `objects_info_enabled`, `sectors_info_enabled` | bool | false | Cover the **whole level** (god-view). |
| `automap_mode` | NORMAL, WHOLE, OBJECTS, OBJECTS_WITH_SIZE | NORMAL | |
| `automap_rotate` | bool | false | |
| `automap_render_textures` | bool | true | |
| `render_hud` | bool | false | |
| `render_minimal_hud` | bool | false | |
| `render_weapon` | bool | true | |
| `render_crosshair` | bool | false | |
| `render_decals` | bool | true | |
| `render_particles` | bool | true | |
| `render_effects_sprites` | bool | true | |
| `render_messages` | bool | false | |
| `render_corpses` | bool | true | |
| `render_screen_flashes` | bool | true | |
| `render_all_frames` | bool | false | |
| `window_visible` | bool | false | |
| `console_enabled`, `sound_enabled` | bool | false | |
| `available_buttons` | `{ BUTTON ... }` | empty | `add_available_button(b, max_value=-1)` in code. `set_button_max_value` caps delta buttons (0 = no limit). |
| `available_game_variables` | `{ VAR ... }` | empty | |
| `mode` | PLAYER, SPECTATOR, ASYNC_PLAYER, ASYNC_SPECTATOR | PLAYER | |
| `ticrate` | int | 35 | ASYNC only. |
| `seed` | int | random | |
| `game_args` | engine CLI args string | "" | `add_game_args` in code (1.2.3+). The docs say use with caution, as it may break the library. |

**Reward getter wording:** the doom_game page describes `get_death_reward` and `get_hit_taken_penalty` as the negation of another getter. Which getter pairs with which, and the resulting signs, are **untested**. Check them before relying on them.

## 4. Buffers

| Buffer | Shape / dtype (docs) | Notes |
|---|---|---|
| screen, `CRCGCB` / `CBCGCR` | (3, H, W) uint8, channels-first | **The default is channels-first.** |
| screen, `RGB24` / `BGR24` | (H, W, 3) | |
| screen, `RGBA32` / `ARGB32` / `BGRA32` / `ABGR32` | (H, W, 4) | |
| screen, `GRAY8` / `DOOM_256_COLORS8` | (H, W) | The latter holds Doom palette indices. |
| depth | (H, W), one 8-bit channel | The docs give no units or polarity. **Probes:** larger means farther, about 8 map units per level, rows 100-119 are 0 with the HUD on at 160x120. |
| labels | (H, W), one 8-bit channel | Each pixel holds `Label.value` for the object drawn there. **Probes:** `value` for one object changes between frames; key on `object_id`. |
| automap | same shape as screen, uses `screen_format` | NORMAL shows only seen architecture. WHOLE and OBJECTS are god-view. |
| notifications | text from the last N tics | Added 1.3.0. **Probes:** on tic 1 it holds the level name, which is a map identifier. |
| audio | last N tics | Needs OpenAL. |

- `get_screen_channels()` gives the channel count of the screen and automap; depth and labels have 1.
- **Why fields can be None:** the game_state page only says what is present "depends on the configuration". `get_state()` is None after the episode ends. The probes saw depth, labels and `state.labels` read **None** under `+viz_nocheat 1`.

### Label fields
- `object_id`: equals `Object.id` when both are enabled.
- `object_name`: shared by many objects, e.g. "Zombie".
- `object_category`: e.g. "Monster", "Weapon", "Player". The current player is "Self". Added 1.3.0. Probes found: Ammo, Armor, Decoration, Gibs, Gore, Hazard, Health, Key, LightSource, Monster, Powerup, Self, Vegetation, Weapon, Explosive.
- `value`: the pixel value in `labels_buffer`.
- `x`, `y`, `width`, `height`: the bounding box in pixels. The slice is `labels_buffer[y:y+height, x:x+width]`. The box size was fixed in 1.3.2.
- `object_position_x`, `object_position_y`, `object_position_z`, `object_angle`, `object_pitch`, `object_roll`, `object_velocity_x`, `object_velocity_y`, `object_velocity_z`: listed with **no description**.
  - The docs give no units or frame. *Inference:* these are map-frame (absolute) values, like POSITION_*. Treat them as absolute and transform them to the ego frame.

### Object fields
- `id`, `name`
- `category` (1.3.1), `sector_id` (1.3.1)
- `position_x`, `position_y`, `position_z`, `angle`, `pitch`, `roll`, `velocity_x`, `velocity_y`, `velocity_z`: no description and no units; the same inference applies.
- Covers every object in the level.

### Sector and Line fields
- **Sector:** `id` (1.3.1), `floor_height`, `ceiling_height`, `lines`.
- **Line:** `x1`, `y1`, `x2`, `y2`, `is_blocking`.
- **No line specials, tags or door/lock info are exposed** (agrees with the probes).

### GameState
- Fields: `number`, `tic`, `screen_buffer`, `depth_buffer`, `labels_buffer`, `automap_buffer`, `audio_buffer`, `notifications_buffer`, `game_variables`, `labels`, `objects`, `sectors`.
- The docs describe none of the buffer fields individually. *Inference:* `game_variables` follows the order of `available_game_variables`.

## 5. ScreenResolution (36 members)

RES_160X120, 200X125, 200X150, 256X144, 256X160, 256X192, 320X180, 320X200, 320X240, 320X256, 400X225, 400X250, 400X300, 512X288, 512X320, 512X384, 640X360, 640X400, 640X480, 800X450, 800X500, 800X600, 1024X576, 1024X640, 1024X768, 1280X720, 1280X800, 1280X960, 1280X1024, 1400X787, 1400X875, 1400X1050, 1600X900, 1600X1000, 1600X1200, 1920X1080 (all with the `RES_` prefix).

## 6. GameVariable list (132 members, page order)

"nocheat" marks the ones the docs say are unavailable under `viz_nocheat`.

**Counters**

| Variable | Meaning |
|---|---|
| KILLCOUNT | Monsters killed this episode; since 1.1.5 includes players and bots. |
| ITEMCOUNT | Items picked up. |
| SECRETCOUNT | Secrets found. |
| FRAGCOUNT | Frags minus suicides. Multiplayer. |
| DEATHCOUNT | Player deaths. Multiplayer. |
| HITCOUNT | Hits landed. 1.1.5+. |
| HITS_TAKEN | Hits received. 1.1.5+. |
| DAMAGECOUNT | Damage dealt. 1.1.5+. |
| DAMAGE_TAKEN | Damage received. 1.1.5+. |

1.3.3 resets the death and combat counters on `new_episode`.

**Player state**

| Variable | Meaning |
|---|---|
| HEALTH | Can exceed 100. |
| ARMOR | Can exceed 100. |
| DEAD | True if dead. |
| ON_GROUND | Not in the air. |
| ATTACK_READY | Attack can be performed. |
| ALTATTACK_READY | Alt attack can be performed. |

**Weapons and ammo**

| Variable | Meaning |
|---|---|
| SELECTED_WEAPON | Selected weapon number. |
| SELECTED_WEAPON_AMMO | Ammo of the selected weapon. |
| AMMO1 ... AMMO9, AMMO0 | Ammo for weapon slot N. |
| WEAPON1 ... WEAPON9, WEAPON0 | Number of weapons in slot N. |

**Pose and motion (all nocheat)**

| Variable | Meaning |
|---|---|
| POSITION_X, POSITION_Y, POSITION_Z | Player position. The docs say only "position of the player"; the absolute map frame is inferred. |
| ANGLE, PITCH, ROLL | Player orientation. |
| VIEW_HEIGHT | Camera Z = POSITION_Z + VIEW_HEIGHT. 1.1.7+. |
| VELOCITY_X, VELOCITY_Y, VELOCITY_Z | Player velocity. |

**Camera (all nocheat, 1.1.7+)**

| Variable | Meaning |
|---|---|
| CAMERA_POSITION_X, CAMERA_POSITION_Y, CAMERA_POSITION_Z | Camera position. |
| CAMERA_ANGLE, CAMERA_PITCH, CAMERA_ROLL | Camera orientation. |
| CAMERA_FOV | Field of view in degrees. |

**Scripting**

| Variable | Meaning |
|---|---|
| USER1 ... USER60 | ACS global ints 1 to 60. Global int 0 is reserved for the reward and is fixed-point; convert with `doom_fixed_to_float`. |

**Multiplayer**

| Variable | Meaning |
|---|---|
| PLAYER_NUMBER | This player's number. |
| PLAYER_COUNT | Number of players. |
| PLAYER1_FRAGCOUNT ... PLAYER16_FRAGCOUNT | Per-player frags. |

## 7. Buttons (summary)

**Delta buttons:**
- TURN_LEFT_RIGHT_DELTA, LOOK_UP_DOWN_DELTA: in degrees. **Probes:** degrees per tic, and **positive turns right** (ANGLE decreases).
- MOVE_FORWARD_BACKWARD_DELTA, MOVE_LEFT_RIGHT_DELTA, MOVE_UP_DOWN_DELTA: in map units.

**Binary buttons:**
- Combat and actions: ATTACK, ALTATTACK, USE, JUMP, CROUCH, TURN180, RELOAD, ZOOM, SPEED, STRAFE, LAND
- Movement and looking: MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, MOVE_UP, MOVE_DOWN
- Weapons: SELECT_WEAPON0 to SELECT_WEAPON9, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON, DROP_SELECTED_WEAPON
- Items: ACTIVATE_SELECTED_ITEM, SELECT_NEXT_ITEM, SELECT_PREV_ITEM, DROP_SELECTED_ITEM

The FAQ notes that some weapons have `noautofire`: ATTACK must be released and pressed again.

## 8. `+viz_nocheat`

**What the docs say:**
- The depth and labels buffers "will contain noise" if `viz_nocheat` is enabled.
- POSITION, ANGLE, PITCH, ROLL, VIEW_HEIGHT, VELOCITY and all CAMERA_* variables are unavailable.
- The FAQ and the release notes do not mention it.

**Repo source** (`viz_main.cpp`, master):
- It is declared as a CVAR, default false. It can be set from the command line only, so pass it via `game_args`.
- In the code read, it **disables automap rendering** and forces the automap cheat level to 0.
- Other files were not read, so do not conclude that the flag only affects the automap.

**CONTRADICTION with engine-probes.md (1.3.3, measured):**
- Under `+viz_nocheat 1`, depth, labels and `state.labels` were **None**, not noisy. Objects and sectors info were empty.
- The pose and velocity variables read 0.0, and the screen was identical.
- **Trust the probe for 1.3.3.**

## 9. Multiple instances and multiprocessing

- `examples/python/multiple_instances.py` (master) says both processes and threads can run many DoomGame instances in parallel.
- It also says that because ViZDoom releases the GIL, there is little or no performance difference between them.
- That example is a 2-player deathmatch, hosted with `-host 2` and joined with `-join 127.0.0.1`. It is about multiplayer, not independent training envs.
- The FAQ does not cover running independent instances, threads or processes.
- The docs give no guidance on port or resource limits for many single-player instances.
- Per the probes, each instance runs a child `vizdoom.exe`.

## 10. Detecting that the level exit was reached

- **The only documented mechanism is `map_exit_reward`** (1.3.0+). The setter docs describe it as the reward "for finishing a map (finding an exit or succeeding in other programmed objective)".
- The original-levels environments use reward 1 on reaching the end of the level and 0 otherwise.
- **The docs do not say explicitly** that the episode ends when the exit is reached. They also document no `is_map_exited()`-style method.
- **Suggested check (inference, untested):** set `map_exit_reward` to a distinctive value. Then treat "episode finished AND not dead AND not timeout AND that reward was received" as exit. Verify it on a short map.
- Whether the eval harness keeps this API setting is unknown; ask the organisers.

## 11. Gotchas

1. **`get_state()` returns None at episode end.** The FAQ says there is no reliable final state and suggests a zero dummy observation.
2. **The default screen format CRCGCB is channels-first** (3, H, W), unlike RGB24 (H, W, 3).
3. **Most setters do nothing after `init()`:** resolution, format, buffers, buttons, variables, mode, ticrate, window.
4. **Underscores in config keys are ignored**, and `=` replaces a list while `+=` appends. A second `load_config` overrides only the keys it sets.
5. **`episode_timeout` counts tics, not agent decisions.** With frame skip 4, a 2100 timeout is 525 decisions.
6. **Death penalty semantics changed in 1.3.3** (now once per death). Counters reset on new episode in 1.3.3. Do not trust older tutorials.
7. **objects_info, sectors_info, automap WHOLE/OBJECTS and Label/Object positions are whole-level or god-view data.** They must be filtered or kept training-only (RULES.md).
8. **The notifications buffer leaks the level name at tic 1** (probes). Filter it before any observation.
9. **`floor_height` sign:** the docs conflict (fixed in 1.3.1 per the release notes, 1.4.0 per the game_state page). Verify on 1.3.3.
10. **`init()` docs say it returns a bool, but the stub shows `None`.** Do not rely on its return value.
11. **`set_config` resolves relative paths from the cwd; `load_config` resolves them from the config file's directory.**
12. **The sector-info SIGBUS on sectors with more than 128 lines was fixed in 1.3.2.** We have 1.3.3.
13. **Viz_nocheat docs vs reality:** see section 8. Plan for None buffers, not noisy ones.
14. **Seeding is not documented as per-episode.** There is no seed argument on `new_episode`, and whether `set_seed` takes effect on a running game is untested.
