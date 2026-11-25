# Engine probes (ADR 0001 / 0002 validation)

Date: 2026-11-25. Run by env-engineer. Machine: Windows 11, Python 3.12.10, vizdoom 1.3.3 (PyPI wheel), gymnasium 1.4.0, numpy 2.2.6.
Script: `env/probes/run_probes.py <section>` (one DoomGame at a time, each closed in `finally`). Unless stated, the map is `freedoom2.wad` MAP01 (bundled with the package), 160x120, RGB24, HUD on, depth + labels on, seed 123.
Probes use cheat console commands (`+sv_cheats 1`, `summon`, `warp`, `kill`) and whole-level objects/sectors info **only to measure the engine**. None of this is in the EVAL path.

## 1. `+viz_nocheat 1` (section `nocheat`)

**What I ran:** the same seed and the same scripted 60-decision action sequence (frame skip 4), once without and once with `+viz_nocheat 1`. Snapshots were taken at tics 1, 121 and 237.

| Item | Without flag | With `+viz_nocheat 1` |
|---|---|---|
| screen buffer | valid | **identical** (mean absolute difference 0.0 at all three snapshots) |
| depth buffer | (120,160) uint8 | **`None`**: missing, not noised |
| labels buffer | (120,160) uint8 | **`None`** |
| `state.labels` | list | **`None`** |
| objects info | 120 to 122 objects | **empty (0)** |
| sectors info | 211 sectors | **empty (0)** |
| HEALTH, ARMOR, SELECTED_WEAPON, SELECTED_WEAPON_AMMO, DAMAGE_TAKEN, WEAPON2, AMMO2, KILLCOUNT, ITEMCOUNT, DEAD | valid | **valid, identical values** |
| POSITION_X/Y/Z, ANGLE, VELOCITY_X/Y/Z | valid | **all read 0.0** |
| notifications buffer | console text | identical text |

**Consequences:**

- The ADR's premise that depth and labels "contain noise" is wrong for 1.3.3. Under the flag they are absent.
- If the harness sets the flag, `rays`, `entities`, `entity_type` and `entity_mask` are lost. So is ego-motion, because VELOCITY and ANGLE read 0.
- `screen`, `prev_action` and the health, armor, ammo and weapon part of `self` survive.
- The wrapper handles this case. It sets `modality` = [0, 1, 0] and zero-fills the missing keys. This is tested in `tests/test_env_episode.py::test_viz_nocheat_fallback`, which runs `DoomEnv` with `+viz_nocheat 1`.

## 2. Depth buffer (sections `depth`, `depth2`, `depth_calib`, plus a HUD on/off comparison)

- **Polarity:** a larger value means farther. Walking into a wall drove the centre depth from 6 to 4 to 2, and it held at 2.
- **Units:** about **8 map units per depth level**. The ratio of true distance to depth had a median of 8.9 over 12 samples (24, 32, 72 and 525 units read 2 to 4, 9 and 63). The true distance was a ray cast against blocking sector lines from still frames. The sample is small, and the rays were cast against walls only.
- **Range:** the largest value observed was 120 (about 960 units). The uint8 ceiling of 255 would be about 2040 units (not observed). `rays` divides by 255.
- **Status bar:** with the HUD on, rows **100 to 119** read depth 0 and label 0 in every column. The 3D view is rows 0 to 99. With the HUD off, no row is all zero.
- **Horizon:** the HUD-on depth image equals the HUD-off image shifted up by exactly 10 rows. The match error was 0.0 at shift 10 and scale 1.0; the neighbouring shifts gave 0.08. So with the HUD on, the horizon sits about 10 rows above its HUD-off position. That is **row ~50** if the HUD-off horizon is the image centre (row ~60, inferred and not measured directly).
- **Superseded by ADR 0006:** bands are now horizon offsets (see R2.1), and `ADR_RAY_BANDS` and `HUD_SHIFTED_RAY_BANDS` no longer exist. The round-1 finding follows for the record.
- **ADR 0001 erratum:** the ray bands 40-55 / 56-63 / 64-79 assume a horizon near row 60. With the HUD rendered, the "horizon" band 56-63 lies below the horizon, on the floor. The wrapper keeps the ADR rows as default (`ADR_RAY_BANDS`) and defines `HUD_SHIFTED_RAY_BANDS` = 30-45 / 46-53 / 54-69 for advisor to choose.
- **CORRECTED (round 3):** the round-1 claim here, that "the weapon sprite does not appear in the depth buffer", was **wrong**. It came from reading one column (col 80) on one frame. The weapon sprite carries the Self label (value 255) at **depth 0**, and it reaches up to row 50, inside the ray bands. Before the round-3 fix this made the centre rays read 0. See R3.1.

## 3. Keys (sections `scan`, `keys`)

- **Maps with keys** (objects info at spawn):
  - freedoom1 E1M1 (blue), E1M2 (blue, red, yellow), E1M3 (blue)
  - freedoom2 MAP02 (red, blue, yellow), MAP03 (blue, red, yellow)
  - freedoom2 MAP01: none
- **Labels:** I did not navigate to a placed key. I summoned each key 96 units ahead with `summon` (TRAIN-style cheat).
  - All six appear in `state.labels` with `object_category == "Key"` and `object_name` BlueCard, YellowCard, RedCard, BlueSkull, YellowSkull, RedSkull.
  - The colour is in the name, so skull keys map to the same colour ids as cards.
  - The label `value` of the same object changed between frames (223, then 198, then 178). Tracking must use `object_id`, which the wrapper does.
- **Pickup:** walking over the keys removed them from labels and from objects info.
- **HUD icons:** at 160x120 the three key slots of the status bar change from grey to clearly coloured blobs (blue, yellow, red). At 60x80 (the `screen` key) each is still a distinct coloured patch of about 1 to 2 pixels.
  - Card and skull were held together in this test, so whether they can be told apart was **not tested**.
  - This is a judgement from viewing upscaled PNGs, not a measured classifier.
- **Notifications buffer:** on pickup it held plain vanilla strings, for example "Blue passcard secured!" and "Red skeleton key secured!" (Freedoom wording).
  - The wrapper parses these into `info["keys_held"]` (info only). A test covers it.
- **"You need a key" message on a locked door: not completed.** Finding a locked door needs line specials, which the API's sector lines do not expose (only coordinates and `is_blocking`). Out of time.

## 4. Notifications buffer contents (sections `nocheat`, `notif`)

- **Tic 1:** the buffer holds console text, including the level name, for example "map01 - Hydroelectric Plant".
  - That is a **map identifier**, so the notifications buffer must never become an observation without filtering.
  - The level name was **not** drawn on the screen at tic 1 (checked visually).
- **Gameplay:** it holds pickup messages (above).
- **`basic_notifications.cfg`:** it held "Shoot: DoomImp". That text comes from the scenario's own ACS `print`, so it is not vanilla output.

## 5. Turn delta units (section `turn`)

- **Units:** `TURN_LEFT_RIGHT_DELTA` is in **degrees per tic**. A value of +1 for 4 tics gave a heading change of -4.00. The other bins scale the same way: +4 gave -16.00 and +10 gave -39.99.
- **Direction:** a **positive value turns right (clockwise)**, so ANGLE decreases. ANGLE is counter-clockwise in degrees.
- **Symmetry:** turning is symmetric.
- **Fractions:** fractional values work (0.5 gave -2.0; 2.5 gave -10.0).
- **SPEED:** has no effect on delta turning.
- **Dead time after `new_episode`:** turning is **ignored for the first 7 tics**. With a +1 command every tic, ANGLE stayed 0 through tic 8, then changed by 1 degree per tic. Earlier probes agreed: a 30-degree command each tic for 12 tics changed nothing, and the first decision of the turn probe was lost. The cause is not known.

## 6. USE edge trigger (sections `use`, `use_test`)

**Superseded by ADR 0006 ruling 2.** The wrapper no longer uses first-tic USE; USE is held for the whole decision (see R2.2). The round-1 finding follows for the record.


- **Door used:** sector 27 of freedoom2 MAP01. It is closed (ceiling equal to floor), 16 by 128 units, at x 704-720, y -224 to -96.
- **Setup:** I warped to (670, -160) facing east (34 units from the door). I recorded the door's ceiling minus floor after each decision.
- **Held USE**, all 4 tics of every decision: 32, 64, 92, 92, 92... The door opened once and stayed open.
- **First-tic-only USE** (ADR 0002) on every decision: 32, 0, 32, 0, 32, 0... **Each decision is a fresh press**, and pressing USE on a moving door reverses it.
- **Conclusion:** USE **is edge-triggered**. Held USE fires once.
- **Consequence for ADR 0002:**
  - First-tic USE opens doors. By the ADR's own criterion, that does not trigger a reversal.
  - But a policy that outputs USE on consecutive decisions makes the door oscillate. Held USE would instead behave as a single press.
  - Reported to advisor; I built as written.
- **Switch press: not completed** (no switch located in time).

## 7. Throughput finding (not an ADR probe, but it bounds training)

- **The problem:** on this Windows 11 laptop, the round trip to the engine (`vizdoom.exe` runs as a child process) often waits one system timer tick, 15.6 ms. That caps a single env at **about 60 to 76 decisions/s, whatever the extraction cost**.
- **The fix:** calling `timeBeginPeriod(1)` in the Python process raised the same loop to 368 to 703 decisions/s across runs. On Windows 11 this call is process-scoped and released on `timeEndPeriod` or exit.
- **What did not help:** opting the child out of power throttling.
- **Unexplained:** one run with the request active still measured 75/s, although its raw-engine phase ran at 830/s.
- **Superseded by ADR 0006 ruling 3:** the constructor argument `timer_resolution_ms` no longer exists. The request is now `EnvConfig.win_timer_ms` (1 for TRAIN, off for EVAL), overridable with `$DOOMSDAY_WIN_TIMER_MS`, and released exactly once in `close()`.
- Round-2 measurements (5 runs each way): `reports/runs/2026-11-25-env-timer-variance-bench.md`.
- Details: `reports/runs/2026-11-25-env-single-bench.md`.

## Round 2 (ADR 0006 validation), 2026-11-25

Every probe below is a saved section of `env/probes/run_probes.py`. The three round-1 inline probes are now saved too: `hud_shift`, `categories` and `tic1_screen`. Of these, `hud_shift` was rerun and gave the same result, a zero-error match at a shift of 10 rows.

### R2.1 Horizon row (sections `horizon`, `hud_shift`)

- **Method:** floor pixels (labels value 1) give 1/depth linear in the row, reaching zero at the horizon. I fitted this per column over 16 poses on freedoom2 MAP01, with the HUD on and with it off. Results are in pixel-edge coordinates.

| Depth range used | Columns | HUD on | HUD off |
|---|---|---|---|
| 4-60 (raw) | all, step 4 | 56.7 | 67.7 |
| 8-60 (+0.5 quantization) | all | 55.7 | 64.2 |
| 12-60 (+0.5) | all | 52.3 | 62.3 |
| 12-60 (+0.5) | 72-87 | 52.1 | 62.1 |
| 16-60 (+0.5) | 72-87 | 51.7 | 61.7 |

- **Reading the table:**
  - The estimate falls toward 50 and 60 as near, coarsely quantized depths are excluded. This bias comes from integer depth levels of about 8 map units each.
  - The best estimates are 51.7 to 52.4 with the HUD on and 61.7 to 62.4 with it off.
  - The HUD on/off difference is exactly 10 rows in every variant.
  - Ceiling fits were unusable, because sky and outdoor areas make the interquartile range span 50 rows.
- **Larger minimum depths** (re-run after review): with depth ≥ 24 (+0.5, columns 40-119, n=31) the fit gives **49.07 (HUD on) and 59.07 (HUD off)**, with zero interquartile spread. Centre columns 72-87 give the same (n=7). With depth ≥ 32 no column had enough floor pixels.
- **Conclusion:** the estimate falls monotonically as near pixels are excluded (56.7, 55.7, 52.3, 49.1), so the earlier values near 52 were quantization bias. 50 and 60 are kept (`env/config.py: horizon_row`).
- **Caveat:** the ≥ 24 result comes from few poses (n=31 columns, likely one far floor region). It is consistent with 50/60 within about 1 row, not a large-sample confirmation.

### R2.2 Held USE: doors and switches (sections `held_use_door`, `switch`)

- **Door:** freedoom2 MAP01, sector 27. USE held for all 4 tics of each decision. Door gap after each decision:

| USE pattern per decision | Door gap |
|---|---|
| 1 1 1 1 1 1 1 1 1 1 1 1 | 32, 64, 92, 92, ... (one open, stays open) |
| 1 0 0 0 1 0 0 0 0 0 0 0 | 32, 64, 92, 92, 60, 28, 0, ... (the second press closes it, as vanilla DR doors do) |
| 1 0 1 0 0 0 0 0 0 0 0 0 | 32, 64, 32, 0, ... (a press while opening reverses it) |

- **Switches:** I stood 36 units in front of each switch line found in the WAD's LINEDEFS lump and held USE for 3 decisions. The tagged sector moved in all 6 cases:
  - MAP02 line 662: special 103, S1 open door; the door ceiling went from -88 to 8.
  - MAP02 lines 728 and 805: special 62, SR lift.
  - MAP03 lines 294, 295 and 332: special 62, SR lift.
- **Sector order:** `state.sectors` order matches the WAD's SECTORS lump order (heights identical on MAP02).
- **Conclusion:** held USE opens doors and presses switches. ADR 0006's reversal criterion is not triggered.

### R2.3 Locked-door text (sections `wadlines`, `locked`, `locked_screen`)

- **Lines used:** freedoom2 MAP02 line 215 (special 27, DR yellow) and line 379 (special 133, S1 blue locked switch), found in the WAD's LINEDEFS lump.
- **Locked-door text** (notifications buffer, wrapped in console separator characters):
  - Without the key at line 215: `Yellow key needed for this door.`
  - At line 379: `Blue key needed.`
- **After `give`:** the cheat itself printed nothing. The door and the switch's tagged sector then opened under held USE (gaps 64 and 124).
- **Buffer size:** the default notifications buffer size is **1 tic**. A message is missed if the buffer is read after a multi-tic action without raising the size. `DoomEnv` sets size = frame_skip; the probe now sets 4.
- **Consequence:** the old `_KEY_MSG` pattern matched both locked texts, so `keys_held` would have been set falsely (item 7 confirmed). The pattern now accepts only pickup wording:
  - Freedoom: "<colour> passcard secured", "<colour> skeleton key secured" (verified).
  - Doom string table: "picked up a <colour> keycard/skull key" (not probed; no Doom IWAD).
- **Regression tests:** the real-engine test is `test_locked_door_message_does_not_set_keys_held`. String tests use the captured text.

### R2.4 Entity categories of shots and missiles (section `projectiles`)

- **Shooting a wall:** gave the labels `BulletPuff` and `Explosive`. The agent's own impacts are visible entities.
- **Summoned classes:** every missile reports category **`Explosive`**:
  - DoomImpBall, CacodemonBall, BaronBall, Rocket, PlasmaBall, ArachnotronPlasma, FatShot, RevenantTracer, BFGBall
  - So do `BulletPuff` and `RevenantTracerSmoke`.
  - `Blood` is `Gibs`. `TeleportFog`, `ItemFog` and `RocketSmokeTrail` are `SFX`.
- **Item 6 confirmed:** the old table mapped every `Explosive` actor to `projectile`.
- **Fix:**
  - `Explosive` is removed from the category table.
  - Missiles are typed `projectile` by class name.
  - Puffs and trails are typed `decoration` by class name.
  - `SFX` maps to `decoration`.
  - Other `Explosive` actors fall to `unknown`.
  - `ExplosiveBarrel` (`Hazard`) gets the new `hazard` id 13.

### R2.5 State after the episode ends (section `final_state`)

- After `is_episode_finished()`, `get_state()` returns `None`.
- So a key picked up during the final decision cannot be read from the notifications buffer once the engine has ended the episode.
- **Time-limit fix:** `DoomEnv` now truncates at `episode_timeout_tics` itself, while the state is still live. The engine's own limit is set one decision later as a backstop.
  - A pickup in the final decision before the time limit is now reported.
  - The final observation is current rather than the previous one.
  - Tests: `test_key_picked_up_on_final_decision_at_time_limit` and `test_time_limit_returns_live_final_state`.
- **Remaining gap:** after death or level exit the engine returns no state, so a pickup in that final decision is still not reported.

## Round 3 (own-weapon pixels, ADR 0007 items), 2026-11-25

All sections below are in `env/probes/run_probes.py`.

### R3.1 Which depth pixels are not the world

**Sections:** `own_pixels`, `zero_depth_breakdown`, `zero_depth_where`, `zero_label0_shape`, `overlay_sources`, `label_zero_depth`.

**Weapon (`own_pixels`)**, freedoom2 MAP01 and MAP02, 150 steps each, without and with continuous firing:

- The Self label (`DoomPlayer`, category `Self`) always had value **255**, and every one of its pixels had depth **0**.
- All its pixels fell inside its box, which spans **rows 50 to 99**. That means it overlaps ray band 1 (rows 46-53) and band 2 (rows 54-69).
- On 1 step of 150 no Self label was listed, and no 255 pixels were present on that step.
- **Ray cells made entirely of weapon pixels:**
  - band 2: 117-130 per 150 steps without firing, 294 per 150 steps with it (20-56% of steps);
  - bands 0 and 1: none.

**Firing overlay (`overlay_sources`):** while firing, extra depth-0 pixels appear in rows 40-99 with a labels value that no listed label claims (0), mostly outside the Self box.

| Attack | Steps with these pixels |
|---|---|
| held | 40 of 100 |
| off | 0 of 200 |

These pixels are dark (mean RGB 21), so not the bright flash. Turning `render_messages` on or off changed nothing, and rows 0-19 had none.

**Translucent sprite (`label_zero_depth`):** per labelled class, over 300 random steps with frequent attack:

- **`BulletPuff` is the only class with depth-0 pixels:** 843 of 266,702. The puff sits on a wall several levels away, so 0 is not its depth.
- Every other class seen had no depth-0 pixel: Blood, BlueCard, RedCard, Clip, Shell, DeadMarine, DoomImp, Zombieman, HealthBonus, Medikit, Stimpack.

**World geometry:** walls are never nearer than the player radius (16 units = 2 levels; walking into a wall bottoms out at 2).

**Rule adopted** (`observation.invalid_depth_mask`): depth 0 never measures the world. Invalid pixels are depth 0, plus every Self-label pixel (included explicitly as a safeguard). `rays()` takes the minimum over valid pixels only.

**A cell with no valid pixel** reads exactly **0.0 (`NO_READING`)**: "no world measurement this step", the same code as absent telemetry. Real readings are `max(depth, 1)/255`, so they are never exactly 0. Nothing is filled in from other cells or earlier steps. In practice NO_READING occurs in band 2 under the weapon, and rarely where a puff's depth-0 pixels cover a whole cell.

### R3.2 Entity depth `d`

- `d` already used only the pixels carrying the entity's own label value, so weapon pixels never entered it directly. Two defects remained:
  - When none of an entity's pixels were visible (fully covered, 2-18 of about 200 entity sightings per 150 steps in `own_pixels`), it fell back to the depth at the box centre. That is usually a weapon pixel, so it read 0, meaning "touching".
  - The `BulletPuff`'s own depth-0 pixels were averaged into its `d`.
- **Fix:** `d` is the mean over the entity's own pixels with depth > 0.
  - If there are none, `d = NO_READING` (0.0) and `dd = 0`.
  - The keep-nearest-24 sort ranks NO_READING last.
- **Weapon bob and firing:** the weapon box moved between rows 50 and 67 (top edge), and firing adds the unlabelled overlay. The rule covers both. A real-engine regression test checks every ray cell against the raw buffers on each step while firing.

### R3.3 Game variables after the episode ends (section `damage_after_end`)

- `get_game_variable` stays readable after the episode ends. HEALTH read -999900 after the console `kill`.
- **The console `kill` does not add to DAMAGE_TAKEN** (0 before and after). Real damage does: a barrel explosion gave 79.
- So `info["last_step_damage_taken"]` is 0.0 after `kill`, and positive after a lethal barrel. Both are tested.

### R3.4 `my_way_home` bundled reward (sections `mwh_lumps`, `mwh_goal`)

- **`my_way_home.cfg`:**
  - `living_reward = -0.0001`
  - `doom_skill = 5`, `episode_timeout = 2100`, `episode_start_time = 10`
  - no `map_exit_reward`, no `death_penalty`
  - engine getters confirm: living -0.0001, map_exit 0.0, death 0.0
- **ACS (`SCRIPTS` lump of `my_way_home.wad`):**
  - On ENTER, the player is moved to one of 17 random start points with a random angle.
  - Picking up the GreenArmor (tid 22) runs `reward = 1.0; Exit_Normal(0);`.
- **Measured** (warp onto the armour, then move):

| Setting | Per-tic rewards | Episode end | Total |
|---|---|---|---|
| Bundled cfg | -0.0001, **+0.9999** (pickup tic: +1 - 0.0001), -0.0001 | level exit | 0.9992 |
| With `map_exit_reward = 1` added | -0.0001, 0.9999, **0.9999** | level exit | 1.9992 (goal paid twice) |

- **Sparsity:** the reward is -0.0001 per tic (-0.0004 per decision at frame skip 4, -0.21 over a full 2100-tic time-out) and a single +1 at the goal, on a 17-start-point layout. It is sparse but not zero.
- **TRAIN_CONFIG** (`bundled_scenario_config("train", "my_way_home")`) carries exactly `(("living_reward", -0.0001),)`. A real-engine test checks that the goal pays once and ends as a level exit with `final_obs_fresh = False`.
- **Not applied from the cfg:**
  - `episode_start_time` (the env has no such field);
  - the cfg's `render_hud = false`, resolution and buttons. The sensors stay identical to EVAL, so the HUD is on in training.

### R3.5 Same seed, same episode? (section `same_seed`, ADR 0008)

**Method:**
- Runs through `DoomEnv.reset(seed=s)`, the same path the trainer uses: `set_seed` on the running game, then `new_episode`.
- After each reset, a fixed 30-decision action sequence (no USE).
- Per config: one instance resets with seed 7, seed 7 again, then seed 8. That instance is closed, and a **fresh** instance resets with seed 7.

| Config | Spawn pose, seed 7 (all three resets) | Spawn pose, seed 8 | 31 observations, 7 vs 7 (same instance) | 7 vs 7 (fresh instance) | 7 vs 8 |
|---|---|---|---|---|---|
| TRAIN (`my_way_home`, ACS random start) | (464.0, -176.0, 148.16) | (543.99, -672.09, 259.23) | identical | identical | differ |
| EVAL (freedoom2 MAP01, monsters) | (-192, -192, 0) | (-192, -192, 0) (fixed map start) | identical | identical | differ |

**Conclusion:**
- `set_seed` on a running game does take effect at the next `new_episode`. The same seed reproduces the episode exactly, across resets and across instances. Different seeds give different episodes: the spawn on `my_way_home`, and engine randomness on MAP01.
- On MAP01 the final pose was the same for seeds 7 and 8, but the observations were not; what differed was not isolated.
- **Effect on ADR 0008 Decision 4 (resume re-seeding):** re-seeding with the same seed on resume would replay the same episodes. Deriving a new seed per session does have an effect.
- `reset()` without a seed draws the next episode seed from the env's own generator, so episodes differ.
- One map per config and two seeds: a small sample.

## Probes run inline in round 1 (now saved as sections)

In round 1 these ran through `python -c`. They are now saved as sections `hud_shift`, `categories` and `tic1_screen`. Only `hud_shift` was rerun in round 2.

- the HUD on/off depth-shift match (section 2)
- the `object_category` vocabulary scan over freedoom1 E1M1-E1M4 and freedoom2 MAP01-MAP05
- the tic-1 screen capture (section 4)

The categories found: Ammo, Armor, Decoration, Gibs, Gore, Hazard, Health, Key, LightSource, Monster, Powerup, Self, Vegetation, Weapon. `DoomImpBall` reported "Explosive".

## Not completed

- Card versus skull distinguishability in HUD pixels.
- Whether a killed monster's label keeps category "Monster" (it affects the entity table).
- Exact horizon row: R2.1 brackets it within about 2 rows but does not pin it.
- These are resolved in round 2 and no longer open: the locked-door text (R2.3), switch presses (R2.2, held USE only; first-tic USE was not retried because it was withdrawn), and the HUD-off horizon (R2.1).

## Round 4 (rules audit 01: E1, R9, engine config file), 2026-11-25

Sections `pwad_map_fallback [reset|init|bundle]`, `occlusion [steps]`, `ini_path` in `env/probes/run_probes.py`; plus a one-off comparison script (old vs new ini path through `DoomEnv`, results below).

### R4.1 A map name absent from the PWAD loads the IWAD's map (audit E1: confirmed)

| Request | What loaded (sectors / objects / level name) |
|---|---|
| freedoom2, no PWAD, MAP02 (reference) | 227 / 148 / "map02 - Filtration Tunnels" |
| `my_way_home.wad`, `set_doom_map("MAP02")` + `new_episode` (the reset path) | 227 / 148 / "Filtration Tunnels": **freedoom2 MAP02** |
| `my_way_home.wad`, `map03` | 315 / 152 / freedoom2 MAP03 (names are case-insensitive) |
| `my_way_home.wad`, MAP02 at `init` | 227 / 148: freedoom2 MAP02 |
| `train-all.wad` bundle (lumps DD0001..DD0080), MAP02 | 227 / 148: **freedoom2 MAP02** |
| `my_way_home.wad`, MAP99 (exists nowhere) | engine exits: `ViZDoomUnexpectedExitException` |

- So the held-out Freedoom Phase 2 maps were reachable from any training config by name. Closed in `env/config.py` (`wad_map_names`, `allowed_maps`, `check_reset_map`): a training config's `maps` and every reset map must be a map marker of its PWAD (or, with no PWAD, of its own `maps` tuple); otherwise `ValueError`. EVAL is unrestricted. Tests: `tests/test_env_map_guard_and_ini.py`.
- Side observation: after the MAP99 engine crash, the next `init()` in the same Python process hung (engine at 100% CPU, 1.5 GB); `init` with MAP02 alone in a fresh process returned in under 1 s. Not investigated further.

### R4.2 Are objects behind walls in `state.labels`? (audit R9)

Random walk (70% forward, random turns, some attack/USE, `god`), 300 decisions each on freedoom2 MAP01-03 and `maps/train/S3-train-10300{1..4}.wad`. Ground truth from whole-level objects/sectors (probe only): "walled" = the segment from the player to the object's centre crosses a blocking line.

| Count (all maps) | Value |
|---|---|
| object-steps with the object in the view cone (+-40 deg) and walled | 26,277 |
| ... of which listed in `state.labels` | 77 |
| walled objects listed in labels (any bearing) | 82, of which **81 had own pixels in the labels buffer** (centre behind a wall corner, an edge visible) |
| non-Self labels | 3,492 |
| labels with **zero** pixels of their own value | 114 (3.3%) |
| ... box covered by the weapon (Self value) | 32 |
| ... box covered by depth-0 firing overlay | 5 |
| ... box covered by another listed sprite | 77 |
| ... covered by world geometry only | **0** |

- **Conclusion (with limits):** none of the 114 zero-pixel labels was covered by world geometry only; every one had a weapon, overlay or other-sprite pixel somewhere in its box. Most such boxes are 1-4 px (distant pickups), where that is conclusive; for larger boxes the `.any()` test only shows a covering pixel exists. One zero-pixel label was also "walled" (map03) and was not inspected individually. The "walled" ground truth uses `is_blocking` lines only, so closed doors and height (ledge) occlusion are not covered by it.
- So: labels behave as "sprite drawn after wall clipping", and the zero-pixel rows `entities()` emits (`d = NO_READING`, full `u, v, w, h`, type) are objects hidden behind a nearer sprite (weapon, firing overlay, another thing, e.g. stacked pickups) at the same screen position, in 3.3% of label rows. No case of a row for an object hidden only by a wall was found.
- **Applied (2026-11-25, after the coordinator confirmed no run was training):** `entities()` now drops every label with zero pixels of its own value. Rows remain for labels with own pixels but no valid depth (d = NO_READING): this case exists (2 rows in 300 decisions on S3-train-103004, the translucent BulletPuff type).
- **Check (section `occlusion_obs`, 300 decisions each):** freedoom2 MAP01: 36 raw zero-pixel labels, 914 rows emitted = 914 expected (labels with own pixels), 0 mismatching steps; S3-train-103004: 22 raw zero-pixel labels, 265 = 265, 0 mismatches. So no zero-own-pixel row remains.
- Tracking: an entity that is hidden for a decision has no row, so it leaves `EntityTracker.prev`; when it reappears it comes back with is_new = 1 and du = dd = 0 (test `test_entity_hidden_then_back_returns_as_new`).
- Observation space, dtypes and `sensor_signature()` are unchanged (no config field or space changed), so existing checkpoints load; the observation content differs from what they were trained on in about 3% of entity rows.

### R4.3 Engine config file (`set_doom_config_path`)

- Default: every instance passes `-config _vizdoom.ini` (working directory); the file is read at start and **written at close** (probe: not present after `init`, present after `close`).
- `DoomEnv` now gives each instance its own path in a fresh temp directory (`doomsday_vzd_ini_*`), removed in `close()`.
- Old vs new, `DoomEnv` with EVAL and TRAIN configs, each run in an empty working directory, seed 7, 40 fixed decisions: the written ini files are **identical** (582 lines, 0 diff lines, timestamp comment excluded), **observations identical** on every step, and the new way writes no `_vizdoom.ini` in the working directory.
- Rerun with the old way reading a copy of the repository's existing `_vizdoom.ini` (the real old condition): ini and observations again identical for EVAL and TRAIN. The shared file had `screenblocks=10` at one reading and `12` later: it records whichever instance wrote last, which is the race this change removes.
- Leak: if a process is killed before `close()`, its `doomsday_vzd_ini_*` temp directory (one ~11 KB file) is left behind.

## Round 5 (observation v2: line-of-sight door channel, key bits), 2026-11-25

Probe script: `maps/probes/probe_doors.py` (one engine, TRAIN config with sector info on, scripted teacher driving on generated TRAIN maps S1-train-101001, S1-train-101004, S2-train-102001; kept under `maps/probes/` because it imports `teachers`). Engine: vizdoom 1.3.3, unmodified.

### R5.1 How a door looks in sector info (`state.sectors`)

- `Sector` fields in 1.3.3: `id`, `floor_height`, `ceiling_height`, `lines`; `Line`: `x1, y1, x2, y2, is_blocking`. No line specials, tags or texture names.
- `Sector.id` equals the WAD sector index (the gap-0 sector ids matched `maps.doors.door_specs` on all three maps). The env keys its door memory by `id`, not by list position.
- **Closed door = ceiling_height == floor_height** (gap exactly 0). On the three maps every gap-0 sector was a WAD door sector and every door sector started at gap 0.
- Each door sector has 4 lines: 2 with `is_blocking=True` (the jambs, one-sided) and 2 with `is_blocking=False` (the two faces, two-sided).
- Opening (held USE): the gap rises 8 units per decision (2 per tic at frame skip 4): 8, 16, 24 ... to the neighbouring ceiling minus 4 (68-92 here). DR doors close again after a wait, back to 0.
- Once open, nothing in sector info distinguishes a door from any other sector. So the env remembers, per episode and privately, every sector seen at gap 0, and treats it as a door while its gap is <= 44.
- Eye-level visibility while opening: up to gap 40, the face still fills the middle depth band (residual against the face within +-2 levels in 98-100% of columns). From gap 48 the band sees the passage beyond (median residual +16 to +26 levels). Caveat: in this tabulation "gap" is the largest gap among the map's door sectors at that step, not necessarily the gap of the door that was hit. Hence `DOOR_CAST_MAX_GAP = 44`. The cue honestly means "closed or partly closed door surface at eye level".

- **Vanilla check (freedoom2 MAP01, engine-test config):** 8 sectors have gap 0 at reset. 6 of them have 2 or 4 non-blocking lines, among them the probed door 27. Sectors 17 and 18 have no non-blocking lines, so they contribute no faces and never light. Facing door 27 from 34 units, every column 20-59 lights (test `test_vanilla_map01_door_lights_under_v2`). Whether non-door sectors with ceiling == floor and non-blocking lines exist on other vanilla maps was not surveyed.

### R5.2 Depth buffer vs 2-D ray cast

- Column bin k (pixels 2k, 2k+1) is at horizontal angle atan((2k+1-80)/80) off the view axis (90-degree FOV); screen right = clockwise. Check: the perpendicular distance to a door face was constant across columns, as a flat face must be.
- The depth buffer is **perpendicular (view-plane) distance**, not Euclidean.
- Fit on 7,787 column pairs (door closed, face matching): **depth level = 0.13979 x perpendicular units - 1.134**, about 7.15 units per level (not 8). Residual: 1st-99th percentile -2.3 to +0.75 levels, maximum +0.81. Most pairs are close range (USE distance). Beyond 150 units, a small cluster sits at -3 to -6. Its cause was **not established**; candidates are the 2-column x 8-row band minimum picking a nearer frame edge, oblique views, or floor/ceiling pixels in the band.
- Faces hidden by a nearer wall gave residuals of -20 and below (median -37).
- **Tolerance used:** lit iff `pred - (2 + 0.05 pred) <= reading <= pred + 1.5` (levels). Over the probe set this lights 7,736 of 8,198 closed-door pairs, and 628 of the 941 beyond 150 units; most of the rest are occluded.
- **Range limit (derived from geometry, not measured):** the middle band is horizon +-4 rows. Beyond about 820 units (level ~113, eye height 41) floor and ceiling pixels in the band should be nearer than a door, so far doors would not light. The probe pairs reach only about 400 units.

### R5.3 Locked-door message and key pickup in the notifications buffer

- Pressing USE at the blue-locked door of S2-train-102001 without the key (key anchor removed): the notifications buffer holds `Blue key needed for this door.` on **every decision whose USE press acted**, and nothing on the decision after. Box-drawing separator lines (`\x1d\x1e...\x1f`) surround it. So the "needs key" bit is naturally a one-decision pulse.
- Pickup: `Blue passcard secured!` on the pickup decision, as in R2.3.
- Parsing for v2 (`env/door_cue.keys_needed`): a line containing "need" plus a colour word. Only booleans reach the observation, never the text.

### R5.4 Cost of sector info (single engine, A/B interleaved, 3 x 800 steps per version, step p50)

| Map | Sectors / lines | v1 step p50 | v2 step p50 | Ratio |
|---|---|---|---|---|
| generated S1-train-101004 (DD0023) | 12 sectors | 0.82 ms | 1.01 ms | 1.23 |
| generated S2-train-102001 (DD0040) | 13 sectors | 0.82 ms | 1.04 ms | 1.26 |
| freedoom2 MAP01 | 211 / 2,036 | 0.96 ms | 5.28 ms | 5.47 |

- On MAP01, enabling sector info adds about 2.4 ms to `advance_action` and 1.0 ms to `get_state` (raw engine timing, no Python observation code). It scales with the size of the map. The Python door code costs about 0.5-0.8 ms there (sector loop about 0.2 ms, cast about 0.3 ms with many doors) and about 0.08 ms on the generated maps.
- `set_sectors_info_enabled(False)` after `init()` had no effect (sectors still returned), so the cost cannot be paid only every N decisions.
- Measured while a 32-engine training run was live, so absolute numbers are noisy; p50 and interleaving reduce that.
