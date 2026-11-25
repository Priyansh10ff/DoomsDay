# Lesson 02: The observation (what the agent sees)

> **Changed since first written (2026-11-25).**
>
> *First revision (ADR 0006):*
> - **Ray rows are measured from the horizon.** The bands are offsets from a horizon row that depends only on whether the status bar is drawn. With the status bar on, that gives rows 30-45, 46-53 and 54-69.
> - **14 entity types.** `hazard` was added at the end.
> - **Bullet puffs are no longer typed as projectiles.**
> - **The `screen` key is optional** (`screen_enabled`).
>
> *Second revision (round-3 probes, `docs/reference/engine-probes.md` R3):*
> - **The zero-rays problem flagged in the first revision was confirmed and fixed.** Three things put false "depth 0" into the depth buffer:
>   - the player's own weapon (label value 255, depth 0);
>   - extra unlabelled depth-0 pixels while firing;
>   - some pixels of bullet puffs.
>
>   `env/observation.py` now ignores every depth-0 pixel and every pixel of the player's own label. A ray cell or entity distance with no valid pixel reads exactly **0.0, called `NO_READING`**; real readings are always at least 1/255.
> - Sections 3.1, 3.3, 4.2 and 7 and question 1 were rewritten. The worked example in 4.2 is a fresh run of the fixed code.

Files taught: `env/observation.py` (all of it), `env/config.py:36-50` (ray bands), and `env/doom_env.py:274-308` (`_build_obs`), plus the tests that guard them (`tests/test_env_rays.py` and others named below).

Design records: ADR 0001 (observation) as amended by ADR 0006. **All seven ADRs (0001 to 0007) are marked "Proposed, pending ratification by the user."** What was measured on the real engine is in `docs/reference/engine-probes.md`.

Builds on lesson 01: the two paths (policy path versus `info`), `reset`/`step`, frame skip, the buffers the engine hands us, and the sensor field group.

---

## 1. What problem this piece solves

The agent must find its way around rooms it has never seen, notice an imp throwing a fireball, and know when its health is low. It must do this:

- **without a god-view**: only what is in front of the player's eyes,
- **without map coordinates**: otherwise it could learn "at (1024, -512), turn left", which is useless on a new map and against the rules (`docs/RULES.md`, "Observations"),
- **cheaply**: the organizers warn that learning from raw pixels spends most of the training time learning to see edges (`docs/TRACK_BRIEF.md`, line 12).

So the question is: which numbers do we hand the network each step so that it has enough to play, and nothing that lets it cheat or memorize? And, as the zero-rays episode showed: are those numbers actually **true**?

## 2. The idea in plain words

Imagine walking through a dark building with three things:

1. **A row of laser distance meters** fixed to your head, pointing straight out in a fan across your view. Each one says how far the nearest surface is in its direction. These are our **rays**.
2. **A spotter** who tells you about each *thing* you can see (not walls): "a monster, slightly right of centre, quite far, moving left", "a medkit, low left, close". This is our **entity set**.
3. **Your own body sense**: how hurt you are, which weapon you hold, how fast you are moving and turning. This is our **`self` vector**.

You may also get a **small, blurry photo** of the view (the **screen**, when switched on), and a card that says which of these instruments are working right now (the **modality flags**).

Where the analogy stops being accurate:

- The distance meters only cover what is on screen, not what is behind you.
- Our "spotter" is not smart. It does not rank threats or say "shoot this one". It only reports where each thing appears on screen, how far it is, and a coarse type. Deciding what matters is the network's job (rule 4.1).
- The distances are not exact meters. They are 8-bit numbers from the engine's depth image, roughly 8 map units per step.
- When your own gun is in front of a laser meter, our meter now says "no reading" rather than "something is touching me". A real meter would just read the gun.

## 3. A walk through our actual code

All of `env/observation.py` is made of **pure functions**: they take arrays in and give arrays out, with no hidden access to the game. The only state is the small `EntityTracker` (3.3). `_build_obs` in `doom_env.py` calls them.

The module docstring (`observation.py:1-8`) states the input rule: only the depth buffer, the labels buffer, the screen-box fields of labels, the screen, and the agent's own game variables. "World-frame fields of labels are never read here." A test enforces that (3.7).

The observation dictionary, as built at `doom_env.py:297-307`:

| Key | Shape, type | One-line meaning |
|---|---|---|
| `rays` | (3, 80) float32 | Nearest distance in each of 3 row bands by 80 column slices of the view; 0.0 means "no reading". |
| `entities` | (24, 8) float32 | Up to 24 visible things, 8 numbers each. |
| `entity_type` | (24,) int64 | A coarse type id per row (monster, health, blue key...), 14 possible values. |
| `entity_mask` | (24,) bool | Which of the 24 rows are real. |
| `self` | (28,) float32 | The agent's own health, ammo, weapons and motion. |
| `prev_action` | (6,) int64 | Last step's action (lesson 03). |
| `screen` | (3, 60, 80) uint8 | The colour picture at half size. **Only present when `screen_enabled` is true.** |
| `modality` | (3,) float32 | Flags: telemetry valid, screen valid, ego-motion valid. |

**Telemetry** in this project means the numbers we pull directly from the engine instead of from pixels: rays, entities and the motion part of `self`.

### 3.1 Rays from the depth buffer: `observation.py:79-124` and `config.py:36-50`

The **depth buffer** is a 120x160 image the engine draws alongside the normal picture. Each pixel holds a number from 0 to 255 for how far away the surface at that pixel is. The probe found that **bigger means farther**, about **8 map units per step**, and the largest value seen was 120 (`engine-probes.md`, section 2). One DOOM **map unit** is the game's basic length unit in its levels.

**Which pixels are not the world: `invalid_depth_mask` (`observation.py:106-124`).** The round-3 probe (`engine-probes.md`, R3.1) found three sources of depth 0 that do not measure the world:

1. **The player's own weapon.** It is drawn with the label of the player itself (`DoomPlayer`, category `Self`, labels value **255**), and every one of its pixels has depth 0. Its box spans rows 50 to 99, so it reaches into ray bands 1 and 2. In the probe's runs, a band-2 cell made entirely of weapon pixels appeared on 20-56% of steps.
2. **A firing overlay.** While attacking, extra depth-0 pixels appear in rows 40-99 with a labels value no listed label claims. They appeared on 40 of 100 steps with attack held, and 0 of 200 without.
3. **Bullet puffs.** The puff sprite is partly see-through, and 843 of its 266,702 pixels in the probe had depth 0, although it sits on a wall several levels away. No other class showed any depth-0 pixel.

The probe also found that real walls are never nearer than the player's radius, 16 map units, which is 2 depth levels; walking into a wall bottoms out at 2. So **depth 0 never measures the world**. `invalid_depth_mask` marks a pixel invalid if its depth is 0 (`:120`), and also marks every pixel carrying the player's own label value, as a safeguard (`:121-123`).

> This replaces the first revision's "Look out" box. The probe confirmed my inference that the zeros came from the weapon, and found the two other sources. It also corrected its own round-1 note that "the weapon sprite does not appear in the depth buffer" (`engine-probes.md:39`). That note came from reading one column of one frame.

**`rays(depth, bands, own)` (`observation.py:79-99`):**

- `:91`: make an output of shape (number of bands, 80).
- `:93-95`: for each band of rows `[r0, r1)`, take those rows. Wherever the mask `own` is true, put in a stand-in value (`_MASKED` = 1000, `:103`) that is larger than any real depth.
- `:96`: cut the 160 columns into 80 slices of 2 pixels each, and take the **minimum** over the rows and the 2 columns. The minimum is the nearest valid surface in that cell; the stand-in value never wins against a real one.
- `:97`: if the minimum is still the stand-in, the cell had **no valid pixel at all**: write 0. Otherwise write the minimum, but at least 1.
- `:98`: divide by 255.

So the encoding is (`:85-89`):

- a **real reading** is max(depth, 1) / 255, so **never exactly 0**;
- **`NO_READING` = exactly 0.0** (`:102`) means "no world measurement in this cell this step". In practice that is band 2 under the weapon, and rarely a cell fully covered by a puff's bad pixels. Nothing is filled in from neighbouring cells or earlier steps.

The same 0.0 is used when telemetry is missing entirely (3.6). The `modality` flags tell those two cases apart.

So each of the 80 columns is like one laser meter pointing in a direction across the view, and the 3 bands look at different heights. Each 2-pixel column slice covers a fixed slice of the field of view, so the ray at column bin 40 always points straight ahead, wherever the player is on the map.

**Where the bands sit: measured from the horizon.** The **horizon** is the screen row that looks straight ahead, level with the player's eyes. Above it you see ceiling and the tops of walls; below it you see floor.

- `RAY_BAND_OFFSETS` (`config.py:37`) = (-20, -4), (-4, +4), (+4, +20): rows above, around and below the horizon.
- `horizon_row(render_hud)` (`config.py:40-45`) is the single place the horizon is defined: **row 50 with the status bar on, row 60 with it off**.
- `ray_bands_for` (`config.py:48-50`) adds the offsets to the horizon. With the status bar on (the default) the bands are rows **30-45, 46-53 and 54-69**.
- `EnvConfig.ray_bands` (`config.py:111-113`) is a read-only property computed from `render_hud`, so the two can never disagree. `tests/test_config_and_source.py:101` checks both cases and that TRAIN and EVAL share them.

Why the horizon moves: with the status bar on, the 3D view is only rows 0-99 and the whole image moves up exactly 10 rows. A direct fit on floor pixels gave about 49.1 with the status bar on and 59.1 with it off (`engine-probes.md`, R2.1). The probe calls this "consistent with 50/60 within about 1 row, not a large-sample confirmation". The first version of the code used ADR 0001's rows (40-55, 56-63, 64-79), which assumed a horizon at row 60; ADR 0006 ruling 1 replaced them.

**Tests** (`tests/test_env_rays.py`):

- `:31-46`: on a made-up buffer, cells made only of weapon or overlay pixels read `NO_READING`, a real depth of 1 stays a non-zero reading, and the world elsewhere is untouched. It also shows that **without** the mask the weapon reads as "contact", the old bug.
- `:49-61`: a puff's depth-0 pixels are ignored, but its valid pixels still count.
- `:99-136`: on the real engine, firing for 24 steps near a door, **every ray cell is checked against the raw buffers** on every step. The test also fails if the weapon or the zero pixels never appeared, so it cannot pass by accident.

### 3.2 The entity type table: `observation.py:20-75`

The **labels buffer** is another 120x160 image from the engine. Each pixel holds a small number saying which object, if any, was drawn there; 0 means "no object" (walls, floors). Alongside it, the engine gives a **label list** (`state.labels`): one record per object on screen, with its on-screen box (`x`, `y`, `width`, `height`), the number it uses in the labels buffer (`value`), a stable `object_id`, its class name (for example `Zombieman`) and a category (for example `Monster`).

Because labels exist only for things the engine actually drew this frame, using them gives us **line-of-sight filtering for free**. A monster behind a wall is not drawn, so it has no label.

`ENTITY_TYPES` (`observation.py:22-26`) is a fixed list of **14** coarse types, numbered in order:

```
0 pad, 1 monster, 2 projectile, 3 health, 4 armor, 5 ammo, 6 weapon,
7 key-blue, 8 key-yellow, 9 key-red, 10 other-pickup, 11 decoration, 12 unknown, 13 hazard
```

`hazard` was added **at the end** (ADR 0006), so every earlier id kept its number. Models must size themselves from the table's length (comment at `:20-21`; lesson 04).

`entity_type_id(name, category)` (`observation.py:61-75`) decides in this order:

1. `:62-70`: if the category is `Key`, look for "blue", "yellow" or "red" in the name. Cards and skull keys of one colour share an id. A key of no known colour becomes `unknown`.
2. `:71-72`: names in `_PROJECTILE_NAMES` (`:51-54`: imp fireballs, rockets, plasma and so on) become `projectile`.
3. `:73-74`: names in `_EFFECT_NAMES` (`:57`: `BulletPuff`, `RevenantTracerSmoke`, `RocketSmokeTrail`) become `decoration`.
4. `:75`: otherwise look the category up in `_CATEGORY_TABLE` (`:34-49`). **Anything not in the table becomes `unknown`, never an error.**

Why names and not categories for shots: the engine reports **every missile and also impact effects** under the category `Explosive` (`engine-probes.md`, R2.4). `Explosive` is deliberately absent from the table (comment at `:29-33`). `SFX` (teleport fog, item fog) maps to `decoration` (`:47`), and `Hazard` (the exploding barrel) to `hazard` (`:48`). Tests: `tests/test_config_and_source.py:224` and `:237`.

### 3.3 The entity set: `observation.py:128-187`

The **entity set** is the list of visible things. It changes size from frame to frame. A network needs fixed-size arrays, so we use a fixed table of `max_entities` = 24 rows with a **mask**: a true/false per row saying whether that row holds a real thing. Unused rows are all zeros ("padding"). Lesson 04 shows how the model reads such a set.

`EntityTracker` (`:128-135`) remembers, for each `object_id` seen in the previous step, its screen position `u` and its distance `d`.

`entities(labels, labels_buf, depth, tracker, max_entities)`:

- `:147-149`: start with all-zero arrays: features (24, 8), types (24,), mask (24,).
- `:151-154`: for each label, **skip the player's own label** (`DoomPlayer` or category `Self`). As 3.1 explained, that label is the weapon.
- `:155-157`: skip labels with an empty box.
- `:158-163`: **distance `d`**. Take the depth pixels inside the box, keep only those that carry **this object's** label value **and** have a depth above 0, and average them. If there are none (the object is fully covered, for example by the weapon), `d` = `NO_READING`. Before the fix, it fell back to the depth at the box centre, which was usually a weapon pixel, so a hidden monster read as "touching" (`engine-probes.md`, R3.2).
- `:164-165`: **screen position**. `u` is the box centre's horizontal position, from -1 (left edge) to +1 (right edge). `v` is the vertical centre, from -1 (top) to +1 (bottom).
- `:166-167`: store the row, together with the box width and height as fractions of the screen, and the type id.
- `:168-172`: **sensor-range cap**. If more than 24 things are visible, keep the 24 nearest, with `NO_READING` ranked last, then put them back in the engine's label order. They are deliberately not sorted by danger or type (ADR 0001).
- `:173-185`: **motion**. For each kept row, look up its `object_id` from last step. If it is new, `du = dd = 0` and `is_new = 1`. Otherwise `du` is the change in `u`, and `dd` is the change in `d` **only if both readings are real** (`:181`); otherwise `dd = 0`. Then write the 8 features `(u, v, w, h, d, du, dd, is_new)`, the type, and set the mask.
- `:186`: replace the tracker's memory with this step's objects.

Tests: `tests/test_env_rays.py:64-74` (an entity fully under the weapon reads `d = 0` and `dd = 0`), and `tests/test_config_and_source.py:276` (nearest-24 and motion tracking).

The label `value` of one object changed between frames in the probe (223, then 198, then 178), so tracking uses `object_id`. Whether the engine ever reuses an `object_id` is not yet known (ADR 0001, risks).

Note that `du` is motion **on the screen**, not in the world. If the agent turns, every visible thing slides across the screen, even if it is standing still. The network has to learn to combine `du` with its own turn reading in `self`.

### 3.4 The `self` vector: `observation.py:191-222`

`self_vector` (`:210-222`) packs 28 numbers:

| Index | Content | Scaling |
|---|---|---|
| 0 | health | / 100 (so 1.0 = full health; can go above 1) |
| 1 | armour | / 100 |
| 2 | ammo of the selected weapon | clipped to 0-200, / 200 |
| 3 | damage taken since the last decision | clipped to 0-200, / 100 |
| 4-13 | which weapon slot is selected | one-hot: a 1 in one position, 0 elsewhere |
| 14-23 | which weapons are owned | 1 if owned, else 0 |
| 24-27 | ego-motion: forward speed, leftward speed, upward speed, turn since last step | see below |

"Damage since last decision" comes from the running total `DAMAGE_TAKEN`. `_build_obs` subtracts the previous total (`doom_env.py:293-294`). It is 0 in the observation returned by `reset()`. It is the agent's only sense of being hit from behind.

**Ego-motion** (`ego_motion`, `observation.py:195-207`) means the agent's own movement, described from its own point of view ("I am moving forward and drifting left"), not on the map ("I am moving east").

- The engine reports velocity on the map's axes (`VELOCITY_X`, `VELOCITY_Y`) and the heading `ANGLE` in degrees.
- `:201-204` rotates the velocity by the heading. `fwd` is the part along the direction the player faces; `left` is the part toward the player's left. This is a **body-frame** reading.
- `:205`: the turn since last step is the heading difference, wrapped into -180 to +180 by `wrap_deg` (`:191-192`). In the observation returned by `reset()` it is 0.
- `:206-207`: divide speeds by 20 and the turn by 180, then clip to [-1, 1].

The heading itself never comes out. ADR 0001 ruling (d) calls this proprioception (the body's sense of its own motion) and allows it, with a fallback: if the organizers object to the rotation, these four numbers are dropped. `tests/test_config_and_source.py:201-213` checks that rotating the whole world by any angle leaves ego-motion unchanged.

### 3.5 The screen, optional: `observation.py:226-229`, `doom_env.py:305-306`

`screen_small` halves the 3x120x160 colour image to 3x60x80. It groups pixels into 2x2 blocks, adds the four values (in 16-bit, so they do not overflow), adds 2, and shifts right by 2 bits. Shifting right by 2 divides by 4, and the "+2" makes it round to the nearest whole number.

The status bar is included in the picture, on purpose: its key icons are one way to know which keys are held (ADR 0001 ruling (b)).

Why have a picture at all? Because **some things exist only in pixels**. A locked door's colour and an exit switch look like ordinary walls in the depth and labels buffers (ADR 0001).

Why it is optional (ADR 0006 ruling 4): making it costs about 0.24 ms per decision, most of the observation-building time. With the screen off, one environment ran at 1171 and 1246 decisions per second, against a median of 877 with it on (`reports/runs/2026-11-25-env-timer-variance-bench.md`). Stage A0 does not use pixels, so it trains with the screen off (`configs/a0_env.py`). When `screen_enabled` is false, the `screen` key is **absent** from both the observation and the observation space (`doom_env.py:117-118`, `:305-306`). It is a sensor field (lesson 01), so it must match between training and evaluation. Test: `tests/test_env_episode.py:181`.

### 3.6 Putting it together, and the modality flags: `doom_env.py:274-308`

`_build_obs(state, v)`:

- `:279`: telemetry is usable only if the depth buffer, labels buffer and label list all exist.
- `:281-286`: if so, build the invalid-pixel mask, rays, entities and ego-motion, and remember the heading for next step's turn reading.
- `:287-292`: if not, **fill those keys with zeros**.
- `:293-296`: damage since last step, then the `self` vector.
- `:297-306`: the dictionary, with `screen` only if switched on.
- `:307`: `modality = [telemetry_ok, screen_enabled, telemetry_ok]`.

Why this matters: the judges might run the engine with `+viz_nocheat 1`, a ViZDoom setting that blocks "cheating" data. **The probe found that the depth buffer, labels buffer and label list are then removed entirely, and position, angle and velocity read 0.0** (`engine-probes.md`, section 1; ADR 0006 ruling 5). So under that setting, the wrapper returns all-zero rays, no entities, zero ego-motion, and `modality = [0, 1, 0]` (screen on). `tests/test_env_episode.py:272` checks this on the real engine.

A ray of 0.0 therefore has two meanings, and the flag separates them:

- `modality[0] = 1`: telemetry is present; this particular cell had no valid pixel (`NO_READING`).
- `modality[0] = 0`: there is no depth sensor at all this step.

In neither case does 0.0 mean "a wall is touching your face": real readings are at least 1/255.

### 3.7 Why none of this contains absolute coordinates

Key by key:

- `rays`: distances from the player's eye along fixed directions on the screen.
- `entities`: screen positions, sizes and depths. The `object_id` is used inside the tracker but is never output.
- `entity_type`: a coarse category, the same on every map.
- `self`: the body's own state. Velocity is turned into the body frame; the heading is used and dropped.
- `prev_action`: the agent's own last choice.
- `screen`: the picture a human would see.
- `modality`: three flags.

The guards:

- `tests/test_config_and_source.py:185` reads the source of `observation.py` and fails if it contains `atan2`, `object_position`, `object_angle`, `object_velocity`, `POSITION_`, `automap`, `.sectors` or `.objects`. **atan2** is the math function that turns a position difference into an angle, the core of a banned "aimbot"; it appears nowhere in the observation code.
- `tests/test_env_engine.py:79-112` searches every number in the non-pixel keys for the privileged values (lesson 01). It is a smoke test, not a proof.
- `tests/test_env_engine.py:115-118`, "same observation under map translation", is still **skipped**: it needs a test map from `map-forge` that does not exist yet.
- The notifications buffer is **not** in the observation; at tic 1 it holds the level name (`engine-probes.md`, section 4).

Honest limits. ADR 0001 notes that `u` grows steadily with the direction to the object, so a very strict judge could still call it "angle information" (ruling (c), residual risk). And the screen can show textures that a network might come to recognise as a particular map.

## 4. A tiny worked example with real numbers

### 4.1 One ray cell, at reset

From the first revision's run of `DoomEnv(EVAL_CONFIG)`, `reset(seed=0)` (real output; the fix does not change this frame, because there is no weapon on screen at reset, and the second run below printed the same `[21 63 48]`):

```
reset depth col80 rows 28..72: [22, 22, 22, 22, 22, 22, 22, 22, 21, 23, 25, 27, 30, 34, 39, 45, 54, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 120, 111, 102, 95, 89, 83, 78, 74, 48, 46, 34]
reset rays[:,40]: [0.082 0.247 0.188] x255 [21 63 48]
```

Column 80 is the middle of the screen, which is ray column bin 40 (pixel columns 80 and 81). Reading it:

- Rows 28-44 go from 22 up to 54. This looks like ceiling, which gets farther away as you look down toward eye level.
- Rows 45-60 are a flat 63: a vertical surface about 63 x 8 ≈ 500 map units away (by hand, from the probe's ~8 units per level).
- From row 61 the value jumps to 120 and then shrinks: floor seen through an opening, getting nearer as you look down.
- Band 0 (rows 30-45): minimum 21, so 21 / 255 = 0.082. Band 1 (rows 46-53): 63, so 0.247. Band 2 (rows 54-69): minimum 48, so 0.188.

### 4.2 The weapon, before and after the fix

I ran the fixed code: one `DoomEnv(EVAL_CONFIG)`, `reset(seed=0)`, then the same seven actions as lesson 01. One engine instance, closed at the end. Real output:

```
reset final_obs_fresh: True  rays[:,40] x255: [21 63 48]
step 1: fresh=True exact-zero rays=0 DoomPlayer(value,x,y,w,h)=[]
step 2: fresh=True exact-zero rays=0 DoomPlayer(value,x,y,w,h)=[(255, 74, 88, 12, 12)]
step 3: fresh=True exact-zero rays=0 DoomPlayer(value,x,y,w,h)=[(255, 68, 74, 25, 26)]
step 4: fresh=True exact-zero rays=0 DoomPlayer(value,x,y,w,h)=[(255, 67, 67, 27, 33)]
step 5: fresh=True exact-zero rays=0 DoomPlayer(value,x,y,w,h)=[(255, 67, 67, 27, 33)]
step 6: fresh=True exact-zero rays=4 DoomPlayer(value,x,y,w,h)=[(255, 67, 50, 27, 50)]
step 7: fresh=True exact-zero rays=0 DoomPlayer(value,x,y,w,h)=[(255, 67, 61, 28, 39)]
step7 band1 bins30-49 x255: [11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9]
step7 band2 bins30-49 x255: [11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9]
step7 NO_READING bins per band: [[], [], []]
```

Reading it:

- The `DoomPlayer` label always has value **255**, and its box sits at the bottom centre and grows when the agent fires (step 6, after the shot at step 5, reaches up to row 50). That is the weapon.
- **Before the fix**, the same run had exact zeros on steps 4 to 7 (2, 2, 13 and 6). At step 7, band 2 read `[..., 10, 0, 0, 0, 0, 0, 0, 10, ...]` in bins 37-42 (lesson 02's first revision).
- **After the fix**, step 7's band 2 reads 10 in those bins: the wall about 80 map units ahead (by hand), seen through the gaps around the weapon. No cell is `NO_READING` at step 7.
- At **step 6**, 4 cells still read exactly 0. Those are cells where *no* pixel was valid: weapon plus firing overlay covered them completely. Since the fix, 0.0 means "no reading" there, not "wall at your face". I did not print which 4 cells they were.
- `final_obs_fresh` is a new `info` key (lesson 01): it is True on every step here, because no episode ended.

### 4.3 One entity, all 8 numbers

The same run, real output:

```
step 6   entities: [[ 0.112 -0.225  0.075  0.108  0.042  0.     0.     1.   ]] types: [11]
step 7   entities: [[ 0.225 -0.317  0.1    0.15   0.039  0.112 -0.003  0.   ]] types: [11]
```

From the first revision, the label for this entity at step 6 was `(1, 'BulletPuff', 'Explosive', 83, 40, 12, 13)` (object_id, name, category, x, y, width, height). Checking step 6 by hand:

- `u` = (83 + 12/2) / 160 x 2 - 1 = 89/80 - 1 = **0.1125**: a little right of centre.
- `v` = (40 + 13/2) / 120 x 2 - 1 = 46.5/60 - 1 = **-0.225**: a little above the middle row.
- `w` = 12/160 = **0.075**, `h` = 13/120 = **0.108**.
- `d` = **0.042**: about 10.7 depth levels, roughly 85 map units (by hand). This is now the mean over the puff's pixels with depth above 0, so its own bad pixels no longer pull it down. In this frame the printed value did not change.
- `du = dd = 0`, `is_new = 1`; type **11 = `decoration`** (`BulletPuff` is in `_EFFECT_NAMES`).

Step 7, same `object_id`: `u` = 0.225 (box `(90, 32, 16, 18)`), so `du` = **0.1125**. Both readings are real, so `dd` = 0.039 - 0.042 ≈ **-0.003**.

### 4.4 The `self` vector after reset, ego-motion, screen average

These did not change with the fix. From the first revision, real output:

```
self: [1.   0.   0.25 0.   0.   0.   1.   0.   0.   0.   0.   0.   0.   0.   0.   1.   1.   0.   0.   0.   0.   0.   0.   0.   0.
 0.   0.   0.  ]
```

Reading it:

- Health 100 (`1.0`), no armour, 50 rounds (0.25 x 200).
- No damage yet.
- Slot 2 selected (index 6), slots 1 and 2 owned (indices 15 and 16).
- No motion.

Ego-motion, by direct call (real output): `O.ego_motion(10.0, 0.0, 0.0, 90.0, 80.0) -> [0. -0.5 0. 0.056]`. The player faces the map's +y direction and moves along +x, which is to its right, so leftward speed is -0.5, and the turn is 10/180 = 0.056.

Screen average, by hand with made-up values: a 2x2 block 10, 11, 12, 14 sums to 47, and (47 + 2) >> 2 = **12**, the correct rounding of 11.75.

## 5. Why we chose this design over the alternatives

From ADR 0001 as amended by ADR 0006 (**Proposed, not ratified**):

- **Option A, pure telemetry**: cheapest, but it would see nothing under `+viz_nocheat`, and it **cannot see lock colours or exit switches at all**.
- **Option B, pure pixels**: survives any setting, but published DOOM results needed 100 million to 1 billion frames.
- **Option C, tiered hybrid (chosen)**: telemetry as the main input, a small picture available, and presence flags.
- **Option D, casting our own rays against the level geometry in Python**: 360-degree awareness, but slow, needs its own line-of-sight filtering, and also dies under `+viz_nocheat`. Kept as a later add-on.

Specific rulings the code follows:

- **Labels buffer, never the objects list** (engine-provided line of sight versus a whole-level god-view).
- **Direction only as screen position, distance only as depth** (ruling (c)). No `atan2`, no bearing from world positions, no "in the crosshair" flag, no threat score, no sorting by priority.
- **Keep the nearest 24**, a uniform sensor limit.
- **Ray bands from the horizon** (ADR 0006 ruling 1).
- **Screen optional, and absent when off** (ADR 0006 ruling 4).
- **`hazard` appended, not inserted.**

The `NO_READING` rule came from the probes, not an ADR. ADR 0007 records the weapon problem as known and assigns the fix to `env-engineer`. The choice was between alternatives such as "fill in from a neighbouring cell" or "use the previous step's value". The code chose **not to invent a value**: if no valid pixel exists, it says so (0.0), and the network learns what that means. Filling in would put hand-written guesses into the sensor.

## 6. How it connects

- **Lesson 01** gave the split between `obs` and `info`.
- **Lesson 03** covers `prev_action`.
- **Lesson 04** shows the encoders that read `rays`, the entity set and `self`. The model sizes itself from the observation space.
- **Lesson 06** shows the trainer's telemetry masking: it can blank out rays, entities and ego-motion and set the modality flags to 0. A0 trains with it off.

## 7. Common mistakes, and how we would notice them

| Mistake | How it would show up |
|---|---|
| Reading entity rows without the mask. | Padding rows look like an object of type `pad` at the screen centre at distance 0. Log the mean of `entity_mask.sum()`. |
| Reading a ray of 0.0 as "touching a wall". | It never means that any more: real walls read at least 2/255 (`engine-probes.md`, R3.1). 0.0 means "no reading" (telemetry present) or "no sensor" (`modality[0] = 0`). Log the share of `NO_READING` cells per band; most should be in band 2, in the centre columns, more often while firing. |
| A new source of false depth (a different weapon, another see-through sprite). | `tests/test_env_rays.py:99-136` checks the rule against the raw buffers, but only for the pistol on MAP01. With other weapons, look for ray values of exactly 1/255 (a leaked depth-0 pixel would be clamped to 1). |
| Treating zero rays as real when `modality[0] == 0`. | Under `+viz_nocheat` every ray reads 0. Log `modality` per episode. |
| Thinking `du` is the object's own motion. | While turning, every entity's `du` jumps the same way. That is correct; the network must combine it with `self[27]`. |
| Training with `screen_enabled=False` and evaluating with it on. | `assert_same_sensors` raises. |
| Adding a "helpful" feature like "bearing to nearest monster". | It breaks rule 4.1. Every new observation feature should go to `rules-auditor` first. |

## 8. Check your understanding

1. With the status bar on, which rows does the middle ray band cover, and why would they change if the status bar were turned off? What value would you expect in `rays[1, 20]` if a wall is about 80 map units away in that direction?
2. Under `EVAL_CONFIG`, an observation has `rays[2, 40] = 0.0` and `modality = [1, 1, 1]`. What does that 0.0 mean, and what does it *not* mean? What if `modality` were `[0, 1, 0]`?
3. A monster is completely hidden behind the agent's weapon on screen, but its label is still listed. What `d` and `dd` does its row get now, and what did it get before the round-3 fix?

---

## Answers

1. Rows 46-53: the horizon row 50 plus the offsets -4 to +4 (`config.py:37`, `:40-50`). With the status bar off, the 3D view uses the whole image and moves down 10 rows, so the horizon becomes row 60 and the band becomes rows 56-63. For the wall: about 8 map units per depth level, so 80 units is about 10 levels; 10 / 255 ≈ **0.04**. (The probe's calibration used 12 samples, so this is approximate.)
2. Telemetry is present, so this is `NO_READING`: no pixel in that cell had a valid depth, almost certainly because the weapon (and maybe the firing overlay) covered all of it. It does **not** mean a wall is touching the agent; a real reading is at least 1/255, and real walls read at least 2/255. With `modality = [0, 1, 0]`, there is no depth data at all this step (for example `+viz_nocheat 1`), so every ray is 0.0 for that reason.
3. Now: no pixel of the monster has both its own label value and a depth above 0, so `d = NO_READING` (0.0), and `dd = 0` because one of the two readings is not real (`observation.py:158-163`, `:181`). Before the fix, `d` fell back to the depth at the box centre, which was a weapon pixel at depth 0, so the monster read as "touching", and `dd` could show a large false jump.
