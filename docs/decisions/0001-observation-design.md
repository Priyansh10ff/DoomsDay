# ADR 0001: Observation design (tiered hybrid, ego-centric, screen-space entities)

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

## Context

The policy must play unseen maps on the unmodified engine (RULES: no absolute coordinates, no god-view, no mods at evaluation, every action from the network). The brief (lines 12, 24, 31) recommends ego-centric telemetry from the engine and allows a hybrid with pixels.

Measured today on the only training machine (Windows 11 laptop, 24 threads, 15.4 GB RAM, torch CPU-only): ViZDoom 1.3.3 gives screen (3,120,160) uint8, depth (120,160) uint8, labels (120,160) uint8, objects info, sectors info. Labels carry a screen-space box, `object_id`, `object_name`, `object_category`, plus absolute world position, angle and velocity. Game variables include HEALTH, ARMOR, ammo, weapons, POSITION_X/Y/Z, ANGLE, VELOCITY_X/Y/Z. There is no game variable for keys held.

Things we do not know, and that bound this decision:

- **Whether the judges' harness sets `+viz_nocheat`.** ViZDoom's docs say objects/sectors info are then unavailable and depth and labels "contain noise" (`docs/research/vizdoom-agents.md`, `docs/osc/vizdoom-and-rl-references.md`). Its effect on game variables, the notifications buffer and the HUD is UNVERIFIED.
- **Whether we ship our own inference loop** (we configure `DoomGame`) or the harness fixes buffers, resolution, buttons and frame skip.
- **The deadline.** The user says "more than 5 days". Budgets below assume 5 days and must be rescaled when a date is given.

Evidence caveat: no opened source shows a telemetry-only policy in Doom; all three reference Doom agents are pixel agents (`vizdoom-agents.md`). Telemetry-first rests on the brief's recommendation and on non-Doom evidence (hide-and-seek), not on a Doom result.

A gap found while writing this ADR: **nothing in the label or sector fields reported so far encodes a locked door's colour or an exit switch.** A closed door is a wall in the depth and labels buffers. Those cues exist only in the rendered image (door-frame key markings, exit textures). This is the main reason the decision below is a hybrid and not pure telemetry.

Throughput of every option below is **unmeasured**. No telemetry-extraction FPS figure exists in the research.

## Options considered

**A. Pure telemetry (rays + entity set + game variables).** Build about 1 day. Throughput: expected highest (no CNN), unmeasured. Cost: dies completely if the harness sets `viz_nocheat`; cannot see lock colours or exit switches at all.

**B. Pure pixels (RGB, optionally predicted depth).** Build about 0.5 day for the env. Throughput: environment similar, learner far slower on CPU; published pixel results needed 10^8 to 10^9 frames. Survives any harness. Cost: most of our compute goes to learning vision, which the brief warns against, and it gives the weakest "ego-centric spatial encoder" story.

**C. Tiered hybrid (chosen).** Telemetry is the primary stream; a small low-resolution RGB stream is always emitted by the environment and can be switched on in the model; per-modality presence flags let one network run with telemetry missing. Build about 1.5 days for the env including probes. Throughput: environment pays one extra resize; learner cost depends on whether the pixel branch is on (see ADR 0003, 0004).

**D. Sector-geometry raycasting (360 degree rays cast against sector lines in Python).** Build 1.5 to 2 days, Python raycasting is the likely step-loop bottleneck (unmeasured), needs line-of-sight filtering to stay within partial observability, and dies under `viz_nocheat`. Not chosen for the first version; kept as a later add-on for rear awareness.

## Decision

Adopt **option C**. The environment emits the dictionary below on every decision step. Absolute pose and any privileged data go only into `info`, which the policy never receives.

### Policy observation (exact contract for `env-engineer`)

| Key | Shape, dtype | Content |
|---|---|---|
| `rays` | (3, 80) float32 | From the depth buffer only. 80 column bins (2 px each) by 3 row bands (above horizon, horizon, below horizon; proposed rows 40-55, 56-63, 64-79). Value = nearest depth in the cell, scaled to [0,1]. `env-engineer` confirms depth polarity and range and documents it. |
| `entities` | (24, 8) float32 | One row per visible label (player's own label excluded): `u` box centre x in [-1,1], `v` box centre y in [-1,1], `w`, `h` box size as a fraction of the screen, `d` depth sampled from the depth buffer inside the box on pixels whose labels value matches, `du`, `dd` change of `u` and `d` since the previous decision for the same `object_id` (0 if new), `is_new`. Zero-padded. |
| `entity_type` | (24,) int64 | Coarse category id from `object_category` / `object_name` through a fixed table: pad, monster, projectile, health, armor, ammo, weapon, key-blue, key-yellow, key-red, other-pickup, decoration, unknown. Unknown names map to `unknown`, never to an error. |
| `entity_mask` | (24,) bool | True where a row is real. |
| `self` | (28,) float32 | health/100, armor/100, selected-weapon ammo (clipped, scaled), damage taken since last decision (scaled), selected weapon one-hot (10), weapons owned (10), ego-motion (4, see ruling d). |
| `prev_action` | (6,) int64 | The head indices applied on the previous decision (ADR 0002); zeros at reset. |
| `screen` | (3, 60, 80) uint8 | RGB screen, 2x area-averaged from 160x120, HUD rendered. Emitted always; used by the model only when its pixel branch is on. |
| `modality` | (3,) float32 | Presence flags: telemetry valid, screen valid, ego-motion valid. |

If more than 24 labels are visible, keep the 24 with the smallest `d`. This is a sensor-range limit applied to all types alike. Rows are left in label order; they are not sorted by threat or type.

Entities come from the **labels buffer, never from objects info**: labels exist only for objects currently rendered, so line-of-sight filtering is done by the engine. Objects info and sectors info are whole-level (god-view) and are used only for training-time rewards and auxiliary labels through `info`.

### `info` (training only, never an input to the network)

Absolute x, y, z, angle; kill and item counts; map name and seed; keys held (when the training map can report it); any sector-derived labels for auxiliary losses. A unit test must assert that no `info` key and no absolute variable appears in the observation tensors.

### Rulings on the open points

**(a) `viz_nocheat` hedge.** Three layers.
1. Ask the organizers now (question below) and run a local probe: repeat the smoke test with `+viz_nocheat 1` and record exactly what changes in depth, labels, objects, sectors, each game variable we use, the notifications buffer and the HUD.
2. What survives from pixels alone: `screen`, `prev_action`, and whichever game variables the probe shows are still valid. The network is built with a pixel branch and `modality` flags from day one (ADR 0003), so the fallback is a configuration of the same agent and not a rewrite.
3. If the answer is "flag is on", or there is no answer by the end of day 2, switch on modality dropout: in a fraction of training episodes the telemetry keys are zeroed and flagged absent, and the pixel branch gets training-only auxiliary targets (predict `rays` and entity presence from the screen, the Arnold-style use of privileged labels). Cost if triggered: the pixel branch must then carry the policy, the learner needs the GPU, and the frame budget rises by an amount we cannot estimate without measuring.

**(b) Keycards.** Key possession is **not** fed as a privileged feature, because the unmodified engine offers no such variable. The agent knows it holds a key through three rule-safe routes, in this order of reliance:
1. **Memory.** A key is an entity of type key-colour in the set; when the agent walks onto it, it disappears. The recurrent state must carry that. To make this learnable, a **training-only auxiliary head** predicts "keys held" (3 bits) from the memory state. Labels come from `info` on training maps (ACS/USER variable, or env-side tracking from objects info). Nothing at inference depends on them (rule 4.2).
2. **HUD pixels.** The vanilla status bar shows key icons. Whether they are legible at 160x120 is UNVERIFIED; if the probe shows they are not, a fixed full-resolution crop of the status-bar key area is added as a fourth sensor key. A fixed crop is a sensor, not a decision.
3. **Notifications buffer.** Whether ViZDoom 1.3.3 delivers pickup or "you need a key" messages there is UNVERIFIED. Probe; use only if it is plain vanilla output.

**(c) Entity angle and distance.** Legitimate, with a hard line. The rule bans computing the answer (a bearing to snap aim to); it does not ban describing where things appear in the agent's own view. The line we adopt:
- Entity direction is given **only as the screen-space position of its label box** (`u`, `v`, `w`, `h`), and distance **only as the depth-buffer value on that object**. These are what the eye sees; they are the same information a pixel CNN would extract.
- **No `atan2` and no bearing computed from world positions anywhere in the pipeline.** `object_position_*`, `object_angle` and the absolute `object_velocity_*` are never read by the observation code. This makes the audit a grep.
- No feature in action units ("turn N degrees"), no "in crosshair" or "aimed" flag, no per-entity threat score, no sorting by target priority, no special treatment of monsters versus other types beyond the category id.
- Entity motion is the finite difference of `u` and `d` by `object_id`, not world velocity.

Residual risk stated honestly: `u` is monotonic in bearing, so a strict reader could still object. Both the screen-box route and the world-position route die under `viz_nocheat`.

**(d) Ego-motion.** Allowed as proprioception. `self` carries forward speed and strafe speed in the body frame, vertical speed, and turn delta since the previous decision (wrapped), all scaled. They are computed inside the wrapper from VELOCITY_X/Y/Z and ANGLE; this needs one rotation by the player's heading, which is a local-frame transform of the agent's own motion and involves no target. The absolute variables never leave the wrapper except into `info`. No integrated position and no heading (absolute or relative to spawn) is given to the network as a number. If the organizers object to the rotation, the fallback is to drop these four values; `prev_action` and ray changes still convey motion.

### Wrapper

A thin `gymnasium.Env` written by us directly on `DoomGame` (the bundled `VizdoomEnv` does not expose labels objects, frame skip is 1 by default, and we need our own action encoding). 160x120, default FOV, hidden window, depth and labels buffers on, objects/sectors info on only when a training reward needs them. One config switch `vanilla_eval=True` turns off every training-only feature and asserts none is active.

### Questions for the organizers

1. Does the evaluation harness set `+viz_nocheat`, or otherwise restrict depth buffer, labels buffer, objects info, sectors info or game variables?
2. Do we submit our own inference loop that creates and configures `DoomGame`, or does the harness fix resolution, buffers, button set and frame skip? Is the HUD rendered?
3. Is transforming engine data into the player's frame inside the observation pipeline (screen-space entity boxes, depth rays, body-frame velocity) accepted as sensory encoding under the "no trigonometry aimbot" rule?
4. Which IWAD and map format do the evaluation maps use, and what is the per-map time limit?

## Risks

- **Harness risk.** If `viz_nocheat` is on, `rays`, `entities` and possibly `self` are lost and the agent falls back to its weakest branch. This is the largest single risk to the entry.
- **Rule interpretation.** Screen-space `u` and body-frame velocity may be read as trigonometry by a strict judge. Mitigated by ruling (c) and question 3, not eliminated.
- **Perception gap.** Lock colour and exit switches are visible only through `screen`; if the pixel branch is weak the agent cannot do the key/door objective however good its telemetry is.
- **Forward-only view.** Rays and entities cover only the field of view. Enemies behind are invisible except through the damage signal and memory.
- **Generalization.** The category table is keyed on class names from Freedoom; evaluation maps with other actors fall into `unknown`. `object_id` tracking may break if ids are reused (probe).
- **Evidence.** Telemetry-only Doom policies are unevidenced in the literature we opened.
- **Throughput.** Per-step Python over labels and boolean masks is unmeasured.

## Validation experiment

Run by `env-engineer`, about half a day, results in `docs/reference/engine-probes.md`:

1. **Nocheat probe:** same episode with and without `+viz_nocheat 1`; for each buffer and each game variable in the contract record valid / noised / missing.
2. **Key and door probe** on a bundled Freedoom map with keys: do key items appear as labels and under what name and category; are HUD key icons distinguishable at 160x120 and at 60x80; what does the notifications buffer contain on pickup and on using a locked door.
3. **Leak test:** unit test asserting no absolute coordinate, angle, map name or `info` value is present in any observation key; a grep-level check that the observation module contains no `atan2` and no read of `object_position_*`.
4. **Throughput:** single env, random actions, frame skip 4, all contract keys on; report decisions per second and the cost of each key.

Reverse or amend if: the nocheat probe plus the organizers' answer show telemetry is unavailable at evaluation (then the pixel branch becomes primary and this ADR is superseded); a single env runs below roughly 250 decisions per second (then drop per-entity masked depth for a box-centre sample and reduce to 16 entities); keys are not visible as labels (then ruling (b) route 1 loses its input and the HUD crop becomes mandatory).
