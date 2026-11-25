# ADR 0002: Action space and frame skip (six categorical heads, frame skip 4)

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

## Context

The agent must move, aim, shoot, and open doors and press switches (USE) on the unmodified engine. Every action must come from the network's forward pass, and no action masking may encode strategy (rule 4.1). ViZDoom 1.3.3 on this machine offers ATTACK, USE, SPEED, MOVE_*, TURN_*, SELECT_WEAPON0-9, SELECT_NEXT/PREV_WEAPON and the `*_DELTA` buttons.

Evidence (`docs/research/vizdoom-agents.md`): frame skip 4 is the default in four independent sources; Sample Factory used skip 2 only for duels. Flat enumerations of button combinations grow as 2^n (DFP used 256); Sample Factory factorized into independent heads. None of the opened papers evaluated doors or keycards, so the USE design has no published support.

Unknown: whether the judges' harness dictates the button set or frame skip. Deadline unknown ("more than 5 days"). Throughput numbers are unmeasured on this machine.

## Options considered

**A. Small flat discrete set (about 9 to 12 hand-picked combinations, IMPALA style).** Build 1 to 2 hours. Throughput: same environment cost, smallest policy head. Cost: we would hand-pick which combinations exist (no strafe-while-turn-while-shoot unless listed), turning is coarse, and adding USE and run doubles the list each time.

**B. Flat enumeration of all combinations.** Build 1 hour. 504 or more actions in one softmax; slow to explore, and each button's meaning must be relearned per combination.

**C. Factorized multi-discrete heads with binned delta turning (chosen).** Build about 3 hours in the env plus 2 hours in the policy head (joint log-probability and entropy are sums over heads). Throughput: negligible difference from A.

**D. Continuous turn head (Gaussian on TURN_LEFT_RIGHT_DELTA) plus discrete heads.** Build about half a day more; mixed distributions complicate PPO and entropy tuning. Not justified before a baseline learns.

Frame skip alternatives: **2** (finer aim, half the decisions per second of game time, twice the horizon in steps), **4** (chosen), **variable/learned repeat** (a head choosing 1/2/4/8; interesting for the innovation score but adds a credit-assignment problem; deferred).

## Decision

**Six independent categorical heads, sampled jointly each decision, no masking of any kind:**

| Head | Size | Meaning | Buttons |
|---|---|---|---|
| `move` | 3 | none, forward, backward | MOVE_FORWARD, MOVE_BACKWARD |
| `strafe` | 3 | none, left, right | MOVE_LEFT, MOVE_RIGHT |
| `turn` | 7 | -10, -4, -1, 0, +1, +4, +10 degrees per tic | TURN_LEFT_RIGHT_DELTA |
| `attack` | 2 | off, on | ATTACK |
| `use` | 2 | off, on | USE |
| `speed` | 2 | walk, run | SPEED |

Joint space 3 x 3 x 7 x 2 x 2 x 2 = 504. `prev_action` in ADR 0001 is this six-index vector.

- **Frame skip 4**: one decision is applied for 4 tics (about 8.75 decisions per second of game time). A turn bin of 1 degree per tic therefore turns 4 degrees per decision, which sets the finest aim step; the 10 bin gives 40 degrees per decision.
- **USE is applied on the first tic of the decision only and released for the remaining tics.** Doom's use action is believed to be edge-triggered (UNVERIFIED on this build), so a held USE across consecutive decisions might never re-fire. The release is uniform plumbing in the action encoding, applied identically at training and inference; it is not a rule about when to use.
- **No weapon-selection head in the first version.** The engine's own behaviour on pickup and on empty ammo is vanilla and is relied on. Random weapon switching early in training would damage combat learning. The policy head code must allow adding a head later (new head initialized biased to "no-op") without invalidating a checkpoint.
- **At inference the action is sampled from, or is the argmax of, the network's output distributions.** Which of the two is chosen per head is fixed in the inference config and evaluated on held-out maps; no code inspects the observation to override a head.
- Turn bin values and frame skip live in one config object shared by training and inference.

Question for the organizers: does the evaluation harness fix the available buttons, the action format or the frame skip, or do we run our own loop and choose them? Are delta buttons permitted?

## Risks

- **Harness mismatch.** If the harness fixes buttons (for example binary TURN_LEFT/RIGHT only) or frame skip, the `turn` head must be retrained. Mitigation: ask now; keep the head sizes in config.
- **Aim granularity.** 4 degrees per decision may be too coarse for distant targets. Evidence that skip 2 helps aim is an inference from where it was used, not a measured result.
- **Independent heads cannot express correlations** between buttons within a step (for example "strafe left only when turning right"). Accepted; autoregressive heads are a later option.
- **USE semantics.** If the edge-trigger belief is wrong, or first-tic-only USE fails on some door types, doors will look broken to the agent.
- **Delta turn units** (degrees per tic, float or integer) are assumed, not verified.
- **Rule compliance.** None expected: no masking, no scripted override. The USE release must stay unconditional to remain so.

## Validation experiment

Run by `env-engineer` while building the wrapper, about 2 hours, results in `docs/reference/engine-probes.md`:

1. **Turn calibration:** apply each turn bin for one decision and read the heading change from `info`; confirm units and symmetry.
2. **USE probe:** stand at a door on a bundled Freedoom map; compare (i) USE held for all 4 tics on consecutive decisions, (ii) USE on the first tic only. Record whether the door opens once, repeatedly, or not at all, and whether a switch can be pressed.
3. **Random-policy sanity:** 10,000 decisions with uniform random heads; confirm every head changes the game state and report decisions per second.

Later, by `evaluator` once a baseline learns: train the same agent at frame skip 2 and 4 on a combat scenario for an equal number of game tics and compare kills per episode and hit rate on held-out maps.

Reverse if: the USE probe shows first-tic USE does not open doors (then hold USE for the whole decision and force a release tic between decisions); skip 2 beats skip 4 by a clear margin at equal tics (then move to skip 2 with a higher discount, ADR 0004); the organizers fix the action set.
