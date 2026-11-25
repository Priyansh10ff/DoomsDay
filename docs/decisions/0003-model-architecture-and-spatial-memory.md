# ADR 0003: Model architecture and spatial memory (staged: set-attention + GRU first, ego-centric neural map second)

Date: 2026-11-25. Status: **Proposed, pending ratification by the user.** Author: chief architect (advisor).

## Context

Inputs and outputs are fixed by ADR 0001 (observation dictionary) and ADR 0002 (six categorical heads). Framework is PyTorch (decided). The deliverable must show a custom topology that keeps spatial memory across rooms and handles a variable number of enemies; the judges reward bespoke modules. All of it must be written during the event (rule 4.3) and every action must come from the forward pass (rule 4.1).

Compute: one laptop, CPU-only torch today, 8 GB GPU present but unused, user requires the machine not be driven hard. Deadline unknown ("more than 5 days"); budgets assume 5 days.

Evidence weighed (`docs/research/spatial-memory.md`, `vizdoom-agents.md`):

- Grid memory written at the agent's location beat LSTM on unseen Doom mazes (66.6% vs 52.4%) at a reachable 7M frames. **But** the published version used absolute game coordinates, which we may not. The ego-centric form is described in that paper and not evaluated. Evidence for what we would build is therefore weak.
- Active Neural SLAM's result depends on a classical planner (banned). Its reusable idea is supervising a map with ground truth as training-only labels.
- MERLIN (partial read, no numeric table extracted): memory trained only by the RL gradient fails over long delays; a predictive auxiliary loss fixes it.
- Entity sets with shared embedding, masked attention and pooling are well supported outside Doom (hide-and-seek); there is no Doom evidence for them.
- No opened source tests any memory on Doom keycard-then-door backtracking.

Throughput of every option is unmeasured.

## Options considered

**A. MLP/GRU over a flattened, padded observation.** Build half a day. Fastest. No permutation invariance over entities, weakest innovation story, and the research shows GRU-only below structured memory in every comparison opened.

**B. Set-attention entity encoder + ray encoder + GRU core, with auxiliary heads (Stage A below).** Build about 1 day, under 1M parameters, CPU-trainable. Custom, but the spatial memory is only a recurrent vector.

**C. B plus an ego-centric neural map (Stage B below).** Build 1.5 more days, including debugging of map shifting. Learner cost rises (backpropagation through the map over the rollout; unmeasured, expected 1.5 to 3 times the Stage A update time). Highest innovation value; weakest evidence in its rule-compliant form.

**D. Transformer over a long observation history (no explicit map).** Build about 1.5 days. Memory and compute grow with context length; a keycard detour is thousands of decisions. Not chosen.

**E. Full hierarchy now (manager choosing options or map goals, worker executing).** Build 2 to 3 days on top of C, with weak evidence at small budgets and a hard learned-gate requirement. Deferred, see below.

## Decision

Build in stages. Each stage is a working agent; a later stage is started only when its gate is met.

### Stage A0: first agent that trains (build about 1 day, `worker`)

- **Ray encoder:** 1D convolutions over `rays` (3 channels, 80 columns) to a 64-vector.
- **Entity set encoder:** per entity, a learned type embedding (16) concatenated with the 8 features, through a shared MLP to 64-d tokens; one masked multi-head self-attention block (4 heads) with residual; pooling by a learned query attending over the tokens (masked). A learned "no entity" token is always present so an empty set is well defined. Permutation-invariant, any count from 0 to 24.
- **Self encoder:** MLP on `self` (28) to 32; `prev_action` through six small embeddings to 16; `modality` flags concatenated.
- **Fusion:** MLP to 192, then a **GRU (256)**. Hidden state is reset on episode end.
- **Heads:** six categorical policy heads; value head for extrinsic return and a second value head reserved for intrinsic return (ADR 0005).
- **Auxiliary heads, training-only labels from `info`:** keys held (3 bits) from the GRU state; damage in the next k steps; number of visible monsters. These supervise memory and perception; nothing reads their outputs to choose actions.

### Stage A1: pixel branch on (build about half a day)

A small CNN (3 convolutions) over `screen` (3, 60, 80) to a 64-vector joins the fusion. Purpose: lock colours, exit switches, HUD keys, and the `viz_nocheat` fallback (ADR 0001). Auxiliary targets for this branch: reconstruct `rays` and entity presence from pixels. Turned on before any key/door curriculum; needs the learner benchmark in ADR 0004 to say whether CPU is enough.

### Stage B: ego-centric neural map (build about 1.5 days, time-boxed)

- A grid of learned feature vectors, proposed 21 x 21 cells x 16 channels, cell size proposed 128 map units, kept per environment alongside the GRU state and zeroed at episode start.
- **Addressing is by dead reckoning from ego-motion only** (the body-frame velocity and turn delta of ADR 0001). The grid is stored aligned to the heading at episode start and translated so the agent stays at the centre; sub-cell remainders are accumulated. No absolute coordinate is used, and the integrated pose is never given to the network as a number.
- **Write:** at the agent's cell, a GRU-style gated update whose content is a learned function of the current fused features. **Read:** the grid is resampled into the agent's current heading (one resampling per step, so blur does not accumulate in storage), passed through a small CNN to a vector, plus a content-based attention read; both feed the GRU core.
- **Auxiliary supervision (training only):** decode from the map a coarse local occupancy and "door seen here" target built from sectors info and absolute pose in `info`.

**Ruling on rule 4.1 for the map.** The addressing arithmetic (shift the grid by how far the agent moved) is sensory bookkeeping in the same class as a local-frame transform: it describes where the agent has been and decides nothing. What is written and how it is read are learned parameters. No search, planner, frontier rule or distance transform ever runs over the grid, and the grid is consumed only by network layers. We will state this in the submission and ask the organizers to confirm (question below). A variant that writes depth rays into cells geometrically is **not** adopted now, because it puts more hand-written geometry into the memory and is harder to defend.

### Gates

- **A0 to A1:** A0 learns on bundled scenarios without doors (clearly above a random policy on a navigation and a combat scenario) and the trainer throughput is measured.
- **A1 to B:** A1 reaches the exit on at least half of episodes on held-out simple multi-room maps without keys, **and** at least 2.5 days remain before the deadline. If the first condition is not met by the end of day 3, Stage B is cut and the remaining time goes to curriculum and reward work.
- **Keeping B:** B stays in the submitted agent only if it beats A1 by at least 10 points of exit-reach rate on held-out maps that require backtracking, at equal environment steps. Otherwise A1 is submitted and B is reported as an ablation.

### Hierarchy

**Deferred to a Stage C, not built now.** If A1 and B are done with at least 1.5 days left, the candidate is a learned slow head: every k decisions the network samples a small categorical "intent" that conditions the fast heads. The switch must be a sampled network output; a hand-written gate such as Arnold's "no enemy visible, so navigate" is forbidden. The research gives no evidence that learned hierarchy helps at our budget, so this is for the innovation score only and must not displace A or B.

Question for the organizers: is a memory grid whose cell addressing is computed from the agent's own motion (dead reckoning), with learned contents and learned read-out and no planner, acceptable under "zero hardcoded scripts"?

## Risks

- **Stage B may not beat the GRU.** The ego-centric neural map is untested in its source paper; dead-reckoning drift (collisions, lifts, teleporters) will corrupt addressing over long episodes.
- **Rule interpretation of the map.** A strict judge could call dead-reckoned addressing hand-written logic. Mitigated by the ruling and the question above; fallback is A1.
- **Pose leak.** If training maps have a fixed spawn, a spawn-aligned grid is equivalent to an absolute frame on those maps. Requires randomized spawn position and heading in training (ADR 0005) and held-out evaluation.
- **Auxiliary heads as hidden rules.** If any auxiliary output were fed back into action selection outside the network, it would breach rule 4.1. They are loss terms only.
- **Recurrent training instability** (reported in Procgen for LSTM) and stale hidden states in PPO; handled in ADR 0004.
- **Pixel branch on CPU** may dominate learner time; unmeasured.
- **Time.** Stages A0 + A1 + B total about 3 days of build before long training runs; with a 5-day horizon that leaves little slack.

## Validation experiment

1. **Set-encoder unit checks** (`worker`, 1 hour): output unchanged under permutation of entity rows; finite with zero entities; unchanged when padded rows hold garbage.
2. **Memory probe task** (`worker` + `evaluator`, half a day once A0 trains): a small set of maps where a cue seen at the start decides which of two distant branches holds the exit. Compare A-stage agent with GRU reset every 16 steps, the full GRU, and Stage B, at equal steps, on held-out layouts.
3. **Linear probe** on the frozen memory state for "keys held" and "a door was seen": if a linear readout is at chance, the memory is not storing it.
4. **Learner cost:** update time per rollout for A0, A1, B on this machine.

Reverse if: Stage B fails the "keeping B" gate (ship A1); Stage B makes the update more than 3 times slower than A1 with no gain; the organizers reject dead-reckoned addressing (ship A1 and describe the GRU plus auxiliary losses as the memory).
