# ADR 0005: Exploration signal and curriculum / map sources (short, provisional)

Date: 2026-11-25. Status: **Proposed, pending ratification by the user. Provisional:** this is a shorter ADR; reward shaping terms beyond the exploration bonus are **deferred** to a later ADR with `reward-designer`, and level-sampling curricula (PLR) are deferred until a map pool exists.

## Context

The brief requires exploration mechanisms that drive the agent into unfamiliar rooms, and zero-shot play on unseen maps (rule 4.4). Intrinsic rewards and map generation are training-only and do not touch the action path (rules 4.1, 4.2).

Evidence weighed (`docs/research/exploration.md`, `generalization-and-hierarchy.md`, `docs/osc/curiosity.md`, `map-generation.md`):

- ICM solved sparse ViZDoom navigation, but was farmed by firing in DMLab. Our agent has an ATTACK button and limited ammo.
- RND avoids that by construction (an inference, not a Doom measurement); its results are at about 2x10^9 frames and its authors say it does not solve long-horizon exploration.
- Episodic bonuses (Episodic Curiosity, NGU's episodic part, E3B, Sample Factory's landmark wrapper) reset each episode. That they transfer better to unseen maps is an argument from mechanism, **not measured**. On the one keys-and-doors task in the EC paper, curiosity did not beat PPO.
- Number of distinct training levels is the main lever for generalization (Procgen; 2D games, order-of-magnitude guide only). Doom papers agree in direction.
- OBSIDIAN v21 can generate Doom 2 levels headlessly, but: it is an unsigned third-party Windows executable with no published checksum; whether its maps reliably contain keys, locked doors and exits, load in ViZDoom's engine, and are completable is UNVERIFIED; it can hang on error paths; it is GPL (must not be vendored).
- omgifol (MIT, pure Python) reads and writes WADs but does not build nodes. Whether ViZDoom's engine rebuilds missing nodes on load is UNVERIFIED.
- Bundled with ViZDoom: `freedoom1.cfg`, `freedoom2.cfg`, `oblige.cfg`, and small scenarios (`my_way_home`, `deadly_corridor`, `health_gathering`). Whether the Freedoom IWADs are inside the installed wheel is UNVERIFIED.

A tension to resolve: the Freedoom campaign maps are the only maps with real keys, doors and exits available without running a generator, and they are also the natural held-out set.

Deadline unknown ("more than 5 days"). Throughput effects are unmeasured.

## Options considered

Exploration:

- **E1. Episodic coverage bonus from absolute position in `info`** (count visits per coarse cell, bonus shrinking with count, reset each episode). Build 2 hours, no network, negligible throughput cost. Uses privileged position for a training reward only. Simple; little innovation credit.
- **E2. Learned episodic novelty in embedding space** (our own elliptical or nearest-neighbour bonus over features trained with an inverse-dynamics loss, memory emptied each episode). Build about 1 day. Small per-step cost (unmeasured). Coordinate-free and a real "intrinsic curiosity" module. Evidence at our scale is thin.
- **E3. RND.** Build half a day, cheap. Lifelong novelty fades and gives nothing on a new map by itself.
- **E4. ICM.** Build half a day. Rejected as the main signal because of the firing-farm failure.

Maps:

- **M1. Bundled scenarios only.** Zero build. No doors or keys in the small ones; far too few layouts.
- **M2. Our own small generator with omgifol** (rooms, corridors, doors, keys, exit, random spawns), written by `map-forge`. Build 1 to 2 days. Full control over key/door structure and no third-party executable, **if** the engine accepts maps without prebuilt nodes. Maps will be plainer than real levels.
- **M3. OBSIDIAN as an external tool.** Build about half a day for a batch driver with timeouts plus validation. Rich, varied levels. Requires the user to approve and run an unsigned executable.
- **M4. Hand-made maps in an editor.** Days of manual work; also needs third-party tools. Rejected.

## Decision

**Exploration:** start with **E1** as soon as the trainer runs, because it is the cheapest way to get the agent moving, and build **E2** as the submitted curiosity module once Stage A1 learns. E1 then stays as the baseline E2 must beat. In both cases:
- the bonus is episodic (memory cleared at reset);
- it has its own value head and discount (ADR 0004), and its weight is annealed, with a vanilla-reward-only evaluation always run;
- the features it is computed on must exclude anything the agent can change just by firing (no ammo count, no own-projectile entities), to avoid the ICM failure.
- ICM is not used. RND is kept only as a possible lifelong modulator if time remains.

**Curriculum ladder** (each rung trains on many layouts and random spawn position and heading where the map allows):
1. Bundled small scenarios for Stage A0: navigation (`my_way_home`), survival (`health_gathering`), combat (`deadly_corridor`). Purpose: prove the pipeline learns.
2. Generated multi-room maps without keys, then with one key and one locked door, then with monsters. Training-only aids (auto-opening doors, extra ammo) are allowed early and must be off by the last rung.
3. Mixed full levels.

**Map sources:**
- **First, a half-day probe by `map-forge` of M2:** write a two-room map with omgifol and try to load and finish it in ViZDoom without a node builder. If it works, our own generator is the primary source: no third-party executable, exact control of key/door dependencies, and it counts as original work.
- **OBSIDIAN (M3) is proposed, not assumed.** It is the best route to realistic variety, but downloading and running it is **the user's decision**. If approved: the zip goes to `third_party/downloads/` and is never run in place; every call has a timeout and kill; a vanilla or Boom-compatible target is selected to avoid GZDoom-only features; generated WADs stay out of git; every map is validated with omgifol (has exit, keys match locked doors) and by loading it in ViZDoom before use.
- If the M2 probe fails and OBSIDIAN is not approved, a node builder is unavoidable and is itself a third-party executable; the user would have to choose one.

**Held-out split (resolving the Freedoom tension):** Freedoom **Phase 2** maps (`freedoom2.cfg`) are **never trained on** and are the standing held-out set, together with a fixed set of generator seeds that are excluded from training. Freedoom **Phase 1** maps may be used for training in the last curriculum rung. No hyperparameter is tuned on a single held-out map.

## Risks

- **Style mismatch.** The organizers' evaluation maps may look like neither our generator's output nor OBSIDIAN's. Freedoom Phase 2 is our only hand-made check.
- **Too few layouts.** If neither M2 nor M3 works, we are left with a handful of maps and will overfit (the Procgen and DFP evidence).
- **Curiosity may not help key/door progression** (the one relevant measurement was negative) and can pull the agent toward hazards ("dancing with skulls" in RND).
- **E1 uses absolute position.** It must stay in the reward path only; if it leaked into observations it would breach the rules. Covered by the leak test in ADR 0001.
- **Training aids left on.** A policy that learned with auto-opening doors may never press USE. The last rung and all evaluation must be vanilla.
- **Third-party executable.** Security and stability risk on the user's only machine; GPL status of generated WADs is unverified.
- **Time.** E2 plus a generator is about 2 to 3 days of build across two agents.

## Validation experiment

1. **M2 probe** (`map-forge`, half a day): omgifol-built two-room WAD with a door, a key and an exit; does ViZDoom load it, can a human-driven or random run finish it, does the exit end the episode.
2. **Exploration ablation** (`evaluator`, once the trainer runs): on `my_way_home` and on held-out generated maps, compare no bonus, E1, and later E2 at equal steps; measure exit-reach rate, distinct cells visited per episode (from `info`), and shots fired per episode.
3. **Generalization gap** tracked from the first generated-map run: success on training seeds versus held-out seeds versus Freedoom Phase 2, reported in `reports/eval/`.

Reverse if: E2 does not beat E1 on held-out maps (submit E1 plus an honest ablation, or try the nearest-neighbour variant); shots per episode rise with the bonus on (the features leak a firing signal; fix the feature set); the M2 probe fails and the user declines OBSIDIAN (escalate the node-builder choice to the user immediately, since without a map source the generalization objective cannot be met).
