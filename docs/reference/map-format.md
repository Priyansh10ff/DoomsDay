# Map format facts verified for ViZDoom 1.3.3 (map-forge, 2026-11-25)

All results come from runs on this laptop (Windows 11, Python 3.12, vizdoom 1.3.3, `freedoom2.wad` from the wheel), with one engine instance at a time. Scripts: `maps/probes/probe_nodes.py`, `maps/probes/probe_extended.py`, plus the per-map QA in `maps/qa/validate.py`.

## Node building: PWADs without nodes load

- A PWAD written by our own code with only THINGS, LINEDEFS, SIDEDEFS, VERTEXES and SECTORS **loads and plays**. Two variants were tested:
  - SEGS, SSECTORS, NODES, REJECT and BLOCKMAP present but empty (0 bytes);
  - those five lumps absent altogether.
- In both variants the engine built the nodes itself:
  - the player spawned exactly on the start thing;
  - `state.sectors` count equalled the SECTORS lump;
  - a DR door (special 1) opened;
  - a keycard was picked up;
  - a W1 exit line (special 52) ended the episode with `map_exit_reward` 1.0.
- **No external node builder (ZDBSP etc.) is needed.** Our WAD writer emits empty lumps in vanilla order.
- Monsters work with empty REJECT/BLOCKMAP. After a warp to 96 units from a Zombieman on an S3 map, player health went 100 to 61 within 60 decisions.

## Doors, keys, exits (all vanilla Doom-format specials)

- **DR door (special 1):** opens from the front side of the door line. Our door faces always have the door sector on the back side.
- **DR keyed doors (26 blue, 27 yellow, 28 red):** without the key the door stays shut and the notifications buffer shows "<Colour> key needed for this door." After `give <colour>card` (QA cheat only) the door opened.
  - This was checked on **all 60 generated S2/S3 maps** by `engine_check`.
  - Skull keys (38, 39, 40) are placed on 30% of keyed maps.
- **Exits:**
  - The W1 exit (52) on the lines around a 64x64 pad ends the episode as a level exit.
  - The S1 exit switch (11, texture SW1EXIT) ends it after one USE press.
  - Both exit types were checked on all 120 generated maps.
- **USE is edge-triggered** (it confirms engine-probes section 6):
  - USE held from spawn, out of range, never opened the door; the press is spent where nothing can be used.
  - Pressing every other decision makes the door oscillate.
  - Opening needs one press while in range. For env-engineer and reward-designer: holding USE constantly is a dead strategy on our maps, as on real ones.
- **Engine class names, as observed** in `state.objects` across the 120 maps of batch 1 (the per-map JSON, `validation.engine.objects`):
  - DoomPlayer;
  - BlueCard, YellowCard, RedCard, BlueSkull, YellowSkull, RedSkull;
  - Zombieman (lower-case m), ShotgunGuy, DoomImp;
  - Stimpack, Medikit, HealthBonus, GreenArmor, Clip, Shell, Shotgun.
  - `Demon` (thing 3002) is in the table but **no map places it**, so it is unverified.

## Several maps in one PWAD

- A PWAD can hold several maps with lump names `DD0000`, `DD0001`, ... and `MAP33`.
- In **one** engine session, each map loads via `set_doom_map(name)` + `new_episode()`.
- `DoomEnv(config with map_wad=bundle, maps=(...))` plus `reset(options={"map": name})` switches maps per episode **without any env change**.
- A level exit inside a bundle ends the ViZDoom episode; it does not continue to the next map.
- Names other than MAP01-MAP32 avoid the IWAD MAPINFO entries (level names, MAP07 specials).

## Our generator's format choices

- Vanilla Doom binary format only:
  - no BEHAVIOR lump, no ACS, no MAPINFO;
  - every texture and flat checked against `freedoom2.wad` (963 textures, 240 flats);
  - things carry flags 7, so they appear on all skills.
- Lattice of 32 units, with axis-aligned walls only. Every space is separated from every other by at least one void cell, so there are no accidental openings.
- Linedefs are split at every shared vertex, so there are no T-junctions.
- Per-episode random start positions are **not possible on vanilla maps**: a single player-1 start, with no ACS. Spawn variety comes from many maps.

## Not verified

- Behaviour under `+viz_nocheat 1`. The QA uses cheats; DoomEnv rollouts ran without the flag.
- Whether the judges' engine build also builds nodes on load. If it does not, a node builder would be needed for our held-out WADs, though not for the judges' own maps.
- Long-run stability: each map got one short engine session only.
