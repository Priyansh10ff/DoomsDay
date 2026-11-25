# OSC: Map generation and WAD tooling (OBLIGE / ObAddon / OBSIDIAN, omgifol, Freedoom)

Scouted 2026-11-25 by osc-scout. Facts come from web reads (GitHub API, raw source and READMEs, PyPI, oblige.sourceforge.net) unless tagged "general knowledge, not verified in this run". **No clones; nothing was downloaded or executed.** "Last push" is the GitHub API `pushed_at`. Unconfirmed values are marked "not verified".

## Recommendation (for the map-generation / curriculum ADR)

1. **Primary procedural generator: OBSIDIAN v21**, the maintained continuation of OBLIGE combined with ObAddon (https://github.com/GTD-Carthage/Obsidian-Content). Reasons:
   - It has a real **headless batch mode** (`-b/--batch <output.wad>`).
   - It ships a Windows x64 binary.
   - It has a built-in node builder. The v20 repo embeds ZDBSP; which builder v21 uses is not verified (its `source/` has `bsp_*.cc` files and no `zdbsp` directory).
   - It produces full Doom 2 levels. Whether these reliably contain keycards, locked doors and exits is **not verified**. Check generated WADs with omgifol's thing and line tables before the ADR relies on it.

   Treat OBSIDIAN as an **external tool we run, not a library we import**. It is GPL-2.0-or-later C++ and Lua, and none of its source should enter our repo.
2. **Use omgifol (MIT, pure Python) as the dependency for WAD inspection and editing.** Typical uses:
   - listing maps in a generated WAD and extracting map lumps;
   - counting keys, doors and exits;
   - small post-edits for training-only curricula (e.g. removing monsters);
   - reading geometry for map statistics.

   It **does not build nodes**, so geometry edits need a node builder afterwards.
3. **Use Freedoom (BSD-3) as the IWAD and as a source of held-out validation maps.** ViZDoom uses Freedoom graphics by default and ships `freedoom2.cfg` / `freedoom1.cfg` scenario configs. The hand-made Freedoom Phase 1 and Phase 2 maps (Phase 2 has a full Doom 2-style megawad of about 32 maps, which is general knowledge, not verified in this run) are a good **held-out** check that the agent is not overfitting to OBSIDIAN's style.
4. **Avoid PyOblige** and the archived ObAddon repo. PyOblige has had no push since 2018, is GPL, compiles C++ on pip install, and its Windows build is a zip you copy into `site-packages`. ObAddon has been folded into OBSIDIAN.
5. **Before committing to OBSIDIAN, test that its output loads and is completable in ViZDoom.** The risks (see item 1):
   - map format: Doom vs UDMF;
   - GZDoom-oriented features such as 3D floors and extra floors;
   - completability in ViZDoom's ZDoom fork.

### What would have to be executed on the user's machine (map generation)

Nothing has been downloaded or run. To generate maps, a human (or map-forge with the user's explicit approval) would have to:

1. **Download** `obsidian-v21-win64-20251226.zip` from https://github.com/GTD-Carthage/Obsidian-Content/releases (tag `Obsidian-v21-20251226`, published 2025-12-26). The release has **a single zip asset with no published checksum or signature**, unlike Freedoom's signed releases. If it is first saved under `third_party/downloads/`, that copy is untrusted and must **never be executed in place** (CLAUDE.md). **The user decides** where the tool is installed and whether running it is approved.
2. **Run the Windows GUI executable in batch mode**, for example `OBSIDIAN.EXE --batch <out.wad> [--load <settings.txt>] [--randomize-arch] [--randomize-combat] [--randomize-pickups] [--randomize-all] [-v]`.
   - Settings can be changed persistently with `-u/--update <c|o> <key> <value>`.
   - **There is no `--seed` flag.** The seed comes from the config or loaded settings, and `-k/--keep` keeps the seed from a loaded file.
   - Exact config key names can be dumped with `-p/--printref` or `--printref-json`. That is also an execution of the exe.
3. **Always wrap each call in a timeout and kill.** On Windows (non-console-only build), batch mode calls `AllocConsole()`. Behaviour read from `source/main.cc` on branch `v21-content-unstable`:
   - **Success:** it calls `Main::Shutdown(false); return 0;` and exits normally.
   - **"No output filename given":** it prints "Close window when finished..." and enters `do { } while (true);`, an infinite busy loop. Confirmed.
   - **Build failure:** the same loop follows `FatalError("FAILED!")` in the source. `FatalError`'s body was not shown, so it may exit first. That makes the hang on build failure **not verified**.

   A batch driver therefore needs a per-map timeout and process kill. Alternatively, use the Linux/Docker build, which avoids the Windows-only loop branch.
4. **Alternative path:** build the Docker image from the repo (`docker build -t obsidian-level-maker .`, then `docker run --rm -v ./output:/wads obsidian-level-maker --batch /wads/x.wad --randomize-all`). This requires Docker Desktop or WSL on the laptop, and building the repo's Dockerfile is also executing third-party code.
5. **Load the output in ViZDoom** with `freedoom2.wad` as the IWAD, using `set_doom_scenario_path(<out.wad>)` and `set_doom_map("map01")` etc., as `scenarios/oblige.cfg` does. OBSIDIAN targets Doom 2; whether OBSIDIAN-specific features behave correctly on Freedoom Phase 2 is not verified.
6. **If omgifol is used to edit geometry,** a separate node builder must be run afterwards. That is another external executable to choose and approve; OBSIDIAN's internal builder is not exposed as a standalone tool (not verified). Whether ViZDoom's ZDoom fork rebuilds missing nodes on load is also **not verified**. Test it before relying on it.

Generated WADs are training data, not decision logic, so map generation does not touch Rule 4.1. They also directly support Rule 4.4: varied, randomized training maps plus held-out maps.

---

## 1. OBSIDIAN (current): GTD-Carthage/Obsidian-Content

- **URL:** https://github.com/GTD-Carthage/Obsidian-Content (default branch `v21-content-unstable`)
- **Stars:** 29
- **Last push:** 2026-11-24
- **License:** GPL-2.0 per the GitHub API. The upstream Obsidian `LICENSE.txt` is GPLv2 "or (at your option) any later version".
- **Status:** The legacy repo's README says "THIS REPOSITORY IS CURRENTLY INACTIVE AND UNSUPPORTED" and points here as primary. This repo describes itself as "A community continuation of the OBLIGE Level Maker, with new features and quality of life updates", and says most development targets **Doom 2 on UZDoom/GZDoom** first.
- **Batch / CLI (from `source/main.cc` help text):**
  - `--home`, `--install`, `--config`, `--options`, `--log`
  - `-b --batch <output>` ("Batch mode (no GUI)")
  - `-a --addon`, `-l --load`, `-k --keep`
  - `--randomize-all|arch|combat|pickups|other`
  - `-d --debug`, `-v --verbose`
  - `-p --printref`, `--printref-json`
  - `-u --update <section> <key> <value>`

  Batch mode returns before the FLTK main window is created.
- **Platform:**
  - Windows x64: a binary zip in Releases (single asset `obsidian-v21-win64-20251226.zip`, no checksum or signature published).
  - Linux, BSD and Haiku: build from source (`COMPILING.md`).
  - Docker is supported.
- **Output format:** a WAD file.
  - Generated WAD/PK3 files embed OBSIDIAN's config, so settings, including the seed, can be recovered from a map.
  - `ports/zdoom.lua` labels the port "GZDoom" and enables `bridges`, `extra_floors`, `liquid_floors` and `png_images`. It uses Legacy-style extrafloor types (281, 301) and FraggleScript for things on 3D floors. These are **features ViZDoom's older ZDoom fork may not support identically; must test.**
  - A comment there says Hexen format is "later", which implies binary Doom-format maps. The default output format is **not verified**.
  - The other port modules are `boom.lua`, `edge.lua` and `xlat.lua`. Choosing a vanilla/Boom target may avoid GZDoom-only features and matches "vanilla rules" at evaluation better.
- **What is reusable:** generated WADs (via the executable) as a training-map distribution.
  - The `--randomize-*` flags give domain randomization.
  - Settings such as level size, monster quantity and item amounts can drive a curriculum; exact keys come from `--printref`, not verified.
- **Rule risk:**
  - **Licence (GPL):** running the program does not affect our licence. **Copying any of its Lua modules, prefabs or C++ into our repo would make those files GPL** and would require distributing the combined work under GPL terms. Do not vendor its source.
  - **Generated WADs:** whether WADs that embed GPL prefab geometry inherit GPL obligations is **not verified**. Keep them out of git, or record their provenance, until the team decides.
  - **Originality:** none. It is a level generator, not an agent.
  - **Evaluation mismatch:** OBSIDIAN style may differ from the organizers' unseen evaluation maps. Mix in Freedoom and other hand-made maps for validation.
  - **Execution:** an untrusted, unsigned Windows binary, with possible hangs on error paths (see above).
- **Label: IMPORT_AS_DEPENDENCY,** as an external **tool** whose output we consume. It is not imported as code, not vendored, and not committed.

## 2. OBSIDIAN legacy (v20): obsidian-level-maker/Obsidian

- **URL:** https://github.com/obsidian-level-maker/Obsidian (`dashodanger/Oblige` redirects here; default branch `obsidian`; branches `oblige-770-enhanced`, `obsidian`, `v21`)
- **Stars:** 199
- **Last push:** 2026-02-13
- **License:** GPLv2-or-later (`LICENSE.txt`). The GitHub API reports `NOASSERTION`.
- **Releases:** `Obsidian-v20-20250522` (win32/win64 zips) and `Oblige-770-Enhanced` (2021-02-12, win32/win64 zips).
- **What is reusable:** a fallback if v21 is unstable. Source layout: `source_files/obsidian_main` plus embedded `zdbsp`, `lua`, `slump` and other libraries.
- **Rule risk:** same GPL considerations as item 1. The README says v20 is no longer supported.
- **Label: AVOID** (inactive; use item 1). Its release zips are acceptable only as a fallback.

## 3. ObAddon: caligari87/ObAddon

- **URL:** https://github.com/caligari87/ObAddon
- **Stars:** 81
- **Last push:** 2021-05-05. The repo is **archived**.
- **License:** GPL-3.0
- **What is reusable:** nothing directly. These are community prefabs for Oblige 7.70, merged into OBSIDIAN according to OBSIDIAN's description.
- **Rule risk:** GPL-3.0 if copied. Archived.
- **Label: AVOID** (unmaintained; superseded by OBSIDIAN).

## 4. OBLIGE 7.70 (original, Andrew Apted)

- **URL:** https://oblige.sourceforge.net/ (GitHub mirror https://github.com/samboy/Oblige; mirror stars and date not verified)
- **Stars:** not verified (SourceForge)
- **Last release:** 7.70, described as the final stable version; development has stopped. Exact date not verified.
- **License:** GPL; the bundled notice reportedly points to GPL v2. This is from a search summary of the docs and man page, not verified against the LICENSE file.
- **Batch mode:** yes. Official usage docs give `Oblige --batch OUTPUT.wad game=doom2 length=episode theme=hell` (settings as `name=value`, no spaces around `=`) and `Oblige --batch OUTPUT.wad --load MYCONF.txt`. The docs say nothing about Windows-specific batch behaviour.
- **What is reusable:** a historically stable generator. ViZDoom's `oblige.cfg` and `pyoblige.py` were built around it (via PyOblige).
- **Rule risk:** GPL if source were copied. Unmaintained.
- **Label: AVOID** (superseded by OBSIDIAN, which keeps the `--batch` interface).

## 5. PyOblige: mwydmuch/PyOblige

- **URL:** https://github.com/mwydmuch/PyOblige (PyPI `oblige`)
- **Stars:** 12
- **Last push:** 2018-07-02
- **License:** GitHub reports none. The README says GNU GPL.
- **What is reusable:** the knob list shown in ViZDoom's `examples/python/pyoblige.py`:
  - generator setup: `oblige.DoomLevelGenerator`, `set_seed`;
  - config: `set_config({"size": "micro", "health": "more", "weapons": "sooner"})`;
  - presets such as `childs_play_wad`;
  - `generate(path)`, which returns the map count.

  Our own Python driver calling the OBSIDIAN exe in batch mode can do the same job.
- **Rule risk:**
  - `pip install oblige` compiles C++ (needs g++, make and libfltk on Linux).
  - The Windows install is a precompiled zip copied into `site-packages`, so an untrusted 2018 binary would be loaded into our interpreter.
  - GPL.
  - Python 3.12 compatibility is not verified.
- **Label: AVOID** (unmaintained, unsafe Windows install, GPL).

## 6. omgifol: devinacker/omgifol

- **URL:** https://github.com/devinacker/omgifol (PyPI `omgifol`)
- **Stars:** 91
- **Last push:** 2026-07-26. Latest PyPI release 0.5.1, uploaded 2023-05-23.
- **License:** MIT
- **Platform:** pure Python (`py3-none-any` wheel), `requires_python >=3.3`. Windows-safe. Python 3.12 is not explicitly tested (not verified).
- **What is reusable:**
  - `omg.wad` / `omg.wadio`: WAD and lump I/O (from the module list; not inspected).
  - `omg.mapedit.MapEditor` (inspected):
    - `from_lumps()` / `to_lumps()` for vertexes, sidedefs, sectors, things and linedefs;
    - Doom and Hexen/ZDoom formats (`ZThing`, `ZLinedef` when a BEHAVIOR lump is present);
    - `draw_sector()` adds a closed polygon sector; `paste()` merges maps with an offset.
  - Not inspected:
    - `omg.udmf`;
    - `omg.thinginfo` and `omg.lineinfo` (by name, these are thing and line type tables, likely useful for counting keys, doors and exits);
    - demo scripts `drawmaps.py`, `merge.py`, `mirror.py` and `listdir.py` (`mirror.py` may be useful for map-mirroring augmentation).
- **Limits:** **No node builder.**
  - NODES, SSECTORS, SEGS, BLOCKMAP and REJECT are passed through, set empty when missing, and go stale after geometry edits. The docstring says an external node builder is needed.
  - GL nodes are read but not written.
  - ACS is passed through; there is no compiler.
- **Rule risk:** none. It is a WAD I/O library with no agent logic. Curriculum edits are training-only map modifications, which Rule 4.2 allows provided evaluation uses unedited maps.
- **Label: IMPORT_AS_DEPENDENCY.**

## 7. Freedoom: freedoom/freedoom

- **URL:** https://github.com/freedoom/freedoom (official releases: https://github.com/freedoom/freedoom/releases)
- **Stars:** 1387
- **Last push:** 2026-11-23
- **License:** BSD-3-Clause ("modified BSD", `COPYING.adoc`). GitHub reports `NOASSERTION`. Permissive, so there is no copyleft consequence.
- **Latest release:** v0.13.0, published 2024-01-29.
  - Assets: `freedoom-0.13.0.zip` (+ `.sig`), `freedm-0.13.0.zip` (+ `.sig`), `freedoom-0.13.0-CHECKSUM`.
  - Releases are signed; verify the signature and checksum after download.
- **What is reusable:**
  - **The `freedoom2.wad` IWAD.** ViZDoom's README says Freedoom graphics are the default and can be replaced with `doom2.wad`. Whether the wheel bundles `freedoom2.wad` was not verified directly, but the `freedoom2.cfg` scenario expects it.
  - **Freedoom Phase 1 and Phase 2 maps** as held-out validation with real keycards, doors and exits (map count is general knowledge, not verified). ViZDoom's `freedoom2.cfg` sets `map_exit_reward = 1`, a 126000-tic (about 60 min) timeout and the full button set.
  - FreeDM deathmatch maps for combat-only curricula.
- **Rule risk:** none for use as an IWAD and evaluation set. **Do not train on maps reserved for held-out validation** (Rule 4.4). The team should decide the split, e.g. OBSIDIAN for training and Freedoom for validation only.
- **Label: IMPORT_AS_DEPENDENCY** (as game data / IWAD).

---

## Sources
- https://github.com/GTD-Carthage/Obsidian-Content, https://raw.githubusercontent.com/GTD-Carthage/Obsidian-Content/v21-content-unstable/README.md, https://raw.githubusercontent.com/GTD-Carthage/Obsidian-Content/v21-content-unstable/source/main.cc, https://raw.githubusercontent.com/GTD-Carthage/Obsidian-Content/v21-content-unstable/ports/zdoom.lua, https://api.github.com/repos/GTD-Carthage/Obsidian-Content/releases
- https://github.com/obsidian-level-maker/Obsidian, https://raw.githubusercontent.com/obsidian-level-maker/Obsidian/obsidian/LICENSE.txt, https://api.github.com/repos/dashodanger/Oblige/releases
- https://api.github.com/repos/caligari87/ObAddon
- https://oblige.sourceforge.net/doc_usage.html, https://github.com/samboy/Oblige, https://doomwiki.org/wiki/Oblige
- https://github.com/mwydmuch/PyOblige, https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/main/examples/python/pyoblige.py
- https://github.com/devinacker/omgifol, https://raw.githubusercontent.com/devinacker/omgifol/master/omg/mapedit.py, https://pypi.org/pypi/omgifol/json
- https://github.com/freedoom/freedoom, https://raw.githubusercontent.com/freedoom/freedoom/master/COPYING.adoc, https://api.github.com/repos/freedoom/freedoom/releases
- https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/main/README.md, https://api.github.com/repos/Farama-Foundation/ViZDoom/contents/scenarios
