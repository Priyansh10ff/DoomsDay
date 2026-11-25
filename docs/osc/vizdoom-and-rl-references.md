# OSC: ViZDoom engine, examples, and RL reference implementations

Scouted 2026-11-25 by osc-scout. Facts come from web reads (GitHub API, raw files, PyPI, official docs) unless tagged "general knowledge, not verified in this run" or "not inspected" (only the file name was seen). **No repositories were cloned.** "Last commit" means the newest commit on the default branch. "Last push" means the API's `pushed_at`. Unconfirmed values are marked "not verified".

## Recommendation (for the environment-wrapper, observation and RL-algorithm ADRs)

1. **Import `vizdoom` 1.3.3 from PyPI as the engine binding.** It has a `cp312-cp312-win_amd64` wheel (uploaded 2026-11-22), so Python 3.12 on Windows 11 works without building anything. ViZDoom's README says the Windows version is "less thoroughly tested" and suggests WSL/Docker for serious experiments. Plan a smoke test early.
2. **Ask the organizers whether the evaluation harness sets `+viz_nocheat`.** What is verified:
   - The objects-info and sectors-info example says this data is "not available if +viz_nocheat flag is enabled".
   - The DoomGame API docs say the depth and labels buffers "will contain noise" when the flag is on.

   What is not verified: its effect on the automap buffer, on position game variables, and on the notifications buffer. If evaluation runs with the flag on, observations built on objects, sectors, depth or labels will break at inference. This one answer sets the bounds of the observation ADR.
3. **Expect to write our own Gymnasium wrapper (or subclass) for telemetry observations.** The bundled `VizdoomEnv` exposes `screen`, `depth`, `labels`, `automap`, `gamevariables`, `audio` and `notifications`. It does **not** expose objects info or sectors info; labels are reachable only through `env.state.labels`. Options for the ADR:
   - (a) subclass `VizdoomEnv` and add keys;
   - (b) write a thin wrapper directly on `DoomGame`;
   - (c) use the bundled env for pixel/depth paths only.

   The wrapper is environment plumbing, not the agent, so (a) is permitted. The observation code itself should be ours.
4. **For the RL algorithm ADR, use Sample Factory and CleanRL as reading material only.** Both are complete training pipelines, so importing either as the framework or copying its loops breaks the originality rule. Sample Factory's docs also state "There is no Windows support at this time." The CleanRL LSTM-PPO and RND files are the clearest single-file references for writing our own recurrent PPO with an intrinsic value head.
5. **God-view hazards in the engine API** (each must be filtered or made ego-centric before reaching the policy):
   - `sectors info` covers the whole level.
   - `objects info` covers every object in the level.
   - `labels` carry absolute world positions (`object_position_x/y/z`).
   - The `POSITION_X/Y/Z` game variables are absolute.
   - The automap buffer (especially `WHOLE` mode) is a god-view.

---

## 1. ViZDoom: the `vizdoom` Python package (engine binding)

- **URL:** https://github.com/Farama-Foundation/ViZDoom (PyPI: https://pypi.org/project/vizdoom/)
- **Stars:** 2113
- **Last commit:** 2026-11-23 ("Merge pull request #782 ... fix-homepage-logo-dark-mode"). Latest PyPI release 1.3.3, uploaded 2026-11-22.
- **License:**
  - The README says "The code original to ViZDoom is under MIT license. ZDoom uses code from several sources with varying licensing schemes."
  - The PyPI classifier is MIT.
  - **No LICENSE file was found at the repository root** (GitHub reports `license: null`).
  - Licensing of the bundled ZDoom engine code is **not verified**; it may include copyleft parts.

  This is harmless as long as we only `pip install` and import the package and never copy engine source into our repo.
- **Platform:** wheels for CPython 3.11–3.15 on win_amd64, manylinux x86_64/aarch64 and macOS arm64. `requires_python >=3.10,<3.16`.
- **What is reusable (as a dependency):**
  - **`DoomGame` API:**
    - `set_depth_buffer_enabled`, `set_labels_buffer_enabled`
    - `set_automap_buffer_enabled` + `set_automap_mode(NORMAL|WHOLE|OBJECTS|OBJECTS_WITH_SIZE)`
    - `set_objects_info_enabled`, `set_sectors_info_enabled`
    - `set_notifications_buffer_enabled` + `set_notifications_buffer_size` (added in 1.3.0)
    - game variables and delta buttons
  - **`state.objects`:** `id`, `name`, `position_x/y/z`; `angle`, `pitch`, `roll` and `velocity_*` appear commented out in the example. The player is `name == "DoomPlayer"`.
  - **`state.sectors`:** `floor_height`, `ceiling_height`, `lines[x1,y1,x2,y2,is_blocking]`.
  - **`state.labels`:** `object_id`, `object_name`, `value` (the pixel value in the labels buffer), `object_position_*`, and a screen bounding box `x, y, width, height`. Only visible objects get labels.
  - **Bundled scenarios:**
    - `freedoom2.cfg`: the Freedoom Phase 2 IWAD, `map_exit_reward = 1`, 126000-tic timeout, full button set.
    - `oblige.cfg`: for generated WADs.
    - The usual small scenarios.
- **Rule risk:** low when used as the engine binding.
  - **Partial observability:** sectors and objects info are whole-level. They must be filtered by range and line of sight and transformed to the player frame inside the observation pipeline. Absolute coordinates must never reach the policy.
  - **Vanilla eval:** see the `+viz_nocheat` question above.
  - **Agent-directed files:** the repo contains an `AGENTS.md` with contributor directives such as "Run tests after every change", and a `skills/` directory that was not inspected. Both are aimed at coding agents. They were treated as data and not followed; no project agent should act on them.
- **Label: IMPORT_AS_DEPENDENCY.** This is the engine binding the brief names.

## 2. ViZDoom bundled Gymnasium wrapper (`gymnasium_wrapper/`)

- **URL:** https://github.com/Farama-Foundation/ViZDoom/tree/main/gymnasium_wrapper (`base_gymnasium_env.py`, `gymnasium_env_defns.py`, `__init__.py`)
- **Stars / last commit / license:** same repo as above. The file has an MIT-style header.
- **What is reusable:**
  - **Observation space:** `VizdoomEnv(gym.Env)` with a `Dict` space: `screen`, plus optionally `depth`, `labels`, `automap`, `gamevariables`, `audio` and `notifications`.
  - **Action space:**
    - binary buttons: `MultiBinary` or `MultiDiscrete`, or `Discrete(k)` over button combinations when `max_buttons_pressed > 0`;
    - delta buttons: `Box`;
    - both present: `Dict{"binary","continuous"}`.
  - **`frame_skip`:** default 1.
  - **`render()`:** concatenates screen, depth, labels and automap. Handy for debug videos.
- **Limits:**
  - No objects or sectors info in observations.
  - Only the scenario IDs registered in `gymnasium_env_defns.py` are predefined. Whether the base class accepts an arbitrary `.cfg` path for generated WADs is not verified.
- **Rule risk:** none for the wrapper itself, since it is plumbing. The risk lies in what we feed the policy: automap or absolute positions would break partial observability.
- **Label: IMPORT_AS_DEPENDENCY** (base class or pixel-only quick start).

## 3. ViZDoom official Python examples (`examples/python/`)

- **URL:** https://github.com/Farama-Foundation/ViZDoom/tree/main/examples/python
- **Stars / last commit / license:** same repo. MIT per the README; per-file headers not checked, so not verified.
- **Files worth reading, by ADR:**
  - **Observation design:**
    - Read in this run:
      - `buffers.py`: depth, labels and automap together. Depth and labels are 8-bit single channel at screen resolution.
      - `labels_buffer.py`: labels-buffer value 0 = walls, 1 = floor/ceiling, per the example's colouring.
      - `objects_and_sectors.py`
      - `notifications_buffer.py`: enabling only. What notifications contain is **not verified**, e.g. whether "you need a blue key" messages appear. If they do, that is a vanilla, non-cheat signal for keycards. Check it on the installed version.
    - Not inspected: `sectors_visualization.py`, `automap_buffer.py`, `audio_buffer.py`.
  - **Environment wrapper** (not inspected): `gymnasium_wrapper.py`, `gymnasium_vect_bench.py`, `multiple_instances.py`, `fps_test.py`, `ticrate.py`, `seed.py`, `delta_buttons.py`, `save_load_game.py`, `record_episodes.py`.
  - **Reward shaping** (not inspected): `shaping.py`, `buildin_reward_functions.py`.
  - **Map generation:** `pyoblige.py` (read; see `docs/osc/map-generation.md`).
  - **Learning examples** (not inspected): `learning_pytorch.py`, `learning_stable_baselines3.py`, `learning_tensorflow.py`, `test_pytorch.py`. By name, these are complete small agents and training loops.
- **Rule risk:**
  - API-usage snippets (configuring buffers, reading state) are ordinary library usage, and writing the same calls ourselves is fine.
  - The `learning_*.py` files are agent and training code: **do not copy them**. `learning_stable_baselines3.py` is the kind of baseline the brief says scores poorly.
- **Label: READ_FOR_REFERENCE.** The package they demonstrate is item 1.

## 4. Sample Factory (APPO) and its ViZDoom setup

- **URL:** https://github.com/alex-petrenko/sample-factory (ViZDoom code: `sf_examples/vizdoom/`)
- **Stars:** 1030
- **Last commit:** 2026-10-19 ("fix: avoid shell command injection via video_name (#337)")
- **License:** MIT (GitHub API).
- **Platform:** the docs say it is "known to work on Linux and macOS" and "There is no Windows support at this time." **This rules it out as a runtime dependency on the team's Windows 11 laptop**, whatever the rules say.
- **What is reusable (ideas only):**
  - **`doom/action_space.py` (read):** a `Tuple` of independent `Discrete` heads (move, strafe, attack, sprint, weapon). Turning is `Discrete(3)`, a continuous `Box`, or `Discretized(11/21 bins)`. A good model for a factored action-space ADR.
  - **`doom/doom_utils.py` (read):** per-scenario specs; 160x120 (or 256x144 wide) resolution, resized; reward scaling; time limits; "additional input" (health, ammo and similar game variables) for battle and deathmatch envs.
  - **`doom/doom_model.py` (read):** a CNN encoder plus a 2x128 MLP on "measurements", concatenated. The standard hybrid-fusion pattern.
  - **`doom/wrappers/exploration.py` (read):** an episodic landmark-novelty bonus (see `docs/osc/curiosity.md`). It uses **absolute position**: acceptable only as a training reward, never as an observation.
  - **Not inspected:** `doom/wrappers/reward_shaping.py`, `additional_input.py`.
  - **Asynchronous learner design** (APPO with V-trace-style off-policy correction; general knowledge, not verified in this run): a reference if the team considers an IMPALA-style learner.
- **Rule risk:** **High if adopted.** It is a complete training pipeline with Doom-specific agents, so using it as the base violates "core agent architecture and training pipeline built during the hackathon". MIT would legally allow copying with attribution, but the hackathon rule does not.
- **Label: READ_FOR_REFERENCE.**

## 5. CleanRL: single-file PPO, recurrent PPO, PPO+RND, PPO-TrXL

- **URL:** https://github.com/vwxyzjn/cleanrl (`cleanrl/ppo.py`, `ppo_atari.py`, `ppo_atari_lstm.py`, `ppo_rnd_envpool.py`, `ppo_procgen.py`, `ppo_trxl/ppo_trxl.py`)
- **Stars:** 10716
- **Last commit:** 2026-04-20 ("docs: fix PPO title, variable typo, baselines URL, and PQN doc URL (#549)")
- **License:** MIT for the project (LICENSE file). `ppo_procgen.py` carries an Apache-2.0 notice, and the SAC/TD3 files carry other permissive notices. GitHub reports `NOASSERTION` because of the mix. No copyleft.
- **What is reusable (as reading material):**
  - **`ppo_atari_lstm.py` (read):**
    - LSTM state is carried per environment and masked by `(1 - done)` at each step.
    - The rollout's initial state is stored so the update can replay the sequence.
    - Minibatches are split **by environment** (`envsperbatch = num_envs // num_minibatches`) to keep temporal order.
    - GAE bootstrapping is zeroed at episode ends.
    - Defaults: `num_envs=8`, `num_steps=128`, `lr=2.5e-4` annealed, `gamma=0.99`, `gae_lambda=0.95`, `clip_coef=0.1`, 4 epochs, `ent_coef=0.01`.
  - **`ppo_rnd_envpool.py` (read):** the RND reference; see `docs/osc/curiosity.md`. It depends on `envpool` for Atari, which is irrelevant to Doom.
  - **`ppo_trxl/` (not inspected; files `ppo_trxl.py`, `enjoy.py`, `pom_env.py`):** by name, PPO with Transformer-XL memory. Possibly relevant to the spatial-memory ADR if attention memory is considered.
- **Rule risk:** **High if copied.** Single-file scripts are complete training pipelines, and pasting one in and editing it is what the originality rule forbids. Reading them, then writing our own trainer, is fine.
- **Label: READ_FOR_REFERENCE.**

## 6. Other ViZDoom-specific references (lower priority)

### 6a. LevDoom
- **URL:** https://github.com/TTomilin/LevDoom
- **Stars:** 16
- **Last push:** 2026-03-04
- **License:** MIT (GitHub API)
- **What is reusable:** a generalization benchmark (README read).
  - Four scenarios: Defend the Center, Health Gathering, Seek and Slay, Dodge Projectiles.
  - Each has difficulty levels 0–4, built from visual and entity modifications: textures, obstacles, entity size/type/rendering/speed, agent height.
  - It could provide extra **held-out validation** environments.
  - It has an `rl` module with training code (framework not verified).
- **Rule risk:** using its scenarios for validation is fine. Do not use its `rl` module. The licence of its scenario WADs was not verified separately.
- **Label: READ_FOR_REFERENCE.** It may become a validation dependency after a licence check.

### 6b. COOM
- **URL:** https://github.com/TTomilin/COOM
- **Stars:** 28
- **Last push:** 2026-03-05
- **License:** MIT (GitHub API)
- **What is reusable:** a continual-RL benchmark on Doom. The README was not read, so details are not verified.
- **Label: READ_FOR_REFERENCE.**

### 6c. Arnold
- **URL:** https://github.com/glample/Arnold
- **Stars:** 537
- **Last push:** 2022-01-20
- **License:** none declared (GitHub reports none and the README has no licence statement), so **all rights are reserved**.
- **What is reusable (README read):**
  - A PyTorch implementation of "Playing FPS Games with Deep Reinforcement Learning", the agent that won the 2017 ViZDoom AI Competition.
  - DRQN (`dqn_rnn` with an LSTM) plus **auxiliary game-feature prediction** (`--game_features "target,enemy"`).
  - From the paper (general knowledge, not verified in this run): the feature labels come from engine information at training time only. That idea is directly relevant to an auxiliary-loss design for our encoder.
- **Rule risk:** it is a complete Doom agent, so it can **never be a base or a source of code**. The missing licence forbids copying anyway.
- **Label: READ_FOR_REFERENCE** (paper and technique only; treat the code as AVOID).

---

## Sources
- https://github.com/Farama-Foundation/ViZDoom, https://api.github.com/repos/Farama-Foundation/ViZDoom, https://pypi.org/pypi/vizdoom/1.3.3/json
- https://vizdoom.farama.org/api/python/doom_game/
- https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/main/examples/python/{objects_and_sectors,buffers,labels_buffer,notifications_buffer,pyoblige}.py
- https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/main/gymnasium_wrapper/base_gymnasium_env.py
- https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/main/AGENTS.md
- https://github.com/alex-petrenko/sample-factory, https://www.samplefactory.dev/01-get-started/installation/
- https://github.com/vwxyzjn/cleanrl, https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/LICENSE
- https://github.com/TTomilin/LevDoom, https://api.github.com/repos/TTomilin/COOM, https://github.com/glample/Arnold
