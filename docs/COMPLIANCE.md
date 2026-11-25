# Rules compliance status

**Compliance is not established.** This file maps each hackathon rule to the code
that addresses it and copies the verdict of the latest audit. It is not a claim that
the entry complies.

- Latest audit: `reports/audit/2026-11-25-rules-audit-01.md` (2026-11-25, by
  `rules-auditor`, static inspection). Totals: **7 PASS, 1 FAIL, 5 UNCLEAR.**
- That audit predates `teachers/`, `evaluate/`, BC/DAgger (`train/bc.py`) and the
  generated-map training path (`--maps generated`). Its PASS verdicts do not cover that
  code. Its line numbers are now stale, so this file cites functions.
- Changes listed below as "since audit 01" have **not been audited**.
- **TODO**: run `rules-auditor` for audit 02 on the final code and checkpoint, then
  replace the verdict column. Until then no rule below should be read as passed for the
  submitted agent.

Rules are from `docs/RULES.md`, wording from `docs/TRACK_BRIEF.md` section 5.

## Decision path (inference)

| Rule | Mechanism in the code | Audit 01 verdict | Since audit 01 (unaudited) |
|---|---|---|---|
| R1. Every action comes from trained network parameters | `agent/inference.py` `PolicyRunner.act`: forward pass, then per-head sample or argmax, fixed at construction. `env/actions.py` `encode`: fixed index-to-button map. `evaluate/policies.py` `NetworkPolicy` wraps `PolicyRunner`. | PASS (for the components that existed; wiring under R6) | `evaluate/` now provides the wiring; not audited. |
| R2. No hand-written if/else decision trees | No action mask, fallback or override in `agent/`, `env/actions.py`, `env/doom_env.py`. | PASS | `evaluate/episode.py` `run_episode` added; its docstring says the policy receives only the observation and episode-start flag. Not audited. |
| R3. No A*, Dijkstra or other classical pathfinding | Search code exists only in training-time tools: `maps/generator.py`, `maps/qa/validate.py` (map QA) and, since audit 01, `teachers/navigator.py` (Dijkstra for the scripted teacher). | PASS (before `teachers/` existed) | `tests/test_teachers_isolation.py` fails if any module under `agent/`, `env/` or `evaluate/` imports `teachers` directly (it checks direct imports in those three packages only). `tests/test_maps_isolation.py` forbids `agent/` and `env/` from importing `maps` or `curriculum`, and `train/` from importing `maps.qa`. `evaluate/` imports `maps.loader` (manifest and hash checks; it imports only the standard library) and `train.checkpoint`. A text search of `agent/`, `env/`, `evaluate/`, `train/` and `configs/` found no import of `teachers` or `maps.qa`; `train/run_doom.py` imports `maps.loader` and `maps.doors`, and `maps.doors` imports `maps.qa.validate` (map geometry for the training-only door reward), so the training entry point reaches `maps.qa` indirectly; the evaluation path does not import `train.run_doom` or `maps.doors`. No test checks indirect imports. |
| R4. No atan2 / trigonometry aimbots | Trigonometry on the inference path is limited to `env/observation.py` `ego_motion`, which rotates the agent's own velocity into its body frame. No entity feature is in action units. `atan2` appears in `maps/qa/validate.py` and the teacher (`teachers/`), both training-only. | PASS (interpretation items I1-I4 below) | Teacher code added (training only); not audited. |

## Engine and evaluation

| Rule | Mechanism in the code | Audit 01 verdict | Since audit 01 (unaudited) |
|---|---|---|---|
| R5. Engine / ACS / game-variable modifications only in training | `env/config.py` `EnvConfig.assert_vanilla` rejects, under EVAL: non-stock IWAD, any scenario PWAD, any console argument except `+viz_nocheat 1`, whole-level objects/sectors info, altered engine reward knobs, extra knobs. `rewards/factory.py` `ShapedEnvFactory` refuses a vanilla-eval config. | PASS | Generated-map training uses our own vanilla-format PWADs with no ACS and vanilla reward knobs (`configs/a0_env.py` `generated_maps_env_config`). The only scenario with ACS used in training is bundled `my_way_home` (first run). No auto-open doors, infinite ammo or cheats appear in any `configs/ops/*.json`. |
| R6. Evaluation on the unmodified engine; nothing needed at inference depends on a mod | `evaluate/run.py` `verify_configs`: for every map, `assert_vanilla` and `assert_same_sensors(checkpoint signature, a0_eval_env_config)` before any episode; `check_spec_matches` on the obs/action spec. One engine at a time. | UNCLEAR (no eval script existed then) | `python -m evaluate.run` now exists and was used for the val numbers in `reports/eval/2026-11-25-bc-dagger.md`. Not audited. **Gap**: there is no command to load an arbitrary judge-supplied WAD; `--split` accepts only `val`, `heldout`, `my_way_home`. |

## Original work

| Rule | Mechanism | Audit 01 verdict | Since audit 01 (unaudited) |
|---|---|---|---|
| R7. Core agent and training pipeline written during the hackathon | No RL framework imported; no pretrained weights; no licence headers or external repository references found. | UNCLEAR: the repository has **zero commits**, so nothing shows when the code was written. | Still zero commits at the time of writing. **TODO**: make a first commit so the history exists. |

## Observations

| Rule | Mechanism | Audit 01 verdict | Since audit 01 (unaudited) |
|---|---|---|---|
| R8. No absolute map coordinates in the observation | `env/doom_env.py` `split_vars` sends pose, counts and map name to `InfoVars`; `_build_obs` receives only `ObsVars` and sensor buffers. Heading is used only inside `ego_motion`. | PASS | Observation v2 (door channel, key bits) is in progress and not in the code. **TODO-VERIFY** after it lands. |
| R9. No god-view radar; partial observability only | Rays from the rendered depth buffer; objects/sectors info off and refused under EVAL; no automap. | UNCLEAR: whether the label list could include an object hidden behind a wall was not verified. | `env/observation.py` `entities` now drops labels with no visible pixel of their own. Not audited. |

## Judging

| Rule | Mechanism | Audit 01 verdict | Since audit 01 (unaudited) |
|---|---|---|---|
| R10. Zero-shot on unseen maps; never tune on a single map | `python -m train.run_doom --maps generated` trains on the 80-map TRAIN split through the curriculum; `maps/loader.py` refuses held-out maps by split, path and hash; the env refuses `heldout_maps/`. | **FAIL** (at audit time training used only `my_way_home`) | Generated-map training runs exist (`reports/runs/INDEX.md`). No held-out evaluation has been run. Verdict stays FAIL until re-audited. |

## Extra items in audit 01

| Item | Audit 01 verdict | Since audit 01 (unaudited) |
|---|---|---|
| E1. Held-out maps cannot be loaded in training | UNCLEAR (env did not check map names against the PWAD) | Env now refuses training map names that are not in the configured PWAD (`EnvConfig` `allowed_maps` / `check_reset_map`). Not audited. |
| E2. Shaping, cheats and curriculum never required at inference | PASS | Door term and BC added (training only). Not audited. |
| E3. `+viz_nocheat` dependency | UNCLEAR (organizer question) | Unchanged. See open risks. |

## Interpretations that need organizer confirmation

From audit 01, section 4: all are treated as compliant only under our reading.

- **I1.** Entity `u, v` screen position: monotonic in bearing; a strict reader could
  call it a pre-computed aim error. It is not in action units and uses no trigonometry.
- **I2.** `du, dd` are tracked by the engine's object id across frames.
- **I3.** Entity type from engine class names; nearest-24 truncation when more are
  visible.
- **I4.** Body-frame ego-motion computed from absolute angle and velocity.
- **I5.** Depth buffer, labels buffer and game variables count as "unmodified engine".
- **I6.** `+viz_nocheat 1` whitelisted as a vanilla game argument.
- **I7.** Per-head sampling from the network's distribution counts as "decided by
  trained parameters".

## Open risks

1. **`+viz_nocheat`.** If the judges run with `+viz_nocheat 1`, the engine removes the
   depth buffer, labels buffer and position/velocity variables
   (`docs/reference/engine-probes.md`, cited in audit 01 E3). Our rays, entities and
   ego-motion become zeros; only health, armor, ammo and weapons remain. The A0 agent
   has no screen encoder and was never trained in that condition, so it could not
   function. Question to the organizers: open.
2. **Scripted teacher.** The teacher (`teachers/`) uses Dijkstra and trigonometry to
   produce training labels. It is never run at inference and is not imported by the
   inference path, but whether the organizers accept a scripted teacher for imitation
   is awaiting their confirmation (CLAUDE.md section 7).
3. **No git history.** The repository has zero commits, so "created during the
   hackathon" is not evidenced by the repository itself (audit 01, R7).
4. **No inference command for an arbitrary map.** `evaluate.run` cannot yet load a
   judge-supplied WAD (owner: `worker`).
5. **Observation v2.** New sensor channels are being added; they must be checked
   against R8 and R9 and against the "describe the world, do not pre-compute the
   answer" line in `CLAUDE.md` section 4.1.
