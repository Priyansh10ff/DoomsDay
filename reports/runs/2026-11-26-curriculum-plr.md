# 2026-11-26: Prioritized Level Replay map order in the curriculum scheduler

Code only; no training, no engine started. The live run (`reports/runs/20261126-night3b-v3-mixed/`) is untouched.

## What changed
- `curriculum/scheduler.py`: new `map_order: "plr"` (default stays `"shuffled_cycle"`; unknown values refused). Exact rule in the module docstring:
  unseen (never issued) maps of the stage first, uniformly; then `P = (1-rho)*P_S + rho*P_C`, with
  `score = (1 - mean(last window outcomes)) + progress_weight*|mean(last window) - mean(window before)|` (1.0 if issued but no outcome yet),
  `P_S ~ (1/rank)^(1/beta)`, rank = 1 + number of strictly higher scores (ties share a rank), `P_C ~ issues on the stage since the map was last issued` (uniform if all 0).
  Stats are per stage, per map; replay and left-stage outcomes update them; staleness is stamped at issue time. `state_dict()` adds a `plr` key only in plr mode (shuffled state byte-identical, pinned by hash); a state without `plr` loads into plr mode with fresh stats; a plr state for a different map pool is refused. `plr_metrics()` (pure) gives per stage `plr_mean_score_*`, `plr_visited_frac_*`, `plr_entropy_*`, `plr_entropy_norm_*`, `plr_max_prob_*`.
- `train/curriculum_vec_env.py`: one guarded line in `metrics()` adds `plr_metrics()` in plr mode only.
- `curriculum/config_mixed_plr.json`: `config_mixed.json` + `map_order: plr`, `plr: {rho 0.3, beta 0.3, window 5, progress_weight 0.0}` (UNMEASURED placeholders; learning-progress term is an opt-in).
- `tests/test_curriculum_plr.py`: 17 tests (incl. shuffled_cycle sequence + state hash pins, shipped-config params, CurriculumVecEnv step loop + state_json round trip in plr mode).

## Tests
Only these were run (CUDA_VISIBLE_DEVICES=-1, one pytest at a time): `pytest tests/test_curriculum_plr.py`: 17 passed. `pytest tests -k curriculum`: 42 passed (includes the 17). Free memory before running: 4.62 / 3.90 GiB.

## Sanity on the real pool (train-all-b2, no engine, simulated success)
Pool S0 220, S1 220, S2 220, S3 120; 0.14 ms per choice. 40,000 simulated choices, per-map latent success rates, >= 9-10 outcomes per map:
- progress_weight 0.0 (shipped): max single-map probability 0.17-0.22, normalized entropy 0.59-0.61.
- progress_weight 0.5: max 0.49-0.52 on three stages (a unique top-ranked map takes half the stage's draws), so it is not the default.
- Either way, 48 outstanding choices gave 4-8 distinct maps per 10-15 issued on a stage: PLR at beta 0.3 repeats maps among parallel envs; raise beta or rho if that is too concentrated.

## Map counts / generation
No maps generated or changed. splits.json: train 780 (S0 220, S1 220, S2 220, S3 120); val 20 (5 per stage); heldout 20 (5 per stage); external_eval 2 entries. No generation failures.

## Enable
`DOOMSDAY_CURRICULUM_CONFIG=curriculum/config_mixed_plr.json` on a fresh run or a resume (a resume of a shuffled_cycle run starts PLR stats fresh; stage, pending choices and rng continue).
