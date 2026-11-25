# Strategy: level completion on unseen maps (2026-11-25 23:50, about 24 h left)

Written by the strategist (`advisor` agent) from the gap analysis (`reports/eval/2026-11-25-gap-analysis.md`), the BC/DAgger report, the obs-v2 benchmark, engine probes Round 5, the research report (`docs/research/2026-11-25-blogs-reddit-doors-and-bc-ppo.md`), `train/ppo.py`, `train/trainer.py`, `teachers/controller.py` and the BC-v2 log to epoch 10. Saved by the coordinator. Not an ADR. Deliberate deviation from ADRs 0010/0011: entropy coefficient 0 and a BC anchor in PPO.

## 1. Diagnosis, in order
1. **USE presses are barely taught.** The teacher labels USE only at a hidden stand-off point within 6 degrees (`teachers/controller.py` lines 226-252). Pad exits 95% vs 7% for matched switch exits; 12 of 20 val maps have switch exits. Caps every stage.
2. **The first door stops about 80% of S1 episodes.** Same cause, plus v1 had no door channel. A second USE press reverses a moving door (probe R2.2), so a sampled USE head may shut doors (unmeasured).
3. **PPO erodes the imitation start.** First policy update KL 0.155; critic explained variance negative for 150 updates; the entropy bonus works against a sharp BC policy. Fixing this only preserves what BC gives, so it ranks third.
4. **Long routes fail.** Pad-only success 91% at 2 hops, 36% at 5: memory/exploration limit, not solvable in 24 h.
5. **Validation cannot separate policies** at 15 episodes per stage.

Checked facts: value warm-up freezes everything except value and aux heads (`trainer.py` `_freeze_all_but_value_and_aux`), so it is safe to lengthen. Switch exits already count as successes (88 in night1-A), so no exit-switch reward term is needed.

## 2. Tonight (until 08:00)
| When | Step | Detail |
|---|---|---|
| Now | Stop night1-C | Keep `checkpoints/checkpoint-u0000143.pt` (before the decline). |
| About 00:20 | BC-v2 baseline | `python -m teachers.collect --stages S0 S1 S2 --episodes-per-map 2 --student reports/runs/20261125-bc-v2/bc_model.pt --no-save --seed 777` |
| 00:00-01:30 | Builds C1-C4 | In parallel (section 3). |
| About 01:30 | Recollect `data/teacher/bc_v3` | 3,000 teacher-driven episodes with the new USE labels. Do not mix with `bc_v2`. |
| Then | BC | `python -m train.bc --data data/teacher/bc_v3 --epochs 40 --device cuda --head-weights 1,1,1,1,1,1 --use-pos-weight 8` |
| Then | One DAgger round | `--beta 0.3 --max-steps 400`, union, 8 epochs, `--init-from-bc`. Re-measure on seed 777. |
| About 03:30 | PPO overnight | Settings below. |

PPO (`--maps generated --shaping full_level_doors --init-from <best of BC-v3/DAgger by TRAIN S1> --start-stage S1`): `--value-warmup-updates 200 --lr 1e-4 --lr-warmup-updates 50 --clip 0.1 --ppo-epochs 2 --entropy-coef 0`, `--bc-coef 1.0` decaying 0.999 per update to a floor of 0.2 on teacher-state minibatches from `bc_v3` plus DAgger data; 4x8 if free RAM >= 5 GiB else 4x6; numbered checkpoints every 50 updates; no part frozen after warm-up; reward unchanged. BC loss on teacher states chosen over KL to frozen BC because the BC model is weak at doors and the student rarely visits post-door states. Expected about 550 decisions/s on v2, about 9M decisions by 08:00.

## 3. Code changes, smallest first
| # | Change | Build | Test | Worth it |
|---|---|---|---|---|
| C1 | `--use-pos-weight` in `train/bc.py`; USE recall/precision in `bc_metrics.jsonl` | 20 min | BC-val press recall >= 0.8 with precision >= 0.3; weight 1 reproduces the current loss | Yes, first |
| C2 | BC auxiliary loss + flags in PPO; skip remaining epochs when `approx_kl` > 0.03 | in progress +15 min | `--bc-coef 0` bit-identical; 30-update smoke: first post-warm-up KL < 0.02, BC loss falls; loader refuses sensor mismatch | Yes |
| C3 | Teacher labels USE from what the student can see: closed door (gap 0) or exit switch line within USE range under the view axis, one fresh press edge, never while a door is moving or open | 60-90 min | Student-driven S1 USE-label share 0.9% -> above 3%; zero labelled presses at gap > 0; teacher-driven success stays 100% on 20 S0-S2 maps | Yes, highest value |
| C4 | Per-episode logging: USE edges, edges in range, door reversals after a press, on-path doors passed, exit kind | 45 min | Scripted press/release/press counts one reversal | Yes; kill criteria need it |
| C5 | On-path-only door reward | 30 min | Side-door case pays 0 | Not tonight |
| C6 | Reverse curriculum by spawn placement | 2-3 h | At-door starts reach 20% within 150 updates | Tomorrow, only if K5 fires |
| C7 | Training-only progress potential (key -> door -> exit) | 2-4 h | Potential never rises toward a locked door without its key | Tomorrow, only if doors pass but episodes time out |

Skip: new curiosity module, more DAgger rounds on old labels, turn-label smoothing. C3 stays in `teachers/`; no USE debounce/filter in the env wrapper or inference path.

## 4. Tomorrow to midnight
- 08:00-09:30: screen checkpoints on val 20 maps x 6 episodes, paired seeds, sampled and argmax, for the BC start and three PPO checkpoints; top two re-run at 15 per map.
- 09:30-16:30: continue PPO from the peak, or switch to C6/C7 as the gates direct. Promote to S2 when TRAIN S1 passes 40%.
- Combat (S3): accept 0 by default. At 13:00, only if val S1 >= 30%, mix in 20% S3 for at most 3 h.
- 14:00 no new mechanisms. 20:00 freeze the model. 21:00 held-out 20 maps x 5 once, plus Freedoom MAP01-03 smoke, reported honestly. 22:00 rules audit and write-up. 23:00 hard stop.

## 5. Kill criteria
- K1: C3 not passing by 02:00 -> PPO from USE-weighted BC-v2, same settings.
- K2: first post-warm-up update KL > 0.03 or clip fraction > 0.3 -> relaunch at `--lr 5e-5`.
- K3: by 150 updates after warm-up, on-path door-pass below 60% of its warm-up value, USE share outside 0.03-0.25, zero-turn share < 0.05, or strafe entropy < 0.05 -> back to last good checkpoint, hold `--bc-coef` 1.0.
- K4: door reversals in > 20% of door episodes -> fix the teacher's press pattern before more PPO.
- K5: at 08:00 best PPO checkpoint no better than its BC start on val S1 -> abandon PPO for doors, build C6, submit the best BC/DAgger model if nothing beats it by 20:00.
- K6: at-door starts < 20% after 150 updates -> drop reverse curriculum.
- K7: val S1/S2 falls > 10 points with S3 mixed in -> remove S3.

## Open questions
- Is the laptop off 17:00-21:00 tomorrow?
- Organizers: are scripted teachers accepted; do judges set `+viz_nocheat 1` (would remove the sector info observation v2 depends on)?
