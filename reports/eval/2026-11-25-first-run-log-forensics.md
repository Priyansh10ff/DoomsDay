# E0 log forensics: first DOOM run `20261125-nogit-7a675759` under ADR 0011's health monitor

Date: 2026-11-25. Author: `evaluator`. Task: ADR 0011, Validation 1 (E0).

**What this is.** Offline analysis of the training logs of one run on the **training scenario** (`my_way_home`). No engine was started, nothing was trained, and no episode was played. **This is not a held-out evaluation.** The EVAL-config checks (unmodified engine, vanilla rules, no ACS mods, held-out split) do not apply because no episode ran, so none are claimed. Nothing here says anything about unseen maps.

## Inputs, script and outputs

- Logs: `reports/runs/20261125-nogit-7a675759/metrics.jsonl` (1,814 lines, `update_idx` 1..1814, no gaps) and `episodes.jsonl` (10,060 lines).
- Random baseline: p0 = 28/200 = 0.140, from `reports/runs/2026-11-25-random-baseline.json` (`level_exit_flag_rate` 0.14, 200 episodes).
- Script (stdlib only, paths as arguments): `reports/eval/scripts/adr0011_monitor_forensics.py`. To reproduce:
  `python -I reports/eval/scripts/adr0011_monitor_forensics.py reports/runs/20261125-nogit-7a675759 reports/eval/2026-11-25-first-run-log-forensics-data`
- Outputs in `reports/eval/2026-11-25-first-run-log-forensics-data/`:
  - `adr0011_levels_primary.csv`: one row per update. Columns are episodes seen, trailing-500 success count, run best, summed entropy and its 100-update mean, each head's 50-update mean, the `watch`/`warning`/`drop_cond`/`low_cond`/`event` booleans, the warning heads, and `level`. **`worker` can diff a test against this file.**
  - `adr0011_levels_variants.csv`: the level per update under the four readings in section 1.
  - `e0_entropy_per_update.csv`: each head's entropy, its fraction of the maximum, and the head with the lowest fraction, per update.
  - `e0_per_update_400_1200.csv`: `v_loss`, `explained_variance`, `grad_norm`, `return_scale`, `approx_kl`, `clipfrac`, entropies, success count and level for updates 400..1200.
  - `e0_summary.json`: every summary figure quoted below.

## Integrity checks (all passed)

- **Alignment.** At metrics update u, `episodes_done` equals the number of episodes stamped `update_idx` <= u-1, on all 1,814 lines (0 mismatches). This confirms lesson 11's "one apart" rule. The per-100-update episode counts reproduce lesson 11's figures: 426, 532, 522, 516, 516, 501, 479, 462, 494, 669, 615 and 718.
- `episodes.jsonl` is sorted by `decision` and by `update_idx`, and `success == level_exit` on every line.
- The logged `entropy` equals the sum of `entropy_head0..5` on every update (largest difference 2.9e-7).
- The exact maxima are ln 3 = 1.0986, ln 7 = 1.9459 and ln 2 = 0.6931. Their sum is 6.2226, and half the sum is 3.1113.
- The logged `grad_norm` is the **pre-clip** total norm, because it is the return value of `clip_grad_norm_` (`train/ppo.py:102-104`), averaged over every epoch x minibatch step (`train/ppo.py:92-108`; 12 steps per update per lesson 11).
- Workers own contiguous blocks of 8 envs (`train/run_doom.py:131`, `train/subproc_vec_env.py:107,139-144`), so worker = `env_index // 8`.

## 1. Ambiguities in ADR 0011 Decision 2, and the readings used

| # | Ambiguity | Primary reading | Alternatives computed |
|---|---|---|---|
| A1 | What "success" is before 500 episodes exist | Not defined until 500 episodes are seen (from update 118) | Partial window: the mean of all episodes so far |
| A2 | The 100-update and 50-update means before 100 or 50 updates exist | Not defined, so the condition is false | Partial mean |
| A3 | Head maxima: exact ln(k), or the printed 1.10/1.10/1.95/0.69/0.69/0.69 and 3.11 | Exact ln(k) | Rounded values |
| A4 | "Below", "at least" | "Below" is strict `<`. "At least 0.15 below run best" means best - success >= 0.15. Compared as integer counts out of 500: count <= best - 75, and count < 120 for p0 + 0.10 = 0.24 | none |
| A5 | "Run best" | The maximum full-window trailing-500 success at updates <= u, current update included | Under A1-partial, the maximum includes partial windows |
| A6 | When the monitor evaluates | Once per metrics update, on the episodes known at that update | none |
| A7 | Whether a one-update gap ends an event, i.e. what "still active 600 updates after it began" means | Maximal contiguous runs of the event condition. Stop fires at start + 600 if the same run is still active then | First onset to last active update, treated as one event |
| A8 | Level precedence for one `level` column | stop > event > warning > watch > none. The three conditions are also written as independent booleans | none |
| A9 | "Which head fell first, per update" | No criterion is given. The first crossing of 50%, 25% and 10% of each head's maximum is reported, both raw and as a 50-update mean, plus the lowest-fraction head per update | none |
| A10 | ADR 0009's ten-fold trigger: "jump" and "settle within 20 updates" | Jump: the value at u is more than 10x the value at u-1, or more than 10x update 1 (before any goal) | Settle, reading S1: back under 10x update 1 at least once within 20 updates. Reading S2: under 10x update 1 and staying there from some update within 20 updates on |

**Sensitivity.** The rounded maxima give exactly the same `level` on all 1,814 updates. Only two per-head warning onsets move: strafe starts at 732 instead of 733, and use at 947 instead of 946. **The partial-window reading (A1/A2) changes 893 of 1,814 levels and makes Stop fire** (section 2). This happens because the first three episodes all succeeded, so the partial "run best" is 1.0 and every entropy warning becomes a collapse event. The primary full-window reading is the one used everywhere below. A1 and A2 should be pinned down in F4.

## 2. Monitor levels, all 1,814 updates and all six heads (primary reading)

| Level | Updates (start-end, inclusive) | Length | Head(s) |
|---|---|---|---|
| Watch | **622-1140** | 519 | (summed entropy; 100-update mean first below 3.1113 at 622; trailing success first >= 250 at 1141) |
| Entropy warning, any head | **630-1372**, **1385**, **1436-1651** | 743, 1, 216 | per head below |
| - move (line 0.1099) | 965-1049, 1618-1651 | | |
| - strafe (0.1099) | 733-800 | | |
| - turn (0.1946) | 743-763, 879-883, 892-1351, 1385, 1436-1533 | | |
| - attack (0.0693) | **630-734** (first head to warn), 838-1252 | | |
| - use (0.0693) | 946-1372, 1474-1647 | | |
| - speed (0.0693) | 655-988 | | |
| **Collapse event** | **814-1105** (one event, contiguous) | 292 | **speed** was the only warning head at onset. Heads warning during the event: speed, attack, turn, use, move. Strafe was not among them. |
| Stop | **never** (primary reading) | | |

Each update gets one `level` (precedence A8). By that column the run reads: none 1-621, watch 622-629, warning 630-813, event 814-1105, warning 1106-1372, none 1373-1384, warning 1385, none 1386-1435, warning 1436-1651, none 1652-1814. The column only shows the highest level, so it hides that the watch condition is also true at every update from 630 to 1140. The booleans in the CSV carry it.

**Collapse event details.**

- **Onset at 814.** Trailing success was 134/500 against a run best of 0.420 (210/500), so the drop condition held (134 <= 135). The low condition (count < 120) did not hold yet. Inside the event it held at 838, 844-845 and 847-1094. It had also held at 118-181, when no head was warning, so no event fired then.
- **End at 1105.** At 1106 success was 137/500 (0.274), above the drop line of 135, while turn, attack and use were all still under their lines. **The event ended because success crept back above best - 0.15 = 0.27, not because it recovered.** Success was still 0.274, and it first reached 0.5 only at update 1141.
- **Near misses.** Before onset the closest approach to the drop line was 3 episodes, at updates 806 and 813. After the event the closest was 2 episodes, at update 1106. The run best then rose to 0.662 at update 1198, which puts the drop line at 256/500 for the rest of the warning period. No update from 1106 to 1651 met it.
- **Stop, both readings of A7.** There was one event of 292 updates, so it is never active 600 updates after onset. Under the merged reading the span is also 814-1105. **No event was active at update 1814.**
- **Partial-window variant, for the record.** Events at 630-1372, 1385 and 1436-1651, with **Stop at update 1230**. Rounded-partial gives the same.

**The ADR's amend clause (Validation, "reverse or amend if").**

- The backtest table implies an event from about 901 to about 1100. The measured event runs from **814** to **1105**. The start is about 87 updates earlier and the end about 5 updates later, both within the 100-update tolerance.
- The Stop level would not have fired under the primary reading.
- So **no threshold reset is triggered** under the primary reading. Under the partial-window reading the Stop level would have fired at 1230, which is the amend condition. F4 must therefore fix A1/A2 as full windows, or fix whatever reading the team picks.

## 3. Trailing-500 success first reaching 0.5

| Criterion | Episode index (0-based / 1-based) | Successes in window | Episode `update_idx` stamp | Aligned metrics update | Episode `decision` stamp | Trained decisions at that update (u x 2048) | Half of the decision stamp |
|---|---|---|---|---|---|---|---|
| >= 0.5 | 5,708 / 5,709 | 250 | 1140 | **1141** | **2,334,976** | 2,336,768 | 1,167,488 |
| > 0.5 | 5,710 / 5,711 | 251 | 1140 | 1141 | 2,336,000 | 2,336,768 | 1,168,000 |

At update granularity, where the monitor reads it, the first update whose trailing-500 count is >= 250 is 1141, and the first > 250 is also 1141.

**The 1,213,904-decision line.**

- The arithmetic is correct: 2 x 1,213,904 = 2,427,808.
- The base figure is not the first trailing-500 crossing. 2,427,808 is the `decision` stamp at the end of the **non-overlapping block** of episodes 5500-6000 (block success 0.654), which is the first such block >= 0.5. The block before it, 5000-5500, is 0.294.
- On a trailing-500 window, the definition ADR 0011 Decision 2 uses for "success", the first crossing is at 2,334,976 decisions, and half of that is **1,167,488**.
- ADR 0011's E1 line is worded "trailing-500 success reaches 0.5 by 1,213,904 decisions", so the stated definition and the number disagree by 46,416 decisions (1,213,904 - 1,167,488). On the trailing reading, the line as written is looser by that amount.
- **For `advisor`:** decide whether the E1 line should be 1,167,488 (trailing window, matching the wording) or keep 1,213,904 and reword it to "first non-overlapping 500-episode block". I have not changed the rule.

## 4. ADR 0011 backtest table, cell by cell

Block means over 100 updates with lesson 11's alignment. Success here is the share of episodes collected in the block, which is the same quantity as the ADR's table, not the trailing-500 value. The level column shows how the per-update monitor classifies that block's updates.

| Updates | ADR: turn / sum | Measured turn / sum | ADR success | Measured success (n) | ADR level | Measured levels in block | Verdict |
|---|---|---|---|---|---|---|---|
| 601-700 | 0.69 / 1.65 | 0.6865 / 1.6459 | 0.372 | 0.3721 (516) | Watch | none 21, watch 8, **warning 71** | Numbers confirmed. **Level corrected:** the attack head warns from 630 and speed from 655, so most of this block is at warning, not watch. |
| 801-900 | 0.21 | 0.2141 | 0.219 | 0.2192 (479) | Borderline | warning 13, **event 87** | Numbers confirmed. **Level corrected:** the event starts at 814, on the speed head, not on the turn head. |
| 901-1000 | 0.02 | 0.0197 | 0.180 | 0.1797 (462) | Collapse event | event 100 | Confirmed. |
| 1001-1100 | 0.008 | 0.0081 | 0.249 | 0.2490 (494) | Collapse event | event 100 | Confirmed. |
| 1101-1200 | 0.015 | 0.0148 | 0.647 | 0.6472 (669) | Entropy warning only; event over | event 5 (1101-1105), warning 95 | Confirmed, apart from the first 5 updates. |
| 1201-1300 | 0.053 | 0.0525 (0.05254) | 0.585 | 0.5854 (615) | Entropy warning only | warning 100 | Confirmed. |

**Corrections to the lines in the table and its text.**

- The run best before the event is **0.420** (210/500, first reached at update 402), not 0.41. The drop line is therefore 0.27 (count <= 135), not 0.26. The 0.41 is the block value for updates 301-400.
- After recovery the trailing run best is **0.662** (from update 1198), not 0.654, so the 1201-1300 line is **0.512**, not 0.504. The 0.654 is the 5500-6000 block value.
- The p0 + 0.10 line of 0.24 is confirmed (count < 120). The turn head's 10% line is 0.1946, which the ADR prints as 0.195. The watch line is 3.1113, printed as 3.11. Neither rounding changes any level (section 1).
- "The event lasts roughly 200 to 300 updates": measured **292**. Confirmed.
- "The watch fires about 300 updates before the trough": watch onset is at 622. That is **331** updates before the summed-entropy minimum (update 953) and **394** updates before the trailing-500 success minimum (87/500 = 0.174, updates 1016-1018).

**Corrections to ADR 0011's Context.**

- *"The move head fell first (0.58 at updates 501-600 while the turn head was still 1.42)."* The two figures are confirmed: 0.5802 and 1.4180. Move did fall before turn: its raw entropy first went below 50% of the maximum at update 512, turn at 606. But move was not first overall. As a fraction of each head's maximum, **speed and attack fell first**:
  - raw entropy below 50% of the maximum: speed at 480, attack 490, move 512, strafe 536, turn 606, use 640;
  - 50-update mean below 10% of the maximum: attack 630, speed 655, strafe 733, turn 743, use 946, **move 965 (last)**.
  - In absolute terms over updates 501-600, move is 0.5802 (52.8% of its maximum), while attack is 0.1215 (17.5%) and speed is 0.1700 (24.5%).
- *"Summed entropy fell from 5.05 (updates 401-500) to a minimum of 0.122 at update 953, turn head 0.0019."* Confirmed: 5.0525, then 0.12197 at update 953, turn 0.001855. The turn head's own minimum is lower and later: 0.00158 at update 961.

## 5. Other E0 outputs (ADR 0011 Decision 2)

### 5.1 Which head fell first, per update

The full first-crossing table is in `e0_summary.json` → `head_first_crossings`. The lowest-fraction head for each update is in `e0_entropy_per_update.csv`. Over updates 400-1200 (801 updates), the head lowest relative to its maximum was:

| Head | Updates |
|---|---|
| speed | 284 |
| attack | 217 |
| turn | 137 |
| strafe | 93 |
| use | 70 |
| move | 0 |

| Head | Raw < 50% | Mean50 < 50% | Raw < 25% | Mean50 < 25% | Raw < 10% | Mean50 < 10% | Raw minimum (update) |
|---|---|---|---|---|---|---|---|
| move | 512 | 549 | 619 | 684 | 934 | 965 | 0.00785 (961) |
| strafe | 536 | 562 | 597 | 634 | 666 | 733 | 0.00995 (737) |
| turn | 606 | 640 | 678 | 705 | 714 | 743 | 0.00158 (961) |
| attack | 490 | 512 | 504 | 537 | 536 | 630 | 0.00121 (1018) |
| use | 640 | 671 | 646 | 716 | 715 | 946 | 0.00165 (1054) |
| speed | 480 | 515 | 513 | 550 | 603 | 655 | 0.00210 (914) |

### 5.2 Value loss, explained variance, gradient norm, `return_scale`, updates 400-1200

The per-update values are in `e0_per_update_400_1200.csv`. The block below starting at 1200 contains update 1200 only.

| Updates | v_loss mean / median / max | EV mean / min | grad_norm mean / max | return_scale min-max |
|---|---|---|---|---|
| 400-499 | 0.0642 / 0.0582 / 0.1765 | 0.8274 / 0.5115 | 0.7911 / 1.0064 | 0.1901-0.1930 |
| 500-599 | 0.0619 / 0.0507 / 0.1885 | 0.8105 / 0.3376 | 0.7755 / 1.0335 | 0.1881-0.1901 |
| 600-699 | 0.0780 / 0.0625 / 0.2635 | 0.7784 / 0.0147 | 0.6442 / 1.0097 | 0.1846-0.1883 |
| 700-799 | 0.0508 / 0.0435 / 0.2622 | 0.8163 / 0.4720 | 0.6329 / 1.0307 | 0.1819-0.1849 |
| 800-899 | 0.0459 / 0.0352 / 0.1775 | 0.8510 / 0.1881 | 0.6159 / 0.9726 | 0.1807-0.1861 |
| 900-999 | 0.0344 / 0.0102 / 0.1841 | 0.8334 / -0.8178 | 0.5117 / 0.9247 | 0.1862-0.1900 |
| 1000-1099 | 0.0488 / 0.0419 / 0.2009 | 0.8716 / 0.4209 | 0.6661 / 1.1674 | 0.1886-0.1900 |
| 1100-1199 | 0.0995 / 0.0984 / 0.2817 | 0.7704 / 0.5373 | 0.8432 / 1.3842 | 0.1900-0.2059 |
| 1200 | 0.0667 | 0.6453 | 0.7236 | 0.2061 |

- `return_scale`: the largest ratio between consecutive updates after update 50 is 1.0123, so the "more than doubling" alarm is clear. This confirms lesson 11.
- Gradient norm: the largest consecutive ratio in the whole run is 2.574 (update 953). The largest ratio to update 1 over updates 2-100 is 3.795 (update 5). **The ten-fold condition is not met for `grad_norm`.**

### 5.3 ADR 0009's ten-fold trigger for value loss: depends on the reading of "settle"

This is the item lesson 11 left unchecked.

- The first goal is episode 0, aligned with update 2. `v_loss` goes from 0.01645 at update 1 to 0.4317 at update 2, a factor of **26.2**. The "jump" part is met.
- For updates 2-7 the ratio to update 1 is 26.25, 13.50, 9.995, 26.64, 26.09 and 0.100.
- **Reading S1** ("back under 10x update 1 at least once within 20 updates"): it is back under at update 4, so the trigger is **not met**.
- **Reading S2** ("stays under"): within the first 100 updates it is above 10x update 1 again at 5, 6, 10, 15, 17, 19, 20, 22, 24-29, 34, 53, 60 and 91. Over the whole run it is above 10x at 107 updates, the last at 1642. Under S2 the trigger is **met**.
- Over the whole run there are **162** consecutive-update jumps of more than 10x. The largest is 511x, at update 120. Within updates 400-1200 the largest is 301x, at update 923.
- So `v_loss` is bimodal from one update to the next, which is consistent with whether a rollout contains a goal. That is a reading, not a test.
- **For `advisor`:** state which reading of "settle" is intended, or redefine the trigger on a smoothed signal such as a rolling median. I have drawn no verdict from it.

### 5.4 Success per worker and per env, before, during and after the event

Phases are by aligned update:

- before: 1-813
- during: 814-1105
- after: 1106-1814
- matched 300-update windows: 514-813 and 1106-1405

| Worker | Before (1-813) | Near before (514-813) | During (814-1105) | Near after (1106-1405) | After (1106-1814) |
|---|---|---|---|---|---|
| 0 | 0.330 (331/1004) | 0.355 (136/383) | 0.214 (75/351) | 0.628 (304/484) | 0.634 (739/1165) |
| 1 | 0.335 (342/1021) | 0.324 (122/376) | 0.191 (65/341) | 0.604 (294/487) | 0.611 (709/1160) |
| 2 | 0.316 (321/1016) | 0.329 (126/383) | 0.257 (94/366) | 0.604 (284/470) | 0.630 (726/1153) |
| 3 | 0.328 (332/1012) | 0.338 (129/382) | 0.235 (84/358) | 0.559 (259/463) | 0.596 (663/1113) |
| All | 0.327 (1326/4053) | 0.337 (513/1524) | 0.225 (318/1416) | 0.599 (1141/1904) | 0.618 (2837/4591) |

**Per env during the event:**

- 32 envs, 37 to 54 episodes each; the full table is in `e0_summary.json` → `per_env`.
- Success ranges from **0.054** (env 22, 2/37) to **0.389** (env 21, 21/54), median 0.209.
- Before the event the range was 0.231-0.400. After it, 0.519-0.738.
- No worker or env stands apart: the fall shows on every worker. Whether the per-env spread during the event (2/37 to 21/54) is larger than sampling noise was **not tested**.

### 5.5 Length of successful episodes (decisions)

| Phase | n | Mean | Median | p10 | p25 | p75 | p90 | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| Before (1-813) | 1326 | 168.6 | 114.5 | 22 | 44 | 275 | 411 | 7 | 524 |
| Near before (514-813) | 513 | 158.0 | 98 | 26 | 44 | 248 | 395 | 7 | 524 |
| **During (814-1105)** | 318 | **75.8** | **55.5** | 13 | 28 | 85 | 171 | 7 | 431 |
| Near after (1106-1405) | 1141 | 187.6 | 148 | 29 | 65 | 295 | 419 | 7 | 525 |
| After (1106-1814) | 2837 | 187.4 | 146 | 32 | 64 | 296 | 421 | 6 | 525 |

Percentiles are nearest-rank on (n-1).

During the event, the successes that did happen were short: the median falls from 98 to 55.5, and p75 from 248 to 85. This is consistent with the collapsed policy reaching the goal only from spawns close to it. No position data was logged, so that is **not measured**.

## 6. Failure modes visible in the logs

- **Near-deterministic policy during 814-1105.** Five of six heads were under 10% of their maximum entropy at some point. Successes during the event were short, and the success rate fell on all four workers. No video or action histogram exists, so *which* action was locked in is **not measured**; F1 is needed.
- **Entropy narrowing started on attack and speed, not on move or turn.** It began on buttons that are irrelevant on `my_way_home` (no enemies; whether running matters there is unmeasured). These buttons have no reward signal on this map, which may be why they narrowed first. That is a reading, not a test.

## 7. For other agents

- `worker` (F4): reproduce `adr0011_levels_primary.csv` exactly, using readings A1-A8. Watch 622-1140; warning 630-1372, 1385 and 1436-1651; event 814-1105; no Stop. Make A1/A2 (full windows only) explicit in code: the partial reading fires Stop at 1230.
- `advisor`:
  - the 1,213,904 vs 1,167,488 base for E1 (section 3);
  - the ADR 0009 value-loss ten-fold trigger: met or not depending on the reading of "settle" (section 5.3);
  - the backtest's level column and the "move head fell first" sentence need correcting (section 4);
  - the event ends on the drop line while success is still 0.274 (section 2).
