# 2026-11-25: Large-layout benchmark, Part 3 (Stage A0, `my_way_home`)

Author: trainer-ops. **All runs are `nogit`**: the repository has no commits, and the whole working tree is uncommitted. The code has changed since Part 2, so Part 2 figures are not like-for-like.

**User approval** (relayed by the coordinator, given 2026-11-25, quoted verbatim): "you can [use] my computer's resources at max possible such that i can run small tasks easily side by side".

**Limits are the coordinator's numbers, not the user's:**
- system available memory must stay at or above 2.0 GiB;
- total CPU must stay at or below about 80%;
- below-normal priority and the 8 GiB tree cap and watcher stay on.

## Method

- **Runs:** one at a time, 90 s each (`--max-seconds 90`), `--device cpu`, rollout 64, default flags, `--allow-multi-instance` equal to the exact instance count. Each was launched with `python -u -m configs.ops_launch configs/ops/bench3_<layout>_t<threads>.json reports/runs/<run>`.
- **Engine checks:** 0 `vizdoom.exe` was confirmed before each run and 5 s after it.
- **Sampling:** `typeperf "\Processor(_Total)\% Processor Time" "\Memory\Available MBytes"` every 3 s, written to `perf.csv` in each run directory. The CPU window runs from 20 s after launch to process exit. The idle reading between runs was 2-17%, typically about 8-11%.
- **End-to-end:** decisions / (sampling time + update time) over updates 3 onward. This is ADR 0011's `B_e2e`. Sampling is the median of `sample_dps` over updates 3 onward (`B_sample`).
- **Validation:** the analysis script reproduced the Part 2 4x8 figures from `20261125-nogit-08ef51b5` (519.6 end to end, sampling 743.4, tree 2653 MiB).
- **Minimum available:** the lower of the trainer summary's `min_available_bytes` and typeperf's minimum.
- **Breach rule, fixed before any data:**
  - memory breaches if minimum available falls below 2.0 GiB;
  - CPU breaches if the in-run median is above 80% (the maximum is also reported).
- **Ladder:** 4x8 at 4 and 8 threads, then 6x8, 8x6, 8x8 and 12x4 at 4 threads, in the listed order, stopping at the first breach. After that, 8 vs 4 threads for the best two that passed.

## Results

| Layout, threads | Run | Inst. | Updates (steady) | Sampling dec/s median (min-max) | Update dec/s | **End-to-end dec/s** | Per instance | Tree peak | Pre-run avail | Min avail | CPU % median (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4x8, 4 | `20261125-nogit-ec1fc798` | 32 | 21 (19) | 642.5 (622.3-733.1) | 1485.8 | **458.7** | 20.1 | 2649 MiB | 5656 MB | 4010 MiB (3.916 GiB) | 30.2 (35.8) |
| 4x8, 8 | `20261125-nogit-64a0589a` | 32 | 21 (19) | 625.5 (606.7-708.7) | 1842.4 | **472.2** | 19.5 | 2652 MiB | 5522 MB | 3.931 GiB (summary; typeperf 4027 MiB) | 45.2 (48.8) |
| 6x8, 4 | `20261125-nogit-16684d7f` | 48 | 19 (17) | 1012.1 (925.5-1075.8) | 1791.7 | **646.4** | 21.1 | 3813 MiB | 5460 MB | 3518 MiB (3.435 GiB) | 33.1 (37.7) |
| 8x6, 4 | `20261125-nogit-985d56e7` | 48 | 21 (19) | 1178.7 (1124.4-1207.6) | 1723.9 | **700.2** | 24.6 | 3908 MiB | 5645 MB | 3.404 GiB (summary; typeperf 3487 MiB) | 33.4 (41.2) |
| 8x8, 4 | `20261125-nogit-72732b54` | 64 | 18 (16) | 1319.3 (1170.5-1523.0) | 1990.4 | **780.3** | 20.6 | 4962 MiB | 5640 MB | 2662 MiB (2.599 GiB) | 38.7 (44.5) |
| 12x4, 4 | `20261125-nogit-9506be71` | 48 | 4 (stopped) | see below | see below | **no figure** | n/a | 4008 MiB | 5379 MB | **1143869440 bytes, 1.065 GiB (BREACH)** | 42.5 (43.6), 5 samples |
| 8x6, 8 | `20261125-nogit-8dd7a3e9` | 48 | 24 (22) | 1241.6 (1131.0-1310.2) | 2377.2 | **818.1** | 25.9 | 3904 MiB | 5902 MB | 3.639 GiB (summary; typeperf 3728 MiB) | 52.4 (58.1) |
| 8x8, 8 | `20261125-nogit-e271ac38` | 64 | 20 (18) | 1363.3 (1292.7-1429.9) | 2602.9 | **895.0** | 21.3 | 5000 MiB | 5812 MB | 3200 MiB (3.125 GiB) | 53.0 (61.4) |

Every run had the following:
- `memory_cap_gib=8.0` and `all_below_normal` true; 37 to 73 processes, all below normal.
- Launcher exit 0, and 0 engines alive after shutdown and after 5 s.
- Stale time-outs 0 and no NaN or inf in any metric.
- Every run except 12x4 stopped on `max_seconds`.

### 12x4: limit breach, ladder stopped

- **What happened:** available memory was steady at about 3.3 GB for updates 1-3, then fell from 3278 to 1157 MB in about 12 s during update 4.
- **Tree working set stayed flat:** 3943, 3993, 3983 and 3995 MiB across updates 1-4. The extra 2.1 GB was not in the tree's working set.
- **Cause not attributed.** It was not shown to be external, and not shown to come from the run.
- **Stop:** the trainer's watcher stopped the run at update 4 ("system available memory low (1143869440 < 1610612736 bytes)"). It wrote a checkpoint, and the system recovered to about 5.9 GB within 6 s.
- **Raw per-update sampling:** 1342, 1255, 1361, 332 decisions/s. Update rates: 1829, 1816, 1757, 1495.
- **Not ranked.** It was not re-run, because it reached 1.065 GiB. Repeating it is the coordinator's call.

### Memory trend (relevant to "room to work")

- **8x8 at 4 threads:** per-update available drifted down, from 3027 to 2669 MB by update 16, with a flat tree (4853-4960 MiB). The two 8x8 runs reached minima of 2.599 and 3.125 GiB from similar starting points (5640 and 5812 MB). Its margin above the 2.0 GiB floor was therefore 0.599 to 1.125 GiB in 90 s. The 4-thread trace was not monotonic (2958 MB at update 14, 2748 at the end); the 8-thread run was flat at 3177-3306 MB.
- **8x6:** flat in both runs, at 3485-3605 MB (4 threads) and 3733-3796 MB (8 threads).
- **Long run `7a675759` (4x8, 2.3 h):** tree was flat at 2612 to 2684 MiB, but available swung from 3602 to 4541 MB. That is about 0.9 GB of background swing on this machine, independent of the run. A swing that size would breach the floor from the lower 8x8 run's minimum (2.599 GiB), but not from the 8-thread run's (3.125 GiB).

### Torch threads

- **4x8:** 8 threads gave 472.2 vs 458.7 end to end (+3%), but CPU rose from 30.2 to 45.2%. For scale, the same 4x8 config at 4 threads moved from 519.6 (Part 2) to 458.7 (now), about -12% across the code change and the time between runs. The 3% gap is not a result.
- **8x6:** 8 threads gave 818.1 vs 700.2 (+17%).
  - The update phase went from 1723.9 to 2377.2 (+38%), which is what more threads should do.
  - Sampling also went from 1178.7 to 1241.6 (+5%), which threads should not change. Treat about 5% of the gap as noise.
- **8x8:** 8 threads gave 895.0 vs 780.3 (+15%), with update 1990.4 to 2602.9.
- **Cost:** CPU rises by 14.3 to 19.0 points (median).

### Learning signals in 90 s (not a verdict)

The seven completed runs (12x4 excluded: 4 updates, entropy 6.13 at the end, KL max 0.0082, 6 episodes, 6 successes):

- Entropy fell from 6.22 to 5.37-5.65; no collapse.
- KL peaked at 0.0101-0.0127, and clip fraction at 0.17-0.22.
- Value loss stayed under 0.59.
- Explained variance ended between -0.003 and 0.48, and is uninformative at this length.
- Successes per episodes, in table order without 12x4: 16/74, 17/75, 23/111, 28/120, 19/140, 35/127 and 25/149. Median episode length was 525 decisions, the time-out.
- No curriculum stage; the scenario is fixed.

## Recommendation

**8 workers x 6 envs (48 instances), 8 torch threads**, as benchmarked in `20261125-nogit-8dd7a3e9` (`configs/ops/bench3_8x6_t8.json`).

- **Throughput:** 818.1 decisions/s end to end, against 458.7 for today's 4x8 reference.
- **Memory headroom:** minimum available 3.639 GiB, flat, which is 1.639 GiB above the 2.0 GiB floor. That covers the 0.9 GB background swing seen in the 2.3 h run, but not an event the size of 12x4's unexplained 2.1 GB drop.
- **CPU headroom:** median 52.4% and maximum 58.1%, about 22 points under the 80% limit. That leaves about 10 of 24 threads idle.
- **Alternative, 8x8 at 8 threads (`20261125-nogit-e271ac38`):** it breached neither limit in its one run and is faster (895.0, +9%). Its margin is thinner: 1.125 GiB in that run, and 0.599 GiB in the 4-thread run of the same layout. The 0.9 GB swing would breach only from the lower figure. It is the coordinator's call whether the thinner margin is acceptable; trainer-ops recommends 8x6 for the user's room to work.
- **ADR 0011 "no verdict" line** for a long run on this exact ops config (8x6, 8 torch threads, no shaping preset, same nogit code). A different layout, thread count or shaping preset needs its own 90 s benchmark.
  - sampling: 0.5 x `B_sample` = 0.5 x 1241.6013 = **620.8006 decisions/s**, the median `sample_dps` in any 10-minute span;
  - end to end: 0.5 x `B_e2e` = 0.5 x 818.0597 = **409.0298 decisions/s** (untruncated: 620.80065 and 409.02983), decisions trained / wall seconds over the whole run.
- **A long run needs its own config.** It must be a new JSON with its own `--max-seconds` and checkpoint interval, which gives a new hash and run name. The ADR 0011 line carries over only if its layout, threads and shaping match.

## Caveats

- **90 s only:** thermals, hours-long drift and desktop responsiveness were not measured. Nobody was at the machine.
- **CPU median is higher at 8 threads:** the cause is likely busy-wait spin in torch's intra-op threads (unconfirmed).
- **The 12x4 memory event is unexplained.** If it came from the worker count, 8 workers is not proven safe for hours. The flat 8x6 traces argue against that, but only over 90 s.
- **No val split exists**, so there is no best-val checkpoint. Metrics are in `metrics.jsonl` and `episodes.jsonl`; there are no TensorBoard logs.

## For routing (coordinator)

- **To `worker`: same-config slowdown since Part 2.** The 4x8, 4-thread config with the same argv ran slower today than in Part 2:
  - sampling median 743.4 to 642.5 decisions/s;
  - update 1731.0 to 1485.8 decisions/s;
  - end to end 519.6 to 458.7 decisions/s.

  The code changed between the two runs. The cause is not established.
- **To `worker` and `debugger`: the 12x4 memory event.** About 2.1 GB of available memory disappeared within about 12 s, with a flat tree working set, and sampling fell to 332/s. The cause is not attributed. Repeating 12x4 is the coordinator's call.
