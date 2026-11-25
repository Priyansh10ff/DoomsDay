# 2026-11-25: Multi-instance layout benchmark (Stage A0, `my_way_home`)

Author: trainer-ops. **Runs use `nogit`**: there are no commits, and the working tree is uncommitted, including the `train/` per-episode logging edits made at 07:43, which all three runs used.

**User approval for multi-instance runs** (ADR 0008 Decision 5), relayed by the coordinator on 2026-11-25, quoted verbatim: "See don't ask me and run everything without my permission till 11.00 a.m. except commiting". Standing instruction, verbatim: "Don't use my lap's component too much that my lap starts hanging or shuts down".

## Setup

- Part 1 showed zero hard-kill survivors with both worker designs, so the default non-daemon design was used.
- Configs: `configs/ops/bench_{2x4,4x4,4x8}.json`, launched through `configs/ops_launch.py`.
- Settings: `--allow-multi-instance` equal to the exact instance count; 4 torch threads (held fixed across layouts); default 8 GiB cap; below-normal priority; rollout 64; 4 minibatches; `--max-seconds 90`. Runs were one at a time.
- CPU: `typeperf "\Processor(_Total)\% Processor Time"`, 5 s samples; idle reading about 10%.
- Machine: Windows 11, 24 threads, 15.4 GB, about 5.8 GiB available before each run.
- Single-instance reference from this session: 1x1 sampling median 177 decisions/s.

## Results

End-to-end = decisions / (sampling time + update time), over steady-state updates (updates 3 onward). Wall-clock agrees: for example, 2x4 did 45 updates x 512 = 23,040 decisions in 90 s.

| Layout | Run | Instances | Sampling decisions/s, median (min-max) | Update decisions/s | End-to-end decisions/s | Updates/min | Tree peak (planned estimate) | Min system available | CPU % `_Total`, median (max) | All below normal | Engines after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1 (Part 1) | `20261125-nogit-2d4e952a` | 1 | 177 (118-259) | 617 | about 133 | 125 | 409 MiB (467) | 5.44 GiB | not measured | yes (3/3) | 0 |
| 2x4 | `20261125-nogit-a44cb77a` | 8 | 401 (369-449) | 701 | 257 | 30.1 | 926 MiB (1036) | 5.07 GiB | 25.2 (29.5) | yes (11/11) | 0 |
| 4x4 | `20261125-nogit-34de7e63` | 16 | 620 (482-652) | 1110 | 391 | 22.9 | 1554 MiB (1742) | 4.81 GiB | 27.2 (29.8) | yes (21/21) | 0 |
| 4x8 | `20261125-nogit-08ef51b5` | 32 | 743 (713-765) | 1731 | 520 | 15.2 | 2653 MiB (2894) | 4.27 GiB | 29.8 (31.4) | yes (37/37) | 0 |
| 4x2 (added after reading ADR 0009) | `20261125-nogit-e08fb378` | 8 | 574 (525-621) | 822 | 338 | 39.6 | 1011 MiB | 5.07 GiB | 26.4 (32.5) | yes (13/13) | 0 |

Sampling per engine instance (ADR 0009's "no verdict" rule is below 50/s per instance for more than 10 minutes):

| Layout | 1x1 | 2x4 | 4x2 | 4x4 | 4x8 |
|---|---|---|---|---|---|
| Decisions/s per instance | 177 | 50 | 72 | 39 | 23 |

Each worker steps its envs one after another, so the per-instance rate falls as envs per worker rise. **4x4 and 4x8 would trip that rule by construction**, without any fault.

None of the stop conditions fired: no memory-cap trip, no processes left behind, no throughput collapse. Each run stopped on `max_seconds`, wrote a checkpoint, and had stale time-outs 0 and no NaN or inf. 4x8 was run because 4x4 left 4.81 GiB free, above the 3 GB threshold, and CPU stayed under 30%.

## Reading

- **Throughput scales with worker processes, not with envs per worker.** Sampling went 177 to 401 to 620 from 1 to 2 to 4 workers. Doubling envs per worker at 4 workers added only 20% (620 to 743). The update share of wall time is about 30% at 4x8 (2.76 s sampling and 1.18 s update per 2048-sample update).
- **CPU stayed at 25-31% of 24 threads** (about 6-7.5 threads busy) for every layout, under the "about half" budget.
- **Memory:** 4x8 peaked at 2.65 GiB of tree, with at least 4.27 GiB still available. That is 2.3 GiB above ADR 0009's 2 GB floor, but it assumes the user does not open large applications. The trainer's watcher stops the run at 1.5 GiB available.
- **Not measured:** thermal behaviour and desktop responsiveness over hours. 90 s cannot show either, and nobody was at the machine.

**Learning signal in 90 s, not a verdict.** Success fell from 1.0 to between 0.15 and 0.5 over the runs, as the first, short, successful episodes were followed by time-outs.

| Layout | Successes / episodes | Entropy, start to end |
|---|---|---|
| 2x4 | 10 / 45 | 6.22 to 5.42 |
| 4x4 | 20 / 72 | 6.22 to 5.72 |
| 4x8 | 19 / 81 | 6.22 to 5.51 |

At 4x8, KL reached 0.013 and clip fraction 0.22 by the last update. Explained variance rose to 0.57. These are worth watching in the long run.

## Recommendation for the first training run (about 2 to 2.5 h)

### Primary: 4 workers x 2 envs

**Layout:**
- 4x2 (8 instances), 4 torch threads, non-daemon workers, default 8 GiB cap, below-normal priority, checkpoint every 600 s.
- Config `configs/ops/a0_4x2_main.json`, run name `20261125-nogit-23b43e64`.
- It meets every ADR 0009 rule as written: 72 decisions/s per instance (at least 50), 5.07 GiB minimum available (at least 2 GB), CPU 26%.

**Command** (working directory `D:\DoomsDay`, run in the background):

```
mkdir -p reports/runs/20261125-nogit-23b43e64
python -u -m configs.ops_launch configs/ops/a0_4x2_main.json reports/runs/20261125-nogit-23b43e64 > reports/runs/20261125-nogit-23b43e64/stdout.log 2>&1
```

**Time limit:** `--max-seconds 8400` (2 h 20 min). Launched by 08:20, it ends by about 10:40, inside the approval window. A later launch needs a new config with a shorter limit, which also gives a new run name.

**Expected throughput** (338 decisions/s end to end):

| Per hour | In 2 h | In 8400 s |
|---|---|---|
| about 1.2M | about 2.4M | about 2.8M |

These clear ADR 0009's 1M-decision minimum.

### Alternative: 4 workers x 8 envs, if `advisor` rules that the per-instance rule should be per worker

- 4x8 (32 instances): config `configs/ops/a0_4x8_main.json`, run name `20261125-nogit-7a675759`, same command shape.
- Throughput: 520 decisions/s, about 1.87M/h, about 3.7M in 2 h.
- Memory: 2.65 GiB tree, 4.27 GiB minimum available. That clears the 2 GB floor by measurement, contrary to ADR 0009's estimate from 3.4 GB available.
- Its per-instance sampling (23/s) is below the 50/s line, and its per-worker sampling is 186/s.
- **Route to `advisor`:** restate ADR 0009's anomaly rule per worker process, for example below 50 decisions/s per worker.

### Caveats

- 90 s cannot show thermal behaviour. Responsiveness was judged by CPU only (at most 32.5%), not observed, because nobody was at the machine.
- Interactive Ctrl-C from a console is unverified (Part 1c: the hook block), so the run is bounded by `--max-seconds`.
- No validation split exists, so no "best val" checkpoint can be kept; only the latest and previous checkpoints exist.
- The trainer writes no TensorBoard logs; metrics are in `metrics.jsonl` and `episodes.jsonl`.
