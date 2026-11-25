# Lesson 08: The training run on DOOM (worker processes, the entry point, laptop protection, checkpoint and resume)

> **Changed since first written (2026-11-25, after ADR 0008).** This lesson was rewritten for the new code:
> - **The bare entry point now starts exactly 1 worker x 1 environment with 2 torch threads.** More engine instances need `--allow-multi-instance N`, with N the exact count (new `train/layout_guard.py`). The first version's "Look out" (32 engines by default) is fixed.
> - **Memory is checked every 5 seconds by a watcher**, including the system's free memory, not once per update.
> - **Checkpoints are written only between updates.** Ctrl-C finishes the current update (or abandons the rollout) and then saves; a second Ctrl-C stops without saving. The previous checkpoint is kept as `.prev`.
> - **Resume re-seeds the environments per session**, so a resumed run does not replay the same episodes. A probe confirmed the same seed replays the same episode exactly.
> - **Checkpoints load with `weights_only=True`.**
> - **Workers are non-daemon by default and exit when their parent dies.**
> - New measurements are in section 4. The 17 decisions/s in the worker's smoke report was measured while another process was using the CPU.
> - Gate A passed (lesson 09), which lifts the block on single-instance DOOM runs.
> - *Later correction:* the two run folders this lesson first called "preliminary" are now written up in `reports/runs/2026-11-25-part1-single-instance-checks.md` (lesson 10).

Files taught: `train/run_doom.py`, `train/layout_guard.py`, `train/run_loop.py`, `train/subproc_vec_env.py`, `train/doom_factory.py`, `train/process_control.py`, `train/checkpoint.py`, `train/trainer.py:41-88` and `:218-254` (stopping, sessions, what is saved), the `SafetyConfig` and `RunConfig` parts of `configs/train_config.py`, and the tests `tests/test_subproc_and_run_loop.py` and `tests/test_layout_guard.py`.

Design records:

- ADR 0004 (process layout, checkpoints);
- ADR 0006 (4 x 8 layout);
- `0007-a0-trainer-rulings-and-first-run-prerequisites.md` (first-run prerequisites);
- ADR 0008 (`0008-reconcile-0007-files-memory-gate-and-safety-defaults.md`, Decisions 1, 4 and 5). Two ADRs share the number 0007 (lesson 09); ADR 0008 reconciles them.

**All ADRs are "Proposed", not ratified by the user.**

**State of things, plainly:**

- **Nothing has been trained on DOOM.** Every DOOM run so far is a single-instance (1 x 1) smoke run of one or two updates, plus one 40-second run by the coordinator (section 4).
- **No multi-instance run has happened** at the time of writing. A benchmark is starting now; any multi-instance run needs the user's go-ahead. (Since then: the Part 1 checks, the layout benchmark and the first 4x8 training run; see lesson 10.)
- Gate A (lesson 09) passed, so single-instance runs are no longer blocked.

Builds on lesson 01 (`DoomEnv`, `TRAIN_CONFIG`, the sensor signature) and lesson 06 (the trainer and the vector-environment interface).

---

## 1. What problem this piece solves

PPO learns from many games at once, and each DOOM game is a separate `vizdoom.exe` program. On one Windows laptop, whose owner has said "don't use my laptop so hard that it hangs or shuts down", that raises five practical problems:

1. **Speed.** One Python program stepping many engines one after another spends most of its time waiting.
2. **Leftovers.** If training crashes or is stopped, engine programs may keep running and keep eating memory.
3. **Hogging the machine.** Training must never make the laptop sluggish.
4. **Accidents.** A single typo must not start 32 engines.
5. **Stopping safely.** The owner must be able to stop at any moment without losing work, and resume later.

## 2. The idea in plain words

Think of a **restaurant kitchen**:

- The **head chef** (the main process) holds the recipe book (the neural network) and decides what every stove does next.
- **Line cooks** (worker processes) each look after a few stoves (engines), pass on the orders and bring back what happened. All cooks work at the same time.
- **House rules**:
  - cooks work quietly so the dining room is not disturbed (low priority);
  - only one stove may be lit unless the manager writes down exactly how many (the layout guard);
  - nobody starts a shift while yesterday's stoves are still on (the leftover check);
  - someone checks the power meter every 5 seconds (the memory watcher);
  - the chef writes the state of the kitchen into a notebook every 10 minutes and when told to stop, but only between dishes (checkpoints on update boundaries);
  - if the chef collapses, the cooks notice and turn off their stoves (workers exit when their parent dies).

Where the analogy stops being accurate:

- The cooks do not think; every decision comes from the chef's network (rule 4.1).
- Resuming from the notebook does not continue the half-cooked dishes: every game starts a new episode, with fresh memory and a fresh seed.
- The house rules see memory and priority, but not temperature (ADR 0008, risks).

### Terms used in this lesson

- **Process**: a running program with its own memory. **Child process**: one started by another; children, grandchildren and so on form a **process tree**.
- **Worker process**: a child process that only runs environments and reports back.
- **Pipe**: a two-way message channel between two processes.
- **Spawn**: how Windows starts a Python child: a fresh interpreter that **re-imports** the modules it needs.
- **Pickling**: turning a Python object into bytes to send to another process.
- **Daemon process**: a child that Python ends automatically when the main process exits normally. **Non-daemon**: one that keeps running on its own unless it decides to exit.
- **Priority**: how much the operating system favours a process. **Below normal** means "run when the user's programs don't need the CPU".
- **Resident memory (RSS)**: the memory a process is really using. **Available memory**: what the whole system still has free.
- **Thread**: a second line of work inside one process; the memory watcher is one.
- **Signal handler**: a function the program runs when the operating system delivers an event such as Ctrl-C.
- **Exit code**: the number a program returns when it ends; 0 means success.
- **MiB / GiB**: 1 MiB = 1024 x 1024 bytes; 1 GiB = 1024 MiB.
- **Checkpoint** / **resume**: save the training state; load it and continue (lesson 06).
- **Session**: one stretch of training between a start (or resume) and a stop.

## 3. A walk through our actual code

### 3.1 The settings: `configs/train_config.py:69-111`

`RunConfig` (`:95-111`) now defaults to **1 worker x 1 environment and 2 torch threads** (`:100-102`). Its comment says ADR 0006's 4 x 8 is "the layout to ASK the user for".

`SafetyConfig` (`:69-92`):

| Field | Default | Meaning |
|---|---|---|
| `below_normal_priority` | True | Main process, workers and engines at below-normal priority. |
| `checkpoint_interval_s` | 600 | A periodic checkpoint every 10 minutes, on an update boundary. |
| `memory_cap_bytes` | 8 GiB | Process tree above this: stop and save. |
| `memory_check_interval_s` | 5.0 | How often the watcher checks. |
| `min_available_bytes` | 1.5 GiB | System available memory below this: stop and save. |
| `emergency_available_bytes` | 0.75 GiB | Below this: close the engines **at once**, without waiting for a safe point. |
| `cooldown_s` | 0.0 | Pause between updates. |
| `engine_process_names` | `vizdoom.exe`, `vizdoom` | What counts as an engine process. |
| `worker_join_timeout_s` | 10 | How long to wait for a worker before forcing it. |
| `worker_start_method` | `spawn` | How workers are started. |
| `worker_daemon` | **False** | Non-daemon workers (ADR 0008 point 6). |
| `parent_poll_interval_s` | 1.0 | How often a waiting worker checks that its parent is alive. |
| `est_engine_bytes`, `est_worker_bytes`, `est_trainer_bytes` | 72, 65, 330 MiB | Rough sizes for the planned-memory estimate. |

### 3.2 The layout guard: `train/layout_guard.py`

ADR 0008 Decision 5 says: one engine instance by default, and more only with an explicit flag stating the exact count.

- **Counting engines, not processes.** `DoomEnvFactory` carries the marker `is_engine_instance = True` (`train/doom_factory.py:14-15`). `count_engine_instances` (`layout_guard.py:26-27`) counts factories with that marker, so several engines inside one worker count too, and toy environments do not.
- `check_layout` (`:41-49`): one engine or fewer always passes. More passes only if `allow_multi_instance` equals the exact engine count. Otherwise it raises `LayoutRefused` with the plan: engines, workers, threads and estimated memory (`plan_text`, `:35-38`; `estimated_bytes`, `:30-32`).
- **Where it runs:** in `SubprocVecEnv.__init__` (`subproc_vec_env.py:102-104`) and in `SyncVecEnv` (`train/vec_env.py:95-100`), before any process starts. So the guard protects every way of building environments, not only the command line.
- The docstring (`layout_guard.py:9-10`) and ADR 0008 add: "The flag prevents accidents; it is not consent." Only `trainer-ops` passes it, for a layout the user approved, and the run report quotes that approval.

### 3.3 Making an environment inside a worker: `train/doom_factory.py`

`DoomEnvFactory` (`:11-19`) is a tiny frozen dataclass holding the `EnvConfig`, so it can be pickled to a worker. It imports `env` (and so ViZDoom) only when called (`:18`), and never torch.

### 3.4 The worker processes: `train/subproc_vec_env.py`

`SubprocVecEnv` implements the same `VecEnv` interface as `SyncVecEnv` (lesson 06), so the trainer cannot tell them apart.

**Why separate processes?** Each engine is its own program, but someone must talk to it. With several workers, several groups of engines are stepped at the same moment. And because workers import **no torch** (module docstring, `:3-5`), each stays small under spawn.

**What a worker does** (`_worker`, `:49-94`):

- `:52`: **ignore Ctrl-C.** On Windows, Ctrl-C reaches every process attached to the console. Only the main process reacts, so it can save and clean up in order.
- `:55-59`: set below-normal priority on itself, build its environments (each starts a `vizdoom.exe`), then set below-normal **directly on its engine children**.
- `:60`: send "ready" with the spaces.
- `:61-76`: wait for commands: `step`, `reset`, `query` (process id and whether torch got loaded), `close`.
- **`:62-64` and `_wait_for_command` (`:41-46`): watch for a dead parent.** While waiting, the worker checks every `parent_poll_interval_s` whether its parent is still alive (`_parent_alive`, `:36-38`). If the main process has been killed hard (a crash, or ended in Task Manager), the worker stops waiting and goes to cleanup. That is how a **non-daemon** worker avoids orphaning its engines.
- `:84-94`: **always** close every environment on the way out.

**The main-process side** (`:97-214`):

- `:102-104`: the layout guard, first.
- `:109-122`: one pipe and one worker per group; daemon or not, from `safety.worker_daemon` (`:117`).
- `:113`: `close` is registered with `atexit` (Python's list of things to run at exit).
- `:123-125`: wait for every "ready"; remember every descendant process (the engines) by id.
- `:183-188`, `step`: send every worker its actions first, then collect all replies, so all workers step at the same time.
- `:164-169`, `kill_engines`: the **emergency** path. Kill every process under the workers now, without a safe point; called by the memory watcher (3.7).
- `:190-214`, `close`, safe to call twice: ask workers to close, wait up to 10 s, terminate surviving workers, terminate surviving remembered engines, close the pipes.

**Daemon or not?** ADR 0008 point 6 says this is "decided by a test, not by recall": kill the parent hard at 1 x 1 with a **real engine** and count surviving `vizdoom.exe` after 15 s. The code defaults to non-daemon with parent polling, keeps the daemon variant selectable (`tests/test_subproc_and_run_loop.py:86-93`), and has a test that a non-daemon worker exits after its parent is hard-killed (`:111-134`). That test uses the **toy** environment, not an engine. The real-engine test is assigned to `trainer-ops` and has not been reported.

### 3.5 Process plumbing: `train/process_control.py`

Torch-free helpers on the `psutil` library:

- `set_below_normal` (`:21`), `is_below_normal` (`:25`): set and check below-normal priority;
- `descendants` (`:32`): every child, grandchild and so on;
- `priority_report` (`:39`): name and priority check for each process;
- `lower_tree_priority` (`:50`): set below-normal on a process and all its descendants;
- `tree_rss_bytes` (`:59`): total resident memory of a process tree;
- `engine_processes` (`:69`): every `vizdoom.exe` on the machine;
- `kill_processes` (`:81`): terminate, wait, then force-kill survivors.

### 3.6 The entry point: `train/run_doom.py`

`python -m train.run_doom` with no flags runs **1 worker x 1 environment, 2 torch threads** (docstring, `:3-8`). A larger layout looks like `--workers 2 --envs-per-worker 4 --allow-multi-instance 8` (`:4`).

`main()` (`:70-117`), in order:

1. `:71-73`: read the options (`:31-50`) into a `RunConfig`. The number of minibatches defaults to the largest divisor of the environment count, up to 4 (`:53-56`; for 1 x 1 that is 1), because minibatches are whole environment sequences.
2. `:74-78`: **the layout guard, before any process starts.** If refused, print the plan and exit with code **4**.
3. `:79`: print the planned layout, for example "1 engine instance(s) in 1 worker process(es), 2 torch thread(s), estimated memory 467 MiB" (by hand: 330 + 65 + 72 = 467).
4. `:80-84`: **refuse to start if any engine process is already running**: exit code **2**.
5. `:85-86`: lower the main process to below normal before spawning workers.
6. `:87-89`: the A0 training config (`my_way_home`, screen off, asserted) and its sensor signature.
7. `:95-96`: start the workers (the guard runs again inside).
8. `:97-102`: import the torch-using modules only now, so spawned workers, which re-import this module, never load torch.
9. `:104-108`: build the trainer; if `--resume`, resume and print the update index, the **session** number and the new environment seed (3.8).
10. `:109-110`: run the training loop, giving it the process ids to check and the emergency engine killer.
11. `:112-117`: **always** close the vector environment, count leftover engines, and exit **0** (none left) or **3** (some left).

### 3.7 The training loop with its guards: `train/run_loop.py`

The module docstring (`:1-12`) summarises the rules.

**One stop flag** (`StopController`, `:49-71`). Everything that wants the run to stop (Ctrl-C, the memory watcher, the update limit, the time limit) **requests** a stop through one shared flag. It does not abort on the spot.

- `on_sigint` (`:66-71`): the **first** Ctrl-C requests a stop with reason "interrupt". The **second** raises `KeyboardInterrupt` at once.

**The memory watcher** (`MemoryWatcher`, `:79-108`) is a background thread. Every `memory_check_interval_s` (5 s) it reads the process tree's memory and the system's available memory (`memory_status`, `:74-76`), then (`check_once`, `:89-101`):

- available below **0.75 GiB**: request a stop **and** call the emergency function, which kills the engines immediately (`subproc_vec_env.py:164-169`);
- tree above **8 GiB**: request a stop;
- available below **1.5 GiB**: request a stop.

It also records the peak tree memory and the lowest available memory seen.

**Stopping only on update boundaries.** In the trainer (`train/trainer.py:100-111`, `:201-215`):

- `collect` checks `should_stop()` at every step. If a stop was requested **during the rollout**, it raises `RolloutAbandoned` (`:41-42`); no weights have changed, so the partly collected rollout is thrown away.
- Once an update has started, it **always completes** (`:202-203`).

So weights are always "on a boundary": exactly as they were after some whole update. The first version could save a half-applied update; that is fixed.

**`run_training` (`run_loop.py:126-200`):**

- `:134-135`: check priorities on every process: read them, set any that is not below normal directly, read again, and log the result (`check_priorities`, `:111-123`).
- `:136-142`: create the stop flag, run one memory check right away, start the watcher, and install the Ctrl-C handler.
- `:152-185`: the loop:
  1. stop if requested;
  2. one training iteration, which may be abandoned (`:156-162`); if the emergency killed the engines underneath, the resulting error is treated as a stop;
  3. append the update's statistics, tree memory and available memory to `metrics.jsonl`, and print a one-line summary (`:163-175`);
  4. request a stop at `max_updates` or `max_seconds`, or write the 10-minute periodic checkpoint (`:176-183`);
  5. cooldown pause (`:184-185`).
- `:186-189`: **after the loop: one checkpoint**, with the stop reason, then log how long the stop took from request to stop.
- `:190-191`: a second Ctrl-C ends here, **without saving**. The previous checkpoint stays intact.
- `:192-200`: stop the watcher, restore the old Ctrl-C handler, fill in the summary.

The longest a normal stop can take is one update (logged as `stop_latency_s`).

### 3.8 Checkpoint and resume: `train/checkpoint.py` and `train/trainer.py:218-254`

**What is saved.** `Trainer.state_dict()` (`trainer.py:218-233`) contains:

- the model and optimizer;
- the return scale;
- the update index and `decisions_done` (the position of any learning-rate decay; ADR 0008 point 1);
- the **session** counter;
- the stale time-out total;
- the three random-generator states, with the NumPy one stored as JSON text.

`save_checkpoint` (`checkpoint.py:36-51`) adds the model config, observation spec, run config and sensor signature, all as **JSON text**. Before writing, it renames the old file to `checkpoint.pt.prev` (`:49-50`), keeping one older generation. It writes a temporary file and renames it into place (`:47-51`), so a crash never leaves a half-written checkpoint.

**Why JSON text: `weights_only=True`.** `torch.load` can unpickle arbitrary Python objects, and unpickling a file can run code hidden in it. With `weights_only=True` (`:55`) it accepts only tensors, numbers, strings, lists and dicts. ADR 0008 point 7 chose this because the submitted checkpoint will be loaded by other people. Everything else (tuples, configs, NumPy state) is stored as JSON text and decoded on load (module docstring, `:3-5`).

**Resume** (`:63-75`) refuses with `CheckpointMismatch` if the sensor signature, model config or observation spec differs. Otherwise it loads the trainer state and calls `resume_session()` (`trainer.py:247-254`), which:

- adds 1 to the session counter;
- derives a **new environment seed** from the run seed and the session number (`session_env_seed`, `:45-49`; session 0 uses the run seed itself);
- resets every environment with that seed, with zeroed hidden states (`restart_episodes`, `:81-88`).

**Why a new seed per session** (ADR 0008 Decision 4). A probe (`docs/reference/engine-probes.md:272-274`) confirmed that the same seed reproduces an episode exactly, across resets and across instances. Re-using the run seed on every resume would replay the same episodes at the start of every session; on a laptop where training is stopped and resumed often, that would bias training.

### 3.9 What the tests check, and what they do not

All tests in `tests/test_subproc_and_run_loop.py` and `tests/test_layout_guard.py` use the **toy environment** or fakes. **No ViZDoom is started.**

| Test | Checks |
|---|---|
| `test_layout_guard.py:42-54` | Defaults are 1 x 1 with 2 threads; the bare command resolves to 1 x 1. |
| `test_layout_guard.py:64-81` | More than one instance without the exact flag is refused before any process starts; with it, the guard passes. |
| `test_layout_guard.py:91-112` | The guard lives in both vector environments and counts engines, not toys. |
| `test_layout_guard.py:83` | With a leftover engine (faked), the entry point refuses. |
| `test_subproc_and_run_loop.py:52`, `:78` | Workers match `SyncVecEnv`, load no torch, run below normal, leave nothing behind; a frozen worker is terminated. |
| `:86-93`, `:111-134` | Non-daemon is the default; a non-daemon toy worker exits when its parent is hard-killed. |
| `:138-156` | Checkpoint, metrics with available memory, priority, and resume to update 3 with session 1. |
| `:159-195` | The memory cap stops and saves; low available memory stops; very low memory fires the emergency; the watcher runs on a clock. |
| `:198-261` | Ctrl-C during an update finishes it and saves weights equal to the model's; Ctrl-C during a rollout abandons it and saves the previous boundary; a second Ctrl-C stops without saving. |
| `:263` | The cooldown pauses. |

**Untested, stated plainly:**

- **Shutdown with real engines after a hard kill**: the parent-polling test uses toy workers. ADR 0008 assigns the real-engine survivor count to `trainer-ops`. **Since done** (`reports/runs/2026-11-25-part1-single-instance-checks.md`, section b): with a real engine, hard-killing only the main process left **0 survivors** from 1 to 20 s, for both non-daemon (`20261125-nogit-a9f0b2fb`) and daemon (`20261125-nogit-7cfb625c`) workers. Non-daemon stays the default. See lesson 10.
- **A real Ctrl-C** with engines attached to the console: still untested. The tests call the handler from code, and the Part 1 interrupt-then-resume check did the same (it passed, with a new session seed on resume). A helper that would have sent a real console Ctrl-C was blocked by a security hook (Part 1, section c). Whether Ctrl-C reaches `vizdoom.exe` directly is still open.
- **A real DOOM time-out** in a training run: the smoke report says none had happened, so its "stale time-outs: 0" was vacuous. **Since done** (Part 1 report, section a; run `20261125-nogit-2d4e952a`): with a 200-tic episode limit, the wrapper's own limit fired every time (the fresh-observation path, with a bootstrap value computed). Over 250 updates the stale count was 0, `valid_frac` was 1.0, the tree peaked at 429,174,784 bytes, every process was below normal, and 0 engines were left. See lesson 10.
- **Priority inheritance by the engine**: unproven, because the worker sets the engine's priority directly before the main process checks it.
- **Any multi-instance run** at the time this lesson was rewritten. Since then, the layout benchmark and the first training run (4x8) have happened; see lesson 10.
- **Any DOOM training.**

## 4. A tiny worked example with real numbers

I did not run any DOOM engine for this lesson; another agent is counting `vizdoom.exe` processes, and a stray instance would corrupt its results. The numbers below are other agents' measurements, labelled by source.

**The worker's smoke runs** (`reports/runs/2026-11-25-a0-single-instance-smoke.md`), all 1 x 1 on `my_way_home`:

- Before ADR 0008:
  - one update collected at 119 decisions/s, with the process tree at 394 MiB;
  - a resumed second update collected at 180/s;
  - a long rollout ended when the agent **reached the goal** (return 0.954, a termination, not a time-out), with a 460 MiB peak.
- After ADR 0008 (bare command, 1 x 1, 2 threads):
  - the plan line read "estimate 467 MiB";
  - all processes were below normal, with nothing set directly;
  - the value scale read 0.100 (the new floor);
  - the tree peaked at 393-395 MiB, and the lowest available memory seen was 3.3-3.45 GB;
  - the resumed run reported session 1 with a new environment seed, and the previous checkpoint was kept as `.prev`;
  - stop latency was 0.016 s, and 0 engines were alive after shutdown;
  - **sampling was 17-18 decisions/s**.
- The report could not explain the 17/s. Another `python.exe` was using about 48% of the CPU at the time.

**The coordinator's quiet-machine run** (passed to me by the coordinator; there is no written report for it among the `.md` files in `reports/runs/`): 1 x 1, about **180-260 decisions/s** over 91 updates in 40 s, tree memory about **410 MiB**, all processes below normal, **0 engines left**. So the 17/s was the busy machine, not the code.

**By hand, for scale:**

- At 1 x 1 with rollout 64, one update consumes 64 decisions. At about 200 decisions/s, that is about 0.3 s of collection per update.
- The layout estimate for 4 x 8 would be 330 + 4 x 65 + 32 x 72 = 2894 MiB, about 2.8 GiB. That is an estimate from the `SafetyConfig` numbers, not a measurement. No multi-instance run has been measured.

**From the first version of this lesson** (a toy demo of mine, before ADR 0008's checkpoint changes): resume refused a changed sensor signature with `CheckpointMismatch: sensor settings differ: ... ('frame_skip', 4) ... vs ... ('frame_skip', 2) ...`. The current code gives the same refusal, now comparing JSON text (`checkpoint.py:66-68`).

## 5. Why we chose this design over the alternatives

From ADRs 0004, 0006 and 0008 (**all Proposed, not ratified**):

- **Synchronous PPO with subprocess workers and central batched inference** (ADR 0004 option B); workers never import torch.
- **Default layout 1 x 1** (ADR 0008 Decision 5). Leaving 4 x 8 as the default with a warning was "one typo from 32 engines". The explicit, counted flag costs about an hour to build and makes accidents impossible. It is still not consent.
- **Memory on a clock, including system free memory** (ADR 0008 Decision 5). A per-update check can miss a spike inside a long update. Watching only our own tree misses the case where *other* programs fill the machine.
- **Checkpoints only on update boundaries** (ADR 0008 Decision 5). A half-applied update is a state no training step ever produced. Keeping `.prev` guards against a bad final save.
- **`weights_only=True`** (ADR 0008 point 7; `0007-recipe` had asked for it).
- **Non-daemon workers that poll for a dead parent**, pending the real-engine hard-kill test (ADR 0008 point 6).
- **A new environment seed per session** (ADR 0008 Decision 4), justified by the same-seed probe.

## 6. How it connects

- **Lesson 01**: `TRAIN_CONFIG` (`my_way_home`) and the sensor signature, which every checkpoint stores.
- **Lesson 06**: the trainer, and `SyncVecEnv`, which now runs the layout guard too.
- **Lesson 09**: why the memory test failed, and Gate A, whose pass lifted the block on single-instance runs.
- **Next**: ADR 0008's single-instance engine checks (time-out smoke run, hard-kill survivor count, Ctrl-C then resume), then the multi-instance benchmark that is starting now, which needs the user's go-ahead. Those will be taught once they have results.

## 7. Common mistakes, and how we would notice each protection failing

| What fails | How we would notice it |
|---|---|
| **Starting more engines than intended.** | Without `--allow-multi-instance N` matching exactly, the run refuses with exit code 4 and prints the plan. A typo in N is also refused. |
| **Priority not applied.** | The start-of-run `priority check` line shows `all_below_normal: False` or a non-empty `set_directly`. The laptop feeling sluggish is the human symptom. |
| **Leftover engines.** | At exit: `engine processes alive after shutdown: N` and exit code 3. On the next start: "refusing to start" and exit code 2. |
| **Memory pressure.** | `tree_rss_bytes` and `available_bytes` in `metrics.jsonl`. The stop reason says "memory cap exceeded", "system available memory low" or "emergency: ...". |
| **A slow stop.** | `stop_latency_s` in the summary: normally under one update's duration. |
| **Checkpoints not written.** | `checkpoints_written` stays 0; `checkpoint.pt` is older than 10 minutes. |
| **Resuming with different sensors or model.** | `CheckpointMismatch` at start. That is the protection working. |
| **Expecting a resumed run to replay the same episodes.** | It will not: the session number and environment seed change on every resume (printed at start). |
| **Building a `Trainer` in your own script.** | No priority, no leftover check, no watcher, no checkpoints. The layout guard still runs inside the vector environments. |
| **Misreading throughput on a busy machine.** | The 17/s smoke numbers versus 180-260/s on a quiet machine. Always note what else was running. |

## 8. Check your understanding

1. Someone runs `python -m train.run_doom --workers 2 --envs-per-worker 4`. What happens, and what would make it run?
2. The owner presses Ctrl-C while the trainer is half-way through collecting a rollout. What happens to that rollout, what is saved, and what would a second Ctrl-C change?
3. A run is resumed from a checkpoint saved at update 412 in session 0. What does the resumed run restore, and why does its first episode differ from the first episode of session 0?

---

## Answers

1. 2 x 4 = 8 engine instances with no `--allow-multi-instance`, so the layout guard refuses before any process starts (`run_doom.py:74-78`; `layout_guard.py:41-49`). It prints the plan (8 engines, 2 workers, 2 threads, estimated memory) and exits with code 4. It would run with `--allow-multi-instance 8` exactly. But ADR 0008 says only `trainer-ops` passes that flag, for a layout the user has approved.
2. The first Ctrl-C only sets the stop flag (`run_loop.py:66-71`). `collect` sees it at its next step and raises `RolloutAbandoned` (`trainer.py:100-111`), so the partial rollout is thrown away and no weights change. The loop then writes one checkpoint of the weights from the last completed update (`run_loop.py:186-187`), and the entry point closes the workers and engines. A second Ctrl-C raises `KeyboardInterrupt` immediately and stops **without saving** (`:190-191`), leaving the previous checkpoint file intact.
3. It restores the model, optimizer, return scale, update index 412, decisions done, stale total, random-generator states and the session counter. Then `resume_session` makes it session 1 and derives a new environment seed from the run seed and the session number (`trainer.py:45-49`, `:247-254`). The probe showed that the same seed replays the same episode exactly, so without the new seed every session would start with the same episodes.
