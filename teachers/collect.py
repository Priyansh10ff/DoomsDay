"""Collect scripted-teacher demonstrations on the generated TRAIN maps (TRAINING ONLY).

    python -m teachers.collect --stages S0 S1 S2 --episodes-per-map N --out data/teacher/<name>
                               [--noise 0.2] [--clean-every K] [--workers K] [--seed S]
                               [--burst-max B] [--keep-failures] [--no-save]

Environment: exactly the config `python -m train.run_doom --maps generated` trains the
student with (`generated_maps_env_config(a0_train_env_config(), bundle, all lumps)`:
vanilla rules, no ACS, no cheats, skill 3, 4200 tics), maps from
`maps.loader.training_bundle()` (train split only; held-out and val maps are refused
there). Each episode resets with `options={"map": lump}`; start poses are whatever the
map gives (one player start per vanilla map).

Noise (`--noise p`, `--burst-max B`): at every decision that is not already inside a
burst, with probability p a burst of 1..B decisions (uniform) starts; during a burst a
uniformly random action (attack head kept at 0) is EXECUTED, while the label recorded
is still the teacher's intended action. The teacher's state keeps updating during the
burst from the real pose and the executed previous action. The expected share of
noisy decisions is p*m / (1 + p*m) with m = (B + 1) / 2 (p = 0.2, B = 3: 29%).
`--clean-every K`: every K-th episode of a map (episode index % K == 0) runs at noise 0.

Dataset contract (one file per episode, `ep_<6 digits>.npz`, np.savez_compressed):
  every key of the env observation dict except `screen`, stacked to (T, ...) with the
  env's dtype (obs[t] is the observation action[t] was chosen from); `action` int64
  (T, 6) teacher label; `executed_action` int64 (T, 6); `engine_reward` float32 (T,);
  `done` bool (T,) true on the last step; `label_combat` bool (T,) true where the label
  came from the combat branch (extra key: train/bc_data.py loads only the keys it
  knows); `goal_mode` int64 (T,) the teacher's goal-mode id per decision (table in
  teachers/goal_modes.py, copy in configs/goal_modes.py; added 2026-11-26, so older
  datasets lack it and stay valid). summary.json gains `goal_mode_share_by_stage` and a
  `goal_mode_share` per stage@noise. `index.jsonl`: one line per saved episode
  {"file", "map_id", "stage", "length", "success", "seed", "noise"}. Failed teacher
  episodes are dropped unless `--keep-failures`; they are always counted in
  `summary.json`.

DAgger (`--student CKPT [--beta b] [--max-steps N] [--student-argmax]`): the student
checkpoint (agent.inference.PolicyRunner, sampled actions, CPU, one torch thread per
worker) drives; with probability b per decision the teacher's label is executed
instead. The teacher (ControllerConfig.student_driven) labels EVERY step from wherever
the student is: its route fields are distance fields over the whole grid, so it
re-targets from any position; unstick labels are given only when its own steering
label would push forward into the blockage. Same dataset contract (`action` = teacher
label, `executed_action` = what ran), plus `label_unstick` and `student_driven` (T,)
bool. Every DAgger episode is saved (failures included); `--max-steps` caps an episode
(a capped episode is not a success). With `--no-save` and b = 0 it measures the
student's closed-loop success on the TRAIN maps.

Laptop limits: at most MAX_WORKERS engines, below-normal priority for every worker and
its engine, and a stop when available memory falls under MIN_AVAILABLE_GIB.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import psutil

from teachers.goal_modes import GOAL_MODES, N_GOAL_MODES

MAX_WORKERS = 8
MIN_AVAILABLE_GIB = 2.5
MEMORY_POLL_S = 2.0
STAGES_SUPPORTED = ("S0", "S1", "S2", "S3")   # S3: combat teacher (teachers/combat.py)
DEFAULT_STAGES = ("S0", "S1", "S2")      # --stages default: unchanged from before S3 support
SCREEN_KEY = "screen"
INDEX_NAME = "index.jsonl"
SUMMARY_NAME = "summary.json"
PARTIAL_NAME = "episodes.partial.jsonl"   # every finished episode's summary, for --resume
JOB_GET_TIMEOUT_S = 2.0                   # an empty job queue for this long = all taken
REDUCE_GRACE_S = 20.0                     # min seconds between two worker reductions
SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class Job:
    episode: int          # global episode number -> ep_<episode:06d>.npz
    lump: str
    map_id: str
    stage: str
    seed: int
    noise: float


@dataclass(frozen=True)
class CollectConfig:
    stages: tuple[str, ...]
    episodes_per_map: int
    out: str | None
    noise: float = 0.0
    clean_every: int = 0
    burst_max: int = 3
    workers: int = 1
    seed: int = 0
    keep_failures: bool = False
    save: bool = True
    map_ids: tuple[str, ...] = ()     # optional subset (debugging)
    trace: bool = False               # keep every controller event in the summary
    resume: bool = False              # continue an interrupted run in the same --out
    # DAgger: a student checkpoint drives; the teacher only labels (see module docstring)
    student: str | None = None
    beta: float = 0.0                 # per decision: probability the teacher's label is executed
    max_steps: int = 0                # cap on decisions per episode (0 = env time limit only)
    student_argmax: bool = False      # student acts by argmax instead of sampling
    # "route": shortest path to the exit from the map file (teachers/navigator.py);
    # "explore": decisions from what the player has seen only (teachers/explore.py)
    teacher: str = "route"


# ------------------------------------------------------------------ setup
def training_maps() -> tuple[str, dict[str, str], dict[str, str]]:
    """(bundle path, lump -> map id, lump -> stage) through maps.loader (train mode)."""
    from maps.loader import training_bundle
    return training_bundle()


def env_config(bundle: str, lumps: tuple[str, ...]):
    """The student's generated-maps training config (train.run_doom.build_generated_maps)."""
    from configs.a0_env import a0_train_env_config, generated_maps_env_config
    return generated_maps_env_config(a0_train_env_config(), bundle, tuple(sorted(lumps)))


def make_jobs(cfg: CollectConfig, lump_to_id: dict[str, str], lump_to_stage: dict[str, str]) -> list[Job]:
    bad = [s for s in cfg.stages if s not in STAGES_SUPPORTED]
    if bad:
        raise ValueError(f"stages {bad} not supported by this teacher (supported: {STAGES_SUPPORTED})")
    jobs: list[Job] = []
    lumps = sorted(l for l in lump_to_id if lump_to_stage[l] in cfg.stages
                   and (not cfg.map_ids or lump_to_id[l] in cfg.map_ids))
    n = 0
    for e in range(cfg.episodes_per_map):
        for lump in lumps:
            clean = cfg.clean_every > 0 and e % cfg.clean_every == 0
            jobs.append(Job(n, lump, lump_to_id[lump], lump_to_stage[lump],
                            seed=cfg.seed * SEED_STRIDE + n, noise=0.0 if clean else cfg.noise))
            n += 1
    return jobs


# ------------------------------------------------------------------ one episode
class StudentDriver:
    """The DAgger student: a BC/PPO checkpoint run through the inference runner
    (agent.inference.PolicyRunner, the same path evaluate uses), fresh hidden state and
    a generator seeded from the episode seed for every episode."""

    def __init__(self, checkpoint: str, argmax: bool = False) -> None:
        from train.checkpoint import load_checkpoint
        from agent.model import A0Agent
        from agent.obs_spec import ObsSpec
        blob = load_checkpoint(checkpoint)
        d = json.loads(blob["obs_spec_json"])
        d["head_sizes"] = tuple(d["head_sizes"])
        self.model = A0Agent(ObsSpec(**d), blob["model_config"])
        self.model.load_state_dict(blob["trainer"]["model"], strict=True)
        self.model.eval()
        self.argmax = argmax
        self.runner = None

    def begin(self, seed: int) -> None:
        from agent.inference import PolicyRunner
        self.runner = PolicyRunner(self.model, 1, (self.argmax,) * self.model.spec.n_heads, seed=seed)

    def act(self, obs: dict, start: bool) -> np.ndarray:
        batched = {k: np.asarray(v)[None] for k, v in obs.items() if k != SCREEN_KEY}
        return self.runner.act(batched, np.array([1.0 if start else 0.0], np.float32))[0]


USE_STAT_KEYS = ("use_labels", "use_edges_exec", "press_open_target", "press_into_moving",
                 "door_presses", "switch_presses", "ineffective_presses", "repress_after_close",
                 "reversals_label", "reversals_other", "max_presses_per_closed_spell",
                 "pressable_now", "pressable_but_use_held", "closed_door_ahead_far",
                 "open_door_ahead", "near_closed_target_door")


class UseStats:
    """Per-episode USE bookkeeping (change C3 acceptance and kill criterion K4).

    use_labels           decisions labelled USE = 1
    use_edges_exec       executed fresh USE edges (prev executed use 0 -> 1)
    press_open_target    labelled presses while the route's current door anchor has gap > 0
    press_into_moving    labelled presses where the door the ENGINE's 64-unit use trace
                         would hit has gap > 0 (independent of the label's own range)
    door_presses / switch_presses   labelled presses that executed, by target
    ineffective_presses  labelled, executed door presses after which that door's gap is
                         still 0 (label range too generous), or switch presses after which
                         the level did not end
    repress_after_close  labelled door presses on a door this episode had already opened
                         (legitimate when it shut again; reported for inspection)
    reversals_label / reversals_other   executed fresh edges whose engine-targeted door
                         was moving or open and then changed direction (an open door
                         starting to close counts); `label` if the label pressed, `other`
                         if noise or the student did
    max_presses_per_closed_spell   most labelled executed presses on one door while it
                         stayed closed (1 = one press per approach)
    Diagnostics (every decision, from the pose the label was chosen at):
    pressable_now        a press target is under the view ray within label range
    pressable_but_use_held   ... but the previous executed action held USE (label = release)
    closed_door_ahead_far    the view ray's first stopper is a closed door face, front side,
                         beyond the label range but within FAR_RANGE
    open_door_ahead      the first stopper within the engine range is a door face with gap > 0
    near_closed_target_door  within NEAR_DOOR of the closed route-target door's centre
    """
    NEAR_DOOR = 112.0
    FAR_RANGE = 160.0

    def __init__(self, route) -> None:
        from teachers.use_label import USE_RANGE_ENGINE
        self.route = route
        self.range = USE_RANGE_ENGINE
        self.c = {k: 0 for k in USE_STAT_KEYS}
        self.prev_gaps: dict | None = None
        self.prev_use = 0
        self.opened: set[int] = set()
        self.spell: dict[int, int] = {}
        self.ineffective: list[dict] = []     # details of each ineffective press

    def _engine_target(self, x: float, y: float, ang: float):
        from teachers.use_label import KIND_DOOR, KIND_SWITCH, facing_hit, on_front
        L = self.route.use_lines
        hit = facing_hit(L, x, y, ang, self.range) if L is not None else None
        if hit is None or not on_front(L, hit[0], x, y):
            return None
        k = int(L.kind[hit[0]])
        if k == KIND_DOOR:
            return ("door", int(L.sector[hit[0]]))
        return ("switch", -1) if k == KIND_SWITCH else None

    def step(self, ctrl, info_before: dict, label, executed, gaps_after: dict | None,
             ended: bool) -> None:
        from teachers.use_label import CLOSED_GAP
        c = self.c
        g0 = info_before["door_gaps"]
        lab_use, ex_use = int(label[4]), int(executed[4])
        edge = ex_use == 1 and self.prev_use == 0
        self.prev_use = ex_use
        c["use_labels"] += lab_use
        c["use_edges_exec"] += int(edge)
        an = ctrl.anchor
        if lab_use and an.kind == "door" and float(g0[an.door_sector]) > CLOSED_GAP:
            c["press_open_target"] += 1
        x, y, ang = float(info_before["pos_x"]), float(info_before["pos_y"]), float(info_before["angle"])
        self._diagnose(ctrl, x, y, ang, g0, info_before.get("keys_held"))
        tgt = self._engine_target(x, y, ang) if (lab_use or edge) else None
        if lab_use and tgt and tgt[0] == "door" and float(g0[tgt[1]]) > CLOSED_GAP:
            c["press_into_moving"] += 1
        if edge and lab_use:
            pt = ctrl.press_target
            same = bool(np.array_equal(np.asarray(label), np.asarray(executed)))
            if pt and pt[0] == "switch":
                c["switch_presses"] += 1
                if not ended:
                    c["ineffective_presses"] += 1
                    self._note_ineffective(ctrl, x, y, ang, "switch", same, executed)
            elif pt and pt[0] == "door":
                s = pt[1]
                c["door_presses"] += 1
                c["repress_after_close"] += int(s in self.opened)
                self.spell[s] = self.spell.get(s, 0) + 1
                c["max_presses_per_closed_spell"] = max(c["max_presses_per_closed_spell"], self.spell[s])
                if gaps_after is not None and float(gaps_after[s]) <= CLOSED_GAP:
                    c["ineffective_presses"] += 1
                    self._note_ineffective(ctrl, x, y, ang, f"door{s}", same, executed)
        if edge and tgt and tgt[0] == "door" and gaps_after is not None and self.prev_gaps is not None:
            s = tgt[1]
            g = float(g0[s])
            before = g - float(self.prev_gaps[s])
            after = float(gaps_after[s]) - g
            if g > CLOSED_GAP and ((after < 0 <= before) or (before < 0 < after)):
                c["reversals_label" if lab_use else "reversals_other"] += 1
        if gaps_after is not None:
            for s in list(self.spell):
                if float(gaps_after[s]) > CLOSED_GAP:
                    self.opened.add(s)
                    del self.spell[s]
        self.prev_gaps = g0

    def _note_ineffective(self, ctrl, x, y, ang, what, same, executed) -> None:
        from teachers.use_label import facing_hit
        hit = facing_hit(self.route.use_lines, x, y, ang, self.range)
        self.ineffective.append({"t": ctrl.t, "target": what, "ray_dist": None if hit is None else round(hit[1], 1),
                                 "executed_is_label": same,
                                 "executed": [int(v) for v in executed]})

    def _diagnose(self, ctrl, x: float, y: float, ang: float, gaps: dict, keys) -> None:
        from teachers.use_label import CLOSED_GAP, KIND_DOOR, facing_hit, on_front, press_target
        c, L = self.c, self.route.use_lines
        an = ctrl.anchor
        if an.kind == "door" and float(gaps[an.door_sector]) <= CLOSED_GAP:
            cx, cy = an.door_center
            c["near_closed_target_door"] += int(math.hypot(cx - x, cy - y) <= self.NEAR_DOOR)
        if L is None:
            return
        if press_target(L, x, y, ang, gaps, keys, ctrl.cfg.use_range) is not None:
            c["pressable_now"] += 1
            c["pressable_but_use_held"] += int(ctrl.last_use_exec == 1)
        hit = facing_hit(L, x, y, ang, self.FAR_RANGE)
        if hit is None or int(L.kind[hit[0]]) != KIND_DOOR or not on_front(L, hit[0], x, y):
            return
        g = float(gaps[int(L.sector[hit[0]])])
        if g <= CLOSED_GAP and ctrl.cfg.use_range < hit[1] <= self.FAR_RANGE:
            c["closed_door_ahead_far"] += 1
        elif g > CLOSED_GAP and hit[1] <= self.range:
            c["open_door_ahead"] += 1

    def summary(self) -> dict:
        out = dict(self.c)
        if self.ineffective:
            out["ineffective_detail"] = self.ineffective
        return out


def run_episode(env, route, job: Job, burst_max: int, trace: bool = False,
                student: StudentDriver | None = None, beta: float = 0.0,
                max_steps: int = 0, teacher: str = "route") -> tuple[dict, dict]:
    """Play one episode. Returns (arrays for the npz, episode summary).
    Teacher mode (student None): the teacher drives, with optional noise bursts.
    DAgger mode: at each decision the student's action is executed, except with
    probability `beta` the teacher's label is; the label is always the teacher's.
    teacher "route": `route` is a navigator.Route; "explore": an explore.ExploreMap."""
    from teachers.controller import ControllerConfig, TeacherController
    sizes = tuple(int(n) for n in env.action_space.nvec)
    rng = np.random.default_rng(job.seed)
    ccfg = ControllerConfig(student_driven=student is not None)
    if teacher == "explore":
        from teachers.explore import ExploreController
        ctrl = ExploreController(env.cfg.turn_bins, ccfg, seed=job.seed)
    elif teacher == "route":
        ctrl = TeacherController(env.cfg.turn_bins, ccfg, seed=job.seed)
    else:
        raise ValueError(f"unknown teacher {teacher!r}")
    obs, info = env.reset(seed=job.seed, options={"map": job.lump})
    if info.get("map") != job.lump:
        raise RuntimeError(f"env loaded {info.get('map')!r}, wanted {job.lump!r}")
    info = _augment(env, info)
    ctrl.reset(route)
    ustat = UseStats(route)
    if student is not None:
        student.begin(job.seed)
    obs_buf: dict[str, list] = {k: [] for k in obs if k != SCREEN_KEY}
    labels, executed, rewards, unstick, by_student, combat = [], [], [], [], [], []
    modes: list[int] = []
    seen: set[int] = set()
    burst, noisy = 0, 0
    capped = False
    while True:
        label = ctrl.act(obs, info)
        if student is not None:
            s_act = student.act(obs, len(labels) == 0)
            drive = not (beta > 0 and rng.random() < beta)
            ex = s_act.astype(np.int64) if drive else label.copy()
            by_student.append(drive)
        else:
            if burst == 0 and job.noise > 0 and rng.random() < job.noise:
                burst = int(rng.integers(1, burst_max + 1))
            if burst > 0:
                ex = np.array([rng.integers(n) for n in sizes], dtype=np.int64)
                ex[3] = 0                       # attack head: no firing in the noise either
                burst -= 1
                noisy += 1
            else:
                ex = label.copy()
            by_student.append(False)
        unstick.append(ctrl.label_source == "unstick")
        combat.append(ctrl.label_source == "combat")
        modes.append(int(ctrl.goal_mode))
        seen.update(m.id for m in info.get("monsters") or ())
        for k in obs_buf:
            obs_buf[k].append(np.array(obs[k], copy=True))
        labels.append(label)
        executed.append(ex)
        info_before = info
        obs, r, term, trunc, info = env.step(ex)
        rewards.append(r)
        if not (term or trunc):
            info = _augment(env, info)
        ustat.step(ctrl, info_before, label, ex, None if (term or trunc) else info["door_gaps"],
                   ended=bool(term or trunc))
        if term or trunc:
            break
        if max_steps and len(labels) >= max_steps:
            capped = True
            break
    T = len(labels)
    arrays = {k: np.stack(v).astype(env.observation_space[k].dtype, copy=False) for k, v in obs_buf.items()}
    arrays["action"] = np.stack(labels).astype(np.int64)
    arrays["executed_action"] = np.stack(executed).astype(np.int64)
    arrays["engine_reward"] = np.asarray(rewards, dtype=np.float32)
    done = np.zeros(T, dtype=bool)
    done[-1] = True
    arrays["done"] = done
    arrays["label_unstick"] = np.asarray(unstick, dtype=bool)
    arrays["student_driven"] = np.asarray(by_student, dtype=bool)
    arrays["label_combat"] = np.asarray(combat, dtype=bool)
    arrays["goal_mode"] = np.asarray(modes, dtype=np.int64)
    success = bool(info.get("level_exit", False)) and not capped
    lab, exa = arrays["action"], arrays["executed_action"]
    route = ctrl.route              # explore: the last dynamic route
    summ = {"episode": job.episode, "map_id": job.map_id, "stage": job.stage, "seed": job.seed,
            "noise": job.noise, "length": T, "success": success, "noisy_decisions": noisy,
            "student_decisions": int(sum(by_student)), "unstick_labels": int(sum(unstick)),
            "capped": capped, "agree_all": round(float((lab == exa).all(1).mean()), 4),
            "agree_head": [round(float(v), 4) for v in (lab == exa).mean(0)],
            "dead": bool(info.get("dead", False)), "timeout": bool(info.get("timeout", False)) or bool(trunc),
            "anchor_reached": ctrl.k, "anchors": len(route.anchors),
            "anchor_kind": route.anchors[min(ctrl.k, len(route.anchors) - 1)].kind,
            "phase": ctrl.phase, "unsticks": sum(1 for _, e in ctrl.events if e.endswith("unstick")),
            "final_pos": [round(float(info.get("pos_x", 0.0))), round(float(info.get("pos_y", 0.0)))],
            # combat (S3); kills from the last live info (a kill on the final decision is missed)
            "attack_labels": int(lab[:, 3].sum()), "attack_exec": int(exa[:, 3].sum()),
            "combat_decisions": int(sum(combat)), "monsters_seen": len(seen),
            "kills": int(info.get("killcount", 0) or 0),
            "damage_taken": round(float(info.get("damage_taken_total", 0.0) or 0.0), 1),
            "combat_give_ups": len(ctrl.given_up), "teacher": teacher,
            "goal_mode_counts": [int(v) for v in np.bincount(arrays["goal_mode"],
                                                              minlength=N_GOAL_MODES)]}
    if teacher == "explore":
        kn = ctrl.kn
        walk = int(np.count_nonzero(ctrl.xm.grid.sector != -1))
        summ.update({"replans": ctrl.replans, "goals": len(ctrl.goal_log),
                     "goal_final": "/".join(str(v) for v in ctrl.goal) if ctrl.goal else None,
                     "seen_share": round(int(kn.seen.sum()) / max(walk, 1), 3),
                     "exit_seen": kn.exit_seen, "keys_seen": len(kn.keys_seen),
                     "locks_learned": ctrl.locks_learned, "blacklist": len(ctrl.blacklist)})
    summ.update(ustat.summary())
    if trace:
        summ["events"] = ctrl.events
    elif not success:
        summ["events"] = ctrl.events[-40:]
    return arrays, summ


def _augment(env, info: dict) -> dict:
    """PRIVILEGED labelling inputs added to a live info dict: live door gaps, and the
    monsters / projectiles the student can see (teachers.combat.sightings_of; world
    positions for aiming). Never written to the dataset's observations."""
    from teachers.combat import sightings_of
    from teachers.use_label import door_gaps_of
    mons, proj = sightings_of(env.game)
    return dict(info, door_gaps=door_gaps_of(env.game), monsters=mons, projectiles=proj)


# ------------------------------------------------------------------ worker
def _worker(wid: int, jobs, cfg: CollectConfig, env_cfg, results, stop_all, stop_me) -> None:
    """Pulls jobs from the shared `jobs` queue until it is empty or a stop is set."""
    from teachers.navigator import load_route
    from train.process_control import lower_tree_priority, set_below_normal
    from maps.loader import resolve
    set_below_normal()
    env = None
    routes: dict = {}
    try:
        student = None
        if cfg.student:
            import torch
            torch.set_num_threads(1)
            student = StudentDriver(cfg.student, cfg.student_argmax)
        from env import DoomEnv
        env = DoomEnv(env_cfg)
        lower_tree_priority()
        while not (stop_all.is_set() or stop_me.is_set()):
            try:
                job = jobs.get(timeout=JOB_GET_TIMEOUT_S)
            except queue.Empty:
                break
            if job.map_id not in routes:       # static map data only; knowledge is per episode
                if cfg.teacher == "explore":
                    from teachers.explore import load_explore_map
                    routes[job.map_id] = load_explore_map(resolve(job.map_id, "train"), map_id=job.map_id)
                else:
                    routes[job.map_id] = load_route(resolve(job.map_id, "train"), map_id=job.map_id)
            arrays, summ = run_episode(env, routes[job.map_id], job, cfg.burst_max, cfg.trace,
                                       student=student, beta=cfg.beta, max_steps=cfg.max_steps,
                                       teacher=cfg.teacher)
            summ["saved"] = False
            if cfg.save and (summ["success"] or cfg.keep_failures or student is not None):
                fname = f"ep_{job.episode:06d}.npz"
                np.savez_compressed(os.path.join(cfg.out, fname), **arrays)
                summ["file"] = fname
                summ["saved"] = True
            results.put(("episode", wid, summ))
    except BaseException as e:   # report, then close the engine in finally
        results.put(("error", wid, f"{type(e).__name__}: {e}"))
    finally:
        if env is not None:
            env.close()
        results.put(("done", wid, None))


# ------------------------------------------------------------------ main
def _mode_share(eps: list[dict]) -> dict:
    """Share of decisions per goal mode (teachers/goal_modes.py) over these episodes;
    episodes from before the goal-mode label carry no counts and are skipped."""
    tot = np.zeros(N_GOAL_MODES, np.int64)
    for e in eps:
        c = e.get("goal_mode_counts")
        if c is not None:
            tot += np.asarray(c, np.int64)
    n = max(int(tot.sum()), 1)
    return {name: round(int(tot[i]) / n, 4) for i, name in enumerate(GOAL_MODES)}


def _stats(eps: list[dict]) -> dict:
    out: dict = {}
    for key in sorted({(e["stage"], e["noise"]) for e in eps}):
        sel = [e for e in eps if (e["stage"], e["noise"]) == key]
        ok = [e for e in sel if e["success"]]
        out[f"{key[0]}@noise{key[1]}"] = {
            "episodes": len(sel), "success": len(ok), "success_rate": round(len(ok) / len(sel), 4),
            "decisions_all": int(sum(e["length"] for e in sel)),
            "mean_length_success": round(float(np.mean([e["length"] for e in ok])), 1) if ok else None,
            "saved": sum(e.get("saved", False) for e in sel),
            "decisions_saved": int(sum(e["length"] for e in sel if e.get("saved"))),
            "use_label_share": round(sum(e.get("use_labels", 0) for e in sel)
                                     / max(sum(e["length"] for e in sel), 1), 4),
            "attack_label_share": round(sum(e.get("attack_labels", 0) for e in sel)
                                        / max(sum(e["length"] for e in sel), 1), 4),
            "attack_labels": int(sum(e.get("attack_labels", 0) for e in sel)),
            "combat_decisions": int(sum(e.get("combat_decisions", 0) for e in sel)),
            "kills": int(sum(e.get("kills", 0) for e in sel)),
            "deaths": int(sum(bool(e.get("dead")) for e in sel)),
            "timeouts": int(sum(bool(e.get("timeout")) for e in sel)),
            "damage_taken": round(float(sum(e.get("damage_taken", 0.0) for e in sel)), 1),
            "monsters_seen": int(sum(e.get("monsters_seen", 0) for e in sel)),
        }
        out[f"{key[0]}@noise{key[1]}"]["goal_mode_share"] = _mode_share(sel)
        for k in USE_STAT_KEYS:
            if k == "max_presses_per_closed_spell":
                out[f"{key[0]}@noise{key[1]}"][k] = max((e.get(k, 0) for e in sel), default=0)
            elif k != "use_labels":
                out[f"{key[0]}@noise{key[1]}"][k] = int(sum(e.get(k, 0) for e in sel))
    return out


def collect(cfg: CollectConfig) -> dict:
    if not 1 <= cfg.workers <= MAX_WORKERS:
        raise ValueError(f"--workers must be 1..{MAX_WORKERS}")
    if cfg.save and not cfg.out:
        raise ValueError("--out is required unless --no-save")
    from train.process_control import set_below_normal
    set_below_normal()
    bundle, lump_to_id, lump_to_stage = training_maps()
    env_cfg = env_config(bundle, tuple(lump_to_id))
    jobs = make_jobs(cfg, lump_to_id, lump_to_stage)
    if cfg.out:
        os.makedirs(cfg.out, exist_ok=True)
        if not cfg.resume and os.path.exists(os.path.join(cfg.out, PARTIAL_NAME)):
            raise FileExistsError(f"{cfg.out} holds an earlier run ({PARTIAL_NAME}); use --resume or a new --out")
        if cfg.save and not cfg.resume and os.path.exists(os.path.join(cfg.out, INDEX_NAME)):
            raise FileExistsError(f"{cfg.out} already holds a dataset ({INDEX_NAME}); use a new --out")
    partial = os.path.join(cfg.out, PARTIAL_NAME) if cfg.out else None
    eps: list[dict] = []
    if cfg.resume and partial and os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            eps = [json.loads(line) for line in f if line.strip()]
        done_ids = {e["episode"] for e in eps}
        jobs_todo = [j for j in jobs if j.episode not in done_ids]
        print(f"[collect] resume: {len(eps)} episodes already done, {len(jobs_todo)} to go", flush=True)
    else:
        jobs_todo = jobs
    ctx = mp.get_context("spawn")
    results, stop_all, job_q = ctx.Queue(), ctx.Event(), ctx.Queue()
    for j in jobs_todo:
        job_q.put(j)
    job_q.cancel_join_thread()   # leftover jobs after a stop must not block exit
    n_workers = min(cfg.workers, len(jobs_todo))
    stops = [ctx.Event() for _ in range(n_workers)]
    procs = [ctx.Process(target=_worker, args=(w, job_q, cfg, env_cfg, results, stop_all, stops[w]),
                         daemon=True) for w in range(n_workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    errors: list[str] = []
    alive = set(range(n_workers))
    min_avail = psutil.virtual_memory().available
    low_memory, reductions = False, 0
    last_print, last_reduce = 0.0, 0.0
    pf = open(partial, "a", encoding="utf-8") if partial else None
    while alive:
        try:
            kind, wid, payload = results.get(timeout=MEMORY_POLL_S)
        except queue.Empty:
            kind = None
        avail = psutil.virtual_memory().available
        min_avail = min(min_avail, avail)
        if avail < MIN_AVAILABLE_GIB * 2 ** 30 and time.time() - last_reduce > REDUCE_GRACE_S:
            last_reduce = time.time()
            live = sorted(w for w in alive if not stops[w].is_set())
            if len(live) > 1:
                stops[live[-1]].set()       # reduce: one engine fewer after its episode
                reductions += 1
                print(f"[collect] available {avail / 2**30:.2f} GiB < {MIN_AVAILABLE_GIB} GiB: "
                      f"stopping worker {live[-1]} ({len(live) - 1} left)", flush=True)
            elif not stop_all.is_set():
                low_memory = True
                stop_all.set()
                print(f"[collect] available {avail / 2**30:.2f} GiB < {MIN_AVAILABLE_GIB} GiB with one "
                      f"worker: stopping (rerun with --resume)", flush=True)
        if kind == "episode":
            eps.append(payload)
            if pf:
                pf.write(json.dumps(payload) + "\n")
                pf.flush()
        elif kind == "error":
            errors.append(f"worker {wid}: {payload}")
            print(f"[collect] worker {wid} error: {payload}", flush=True)
        elif kind == "done":
            alive.discard(wid)
        if time.time() - last_print > 30 and eps:
            last_print = time.time()
            ok = sum(e["success"] for e in eps)
            print(f"[collect] {len(eps)}/{len(jobs)} episodes, success {ok}/{len(eps)}, "
                  f"{sum(e['length'] for e in eps)} decisions, {time.time() - t0:.0f}s, "
                  f"avail {avail / 2**30:.1f} GiB", flush=True)
    if pf:
        pf.close()
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    eps.sort(key=lambda e: e["episode"])
    if cfg.save:
        with open(os.path.join(cfg.out, INDEX_NAME), "w", encoding="utf-8") as f:
            for e in eps:
                if e.get("saved"):
                    row = {"file": e["file"], "map_id": e["map_id"], "stage": e["stage"],
                           "length": e["length"], "success": e["success"], "seed": e["seed"],
                           "noise": e["noise"]}
                    if cfg.student:
                        row.update({"student": os.path.abspath(cfg.student), "beta": cfg.beta,
                                    "student_decisions": e["student_decisions"],
                                    "unstick_labels": e["unstick_labels"]})
                    f.write(json.dumps(row) + "\n")
    summary = {
        "config": asdict(cfg), "env_config": env_cfg.name, "bundle": os.path.relpath(bundle),
        "jobs": len(jobs), "episodes_finished": len(eps), "wall_s": round(time.time() - t0, 1),
        "errors": errors, "stopped_low_memory": low_memory, "worker_reductions": reductions,
        "complete": len(eps) == len(jobs),
        "min_available_gib": round(min_avail / 2 ** 30, 2),
        "by_stage_noise": _stats(eps),
        "goal_mode_share_by_stage": {s: _mode_share([e for e in eps if e["stage"] == s])
                                     for s in sorted({e["stage"] for e in eps})},
        "decisions_saved": int(sum(e["length"] for e in eps if e.get("saved"))),
        "episodes_saved": sum(bool(e.get("saved")) for e in eps),
        "failures": [e for e in eps if not e["success"]],
    }
    if cfg.student:
        n = max(sum(e["length"] for e in eps), 1)
        summary["dagger"] = {
            "student": os.path.abspath(cfg.student), "beta": cfg.beta, "max_steps": cfg.max_steps,
            "student_argmax": cfg.student_argmax,
            "success_by_stage": {s: [sum(e["success"] for e in eps if e["stage"] == s),
                                     sum(1 for e in eps if e["stage"] == s)] for s in cfg.stages},
            "unstick_label_share": round(sum(e["unstick_labels"] for e in eps) / n, 4),
            "student_decision_share": round(sum(e["student_decisions"] for e in eps) / n, 4),
            "agree_all": round(sum(e["agree_all"] * e["length"] for e in eps) / n, 4),
            "capped": sum(e["capped"] for e in eps),
        }
        summary["failures"] = [{k: e[k] for k in ("map_id", "stage", "length", "anchor_reached", "anchors")}
                               for e in summary["failures"]]
    if cfg.out:
        with open(os.path.join(cfg.out, SUMMARY_NAME), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    return summary


def parse_args(argv: list[str] | None = None) -> CollectConfig:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    ap.add_argument("--episodes-per-map", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--clean-every", type=int, default=0,
                    help="every K-th episode of each map runs at noise 0 (0 = never)")
    ap.add_argument("--burst-max", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="measure only: write summary.json, no episodes")
    ap.add_argument("--map-ids", nargs="*", default=[])
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run in --out")
    ap.add_argument("--student", default=None, metavar="CKPT",
                    help="DAgger: this checkpoint drives (sampled actions), the teacher labels every step")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="DAgger: per-decision probability that the teacher's label is executed instead")
    ap.add_argument("--max-steps", type=int, default=0, help="cap on decisions per episode (0 = none)")
    ap.add_argument("--student-argmax", action="store_true", help="DAgger: student acts by argmax")
    ap.add_argument("--teacher", choices=("route", "explore"), default="route",
                    help="route: shortest path from the map file; explore: only what the player has seen")
    a = ap.parse_args(argv)
    if not 0.0 <= a.noise <= 1.0:
        ap.error("--noise must be in [0, 1]")
    if not 0.0 <= a.beta <= 1.0:
        ap.error("--beta must be in [0, 1]")
    if a.student and a.noise > 0:
        ap.error("--noise applies to teacher-driven collection only, not with --student")
    if (a.beta > 0 or a.student_argmax) and not a.student:
        ap.error("--beta / --student-argmax need --student")
    if a.max_steps < 0:
        ap.error("--max-steps must be >= 0")
    if a.burst_max < 1:
        ap.error("--burst-max must be >= 1")
    return CollectConfig(stages=tuple(a.stages), episodes_per_map=a.episodes_per_map, out=a.out,
                         noise=a.noise, clean_every=a.clean_every, burst_max=a.burst_max,
                         workers=a.workers, seed=a.seed, keep_failures=a.keep_failures,
                         save=not a.no_save, map_ids=tuple(a.map_ids), trace=a.trace,
                         resume=a.resume, student=a.student, beta=a.beta, max_steps=a.max_steps,
                         student_argmax=a.student_argmax, teacher=a.teacher)


def main(argv: list[str] | None = None) -> int:
    s = collect(parse_args(argv))
    print(json.dumps({k: v for k, v in s.items() if k not in ("failures", "config")}, indent=1))
    return 1 if s["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
