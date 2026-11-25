"""Multi-process vector env (toy env, at most 2 workers, no ViZDoom) and the safe run loop."""
import dataclasses
import json
import os
import subprocess
import sys
import time
from functools import partial

import numpy as np
import psutil
import pytest
import torch

from a0_helpers import ROOT
from configs.train_config import PPOConfig, RunConfig, SafetyConfig
from train import process_control as PC
from train.checkpoint import load_checkpoint, resume
from train.run_loop import (CHECKPOINT_NAME, METRICS_NAME, MemoryWatcher, StopController,
                            run_training)
from train.subproc_vec_env import SubprocVecEnv
from train.toy_memory import ToyConfig, ToyMemoryEnv, toy_model_config
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SAFETY = SafetyConfig(worker_join_timeout_s=5.0)
SIG = (("toy", True),)
GIB = 2**30


def toy_fns(n: int) -> list:
    return [partial(ToyMemoryEnv, ToyConfig(answer_step=3)) for _ in range(n)]


def quiet(msg: str) -> None:
    pass


def _toy_trainer(safety: SafetyConfig, seed: int = 0) -> tuple[SubprocVecEnv, Trainer]:
    venv = SubprocVecEnv([toy_fns(2), toy_fns(2)], safety)
    run = dataclasses.replace(RunConfig(), seed=seed, num_workers=2, envs_per_worker=2,
                              ppo=PPOConfig(rollout_len=8, num_minibatches=2, epochs=1), safety=safety)
    return venv, Trainer(venv, toy_model_config(True), run)


def _run(tr: Trainer, d: str, safety: SafetyConfig = SAFETY, **kw) -> object:
    kw.setdefault("max_updates", 50)
    return run_training(tr, d, SIG, safety, max_seconds=60, log=quiet, install_sigint=False, **kw)


# ---------------------------------------------------------------- vector env
def test_subproc_matches_sync_and_workers_are_safe() -> None:
    venv = SubprocVecEnv([toy_fns(2), toy_fns(1)], SAFETY)
    try:
        ref = SyncVecEnv(toy_fns(3))
        o1, _ = venv.reset(seed=5)
        o2, _ = ref.reset(seed=5)
        for k in o1:
            assert np.array_equal(o1[k], o2[k])
        rng = np.random.default_rng(0)
        for _ in range(9):  # crosses episode ends (episode length 4)
            a = rng.integers(0, 2, (3, 2))
            s1, s2 = venv.step(a), ref.step(a)
            for k in s1.obs:
                assert np.array_equal(s1.obs[k], s2.obs[k])
            assert np.array_equal(s1.reward, s2.reward) and np.array_equal(s1.terminated, s2.terminated)
            assert np.array_equal(s1.final_obs_fresh, s2.final_obs_fresh)
        assert [d["torch_loaded"] for d in venv.query()] == [False, False]
        PC.set_below_normal()  # as the entry point does before spawning
        assert all(PC.is_below_normal(pid) for pid in venv.worker_pids)
        pids = venv.worker_pids
    finally:
        venv.close()
    venv.close()  # idempotent
    assert not any(psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE for p in pids)


def test_close_kills_unresponsive_worker() -> None:
    venv = SubprocVecEnv([toy_fns(1)], SAFETY)
    pid = venv.worker_pids[0]
    psutil.Process(pid).suspend()  # cannot answer "close": must be terminated
    venv.close()
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_daemon_variant_selectable_nondaemon_default() -> None:
    assert SafetyConfig().worker_daemon is False
    for daemon in (True, False):
        venv = SubprocVecEnv([toy_fns(1)], dataclasses.replace(SAFETY, worker_daemon=daemon))
        try:
            assert venv._procs[0].daemon is daemon
        finally:
            venv.close()


PARENT_SCRIPT = """
import sys, time
sys.path.insert(0, {root!r})
from functools import partial
from configs.train_config import SafetyConfig
from train.subproc_vec_env import SubprocVecEnv
from train.toy_memory import ToyMemoryEnv, ToyConfig
if __name__ == "__main__":
    v = SubprocVecEnv([[partial(ToyMemoryEnv, ToyConfig(answer_step=3))]],
                      SafetyConfig(worker_daemon=False, parent_poll_interval_s=0.2))
    print(v.worker_pids[0], flush=True)
    time.sleep(60)
"""


def test_nondaemon_worker_exits_when_parent_is_hard_killed(tmp_path: "os.PathLike[str]") -> None:
    """Toy env only (no engine): the parent is killed hard; the worker must notice and exit."""
    script = os.path.join(str(tmp_path), "parent.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(PARENT_SCRIPT.format(root=ROOT))
    parent = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, text=True)
    worker_pid = None
    try:
        worker_pid = int(parent.stdout.readline().strip())
        assert psutil.pid_exists(worker_pid)
        psutil.Process(parent.pid).kill()
        t0 = time.monotonic()
        while psutil.pid_exists(worker_pid) and time.monotonic() - t0 < 10:
            time.sleep(0.1)
        alive = psutil.pid_exists(worker_pid) and psutil.Process(worker_pid).status() != psutil.STATUS_ZOMBIE
        assert not alive, "worker outlived its hard-killed parent"
    finally:
        if parent.poll() is None:
            parent.kill()
        if worker_pid is not None and psutil.pid_exists(worker_pid):
            try:
                psutil.Process(worker_pid).kill()
            except psutil.NoSuchProcess:
                pass


# ---------------------------------------------------------------- run loop
def test_run_loop_checkpoint_resume_and_priority(tmp_path: "os.PathLike[str]") -> None:
    PC.set_below_normal()
    d = str(tmp_path)
    venv, tr = _toy_trainer(SAFETY)
    try:
        s = _run(tr, d, max_updates=2, extra_pids=venv.process_pids)
    finally:
        venv.close()
    assert s.stop_reason == "max_updates" and s.update_idx == 2 and s.checkpoints_written == 1
    assert s.priority["all_below_normal"] and s.max_tree_rss_bytes > 0 and s.min_available_bytes > 0
    lines = open(os.path.join(d, METRICS_NAME), encoding="utf-8").read().splitlines()
    assert len(lines) == 2 and "available_bytes" in json.loads(lines[0])
    venv2, tr2 = _toy_trainer(SAFETY)
    try:
        resume(tr2, os.path.join(d, CHECKPOINT_NAME), SIG)
        s2 = _run(tr2, d, max_updates=1)
    finally:
        venv2.close()
    assert s2.update_idx == 3 and tr2.session == 1


def test_memory_cap_stops_and_saves_at_boundary(tmp_path: "os.PathLike[str]") -> None:
    tiny = dataclasses.replace(SAFETY, memory_cap_bytes=1)
    venv, tr = _toy_trainer(tiny)
    try:
        s = _run(tr, str(tmp_path), tiny)
    finally:
        venv.close()
    assert s.stop_reason.startswith("memory cap") and s.updates_this_session == 0
    assert os.path.exists(os.path.join(str(tmp_path), CHECKPOINT_NAME))


@pytest.mark.parametrize("avail,expect,fired", [(int(1.0 * GIB), "system available memory low", False),
                                                (int(0.5 * GIB), "emergency", True)])
def test_low_available_memory_stops_and_emergency_closes_engines(tmp_path: "os.PathLike[str]", avail: int,
                                                                 expect: str, fired: bool) -> None:
    venv, tr = _toy_trainer(SAFETY)
    called = {"n": 0}
    try:
        s = _run(tr, str(tmp_path), emergency=lambda: called.__setitem__("n", called["n"] + 1),
                 memory_probe=lambda: (10, avail))
    finally:
        venv.close()
    assert s.stop_reason.startswith(expect) and (called["n"] == 1) == fired


def test_memory_watcher_runs_on_a_clock() -> None:
    assert SafetyConfig().memory_check_interval_s == 5.0
    fast = dataclasses.replace(SAFETY, memory_check_interval_s=0.05)
    readings = iter([(10, 8 * GIB)] * 3 + [(10 * GIB, 8 * GIB)] * 10000)
    stop = StopController()
    w = MemoryWatcher(fast, stop, lambda: None, probe=lambda: next(readings))
    w.start()
    t0 = time.monotonic()
    while not stop.is_set() and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    w.halt()
    assert stop.is_set() and stop.reason.startswith("memory cap")


def test_interrupt_during_update_finishes_update_then_saves(tmp_path: "os.PathLike[str]") -> None:
    venv, tr = _toy_trainer(SAFETY)
    stop = StopController()
    inner = tr.update

    def update_with_interrupt(ro):
        if tr.update_idx == 1:
            stop.on_sigint()          # Ctrl-C arrives while the 2nd update runs
        return inner(ro)

    tr.update = update_with_interrupt  # type: ignore[method-assign]
    try:
        s = _run(tr, str(tmp_path), stop=stop)
    finally:
        venv.close()
    assert s.stop_reason == "interrupt" and s.update_idx == 2 and s.checkpoints_written == 1
    saved = load_checkpoint(os.path.join(str(tmp_path), CHECKPOINT_NAME))["trainer"]
    assert saved["update_idx"] == 2
    for v, p in zip(saved["model"].values(), tr.model.state_dict().values()):
        assert torch.equal(v, p)


def test_interrupt_during_rollout_abandons_it_and_saves_boundary(tmp_path: "os.PathLike[str]") -> None:
    venv, tr = _toy_trainer(SAFETY)
    stop = StopController()
    inner_step = tr.env.step
    count = {"n": 0}

    def step_with_interrupt(actions):
        count["n"] += 1
        if count["n"] == 8 + 3:   # 3 steps into the second rollout (rollout_len 8)
            stop.on_sigint()
        return inner_step(actions)

    tr.env.step = step_with_interrupt  # type: ignore[method-assign]
    try:
        s = _run(tr, str(tmp_path), stop=stop)
    finally:
        venv.close()
    assert s.stop_reason == "interrupt" and s.update_idx == 1 and s.updates_this_session == 1
    saved = load_checkpoint(os.path.join(str(tmp_path), CHECKPOINT_NAME))["trainer"]
    assert saved["update_idx"] == 1
    for v, p in zip(saved["model"].values(), tr.model.state_dict().values()):
        assert torch.equal(v, p)


def test_second_interrupt_stops_without_saving(tmp_path: "os.PathLike[str]") -> None:
    venv, tr = _toy_trainer(SAFETY)
    stop = StopController()
    inner = tr.update

    def double_press(ro):
        stop.on_sigint()
        stop.on_sigint()  # raises KeyboardInterrupt
        return inner(ro)

    tr.update = double_press  # type: ignore[method-assign]
    try:
        s = _run(tr, str(tmp_path), stop=stop)
    finally:
        venv.close()
    assert s.stop_reason.startswith("second interrupt") and s.checkpoints_written == 0
    assert not os.path.exists(os.path.join(str(tmp_path), CHECKPOINT_NAME))


def test_cooldown_field_pauses(tmp_path: "os.PathLike[str]") -> None:
    cool = dataclasses.replace(SAFETY, cooldown_s=0.5)
    venv, tr = _toy_trainer(cool)
    try:
        t0 = time.monotonic()
        _run(tr, str(tmp_path), cool, max_updates=2)
        assert time.monotonic() - t0 >= 0.5
    finally:
        venv.close()
