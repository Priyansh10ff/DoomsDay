"""Multi-process vector env (ADR 0004 layout, ADR 0007 ruling 7 item 1). No torch import.

Each worker process owns `envs_per_worker` envs and steps them in turn. Workers
import numpy/gymnasium/psutil and whatever the env factory imports (vizdoom),
never torch. Shutdown: `close()` is idempotent, registered with atexit, and after
asking workers to exit it terminates any worker or descendant (engine) process
still alive. Workers ignore Ctrl-C so the main process alone decides when to stop.
Workers are non-daemon by default and poll for a dead parent, closing their games
if it is gone; `SafetyConfig.worker_daemon=True` selects the daemon variant (ADR
0008 point 6 decides between them by trainer-ops' hard-kill test).
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import signal
import traceback
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np
import psutil

from configs.train_config import SafetyConfig

from . import process_control as PC
from .layout_guard import check_layout, count_engine_instances
from .vec_env import (ObsBatch, OneStep, ResetOptions, VecStep, collate, env_seeds, per_env, reset_one,
                      stack_obs, step_autoreset)

EnvFn = Callable[[], gym.Env]


def _parent_alive() -> bool:
    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def _wait_for_command(conn: Connection, poll_s: float) -> tuple[str, Any] | None:
    """Block until a command arrives; None if the parent process has died."""
    while not conn.poll(poll_s):
        if not _parent_alive():
            return None
    return conn.recv()


def _worker(conn: Connection, env_fns: Sequence[EnvFn], below_normal: bool, poll_s: float) -> None:
    """Also polls for a dead parent and then closes its games (ADR 0008 point 6), so a
    hard-killed trainer does not orphan engines even with non-daemon workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    envs: list[gym.Env] = []
    try:
        if below_normal:
            PC.set_below_normal()
        envs = [fn() for fn in env_fns]
        if below_normal:
            PC.lower_tree_priority()  # engine children set directly, then checked by main
        conn.send(("ready", (envs[0].observation_space, envs[0].action_space)))
        while True:
            msg = _wait_for_command(conn, poll_s)
            if msg is None:
                break  # parent gone: games are closed in `finally`
            cmd, data = msg
            if cmd == "step":  # data: (actions, reset options), one entry per env
                acts, opts = data
                conn.send(("ok", [step_autoreset(e, acts[i], opts[i]) for i, e in enumerate(envs)]))
            elif cmd == "reset":  # data: (seeds, options), one entry per env
                seeds, opts = data
                conn.send(("ok", [reset_one(e, s, o) for e, s, o in zip(envs, seeds, opts)]))
            elif cmd == "query":
                import sys
                conn.send(("ok", {"pid": os.getpid(), "torch_loaded": "torch" in sys.modules}))
            elif cmd == "close":
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
        try:
            conn.send(("closed", None))
        except Exception:
            pass
        conn.close()


class SubprocVecEnv:
    def __init__(self, env_fns_per_worker: Sequence[Sequence[EnvFn]], safety: SafetyConfig,
                 allow_multi_instance: int | None = None, torch_threads: int = 0) -> None:
        assert len(env_fns_per_worker) > 0 and all(len(f) > 0 for f in env_fns_per_worker), \
            "need at least one worker with at least one env"
        # ADR 0008 Decision 5: refuse >1 engine instance unless the flag states the count.
        check_layout(count_engine_instances(f for fns in env_fns_per_worker for f in fns),
                     allow_multi_instance, len(env_fns_per_worker), torch_threads, safety)
        self.safety = safety
        self._closed = False
        self._sizes = [len(f) for f in env_fns_per_worker]
        self.num_envs = sum(self._sizes)
        ctx = mp.get_context(safety.worker_start_method)
        self._conns: list[Connection] = []
        self._procs: list[mp.process.BaseProcess] = []
        self._seen_descendants: dict[int, psutil.Process] = {}
        atexit.register(self.close)
        try:
            for fns in env_fns_per_worker:
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, daemon=safety.worker_daemon,
                                args=(child, list(fns), safety.below_normal_priority, safety.parent_poll_interval_s))
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)
            spaces = [self._recv(c) for c in self._conns]
            self.single_observation_space, self.single_action_space = spaces[0]
            self._remember_descendants()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------ plumbing
    def _recv(self, conn: Connection) -> Any:
        tag, data = conn.recv()
        if tag == "error":
            raise RuntimeError(f"vec env worker failed:\n{data}")
        if tag == "closed":
            raise RuntimeError("vec env worker exited unexpectedly")
        return data

    def _chunks(self, items: Sequence[Any]) -> list[list[Any]]:
        out, i = [], 0
        for n in self._sizes:
            out.append(list(items[i:i + n]))
            i += n
        return out

    def _remember_descendants(self) -> None:
        """Record every process under our workers, so close() can kill survivors."""
        for wp in self.worker_pids:
            for p in PC.descendants(wp):
                self._seen_descendants[p.pid] = p

    @property
    def worker_pids(self) -> list[int]:
        return [int(p.pid) for p in self._procs if p.pid is not None]

    def process_pids(self) -> list[int]:
        """Workers and all their descendants (engine processes)."""
        pids: list[int] = []
        for wp in self.worker_pids:
            pids.append(wp)
            pids += [c.pid for c in PC.descendants(wp)]
        return pids

    def kill_engines(self) -> None:
        """Emergency (ADR 0008 Decision 5): kill every process under the workers now,
        without a safe point. Safe to call from another thread."""
        self._remember_descendants()
        PC.kill_processes([p for p in self._seen_descendants.values() if p.is_running()],
                          self.safety.worker_join_timeout_s)

    def query(self) -> list[dict[str, Any]]:
        for c in self._conns:
            c.send(("query", None))
        return [self._recv(c) for c in self._conns]

    # ------------------------------------------------------------ VecEnv API
    def reset(self, seed: int, options: Sequence[ResetOptions] | None = None) -> tuple[ObsBatch, list[dict[str, Any]]]:
        opts = self._chunks(per_env(options, self.num_envs, "reset options"))
        for c, seeds, o in zip(self._conns, self._chunks(env_seeds(seed, self.num_envs)), opts):
            c.send(("reset", (seeds, o)))
        results = [r for c in self._conns for r in self._recv(c)]
        return stack_obs([r[0] for r in results]), [r[1] for r in results]

    def step(self, actions: np.ndarray, reset_options: Sequence[ResetOptions] | None = None) -> VecStep:
        assert actions.shape[0] == self.num_envs, f"actions expected ({self.num_envs}, heads), got {actions.shape}"
        opts = self._chunks(per_env(reset_options, self.num_envs, "reset_options"))
        for c, chunk, o in zip(self._conns, self._chunks(list(actions)), opts):
            c.send(("step", (chunk, o)))
        steps: list[OneStep] = [s for c in self._conns for s in self._recv(c)]
        return collate(steps)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._remember_descendants()
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(self.safety.worker_join_timeout_s)
        alive = [psutil.Process(p.pid) for p in self._procs if p.is_alive() and p.pid is not None]
        PC.kill_processes(alive, self.safety.worker_join_timeout_s)
        leftovers = [p for p in self._seen_descendants.values() if p.is_running()]
        PC.kill_processes(leftovers, self.safety.worker_join_timeout_s)
        for c in self._conns:
            try:
                c.close()
            except Exception:
                pass
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
