"""Training loop with the safety mechanisms of ADR 0007 ruling 7 and ADR 0008 Decision 5.

- Priority checked (not assumed) on every process at start.
- Checkpoints only on update boundaries. Ctrl-C sets a stop flag: a rollout in
  progress is abandoned (no weights changed), an update in progress is finished,
  then the checkpoint is written. A second Ctrl-C stops at once without saving,
  leaving the previous checkpoint intact. The time from request to stop is logged.
- A watcher thread checks memory every `memory_check_interval_s`: process tree
  above the cap, or system available memory below `min_available_bytes`, sets the
  stop flag; below `emergency_available_bytes` it also closes engines immediately.
- Optional cooldown pause between updates.
- ADR 0011 F4: the health monitor (train/health_monitor.py) is fed once per update;
  its level and event start go to metrics.jsonl and, at the Stop level, it requests a
  stop through the same StopController (save, then stop). It only reads logged
  numbers; it never reaches an observation or an action.
- ADR 0011 F3: every checkpoint written here is also kept under a numbered name.
- ADR 0011 F7: `recent_engine_return` (unshaped) is always written next to
  `recent_return` (shaped when shaping is on).
- Curriculum runs: `curriculum` (train/curriculum_vec_env.CurriculumVecEnv) adds its
  per-stage success and stage mix to metrics.jsonl, logs promotions, and its state is
  saved with every checkpoint. The health monitor is fed exactly as without it: every
  finished episode's success, all stages (and replays) mixed.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil

from configs.train_config import SafetyConfig

from . import process_control as PC
from .checkpoint import HEALTH_KEY, keep_numbered_copy, save_checkpoint
from .health_monitor import HealthMonitor
from .trainer import RolloutAbandoned, Trainer

CHECKPOINT_NAME = "checkpoint.pt"
METRICS_NAME = "metrics.jsonl"
EPISODES_NAME = "episodes.jsonl"
RECENT_WINDOW = 20  # episodes behind recent_return / recent_success / recent_length


def drain_episodes(trainer: Trainer, f: Any) -> list[dict]:
    """Append one JSON line per finished episode (ADR 0009 Decision 5 item 2) and flush,
    so the file survives an interrupt. Returns the records written."""
    recs, trainer.log.pending = trainer.log.pending, []
    for r in recs:
        f.write(json.dumps(r) + "\n")
    f.flush()
    return recs


def make_health_monitor(trainer: Trainer, blob: dict[str, Any] | None = None) -> HealthMonitor | None:
    """A monitor for this trainer's action heads, restored from a resumed checkpoint's
    blob when it carries one (else fresh). None when disabled in the run config."""
    if not trainer.run.health.enabled:
        return None
    mon = HealthMonitor(trainer.run.health, trainer.spec.head_sizes)
    if blob is not None and HEALTH_KEY in blob:
        mon.load_json(blob[HEALTH_KEY])
    return mon


def recent_episode_stats(trainer: Trainer) -> dict[str, float]:
    lg = trainer.log
    out = {"episodes_done": float(len(lg.returns))}
    if lg.returns:
        k = RECENT_WINDOW
        out["recent_return"] = float(sum(lg.returns[-k:]) / len(lg.returns[-k:]))
        out["recent_engine_return"] = float(sum(lg.engine_returns[-k:]) / len(lg.engine_returns[-k:]))
        out["recent_success"] = float(sum(lg.successes[-k:]) / len(lg.successes[-k:]))
        out["recent_length"] = float(sum(lg.lengths[-k:]) / len(lg.lengths[-k:]))
    return out


@dataclass
class RunSummary:
    stop_reason: str = ""
    updates_this_session: int = 0
    update_idx: int = 0
    checkpoints_written: int = 0
    max_tree_rss_bytes: int = 0
    min_available_bytes: int = 0
    stale_timeouts_total: int = 0
    stop_latency_s: float = 0.0
    priority: dict[str, Any] = field(default_factory=dict)
    numbered_checkpoints_written: int = 0
    numbered_checkpoint_bytes: int = 0      # size of the last numbered checkpoint
    numbered_checkpoints_removed: int = 0   # by the 1 GB thinning rule
    health_levels_reached: list[str] = field(default_factory=list)
    health_event_active_at_end: bool = False  # ADR 0011: a trainer problem if True
    health_stop_update: int | None = None


class StopController:
    """One stop flag shared by the signal handler and the memory watcher."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""
        self.requested_at = 0.0
        self.presses = 0

    def request(self, reason: str) -> None:
        if not self._event.is_set():
            self.reason, self.requested_at = reason, time.monotonic()
            self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def on_sigint(self, signum: int = signal.SIGINT, frame: Any = None) -> None:
        """First Ctrl-C: stop at the next update boundary. Second: stop now, no save."""
        self.presses += 1
        if self.presses >= 2:
            raise KeyboardInterrupt
        self.request("interrupt")


def memory_status(tree_rss: Callable[[], int] = PC.tree_rss_bytes,
                  available: Callable[[], int] = lambda: psutil.virtual_memory().available) -> tuple[int, int]:
    return int(tree_rss()), int(available())


class MemoryWatcher(threading.Thread):
    def __init__(self, safety: SafetyConfig, stop: StopController, emergency: Callable[[], None],
                 probe: Callable[[], tuple[int, int]] = memory_status) -> None:
        super().__init__(daemon=True, name="memory-watcher")
        self.safety, self.stop_ctl, self.emergency, self.probe = safety, stop, emergency, probe
        self._halt = threading.Event()
        self.max_rss = 0
        self.min_available: int | None = None
        self.emergency_fired = False

    def check_once(self) -> None:
        rss, avail = self.probe()
        self.max_rss = max(self.max_rss, rss)
        self.min_available = avail if self.min_available is None else min(self.min_available, avail)
        s = self.safety
        if avail < s.emergency_available_bytes and not self.emergency_fired:
            self.emergency_fired = True
            self.stop_ctl.request(f"emergency: available memory {avail} < {s.emergency_available_bytes}")
            self.emergency()
        elif rss > s.memory_cap_bytes:
            self.stop_ctl.request(f"memory cap exceeded ({rss} > {s.memory_cap_bytes} bytes)")
        elif avail < s.min_available_bytes:
            self.stop_ctl.request(f"system available memory low ({avail} < {s.min_available_bytes} bytes)")

    def run(self) -> None:
        while not self._halt.wait(self.safety.memory_check_interval_s):
            self.check_once()

    def halt(self) -> None:
        self._halt.set()


def check_priorities(pids: list[int], safety: SafetyConfig) -> dict[str, Any]:
    """Check (not assume) below-normal priority; set it directly on any process that
    does not have it, then check again."""
    if not safety.below_normal_priority:
        return {"enabled": False}
    before = PC.priority_report(pids)
    for pid, (_, ok) in before.items():
        if not ok:
            PC.set_below_normal(pid)
    after = PC.priority_report(pids)
    return {"enabled": True, "all_below_normal": all(ok for _, ok in after.values()),
            "set_directly": [f"{before[p][0]}:{p}" for p in before if not before[p][1]],
            "processes": {str(p): f"{n} below_normal={ok}" for p, (n, ok) in after.items()}}


def run_training(trainer: Trainer, run_dir: str, sensor_signature: tuple, safety: SafetyConfig,
                 max_updates: int, max_seconds: float, extra_pids: Callable[[], list[int]] = lambda: [],
                 emergency: Callable[[], None] = lambda: None, log: Callable[[str], None] = print,
                 stop: StopController | None = None, install_sigint: bool = True,
                 memory_probe: Callable[[], tuple[int, int]] = memory_status,
                 monitor: HealthMonitor | None | str = "auto", curriculum: Any = None) -> RunSummary:
    """`monitor="auto"` builds a fresh health monitor from the run config; pass one
    restored by `make_health_monitor(trainer, blob)` on resume, or None for none."""
    if monitor == "auto":
        monitor = make_health_monitor(trainer)
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, CHECKPOINT_NAME)
    summary = RunSummary()
    summary.priority = check_priorities([os.getpid(), *extra_pids()], safety)
    log(f"priority check: {summary.priority}")
    stop = stop or StopController()
    watcher = MemoryWatcher(safety, stop, emergency, memory_probe)
    watcher.check_once()
    watcher.start()
    old_handler = None
    if install_sigint and threading.current_thread() is threading.main_thread():
        old_handler = signal.signal(signal.SIGINT, stop.on_sigint)

    def save(reason: str) -> None:  # only ever called between updates
        save_checkpoint(ckpt, trainer, sensor_signature,
                        None if monitor is None else monitor.to_json(),
                        None if curriculum is None else curriculum.state_json())
        summary.checkpoints_written += 1
        log(f"checkpoint written ({reason}) at update {trainer.update_idx}: {ckpt}")
        if safety.keep_numbered_checkpoints:
            dst, size, removed = keep_numbered_copy(ckpt, run_dir, trainer.update_idx,
                                                    safety.numbered_checkpoint_budget_bytes)
            summary.numbered_checkpoints_written += 1
            summary.numbered_checkpoint_bytes = size
            summary.numbered_checkpoints_removed += len(removed)
            log(f"numbered checkpoint {dst} ({size} bytes); thinned {len(removed)}")

    t_start = last_ckpt = time.monotonic()
    try:
        with open(os.path.join(run_dir, METRICS_NAME), "a", encoding="utf-8") as mf, \
                open(os.path.join(run_dir, EPISODES_NAME), "a", encoding="utf-8") as ef:
            try:
                while True:
                    if stop.is_set():
                        break
                    try:
                        stats = trainer.train_iteration(stop.is_set)
                    except RolloutAbandoned:
                        break  # weights unchanged since the last boundary
                    except (RuntimeError, EOFError, BrokenPipeError):
                        if watcher.emergency_fired:
                            break  # engines were closed under us; weights still on a boundary
                        raise
                    recs = drain_episodes(trainer, ef)
                    summary.updates_this_session += 1
                    rss = PC.tree_rss_bytes()
                    summary.max_tree_rss_bytes = max(summary.max_tree_rss_bytes, rss)
                    stats["tree_rss_bytes"] = float(rss)
                    stats["available_bytes"] = float(psutil.virtual_memory().available)
                    stats.update(recent_episode_stats(trainer))
                    if curriculum is not None:
                        stats.update(curriculum.metrics())
                        for p in curriculum.drain_promotions():
                            log(f"curriculum promotion {p['from']} -> {p['to']} at recorded episode "
                                f"{p['episode']} (window success {p['rate']:.3f}), update {trainer.update_idx}")
                    reading = None
                    if monitor is not None:
                        ents = [stats[f"entropy_head{h}"] for h in range(len(trainer.spec.head_sizes))]
                        reading = monitor.observe(trainer.update_idx, ents, [r["success"] for r in recs])
                        stats.update(reading.metrics(trainer.run.health.window_episodes))
                    mf.write(json.dumps(stats) + "\n")
                    mf.flush()
                    log(f"update {trainer.update_idx}: kl {stats['approx_kl']:.4f} "
                        f"ev {stats['explained_variance']:.3f} scale {stats['return_scale']:.3f} "
                        f"stale {stats['stale_timeouts']:.0f} rss {rss / 2**20:.0f} MiB "
                        f"sample {stats['sample_dps']:.0f}/s update {stats['update_dps']:.0f}/s")
                    if stats.get("value_warmup_ended"):
                        log(f"value warm-up ended after update {trainer.update_idx}: policy and trunk "
                            f"unfrozen, policy-gradient and entropy terms back on")
                    if reading is not None and reading.level != "none":
                        heads = "+".join(reading.warn_heads) or "-"
                        log(f"health {reading.level} at update {reading.update} "
                            f"(event since {reading.event_start}; warn heads {heads})")
                    now = time.monotonic()
                    if reading is not None and reading.stop and trainer.run.health.stop_enabled:
                        summary.health_stop_update = reading.update
                        stop.request(f"health: collapse event active since update {reading.event_start}, "
                                     f"{trainer.run.health.stop_after_updates} updates (ADR 0011 stop level)")
                    elif summary.updates_this_session >= max_updates:
                        stop.request("max_updates")
                    elif now - t_start >= max_seconds:
                        stop.request("max_seconds")
                    elif (safety.checkpoint_every_updates > 0
                          and trainer.update_idx % safety.checkpoint_every_updates == 0):
                        save(f"every {safety.checkpoint_every_updates} updates")
                        last_ckpt = now
                    elif now - last_ckpt >= safety.checkpoint_interval_s:
                        save("periodic")
                        last_ckpt = now
                    if not stop.is_set() and safety.cooldown_s > 0:
                        time.sleep(safety.cooldown_s)
            finally:
                drain_episodes(trainer, ef)  # episodes from an abandoned rollout are real too
        summary.stop_reason = stop.reason
        save(stop.reason)
        summary.stop_latency_s = time.monotonic() - stop.requested_at
        log(f"stopped ({stop.reason}) {summary.stop_latency_s:.2f}s after the request")
    except KeyboardInterrupt:
        summary.stop_reason = "second interrupt: stopped without saving"
    finally:
        watcher.halt()
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    summary.max_tree_rss_bytes = max(summary.max_tree_rss_bytes, watcher.max_rss)
    summary.min_available_bytes = int(watcher.min_available or 0)
    summary.update_idx = trainer.update_idx
    summary.stale_timeouts_total = trainer.stale_timeouts_total
    if monitor is not None:
        summary.health_levels_reached = sorted(monitor.levels_reached)
        summary.health_event_active_at_end = monitor.event_active
    return summary
