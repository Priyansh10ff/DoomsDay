"""Process plumbing for laptop safety (ADR 0004, ADR 0007 ruling 7). No torch import.

Below-normal priority, process-tree memory, engine-process detection and cleanup.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence

import psutil

# Windows: BELOW_NORMAL_PRIORITY_CLASS. POSIX: a positive nice value.
POSIX_BELOW_NORMAL_NICE = 10


def _below_normal_value() -> int:
    return psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else POSIX_BELOW_NORMAL_NICE


def set_below_normal(pid: int | None = None) -> None:
    psutil.Process(pid or os.getpid()).nice(_below_normal_value())


def is_below_normal(pid: int) -> bool:
    v = psutil.Process(pid).nice()
    if sys.platform == "win32":
        return v in (psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.IDLE_PRIORITY_CLASS)
    return v >= POSIX_BELOW_NORMAL_NICE


def descendants(pid: int | None = None) -> list[psutil.Process]:
    try:
        return psutil.Process(pid or os.getpid()).children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def priority_report(pids: Iterable[int]) -> dict[int, tuple[str, bool]]:
    """{pid: (name, is_below_normal)} for every live pid given."""
    out: dict[int, tuple[str, bool]] = {}
    for pid in pids:
        try:
            out[pid] = (psutil.Process(pid).name(), is_below_normal(pid))
        except psutil.NoSuchProcess:
            continue
    return out


def lower_tree_priority(pid: int | None = None) -> None:
    """Set below-normal on a process and all its descendants (direct, not assumed)."""
    for p in [psutil.Process(pid or os.getpid()), *descendants(pid)]:
        try:
            p.nice(_below_normal_value())
        except psutil.NoSuchProcess:
            continue


def tree_rss_bytes(pid: int | None = None) -> int:
    total = 0
    for p in [psutil.Process(pid or os.getpid()), *descendants(pid)]:
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def engine_processes(names: Sequence[str]) -> list[psutil.Process]:
    want = {n.lower() for n in names}
    found = []
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info.get("name") or "").lower() in want:
                found.append(p)
        except psutil.NoSuchProcess:
            continue
    return found


def kill_processes(procs: Iterable[psutil.Process], timeout_s: float) -> None:
    procs = list(procs)
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=timeout_s)
