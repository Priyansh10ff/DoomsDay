"""Engine-instance layout guard (ADR 0008 ruling 12 / Decision 5). No torch import.

More than one ViZDoom engine instance needs `allow_multi_instance == N`, where N
is the exact total number of engine instances. The guard counts ENGINE instances
(env factories marked `is_engine_instance`), so several engines in one process
count too; toy envs do not. It is called where vector envs are built
(SubprocVecEnv, SyncVecEnv), not only by the command-line parser.

The flag prevents accidents; it is not consent (only trainer-ops passes it, for
a layout the user approved).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from configs.train_config import SafetyConfig

ENGINE_MARKER = "is_engine_instance"


class LayoutRefused(RuntimeError):
    pass


def count_engine_instances(env_fns: Iterable[Any]) -> int:
    return sum(1 for fn in env_fns if bool(getattr(fn, ENGINE_MARKER, False)))


def estimated_bytes(workers: int, engines: int, torch_threads: int, safety: SafetyConfig) -> int:
    return (safety.est_trainer_bytes + workers * safety.est_worker_bytes
            + engines * safety.est_engine_bytes)


def plan_text(workers: int, engines: int, torch_threads: int, safety: SafetyConfig) -> str:
    est = estimated_bytes(workers, engines, torch_threads, safety)
    return (f"planned layout: {engines} engine instance(s) in {workers} worker process(es), "
            f"{torch_threads} torch thread(s), estimated memory {est / 2**20:.0f} MiB")


def check_layout(engines: int, allow_multi_instance: int | None, workers: int, torch_threads: int,
                 safety: SafetyConfig) -> None:
    """Raise LayoutRefused unless engines <= 1 or allow_multi_instance == engines."""
    if engines <= 1:
        return
    if allow_multi_instance is None or int(allow_multi_instance) != engines:
        got = "absent" if allow_multi_instance is None else f"{allow_multi_instance}"
        raise LayoutRefused(f"refusing {engines} engine instances: --allow-multi-instance is {got}, "
                            f"must equal {engines}. {plan_text(workers, engines, torch_threads, safety)}")
