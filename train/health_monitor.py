"""Training-health monitor of ADR 0011 Decision 2 (fix request F4).

Fed once per PPO update with the six per-head policy entropies of that update and the
success flags of the episodes that finished in it. It classifies the update as one of

    none < watch < warning < event < stop      (one `level`; the highest that holds)

and also reports each condition separately:

  watch    summed-entropy mean over the last `watch_mean_updates` updates below
           `watch_entropy_fraction` x its maximum, while success is below
           `watch_success_below`;
  warning  any head's entropy mean over the last `warn_mean_updates` updates below
           `warn_head_fraction` x ln(option count);
  event    warning and (success at least `event_drop_from_best` below the run best,
           or success below p0 + `event_margin_over_p0`);
  stop     an event (a maximal run of consecutive event updates) still active
           `stop_after_updates` updates after it began.

Readings pinned by the coordinator (2026-11-25), matching the E0 forensics:
  * full windows only: "success" is the share of successes among the trailing
    `window_episodes` finished episodes and is undefined (every condition that needs
    it is false) until that many exist; a rolling entropy mean is undefined until that
    many updates exist;
  * head maxima are exact ln(k); "below" is strict;
  * success thresholds are compared exactly (fractions over the window), never as floats;
  * run best is the highest full-window success so far, the current update included.

Training-side only. It reads numbers the trainer already logs and can only ask the run
loop to stop; it is never imported by `agent/` and never reaches an observation or an
action. Stdlib only (no torch), so offline log analysis can use it as is.
"""
from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from configs.train_config import HealthMonitorConfig

LEVELS: tuple[str, ...] = ("none", "watch", "warning", "event", "stop")


def exact_fraction(x: float) -> Fraction:
    """The decimal a config float was written as (0.15 -> 3/20), not its binary value."""
    return Fraction(repr(float(x)))


class RollingMean:
    """Mean of the last `n` values; None until `n` values exist (full windows only).
    The running sum is updated by adding the new value, then subtracting the value that
    left the window, so the float result is reproducible from the log alone."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"window must be >= 1, got {n}")
        self.n = n
        self.values: deque[float] = deque()
        self.total = 0.0

    def push(self, x: float) -> float | None:
        self.total += float(x)
        self.values.append(float(x))
        if len(self.values) > self.n:
            self.total -= self.values.popleft()
        return self.total / self.n if len(self.values) == self.n else None

    def state_dict(self) -> dict[str, Any]:
        return {"values": list(self.values), "total": self.total}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.values, self.total = deque(float(v) for v in d["values"]), float(d["total"])


@dataclass(frozen=True)
class HealthReading:
    update: int
    level: str
    watch: bool
    warning: bool
    warn_heads: tuple[str, ...]
    drop: bool
    low: bool
    event: bool
    event_start: int | None          # update at which the active event began
    stop: bool
    success_count: int | None        # successes among the trailing window, None before it is full
    run_best_count: int | None
    entropy_sum: float
    entropy_sum_mean: float | None
    head_means: tuple[float | None, ...]

    def metrics(self, window: int) -> dict[str, Any]:
        """The fields written to metrics.jsonl."""
        return {
            "health_level": self.level,
            "health_event_start": self.event_start,
            "health_watch": self.watch, "health_warning": self.warning, "health_event": self.event,
            "health_warn_heads": "+".join(self.warn_heads),
            "health_success_window": None if self.success_count is None else self.success_count / window,
            "health_run_best": None if self.run_best_count is None else self.run_best_count / window,
            "health_entropy_sum_mean": self.entropy_sum_mean,
        }


class HealthMonitor:
    def __init__(self, cfg: HealthMonitorConfig, head_sizes: Sequence[int]) -> None:
        if not head_sizes or any(int(k) < 2 for k in head_sizes):
            raise ValueError(f"every head needs at least 2 options, got {tuple(head_sizes)}")
        if cfg.window_episodes < 1 or cfg.stop_after_updates < 1 or cfg.p0_episodes < 1:
            raise ValueError("window_episodes, stop_after_updates and p0_episodes must be >= 1")
        self.cfg = cfg
        self.head_sizes = tuple(int(k) for k in head_sizes)
        self.head_max = [math.log(k) for k in self.head_sizes]
        self.sum_max = sum(self.head_max)
        self.names = tuple(cfg.head_names[h] if h < len(cfg.head_names) else f"head{h}"
                           for h in range(len(self.head_sizes)))
        w = cfg.window_episodes
        # exact success lines, as counts over the window
        self._watch_line = exact_fraction(cfg.watch_success_below)
        self._drop = exact_fraction(cfg.event_drop_from_best)
        self._low_line = Fraction(cfg.p0_successes, cfg.p0_episodes) + exact_fraction(cfg.event_margin_over_p0)
        self._w = w
        self.successes: deque[bool] = deque(maxlen=w)
        self.success_count = 0
        self.run_best: int | None = None
        self.sum_mean = RollingMean(cfg.watch_mean_updates)
        self.head_mean = [RollingMean(cfg.warn_mean_updates) for _ in self.head_sizes]
        self.event_start: int | None = None
        self.stop_fired_at: int | None = None
        self.levels_reached: set[str] = set()

    # ------------------------------------------------------------ feeding
    def _push_success(self, s: bool) -> None:
        if len(self.successes) == self._w:
            self.success_count -= int(self.successes[0])
        self.successes.append(bool(s))
        self.success_count += int(bool(s))

    def observe(self, update: int, head_entropies: Sequence[float], new_successes: Iterable[bool]) -> HealthReading:
        assert len(head_entropies) == len(self.head_sizes), \
            f"expected {len(self.head_sizes)} head entropies, got {len(head_entropies)}"
        for s in new_successes:
            self._push_success(s)
        cfg = self.cfg
        ent_sum = sum(float(e) for e in head_entropies)
        sum_mean = self.sum_mean.push(ent_sum)
        head_means = tuple(m.push(e) for m, e in zip(self.head_mean, head_entropies))

        full = len(self.successes) == self._w
        count = self.success_count if full else None
        if count is not None and (self.run_best is None or count > self.run_best):
            self.run_best = count
        succ = None if count is None else Fraction(count, self._w)
        best = None if self.run_best is None else Fraction(self.run_best, self._w)

        watch = (sum_mean is not None and succ is not None
                 and sum_mean < cfg.watch_entropy_fraction * self.sum_max and succ < self._watch_line)
        warn = tuple(self.names[h] for h, m in enumerate(head_means)
                     if m is not None and m < cfg.warn_head_fraction * self.head_max[h])
        warning = bool(warn)
        drop = succ is not None and best is not None and (best - succ) >= self._drop
        low = succ is not None and succ < self._low_line
        event = warning and (drop or low)
        if event:
            if self.event_start is None:
                self.event_start = int(update)
        else:
            self.event_start = None
        stop = event and self.event_start is not None and update >= self.event_start + cfg.stop_after_updates
        if stop and self.stop_fired_at is None:
            self.stop_fired_at = int(update)
        if self.stop_fired_at is not None:
            level = "stop"
        elif event:
            level = "event"
        elif warning:
            level = "warning"
        elif watch:
            level = "watch"
        else:
            level = "none"
        self.levels_reached.add(level)
        return HealthReading(update=int(update), level=level, watch=watch, warning=warning, warn_heads=warn,
                             drop=drop, low=low, event=event, event_start=self.event_start, stop=stop,
                             success_count=count, run_best_count=self.run_best, entropy_sum=ent_sum,
                             entropy_sum_mean=sum_mean, head_means=head_means)

    @property
    def event_active(self) -> bool:
        return self.event_start is not None

    # ------------------------------------------------------------ resume
    def state_dict(self) -> dict[str, Any]:
        return {"head_sizes": list(self.head_sizes), "successes": [bool(s) for s in self.successes],
                "run_best": self.run_best, "sum_mean": self.sum_mean.state_dict(),
                "head_mean": [m.state_dict() for m in self.head_mean], "event_start": self.event_start,
                "stop_fired_at": self.stop_fired_at, "levels_reached": sorted(self.levels_reached)}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if tuple(d["head_sizes"]) != self.head_sizes:
            raise ValueError(f"monitor state is for heads {d['head_sizes']}, not {self.head_sizes}")
        self.successes = deque((bool(s) for s in d["successes"]), maxlen=self._w)
        self.success_count = sum(self.successes)
        self.run_best = d["run_best"]
        self.sum_mean.load_state_dict(d["sum_mean"])
        for m, s in zip(self.head_mean, d["head_mean"]):
            m.load_state_dict(s)
        self.event_start, self.stop_fired_at = d["event_start"], d["stop_fired_at"]
        self.levels_reached = set(d["levels_reached"])

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True)

    def load_json(self, text: str) -> None:
        self.load_state_dict(json.loads(text))
