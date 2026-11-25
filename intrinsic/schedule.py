"""Annealed coefficient for the intrinsic reward (ADR 0005, ADR 0010).

The clock is PER-ENV decisions: a reward wrapper inside an env worker can count
its own steps but has no channel from the trainer. Envs are stepped in lockstep,
so per-env decisions equal total decisions / num_envs. On resume the caller passes
the offset (`start_decisions`), otherwise the schedule would restart at the top.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearAnneal:
    start: float
    end: float
    hold_decisions: int
    anneal_decisions: int

    def __call__(self, decisions: int) -> float:
        """Coefficient after `decisions` per-env decisions. Flat, then linear, then flat."""
        frac = (decisions - self.hold_decisions) / self.anneal_decisions
        frac = min(max(frac, 0.0), 1.0)
        return self.start + (self.end - self.start) * frac
