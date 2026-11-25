"""USE / door discipline terms of preset `full_level_doors_v2`. TRAINING ONLY (rule 4.2:
privileged pose, map geometry and engine door gaps, for reward only).

Three things, each switched by its own RewardConfig field:

  use_miss       -use_miss per USE press edge (0 -> 1 between consecutive decisions; the
                 first decision of an episode counts against "released") made while the
                 USE trace (64 units from the pre-step position along the pre-step view
                 heading) crosses NO closed door face and NO use-activated exit line.
                 Episode sum bounded by use_miss_cap.
  door_reversal  -door_reversal each time a door that was rising or open starts falling
                 within ~1 s (35 tics) of a player press whose USE trace crossed that
                 door. Counted once per reversal; a door closing by itself is not one.
                 Episode sum bounded by door_reversal_cap. Without engine door gaps the
                 count is unknown: nothing is paid and the logged count is None.
  on-path doors  `on_path_flags` tells the wrapper which doors are on the
                 start -> key -> exit route; with door_on_path_only the others pay 0.

The counting itself is train.episode_diag.EpisodeDiag (the logging the numbers were
measured with): this module only turns the per-step change of its counters into pay.
No geometry is re-implemented here. The import is lazy because train.episode_diag
imports rewards.doors.

What this module reads: the USE head of the executed action (read only; the action is
passed to the env untouched by the wrapper, and nothing here filters, delays or
debounces a press), `info`, and the door gaps. What it writes: reward and log values.
Nothing reaches an observation, and the evaluation path never builds it.

Known ways to game it, and why each does not pay:
  * never pressing USE: saves at most use_miss_cap (0.15), forfeits every on-path door
    (0.3 each) and a switch exit (1.0); an in-range press costs 0.
  * holding USE down: one edge, so no further penalty, but the engine also acts on the
    press edge only, so a held button opens nothing.
  * spamming after the cap: free, by design (the cap keeps the term from dominating);
    the uncapped count is still logged (`use_miss_count`).
  * no occlusion test: a press at a door through a thin wall is "in range" and free.
    Lenient on purpose; it pays nothing either.
  * avoiding reversals by never opening a door: forfeits the door pay, which is 6 x
    one reversal.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from configs.reward_config import RewardConfig

USE_MISS, DOOR_REVERSAL = "use_miss", "door_reversal"
F_GAPS = "door_gaps"            # key of the PRIVATE info copy; never in the returned info


def v2_terms_on(cfg: RewardConfig) -> bool:
    return cfg.use_miss != 0.0 or cfg.door_reversal != 0.0 or bool(cfg.door_on_path_only)


class UseDoorTerms:
    def __init__(self, cfg: RewardConfig, table: Any) -> None:
        """`table`: train.episode_diag.diag_table of the training bundle."""
        from train.episode_diag import USE_HEAD, EpisodeDiag
        self.cfg = cfg
        self.table = dict(table)
        self.use_head = USE_HEAD
        self._diag = EpisodeDiag(cfg.door_near_margin)
        self._before: dict[str, Any] = {}
        self._ready = False

    # ------------------------------------------------------------ episode start
    def reset(self, info: Mapping[str, Any], gaps: Mapping[int, float] | None) -> None:
        lump = info["map"]
        if lump not in self.table:
            raise KeyError(f"map {lump!r} has no USE/door record; the table covers {len(self.table)} maps")
        dm = self.table[lump]
        if self.cfg.door_on_path_only and dm.doors and dm.on_path is None:
            raise ValueError(f"map {lump!r}: door_on_path_only is set but the route of this map is unknown")
        self._dm = dm
        self._before = self._private(info, gaps)
        self._diag.reset(dm, self._before, lump)
        self._miss_paid = self._rev_paid = 0.0
        self._miss_n = self._rev_n = 0
        self._ready = True

    @staticmethod
    def _private(info: Mapping[str, Any], gaps: Mapping[int, float] | None) -> dict[str, Any]:
        d = dict(info)
        d.pop(F_GAPS, None)
        if gaps is not None:
            d[F_GAPS] = gaps
        return d

    @property
    def doors(self) -> tuple:
        return self._dm.doors

    @property
    def on_path_flags(self) -> tuple[bool, ...]:
        """Per door, in maps.doors.door_specs order. All True when the route is not needed."""
        if self._dm.on_path is None:
            return (True,) * len(self._dm.doors)
        return tuple(bool(f) for f in self._dm.on_path)

    @property
    def use_miss_count(self) -> int:
        return self._miss_n

    @property
    def door_reversal_count(self) -> int | None:
        """None while no door gaps were seen this episode (unknown, not zero)."""
        return self._rev_n if self._diag.have_gaps else None

    # ------------------------------------------------------------ one decision
    @staticmethod
    def _charge(weight: float, n: int, cap: float, already: float) -> float:
        """Penalty magnitude for n new events, never past the episode cap."""
        return min(weight * n, max(cap - already, 0.0))

    def step(self, use: int, info: Mapping[str, Any],
             gaps: Mapping[int, float] | None) -> tuple[dict[str, float], dict[str, float]]:
        """(raw, paid) of this decision for use_miss and door_reversal. `use` is the
        executed USE bit of the decision that led to `info`."""
        if not self._ready:
            raise RuntimeError("UseDoorTerms.step called before reset")
        dg = self._diag
        miss0, rev0 = dg.edges - dg.in_range, dg.reversals
        after = self._private(info, gaps)
        dg.step(use, self._before, after)
        self._before = after
        new_miss, new_rev = (dg.edges - dg.in_range) - miss0, dg.reversals - rev0
        assert new_miss in (0, 1) and new_rev >= 0
        self._miss_n += new_miss
        self._rev_n += new_rev
        cfg = self.cfg
        m = self._charge(cfg.use_miss, new_miss, cfg.use_miss_cap, self._miss_paid)
        r = self._charge(cfg.door_reversal, new_rev, cfg.door_reversal_cap, self._rev_paid)
        self._miss_paid += m
        self._rev_paid += r
        assert self._miss_paid <= cfg.use_miss_cap + 1e-9 and self._rev_paid <= cfg.door_reversal_cap + 1e-9
        return ({USE_MISS: -float(new_miss), DOOR_REVERSAL: -float(new_rev)},
                {USE_MISS: -m, DOOR_REVERSAL: -r})
