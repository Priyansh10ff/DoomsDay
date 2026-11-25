"""The nine extrinsic reward terms, computed from `info` alone (ADR 0010).

`ExtrinsicReward.step(info, ended)` takes no action and no observation: a reward
that cannot see the action cannot pay for pressing a button (USE on a door, ATTACK).

Per step it returns, for every term, a raw value (signed, in game units) and the
paid value `scale * raw` after the guards. The total is the sum of the paid values.

Guards implemented here (each is part of the reward computation only):
  * pickups pay on GAINS only; losses are paid once, by `damage_taken`;
  * key colours pay once per colour per episode;
  * ammo gain per decision is clipped, so one large jump cannot dominate;
  * the positive pay of repeatable terms (kill, health, ammo, armor) is capped per
    episode (`positive_shaping_cap`);
  * every delta is measured from the reset `info`, never from zero and never across
    an episode boundary;
  * damage has exactly one source per step;
  * the exit pays only when the engine reward agrees (two-signal rule), if configured;
  * a non-zero scale whose `info` field is absent raises instead of paying zero.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from configs.reward_config import CAPPED_TERMS, TERMS, RewardConfig

# `info` fields. The first block exists in env.DoomEnv today; the second is requested
# from env-engineer (ADR 0010, "Changes needed elsewhere").
F_KILLS, F_KEYS = "killcount", "keys_held"
F_DEAD, F_EXIT, F_ENGINE_REWARD = "dead", "level_exit", "engine_reward"
F_LAST_DAMAGE = "last_step_damage_taken"
F_HEALTH, F_ARMOR, F_AMMO, F_DAMAGE_TOTAL = "health", "armor", "ammo", "damage_taken_total"

# Field each term needs in every per-step `info` (end-of-episode flags are checked
# on the ending step instead).
STEP_FIELDS: dict[str, str] = {"keycard": F_KEYS, "kill": F_KILLS, "health": F_HEALTH,
                               "ammo": F_AMMO, "armor": F_ARMOR, "damage_taken": F_DAMAGE_TOTAL}
END_FIELDS: dict[str, str] = {"level_exit": F_EXIT, "death": F_DEAD}


class MissingInfoField(KeyError):
    """A term with a non-zero scale needs an `info` field the env does not supply."""


@dataclass
class StepReward:
    raw: dict[str, float]            # signed, game units, before scale and guards
    paid: dict[str, float]           # scale * raw after guards; sums to `total`
    total: float
    capped_away: float = 0.0         # positive pay removed by the episode cap this step
    exit_signal_disagreement: bool = False


def _ammo_slots(v: Any) -> tuple[float, ...]:
    if isinstance(v, Mapping):
        return tuple(float(v[k]) for k in sorted(v))
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
        return tuple(float(x) for x in v)
    if hasattr(v, "tolist"):
        flat = v.tolist()
        return tuple(float(x) for x in (flat if isinstance(flat, list) else [flat]))
    return (float(v),)


@dataclass
class ExtrinsicReward:
    cfg: RewardConfig
    _prev: dict[str, Any] = field(default_factory=dict)
    _keys_paid: set = field(default_factory=set)
    _capped_paid: float = 0.0
    _ready: bool = False

    def enabled(self, term: str) -> bool:
        return self.cfg.scale(term) != 0.0

    def _need(self, info: Mapping[str, Any], term: str, key: str) -> Any:
        if key not in info:
            raise MissingInfoField(
                f"reward term {term!r} has scale {self.cfg.scale(term)} but info has no {key!r}; "
                f"set the scale to 0 or have env-engineer add the field (ADR 0010)")
        return info[key]

    # ------------------------------------------------------------ episode start
    def reset(self, info: Mapping[str, Any]) -> None:
        """Take every baseline from the first `info` of the episode."""
        self._prev = {}
        for term, key in STEP_FIELDS.items():
            if self.enabled(term):
                self._prev[term] = self._read(term, self._need(info, term, key))
        # Keys already held at the start (a curriculum may hand them out) never pay.
        self._keys_paid = set(self._prev.get("keycard", ()))
        self._capped_paid = 0.0
        self._ready = True

    @staticmethod
    def _read(term: str, value: Any) -> Any:
        if term == "keycard":
            return frozenset(k for k, held in dict(value).items() if held)
        if term == "ammo":
            return _ammo_slots(value)
        return float(value)

    # ------------------------------------------------------------ one decision
    def step(self, info: Mapping[str, Any], ended: bool) -> StepReward:
        if not self._ready:
            raise RuntimeError("ExtrinsicReward.step called before reset")
        cfg = self.cfg
        raw = {t: 0.0 for t in TERMS}
        cur: dict[str, Any] = {}
        for term, key in STEP_FIELDS.items():
            if self.enabled(term):
                cur[term] = self._read(term, self._need(info, term, key))

        raw["time"] = -1.0
        if "keycard" in cur:
            new = cur["keycard"] - self._keys_paid
            raw["keycard"] = float(len(new))
            self._keys_paid |= new
        if "kill" in cur:
            raw["kill"] = max(cur["kill"] - self._prev["kill"], 0.0)
        if "health" in cur:
            raw["health"] = max(cur["health"] - self._prev["health"], 0.0)
        if "armor" in cur:
            raw["armor"] = max(cur["armor"] - self._prev["armor"], 0.0)
        if "ammo" in cur:
            a, b = self._prev["ammo"], cur["ammo"]
            gained = sum(max(y - x, 0.0) for x, y in zip(a, b)) if len(a) == len(b) else 0.0
            raw["ammo"] = min(gained / cfg.ammo_unit, cfg.ammo_step_clip_units)
        if "damage_taken" in cur:
            # One source per step. On an ending step the env reports the last
            # decision's damage itself (the cumulative counter is stale after death).
            if ended and F_LAST_DAMAGE in info:
                dmg = float(info[F_LAST_DAMAGE])
            else:
                dmg = cur["damage_taken"] - self._prev["damage_taken"]
            raw["damage_taken"] = -max(dmg, 0.0)

        disagreement = False
        if ended:
            if self.enabled("death") and bool(self._need(info, "death", F_DEAD)):
                raw["death"] = -1.0
            if self.enabled("level_exit") and bool(self._need(info, "level_exit", F_EXIT)):
                need = cfg.exit_confirm_engine_reward
                if need is None or float(self._need(info, "level_exit", F_ENGINE_REWARD)) >= need:
                    raw["level_exit"] = 1.0
                else:
                    disagreement = True  # counted and logged, not paid

        paid = {t: cfg.scale(t) * raw[t] for t in TERMS}
        # Episode cap on repeatable positive pay.
        want = sum(paid[t] for t in CAPPED_TERMS)
        room = max(cfg.positive_shaping_cap - self._capped_paid, 0.0)
        capped_away = 0.0
        if want > room:
            keep = room / want if want > 0 else 0.0
            for t in CAPPED_TERMS:
                paid[t] *= keep
            capped_away = want - room
            want = room
        self._capped_paid += want

        if "kill" in cur:  # high-water mark: a count that drops and returns never pays twice
            cur["kill"] = max(cur["kill"], self._prev["kill"])
        self._prev.update({t: v for t, v in cur.items() if t != "keycard"})
        return StepReward(raw, paid, float(sum(paid.values())), capped_away, disagreement)
