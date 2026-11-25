"""ShapedRewardEnv: the single integration point for reward shaping (ADR 0010).

A gymnasium wrapper placed around a training env inside its worker process.

  * The observation is returned untouched (the same object the env returned).
    Nothing computed here is written into it.
  * The action is passed to the env and never read by the reward code.
  * Existing `info` keys are never removed or changed; two keys are added:
      info["shaping"]          every step: per-term raw and paid values, engine,
                               extrinsic, intrinsic (raw, coefficient, scaled), total
      info["shaping_episode"]  ending step only: the episode sums of the same, plus
                               cells visited and guard counters
    With the door term on (RewardConfig.door != 0, needs a door table) the per-step
    `raw`/`paid` dicts also carry "door", and the episode sums `paid/door`,
    `raw/door`, `doors_passed` and `doors_in_map`. With it off nothing about doors
    is computed or logged.
  * The inference / evaluation path builds env.DoomEnv directly and never this
    wrapper, so no reward code runs there.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from configs.reward_config import TERMS, CoverageConfig, RewardConfig
from intrinsic.coverage import EpisodicCoverage
from intrinsic.schedule import LinearAnneal

from .terms import F_ENGINE_REWARD, ExtrinsicReward

DOOR = "door"

INFO_STEP, INFO_EPISODE = "shaping", "shaping_episode"


class ShapedRewardEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, reward: RewardConfig, coverage: CoverageConfig,
                 start_decisions: int = 0, doors: Any = None) -> None:
        """`doors`: a maps.doors.DoorTable; required when `reward.door` != 0."""
        super().__init__(env)
        self.reward_cfg, self.coverage_cfg = reward, coverage
        self._ext = ExtrinsicReward(reward)
        self._cov = EpisodicCoverage(coverage) if coverage.enabled else None
        self._doors = None
        if reward.door != 0.0:
            if doors is None:
                raise ValueError(f"{reward.name!r} pays door progress but no door table was given")
            from .doors import DoorProgress
            self._doors = DoorProgress(doors, reward.door, reward.door_near_margin)
        self._anneal = LinearAnneal(coverage.coef_start, coverage.coef_end,
                                    coverage.hold_decisions, coverage.anneal_decisions)
        self.decisions = int(start_decisions)  # per-env decisions, the anneal clock
        self._sums: dict[str, float] = {}

    def intrinsic_coef(self) -> float:
        return self._anneal(self.decisions) if self._cov is not None else 0.0

    def _zero_sums(self) -> None:
        self._sums = {f"paid/{t}": 0.0 for t in TERMS} | {f"raw/{t}": 0.0 for t in TERMS} | {
            "engine": 0.0, "engine_return_unscaled": 0.0, "extrinsic": 0.0, "intrinsic_raw": 0.0, "intrinsic": 0.0, "total": 0.0,
            "capped_away": 0.0, "exit_signal_disagreements": 0.0, "length": 0.0}
        if self._doors is not None:
            self._sums |= {f"paid/{DOOR}": 0.0, f"raw/{DOOR}": 0.0}

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._ext.reset(info)
        if self._cov is not None:
            self._cov.reset(info)
        if self._doors is not None:
            self._doors.reset(info)
        self._zero_sums()
        return obs, info

    def step(self, action: Any):
        obs, engine_reward, terminated, truncated, info = self.env.step(action)
        ended = bool(terminated or truncated)
        info = dict(info)  # the env may reuse its dict on the next step
        info.setdefault(F_ENGINE_REWARD, float(engine_reward))

        sr = self._ext.step(info, ended)
        engine = self.reward_cfg.engine_reward_scale * float(engine_reward)
        extrinsic = engine + sr.total
        raw, paid = dict(sr.raw), dict(sr.paid)
        if self._doors is not None:
            raw[DOOR] = self._doors.step(info)
            paid[DOOR] = self.reward_cfg.door * raw[DOOR]
            extrinsic += paid[DOOR]
        coef = self.intrinsic_coef()
        intrinsic_raw = self._cov.step(info) if self._cov is not None else 0.0
        intrinsic = coef * intrinsic_raw
        total = extrinsic + (intrinsic if self.reward_cfg.sum_intrinsic_into_reward else 0.0)
        self.decisions += 1

        info[INFO_STEP] = {
            "raw": raw, "paid": paid, "engine": engine,
            "extrinsic": extrinsic, "intrinsic_raw": intrinsic_raw, "intrinsic_coef": coef,
            "intrinsic": intrinsic, "total": total, "capped_away": sr.capped_away,
        }
        s = self._sums
        for t in TERMS:
            s[f"paid/{t}"] += sr.paid[t]
            s[f"raw/{t}"] += sr.raw[t]
        if self._doors is not None:
            s[f"paid/{DOOR}"] += paid[DOOR]
            s[f"raw/{DOOR}"] += raw[DOOR]
        for k, v in (("engine", engine), ("engine_return_unscaled", float(engine_reward)),
                     ("extrinsic", extrinsic), ("intrinsic_raw", intrinsic_raw),
                     ("intrinsic", intrinsic), ("total", total), ("capped_away", sr.capped_away),
                     ("exit_signal_disagreements", float(sr.exit_signal_disagreement)),
                     ("length", 1.0)):
            s[k] += v
        if ended:
            ep = dict(s)
            ep["cells_visited"] = self._cov.cells_visited if self._cov is not None else None
            ep["intrinsic_coef"] = coef
            ep["reward_config"] = self.reward_cfg.name
            if self._doors is not None:
                ep["doors_passed"] = self._doors.passed
                ep["doors_in_map"] = self._doors.n_doors
                assert ep[f"paid/{DOOR}"] <= self._doors.episode_max + 1e-9, "door pay above its episode cap"
            info[INFO_EPISODE] = ep
        return obs, float(total), terminated, truncated, info
