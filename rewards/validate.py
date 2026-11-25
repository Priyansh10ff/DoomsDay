"""Config-time checks that refuse an exploitable or double-paying reward setup
(ADR 0010). Called by `rewards.factory.ShapedEnvFactory` before any env is built.

Each check corresponds to one exploit in the ADR's list:
  double pay     - a term we pay is also paid by the scenario's own reward
  heal farming   - healing (or armor) pays more per point than damage costs
  exit delay     - everything obtainable without exiting could out-earn the exit
  suicide        - dying is cheaper than paying the time penalty to the time limit
  door           - (door term only) one door must pay less than the exit; see
                   `check_door_setup` for why doors are outside the summed bound
  use / door     - (full_level_doors_v2 only) the USE-miss and door-reversal penalties
                   are capped per episode; with both caps spent, dying must still cost
                   more than timing out, an exit must still net positive, and passing
                   one on-path door must still out-pay each cap; `check_use_door_setup`
"""
from __future__ import annotations

import math
from collections.abc import Iterable

from configs.reward_config import KNOB_PAYS, N_KEY_COLOURS, TERMS, CoverageConfig, RewardConfig


class RewardConfigError(ValueError):
    pass


def engine_paid_terms(reward: RewardConfig, engine_rewards: Iterable[tuple[str, float]]) -> set[str]:
    """Terms the scenario's own reward already pays, if it is passed through."""
    if reward.engine_reward_scale == 0.0:
        return set()
    paid = set(reward.scenario_script_pays)
    for knob, value in engine_rewards:
        if float(value) != 0.0:
            if knob not in KNOB_PAYS:
                raise RewardConfigError(f"engine reward knob {knob!r} has no entry in KNOB_PAYS; "
                                        f"say which term it pays before passing it through")
            paid |= set(KNOB_PAYS[knob])
    return paid


def exit_value(reward: RewardConfig, paid_by_engine: set[str]) -> float:
    if "level_exit" in paid_by_engine:
        return reward.engine_reward_scale * reward.scenario_exit_value
    return reward.level_exit


def max_non_exit_positive(reward: RewardConfig, coverage: CoverageConfig) -> float:
    """Upper bound on what one episode can earn without exiting, on any map."""
    return (reward.positive_shaping_cap + N_KEY_COLOURS * reward.keycard
            + coverage.coef_start * coverage.episode_max)


def check_door_setup(reward: RewardConfig, paid_by_engine: set[str]) -> None:
    """The door-progress term (rewards/doors.py).

    Checked: a finite non-negative weight, a positive near margin, and one door paying
    strictly less than the exit (the exit stays the largest single payment).
    NOT checked: summed exit dominance. Door pay is deliberately left out of
    `max_non_exit_positive`: with 0.3 per door and up to 6 doors per training map the
    summed bound cannot hold. Door pay is paid once per door and is never forgone by
    exiting after the doors are passed, so it rewards progress, not delaying the exit.
    Its per-episode maximum (weight x doors in the map) is asserted at run time."""
    v, m = float(reward.door), float(reward.door_near_margin)
    if not math.isfinite(v) or v < 0.0:
        raise RewardConfigError(f"door = {v}: scales are non-negative finite magnitudes")
    if v == 0.0:
        return
    if not (math.isfinite(m) and m > 0.0):
        raise RewardConfigError(f"door_near_margin = {m}: must be a positive distance")
    ev = exit_value(reward, paid_by_engine)
    if not v < ev:
        raise RewardConfigError(f"{reward.name!r}: one door pays {v} but the exit pays only {ev}; "
                                f"the exit must stay the largest single payment")


def use_door_penalty_cap(reward: RewardConfig) -> float:
    """Most the USE-miss and door-reversal penalties can take from one episode."""
    return float(reward.use_miss_cap) + float(reward.door_reversal_cap)


def check_use_door_setup(reward: RewardConfig, paid_by_engine: set[str]) -> None:
    """The USE / door discipline terms (rewards/use_doors.py). All off = nothing to check.

    Checked: finite non-negative weights and caps; every penalty that is on has a cap
    of at least one event (so it is bounded and does something); on-path-only needs
    the door term; the exit pays more than both caps together (a successful episode
    nets positive however clumsy); death costs more than both caps together; one door
    pays more than EACH cap (never pressing USE to dodge use_miss, or never opening a
    door to dodge reversals, always loses to getting through one on-path door). The
    bound with the time penalty added is in `check_reward_setup` (suicide)."""
    for w, c in (("use_miss", "use_miss_cap"), ("door_reversal", "door_reversal_cap")):
        wv, cv = float(getattr(reward, w)), float(getattr(reward, c))
        for n, v in ((w, wv), (c, cv)):
            if not math.isfinite(v) or v < 0.0:
                raise RewardConfigError(f"{n} = {v}: scales are non-negative finite magnitudes")
        if wv == 0.0 and cv != 0.0:
            raise RewardConfigError(f"{c} = {cv} but {w} is off: a cap without its term")
        if wv > 0.0 and cv < wv:
            raise RewardConfigError(f"{w} = {wv} needs an episode cap {c} >= one event, got {cv}")
    if reward.door_on_path_only and float(reward.door) == 0.0:
        raise RewardConfigError("door_on_path_only is set but the door term is off")
    cap = use_door_penalty_cap(reward)
    if cap == 0.0:
        return
    ev = exit_value(reward, paid_by_engine)
    if not cap < ev:
        raise RewardConfigError(f"{reward.name!r}: USE/door penalty caps sum to {cap} but the exit pays "
                                f"{ev}: a successful episode could net negative")
    if "death" not in paid_by_engine and not cap < float(reward.death):
        raise RewardConfigError(f"{reward.name!r}: USE/door penalty caps sum to {cap} but death costs "
                                f"{reward.death}: dying would be cheaper than a clumsy episode")
    if float(reward.door) != 0.0:
        for c in ("use_miss_cap", "door_reversal_cap"):
            if not float(getattr(reward, c)) < float(reward.door):
                raise RewardConfigError(f"{reward.name!r}: {c} = {getattr(reward, c)} but one door pays "
                                        f"{reward.door}: avoiding USE (or doors) would beat getting through one")


def check_reward_setup(reward: RewardConfig, coverage: CoverageConfig,
                       engine_rewards: Iterable[tuple[str, float]] = (),
                       max_episode_decisions: int | None = None) -> None:
    """Raise RewardConfigError on the first problem found."""
    engine_rewards = tuple(engine_rewards)
    for t in TERMS + ("positive_shaping_cap", "ammo_unit", "ammo_step_clip_units"):
        v = float(getattr(reward, t))
        if not math.isfinite(v) or v < 0.0:
            raise RewardConfigError(f"{t} = {v}: scales are non-negative finite magnitudes")
    if reward.ammo_unit <= 0.0:
        raise RewardConfigError("ammo_unit must be positive")
    unknown = set(reward.scenario_script_pays) - set(TERMS)
    if unknown:
        raise RewardConfigError(f"scenario_script_pays names unknown terms {sorted(unknown)}")

    # --- double pay
    paid = engine_paid_terms(reward, engine_rewards)
    twice = sorted(t for t in paid if reward.scale(t) != 0.0)
    if twice:
        raise RewardConfigError(
            f"{reward.name!r}: the scenario's own reward already pays {twice} and "
            f"engine_reward_scale is {reward.engine_reward_scale}; set those scales to 0 "
            f"or set engine_reward_scale to 0")
    if "level_exit" in paid and reward.scenario_exit_value <= 0.0:
        raise RewardConfigError("the scenario pays the exit: declare scenario_exit_value")

    check_door_setup(reward, paid)
    check_use_door_setup(reward, paid)

    # --- two-signal exit rule needs the second signal to exist
    need = reward.exit_confirm_engine_reward
    if reward.level_exit != 0.0 and need is not None:
        knob = dict(engine_rewards).get("map_exit_reward", 0.0)
        if knob < need:
            raise RewardConfigError(
                f"exit_confirm_engine_reward = {need} but the env's map_exit_reward is {knob}: "
                f"the exit term could never pay. Set the knob (env-engineer) or set "
                f"exit_confirm_engine_reward to None")

    # --- heal / armor farming: a hurt-then-heal loop must not net positive
    for gain in ("health", "armor"):
        if reward.scale(gain) > reward.damage_taken:
            raise RewardConfigError(
                f"{gain} gain pays {reward.scale(gain)} per point but damage costs only "
                f"{reward.damage_taken}: taking damage and recovering would net positive")

    # --- coverage and its schedule
    if coverage.enabled:
        if coverage.cell_size <= 0 or coverage.budget_cells < 1:
            raise RewardConfigError("coverage needs cell_size > 0 and budget_cells >= 1")
        if not (0.0 <= coverage.coef_end <= coverage.coef_start):
            raise RewardConfigError("coverage needs 0 <= coef_end <= coef_start (it anneals down)")
        if coverage.hold_decisions < 0 or coverage.anneal_decisions < 1:
            raise RewardConfigError("coverage needs hold_decisions >= 0 and anneal_decisions >= 1")

    # --- exit delay
    if reward.require_exit_dominance:
        ev, other = exit_value(reward, paid), max_non_exit_positive(reward, coverage)
        if not ev > other:
            raise RewardConfigError(
                f"{reward.name!r}: exit pays {ev} but an episode can earn up to {other} without "
                f"exiting (cap {reward.positive_shaping_cap} + {N_KEY_COLOURS} keys x "
                f"{reward.keycard} + coverage {coverage.coef_start * coverage.episode_max})")

    # --- suicide: dying must cost more than waiting out the clock
    if reward.time > 0.0:
        if max_episode_decisions is None:
            raise RewardConfigError("a time penalty needs max_episode_decisions for the suicide check")
        extra = use_door_penalty_cap(reward)
        worst = reward.time * max_episode_decisions + extra
        if "death" not in paid and not reward.death > worst:
            raise RewardConfigError(
                f"death costs {reward.death} but the time penalty over a full episode "
                f"({max_episode_decisions} decisions){' plus the USE/door penalty caps' if extra else ''} "
                f"costs {worst}: dying early would be cheaper. "
                f"Lower `time`, raise `death`, or shorten the training episode limit")
