"""USE / door discipline reward terms (rewards/use_doors.py, preset full_level_doors_v2).

Synthetic press sequences on a one- or two-door corridor; no engine, no torch. The
counting is train.episode_diag's, so the geometry here is the same as in
tests/test_episode_diag.py: door slab x in [96, 128], y in [0, 64]; a player at
(70, 32) facing +x has the door face inside USE range."""
from __future__ import annotations

import dataclasses
import json
import os
import pickle

import numpy as np
import pytest

from reward_test_helpers import FULL, ROOT, ScriptEnv, end, mk

from configs.reward_config import (FULL_LEVEL_COVERAGE, FULL_LEVEL_DOORS_REWARD, FULL_LEVEL_DOORS_V2_REWARD,
                                   FULL_LEVEL_REWARD, TERMS)
from maps.doors import AXIS_X, DoorSpec
from rewards.factory import ShapedEnvFactory
from rewards.use_doors import v2_terms_on
from rewards.validate import RewardConfigError, check_reward_setup
from rewards.wrapper import INFO_EPISODE, INFO_STEP, ShapedRewardEnv
from train.episode_diag import USE_HEAD, DiagMap

V2 = dataclasses.replace(FULL_LEVEL_DOORS_V2_REWARD, exit_confirm_engine_reward=None)
W_MISS, CAP_MISS, W_REV, CAP_REV, W_DOOR = V2.use_miss, V2.use_miss_cap, V2.door_reversal, V2.door_reversal_cap, V2.door
LUMP = "M"
DOOR = DoorSpec(sector=7, axis=AXIS_X, a0=96.0, a1=128.0, lo=0.0, hi=64.0)       # on the route
SIDE = DoorSpec(sector=9, axis=AXIS_X, a0=96.0, a1=128.0, lo=1000.0, hi=1064.0)  # a side door
EXIT_LINE = (300.0, 0.0, 300.0, 64.0)
ONE = DiagMap(doors=(DOOR,), on_path=(True,), exit_kind="switch", exit_use_lines=(EXIT_LINE,))
TWO = DiagMap(doors=(DOOR, SIDE), on_path=(True, False), exit_kind="pad", exit_use_lines=())
ENGINE = (("map_exit_reward", 1.0),)


def act(use: int) -> np.ndarray:
    a = np.zeros(6, np.int64)
    a[USE_HEAD] = use
    return a


class GapEnv(ScriptEnv):
    """ScriptEnv plus env.DoomEnv's privileged diag_door_gaps() (never in info)."""

    def __init__(self, infos: list[dict], gaps: list[dict | None] | None) -> None:
        super().__init__(infos)
        self.gaps = gaps

    def diag_door_gaps(self) -> dict | None:
        return None if self.gaps is None else self.gaps[self._t]


def at(x: float = 70.0, y: float = 32.0, angle: float = 0.0, tic: int = 0) -> dict:
    return mk(x, y, angle, map=LUMP, tic=tic)


def play(steps: list[tuple[int, dict, float | None]], start: dict | None = None, start_gap: float | None = 0.0,
         reward=V2, dmap: DiagMap = ONE, gaps: bool = True) -> tuple[list[float], list[dict], GapEnv]:
    """steps: [(use bit, info after, door-7 gap after)]. The last step ends by timeout."""
    infos = [start or at()] + [i for _, i, _ in steps]
    infos[-1] = end(infos[-1], "timeout")
    g = [{7: start_gap, 9: 0.0}] + [{7: gp, 9: 0.0} for _, _, gp in steps] if gaps else None
    inner = GapEnv(infos, g)
    env = ShapedRewardEnv(inner, reward, FULL_LEVEL_COVERAGE, 0,
                          doors=((LUMP, dmap.doors),) if reward.door else None,
                          diag=((LUMP, dmap),) if v2_terms_on(reward) else None)
    env.reset(seed=0)
    rs, out = [], []
    for use, _, _ in steps:
        _, r, _, _, info = env.step(act(use))
        rs.append(r)
        out.append(info)
    return rs, out, inner


def ep(out: list[dict]) -> dict:
    return out[-1][INFO_EPISODE]


def step_paid(out: list[dict], term: str) -> list[float]:
    return [i[INFO_STEP]["paid"][term] for i in out]


# ------------------------------------------------------------------ use_miss
def test_press_at_a_closed_door_costs_nothing() -> None:
    _, out, _ = play([(1, at(tic=4), 8.0), (0, at(tic=8), 16.0), (0, at(tic=12), 24.0)])
    e = ep(out)
    assert e["paid/use_miss"] == 0.0 and e["raw/use_miss"] == 0.0 and e["use_miss_count"] == 0
    assert e["paid/door_reversal"] == 0.0 and e["door_reversal_count"] == 0


def test_press_at_the_exit_switch_costs_nothing() -> None:
    s = at(x=250.0)
    _, out, _ = play([(1, at(x=250.0, tic=4), 0.0), (0, at(x=250.0, tic=8), 0.0)], start=s)
    assert ep(out)["paid/use_miss"] == 0.0 and ep(out)["use_miss_count"] == 0


def test_press_out_of_range_or_facing_away_is_penalised_on_the_press_step() -> None:
    far = at(x=20.0)                                   # trace ends at x = 84 < 96
    steps = [(1, at(x=20.0, tic=4), 0.0), (0, at(x=20.0, tic=8), 0.0),
             (1, at(x=70.0, angle=180.0, tic=12), 0.0),          # pressed at x=20 still: miss
             (0, at(x=70.0, angle=180.0, tic=16), 0.0),
             (1, at(x=70.0, angle=180.0, tic=20), 0.0),          # at the door, facing away: miss
             (0, at(x=70.0, angle=0.0, tic=24), 0.0)]
    rs, out, _ = play(steps, start=far)
    assert step_paid(out, "use_miss") == pytest.approx([-W_MISS, 0.0, -W_MISS, 0.0, -W_MISS, 0.0])
    e = ep(out)
    assert e["use_miss_count"] == 3 and e["raw/use_miss"] == -3.0
    assert e["paid/use_miss"] == pytest.approx(-3 * W_MISS)
    # the reward of a miss step differs from a no-press step by exactly the penalty
    assert rs[0] - rs[1] == pytest.approx(-W_MISS)


def test_held_use_is_one_edge_and_one_penalty() -> None:
    far = at(x=20.0)
    _, out, _ = play([(1, at(x=20.0, tic=4 * k), 0.0) for k in range(1, 30)], start=far)
    assert ep(out)["use_miss_count"] == 1 and ep(out)["paid/use_miss"] == pytest.approx(-W_MISS)


def test_use_miss_cap_is_respected_and_the_count_keeps_running() -> None:
    far = at(x=20.0)
    n = 80
    assert n * W_MISS > CAP_MISS
    steps = [(k % 2, at(x=20.0, tic=4 * k), 0.0) for k in range(1, 2 * n + 1)]
    _, out, _ = play(steps, start=far)
    e = ep(out)
    assert e["use_miss_count"] == n and e["raw/use_miss"] == -float(n)
    assert e["paid/use_miss"] == pytest.approx(-CAP_MISS) and -e["paid/use_miss"] <= CAP_MISS + 1e-12
    per = step_paid(out, "use_miss")
    assert min(per) == pytest.approx(-W_MISS) and per[-1] == 0.0 and per[-2] == 0.0   # free after the cap


def test_typical_and_clean_episode_losses_match_the_measured_press_counts() -> None:
    """Median door episode of the measured run: 35 misses; q10: 2 misses."""
    assert 0.10 <= min(35 * W_MISS, CAP_MISS) <= 0.15
    assert 2 * W_MISS < 0.01
    assert CAP_MISS / W_MISS == pytest.approx(50)


# ------------------------------------------------------------------ door_reversal
REVERSAL = [(1, at(tic=4), 8.0),      # press at the closed door: it opens
            (0, at(tic=8), 16.0),     # still rising
            (1, at(tic=12), 8.0),     # second press: the door falls again -> reversal
            (0, at(tic=16), 0.0),     # keeps falling: the same reversal, not a new one
            (0, at(tic=20), 0.0)]


def test_reversal_is_penalised_once() -> None:
    _, out, _ = play(REVERSAL)
    assert step_paid(out, "door_reversal") == pytest.approx([0.0, 0.0, -W_REV, 0.0, 0.0])
    e = ep(out)
    assert e["door_reversal_count"] == 1 and e["raw/door_reversal"] == -1.0
    assert e["paid/door_reversal"] == pytest.approx(-W_REV)
    # the second press was at a door that was not closed: also a miss, by the logged definition
    assert e["use_miss_count"] == 1


def test_a_door_closing_by_itself_is_not_penalised() -> None:
    steps = [(1, at(tic=4), 8.0), (0, at(tic=8), 64.0), (0, at(tic=100), 64.0), (0, at(tic=160), 56.0),
             (0, at(tic=164), 48.0)]
    _, out, _ = play(steps)
    assert ep(out)["paid/door_reversal"] == 0.0 and ep(out)["door_reversal_count"] == 0


def test_reversal_cap_is_respected() -> None:
    steps, tic = [], 0
    for _ in range(7):                                   # 7 open-then-reverse cycles, far apart in time
        tic += 200
        steps += [(1, at(tic=tic), 8.0), (0, at(tic=tic + 4), 16.0), (1, at(tic=tic + 8), 8.0),
                  (0, at(tic=tic + 12), 0.0)]
    _, out, _ = play(steps)
    e = ep(out)
    assert e["door_reversal_count"] == 7 and 7 * W_REV > CAP_REV
    assert e["paid/door_reversal"] == pytest.approx(-CAP_REV)


def test_without_door_gaps_reversals_are_unknown_and_unpaid() -> None:
    _, out, _ = play(REVERSAL, gaps=False)
    e = ep(out)
    assert e["door_reversal_count"] is None and e["paid/door_reversal"] == 0.0
    assert e["use_miss_count"] == 0          # without gaps every door face counts as closed (lenient)


# ------------------------------------------------------------------ on-path door pay
def _walk(y: float, tic0: int) -> list[tuple[int, dict, float]]:
    return [(0, at(x=x, y=y, tic=tic0 + 4 * k), 64.0) for k, x in enumerate((60.0, 90.0, 112.0, 140.0, 170.0))]


def test_on_path_door_pays_and_off_path_door_pays_nothing() -> None:
    steps = _walk(32.0, 4) + _walk(1032.0, 100)          # through the route door, then the side door
    _, out, _ = play(steps, dmap=TWO)
    e = ep(out)
    assert e["doors_passed"] == 2 and e["doors_in_map"] == 2
    assert e["doors_passed_on_path"] == 1 and e["doors_on_path_in_map"] == 1
    assert e["paid/door"] == pytest.approx(W_DOOR) == pytest.approx(0.3) and e["raw/door"] == 1.0
    per = step_paid(out, "door")
    assert sum(per[:5]) == pytest.approx(0.3) and sum(per[5:]) == 0.0
    # only the side door: nothing
    _, out, _ = play(_walk(1032.0, 4), start=at(y=1032.0, x=40.0), dmap=TWO)
    assert ep(out)["paid/door"] == 0.0 and ep(out)["doors_passed"] == 1 and ep(out)["doors_passed_on_path"] == 0


def test_old_preset_still_pays_the_side_door() -> None:
    old = dataclasses.replace(FULL, name="full_level_doors", door=FULL_LEVEL_DOORS_REWARD.door)
    _, out, _ = play(_walk(32.0, 4) + _walk(1032.0, 100), reward=old, dmap=TWO)
    assert ep(out)["paid/door"] == pytest.approx(0.6)
    assert not any(k in ep(out) for k in ("paid/use_miss", "paid/door_reversal", "use_miss_count",
                                          "doors_passed_on_path"))


def test_unknown_route_is_refused_not_guessed() -> None:
    unk = DiagMap(doors=(DOOR,), on_path=None, exit_kind=None, exit_use_lines=())
    with pytest.raises(ValueError, match="route"):
        play(_walk(32.0, 4), dmap=unk)


# ------------------------------------------------------------------ sums, isolation
def test_extrinsic_is_the_sum_of_every_paid_term() -> None:
    steps = [(1, at(x=20.0, tic=4), 0.0)] + REVERSAL + _walk(32.0, 400)
    rs, out, _ = play(steps, start=at(x=20.0))
    e = ep(out)
    assert e["paid/use_miss"] < 0 and e["paid/door_reversal"] < 0 and e["paid/door"] > 0
    want = sum(e[f"paid/{t}"] for t in TERMS) + e["paid/door"] + e["paid/use_miss"] + e["paid/door_reversal"]
    assert e["extrinsic"] == pytest.approx(want)
    assert sum(rs) == pytest.approx(e["total"]) == pytest.approx(e["extrinsic"] + e["intrinsic"])
    for i in out:
        s = i[INFO_STEP]
        assert s["extrinsic"] == pytest.approx(sum(s["paid"].values()))
    json.dumps(e)                                         # episodes.jsonl copies it verbatim


def test_action_and_observation_pass_untouched_and_gaps_stay_out_of_info() -> None:
    steps = [(k % 2, at(x=20.0, tic=4 * k), 0.0) for k in range(1, 9)]
    infos = [at(x=20.0)] + [i for _, i, _ in steps]
    infos[-1] = end(infos[-1], "timeout")
    inner = GapEnv(infos, [{7: 0.0}] * len(infos))
    env = ShapedRewardEnv(inner, V2, FULL_LEVEL_COVERAGE, 0, doors=((LUMP, (DOOR,)),), diag=((LUMP, ONE),))
    env.reset(seed=0)
    acts = [act(u) for u, _, _ in steps]
    for k, a in enumerate(acts):
        before = a.copy()
        obs, _, _, _, info = env.step(a)
        assert np.array_equal(a, before) and "door_gaps" not in info
        assert set(obs) == {"self"} and float(obs["self"][0]) == k + 1
    assert all(x is a for x, a in zip(inner.actions, acts))   # every press reaches the env; no debounce


def test_state_resets_between_episodes() -> None:
    far = [at(x=20.0)] + [at(x=20.0, tic=4 * k) for k in range(1, 5)]
    far[-1] = end(far[-1], "timeout")
    inner = GapEnv(far, [{7: 0.0}] * len(far))
    env = ShapedRewardEnv(inner, V2, FULL_LEVEL_COVERAGE, 0, doors=((LUMP, (DOOR,)),), diag=((LUMP, ONE),))
    eps = []
    for uses in ((1, 0, 1, 1), (1, 1, 1, 1)):     # episode 1 ends with USE held; episode 2 starts pressed
        env.reset(seed=0)
        for u in uses:
            info = env.step(act(u))[4]
        eps.append(info[INFO_EPISODE])
    assert [e["use_miss_count"] for e in eps] == [2, 1]
    assert eps[1]["paid/use_miss"] == pytest.approx(-W_MISS)


# ------------------------------------------------------------------ exploits
def test_never_pressing_use_loses_to_opening_the_route_door() -> None:
    """Abstaining saves at most the cap; getting through one on-path door pays more, even
    after wasting the whole cap on useless presses first."""
    spam = [(k % 2, at(x=20.0, tic=4 * k), 0.0) for k in range(1, 161)]          # 80 misses: cap spent
    opened = spam + [(0, at(tic=700), 0.0), (1, at(tic=704), 8.0), (0, at(tic=708), 64.0)] + _walk(32.0, 712)
    idle = [(0, i, g) for _, i, g in opened[:-5]] + [(0, at(tic=712 + 4 * k), 0.0) for k in range(5)]
    r_open, out_open, _ = play(opened, start=at(x=20.0))
    r_idle, out_idle, _ = play(idle, start=at(x=20.0))
    assert ep(out_idle)["paid/use_miss"] == 0.0 and ep(out_idle)["paid/door"] == 0.0
    assert ep(out_open)["paid/use_miss"] == pytest.approx(-CAP_MISS) and ep(out_open)["paid/door"] == pytest.approx(0.3)
    assert sum(r_open) > sum(r_idle)


def test_reversing_then_passing_still_beats_not_opening() -> None:
    steps = REVERSAL + [(1, at(tic=100), 8.0), (0, at(tic=104), 64.0)] + _walk(32.0, 108)
    _, out, _ = play(steps)
    e = ep(out)
    assert e["paid/door"] + e["paid/door_reversal"] + e["paid/use_miss"] > 0.2


def test_toggling_a_door_never_pays() -> None:
    steps, tic = [], 0
    for _ in range(20):
        tic += 200
        steps += [(1, at(tic=tic), 8.0), (0, at(tic=tic + 4), 16.0), (1, at(tic=tic + 8), 8.0),
                  (0, at(tic=tic + 12), 0.0)]
    rs, out, _ = play(steps)
    assert max(rs) < 0.0 and ep(out)["paid/door"] == 0.0
    assert ep(out)["paid/door_reversal"] == pytest.approx(-CAP_REV)


# ------------------------------------------------------------------ config / validation
def test_preset_values_and_validation() -> None:
    v = FULL_LEVEL_DOORS_V2_REWARD
    assert v.name == "full_level_doors_v2" and v.door_on_path_only
    assert (v.use_miss, v.use_miss_cap, v.door_reversal, v.door_reversal_cap) == (0.003, 0.15, 0.05, 0.2)
    assert dataclasses.replace(v, name="full_level_doors", use_miss=0.0, use_miss_cap=0.0, door_reversal=0.0,
                               door_reversal_cap=0.0, door_on_path_only=False) == FULL_LEVEL_DOORS_REWARD
    check_reward_setup(v, FULL_LEVEL_COVERAGE, ENGINE, 1050)
    # the exit stays the largest single payment, and larger than everything the penalties can take
    assert v.level_exit == 1.0 > v.door > max(v.use_miss_cap, v.door_reversal_cap)
    assert v.use_miss_cap + v.door_reversal_cap + v.time * 1050 < v.death < v.level_exit
    bad = (dict(use_miss=-0.001), dict(use_miss_cap=float("inf")), dict(use_miss_cap=0.0),
           dict(use_miss_cap=0.001),                       # cap below one event
           dict(use_miss=0.0),                             # a cap without its term
           dict(door_reversal_cap=0.01), dict(door_reversal=float("nan")),
           dict(use_miss_cap=0.3),                         # cap not below one door
           dict(door_reversal_cap=0.3),
           dict(use_miss_cap=0.29, door_reversal_cap=0.29),   # with the time penalty: above death
           dict(door=0.0))                                 # on-path-only without the door term
    for b in bad:
        with pytest.raises(RewardConfigError):
            check_reward_setup(dataclasses.replace(v, **b), FULL_LEVEL_COVERAGE, ENGINE, 1050)
    with pytest.raises(RewardConfigError, match="exit"):
        check_reward_setup(dataclasses.replace(v, door=0.0, door_on_path_only=False, use_miss_cap=0.6,
                                               door_reversal_cap=0.5, death=2.0), FULL_LEVEL_COVERAGE, ENGINE, 1050)
    with pytest.raises(ValueError, match="diag table"):
        ShapedRewardEnv(ScriptEnv([at(), at()]), V2, FULL_LEVEL_COVERAGE, doors=((LUMP, (DOOR,)),))


def test_old_presets_have_the_new_terms_off() -> None:
    for old in (FULL_LEVEL_REWARD, FULL_LEVEL_DOORS_REWARD):
        assert not v2_terms_on(old)
        assert (old.use_miss, old.use_miss_cap, old.door_reversal, old.door_reversal_cap) == (0.0,) * 4
    assert FULL_LEVEL_DOORS_REWARD.door == 0.3 and FULL_LEVEL_REWARD.door == 0.0


def _golden_episode(reward, doors) -> list:
    xs = [-40.0 + 24.0 * k for k in range(12)] + [224.0 - 24.0 * k for k in range(6)]
    infos = [mk(x, 32.0, map="M", health=100.0 - (3.0 if k > 8 else 0), damage=(3.0 if k > 8 else 0),
                kills=float(k > 4)) for k, x in enumerate(xs)]
    infos[-1] = end(infos[-1], "timeout")
    env = ShapedRewardEnv(ScriptEnv(infos), reward, FULL_LEVEL_COVERAGE, 0, doors=doors)
    env.reset(seed=0)
    out = []
    for _ in range(len(infos) - 1):
        _, r, _, _, info = env.step(np.zeros(2, np.int64))   # 2-head action: the old presets never read it
        out.append([r, info["shaping"], info.get("shaping_episode")])
    return out


def test_old_presets_reproduce_the_snapshot_taken_before_this_change() -> None:
    """tests/data_reward_golden_pre_v2.json was written by this same script BEFORE
    rewards/ and configs/reward_config.py were edited for full_level_doors_v2."""
    with open(os.path.join(ROOT, "tests", "data_reward_golden_pre_v2.json"), encoding="utf-8") as f:
        gold = json.load(f)
    now = {"full_level": _golden_episode(FULL, None),
           "full_level_doors": _golden_episode(dataclasses.replace(FULL, name="full_level_doors", door=0.3),
                                               (("M", (DOOR,)),))}
    assert json.loads(json.dumps(now)) == gold            # same keys, same values, bit for bit


# ------------------------------------------------------------------ preset wiring
def test_run_doom_preset_wiring() -> None:
    from configs.a0_env import a0_train_env_config
    from train import run_doom
    ns = run_doom.parse_args(["--maps", "generated", "--shaping", "full_level_doors_v2", "--start-stage", "S1"])
    assert ns.shaping == "full_level_doors_v2"
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--shaping", "full_level_doors_v2"])       # my_way_home has no doors
    r, _ = run_doom.shaping_configs("full_level_doors_v2")
    assert r is FULL_LEVEL_DOORS_V2_REWARD
    assert run_doom.shaping_configs("full_level_doors")[0] is FULL_LEVEL_DOORS_REWARD
    assert run_doom.shaping_configs("full_level")[0] is FULL_LEVEL_REWARD
    _, cfg = run_doom.build_generated_maps(a0_train_env_config())
    fac = run_doom.build_env_factory(cfg, "full_level_doors_v2", 0, 1)   # no engine starts
    assert fac.reward is FULL_LEVEL_DOORS_V2_REWARD
    diag = dict(fac.diag)
    assert set(diag) == set(cfg.maps) == set(dict(fac.doors))
    assert all(dm.on_path is not None and dm.doors == dict(fac.doors)[lump] for lump, dm in diag.items())
    assert any(not all(dm.on_path) for dm in diag.values())             # the bundle does have side doors
    assert pickle.loads(pickle.dumps(fac)) == fac
    with pytest.raises(ValueError, match="diag"):
        dataclasses.replace(fac, diag=None)
    lump = next(l for l, dm in diag.items() if dm.doors)
    broken = tuple((l, dataclasses.replace(dm, on_path=None) if l == lump else dm) for l, dm in fac.diag)
    with pytest.raises(ValueError, match="route"):
        dataclasses.replace(fac, diag=broken)
    old = run_doom.build_env_factory(cfg, "full_level_doors", 0, 1)
    assert old.reward is FULL_LEVEL_DOORS_REWARD and isinstance(old, ShapedEnvFactory)
