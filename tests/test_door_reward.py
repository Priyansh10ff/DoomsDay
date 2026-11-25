"""Door-progress reward (rewards/doors.py, maps/doors.py, preset full_level_doors).

Real door geometry from generated TRAIN maps; synthetic position sequences. No engine,
no torch."""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import pickle

import numpy as np
import pytest

from reward_test_helpers import FULL, ROOT, ScriptEnv, end, mk

from configs.reward_config import (FULL_LEVEL_COVERAGE, FULL_LEVEL_DOORS_REWARD, FULL_LEVEL_REWARD,
                                   TERMS)
from maps.doors import AXIS_X, DoorSpec, door_specs, door_table
from maps.qa.validate import load_map
from rewards.doors import DoorProgress
from rewards.validate import RewardConfigError, check_reward_setup
from rewards.wrapper import INFO_EPISODE, INFO_STEP, ShapedRewardEnv

S1_ONE = "S1-train-101001"     # 1 door
S1_MANY = "S1-train-101002"    # 5 doors
S2_KEYED = "S2-train-102001"   # 2 doors + 1 locked door
MARGIN = FULL_LEVEL_DOORS_REWARD.door_near_margin
W = FULL_LEVEL_DOORS_REWARD.door
LUMP = "DDTEST"
DOORS_FULL = dataclasses.replace(FULL, name="full_level_doors", door=W)


def wad(map_id: str) -> str:
    return os.path.join(ROOT, "maps", "train", map_id + ".wad")


def doors_of(map_id: str) -> tuple[DoorSpec, ...]:
    return door_specs(load_map(wad(map_id)))


def pt(d: DoorSpec, along: float, lat: float | None = None) -> tuple[float, float]:
    """World (x, y) of a point given in the door's frame (lateral default: centre)."""
    lat = (d.lo + d.hi) / 2.0 if lat is None else lat
    return (along, lat) if d.axis == AXIS_X else (lat, along)


def run(doors: tuple[DoorSpec, ...], path: list[tuple[float, float]]) -> tuple[list[float], DoorProgress]:
    tr = DoorProgress(((LUMP, doors),), W, MARGIN)
    tr.reset({"map": LUMP, "pos_x": path[0][0], "pos_y": path[0][1]})
    return [tr.step({"map": LUMP, "pos_x": x, "pos_y": y}) for x, y in path[1:]], tr


def walk(d: DoorSpec, frm: float, to: float, step: float = 16.0) -> list[tuple[float, float]]:
    n = max(1, int(abs(to - frm) / step))
    return [pt(d, frm + (to - frm) * k / n) for k in range(n + 1)]


# ------------------------------------------------------------------ geometry
def test_door_count_matches_generator_record_on_every_train_map() -> None:
    paths = sorted(glob.glob(os.path.join(ROOT, "maps", "train", "*.wad")))
    assert len(paths) == 80
    for p in paths:
        c = json.load(open(p[:-4] + ".json", encoding="utf-8"))["counts"]
        ds = door_specs(load_map(p))
        assert len(ds) == c["doors"] + c["locked_doors"], p
        assert all(d.a1 - d.a0 == 32.0 and d.hi > d.lo for d in ds), p


def test_door_table_is_picklable_and_hashable() -> None:
    t = door_table(wad(S1_ONE), ["MAP01"])
    assert pickle.loads(pickle.dumps(t)) == t and hash(t) == hash(pickle.loads(pickle.dumps(t)))


# ------------------------------------------------------------------ crossing rule
@pytest.mark.parametrize("map_id", [S1_ONE, S1_MANY, S2_KEYED])
def test_pass_through_pays_once_in_either_direction(map_id: str) -> None:
    for d in doors_of(map_id):
        for frm, to in ((d.a0 - 100, d.a1 + 100), (d.a1 + 100, d.a0 - 100)):
            raw, tr = run((d,), walk(d, frm, to))
            assert sum(raw) == 1.0 and tr.passed == 1, (map_id, d)


@pytest.mark.parametrize("map_id", [S1_ONE, S2_KEYED])
def test_approach_and_retreat_pays_nothing(map_id: str) -> None:
    for d in doors_of(map_id):
        # up to the door face, into the slab (an open door), and back out the same side
        path = walk(d, d.a0 - 120, d.a0 - 1) + walk(d, d.a0 + 16, d.a0 - 120)
        raw, _ = run((d,), path)
        assert sum(raw) == 0.0


def test_back_and_forth_pays_once() -> None:
    d = doors_of(S1_ONE)[0]
    path = []
    for _ in range(4):
        path += walk(d, d.a0 - 100, d.a1 + 100) + walk(d, d.a1 + 100, d.a0 - 100)
    raw, tr = run((d,), path)
    assert sum(raw) == 1.0 and tr.passed == 1 and tr.episode_max == W


def test_other_doors_are_paid_separately() -> None:
    ds = doors_of(S1_MANY)
    assert len(ds) == 5
    path = []
    for d in ds:   # teleport-free is not needed: leaving one region forgets its side
        path += walk(d, d.a0 - 100, d.a1 + 100) + walk(d, d.a1 + 100, d.a0 - 100)
    raw, tr = run(ds, path)
    assert sum(raw) == 5.0 and tr.passed == 5 and max(raw) == 1.0


def test_a_step_that_skips_the_slab_still_pays() -> None:
    d = doors_of(S1_ONE)[0]
    raw, _ = run((d,), [pt(d, d.a0 - 46), pt(d, d.a1 + 46)])   # 124 units in one decision
    assert raw == [1.0]


def test_reaching_the_far_side_by_leaving_the_region_pays_nothing() -> None:
    """A closed door: the far side is only reachable around a loop, which leaves the
    near region and forgets the side."""
    d = doors_of(S1_ONE)[0]
    away = d.hi + 300.0
    path = [pt(d, d.a0 - 40), pt(d, d.a0 - 40, away), pt(d, d.a1 + 40, away), pt(d, d.a1 + 40),
            pt(d, d.a1 + 80)]
    raw, _ = run((d,), path)
    assert sum(raw) == 0.0
    # outside the opening laterally (beside the jamb) is not near either
    raw, _ = run((d,), [pt(d, d.a0 - 40, d.hi + 1), pt(d, d.a1 + 40, d.hi + 1)])
    assert sum(raw) == 0.0


def test_unknown_map_and_missing_fields_raise() -> None:
    tr = DoorProgress(((LUMP, doors_of(S1_ONE)),), W, MARGIN)
    with pytest.raises(KeyError):
        tr.reset({"map": "NOPE", "pos_x": 0.0, "pos_y": 0.0})
    with pytest.raises(KeyError):
        tr.reset({"map": LUMP})


# ------------------------------------------------------------------ wrapper + logging
def _door_episode(reward=DOORS_FULL, doors=None) -> list[dict]:
    d = doors_of(S1_ONE)[0]
    xs = walk(d, d.a0 - 100, d.a1 + 100) + walk(d, d.a1 + 100, d.a0 - 100)
    infos = [mk(x, y, map=LUMP) for x, y in xs]
    infos[-1] = end(infos[-1], "timeout")
    table = ((LUMP, (d,)),) if doors is None else doors
    env = ShapedRewardEnv(ScriptEnv(infos), reward, FULL_LEVEL_COVERAGE, 0,
                          doors=table if reward.door else None)
    env.reset(seed=0)
    out = []
    for _ in range(len(infos) - 1):
        *_, te, tr, info = env.step(np.zeros(2, np.int64))
        out.append(info)
        if te or tr:
            break
    return out


def test_wrapper_pays_and_logs_door_fields() -> None:
    out = _door_episode()
    ep = out[-1][INFO_EPISODE]
    assert ep["paid/door"] == pytest.approx(W) and ep["raw/door"] == 1.0
    assert ep["doors_passed"] == 1 and ep["doors_in_map"] == 1
    assert sum(i[INFO_STEP]["paid"]["door"] for i in out) == pytest.approx(W)
    assert ep["extrinsic"] == pytest.approx(sum(ep[f"paid/{t}"] for t in TERMS) + W)


def test_full_level_episode_has_no_door_keys_and_same_values() -> None:
    plain = _door_episode(reward=FULL)
    ep = plain[-1][INFO_EPISODE]
    assert not any("door" in k for k in ep)
    assert all("door" not in i[INFO_STEP]["paid"] and "door" not in i[INFO_STEP]["raw"] for i in plain)
    withd = _door_episode()[-1][INFO_EPISODE]
    for k, v in ep.items():
        if k in ("extrinsic", "total", "reward_config"):
            continue
        assert withd[k] == v, k
    assert withd["extrinsic"] == pytest.approx(ep["extrinsic"] + W)


def test_door_state_resets_between_episodes() -> None:
    """Episode 1 passes the door; episode 2 (same map) pays it again from zero. An
    episode that ends near the door on the near side, followed by a reset onto the far
    side, pays nothing in the new episode."""
    d = doors_of(S1_ONE)[0]
    ep1 = [mk(x, y, map=LUMP) for x, y in walk(d, d.a0 - 100, d.a1 + 100)]
    ep1[-1] = end(ep1[-1], "timeout")
    ep2 = [mk(x, y, map=LUMP) for x, y in walk(d, d.a0 - 100, d.a1 + 100)]
    ep2[-1] = end(ep2[-1], "timeout")
    ep3 = [mk(*pt(d, d.a0 - 40), map=LUMP), end(mk(*pt(d, d.a0 - 30), map=LUMP), "timeout")]
    ep4 = [mk(*pt(d, d.a1 + 40), map=LUMP), mk(*pt(d, d.a1 + 60), map=LUMP),
           end(mk(*pt(d, d.a1 + 80), map=LUMP), "timeout")]

    class Multi(ScriptEnv):
        def __init__(self, eps: list[list[dict]]) -> None:
            super().__init__(eps[0])
            self.eps, self.k = eps, -1

        def reset(self, *, seed=None, options=None):
            self.k += 1
            self.infos = self.eps[self.k]
            self.engine = [0.0] * (len(self.infos) - 1)
            return super().reset(seed=seed, options=options)

    env = ShapedRewardEnv(Multi([ep1, ep2, ep3, ep4]), DOORS_FULL, FULL_LEVEL_COVERAGE, 0,
                          doors=((LUMP, (d,)),))
    eps = []
    for _ in range(4):
        env.reset(seed=0)
        while True:
            *_, te, tr, info = env.step(np.zeros(2, np.int64))
            if te or tr:
                eps.append(info[INFO_EPISODE])
                break
    assert [e["doors_passed"] for e in eps] == [1, 1, 0, 0]
    assert [e["paid/door"] for e in eps] == pytest.approx([W, W, 0.0, 0.0])


# ------------------------------------------------------------------ config / preset
def test_presets_and_validation() -> None:
    assert FULL_LEVEL_REWARD.door == 0.0 and FULL_LEVEL_REWARD.name == "full_level"
    assert dataclasses.replace(FULL_LEVEL_DOORS_REWARD, name="full_level", door=0.0) == FULL_LEVEL_REWARD
    assert FULL_LEVEL_DOORS_REWARD.door == 0.3 < FULL_LEVEL_DOORS_REWARD.level_exit
    check_reward_setup(FULL_LEVEL_DOORS_REWARD, FULL_LEVEL_COVERAGE, (("map_exit_reward", 1.0),), 2000)
    for bad in (dict(door=1.0), dict(door=-0.1), dict(door=0.3, door_near_margin=0.0)):
        with pytest.raises(RewardConfigError):
            check_reward_setup(dataclasses.replace(FULL_LEVEL_DOORS_REWARD, **bad), FULL_LEVEL_COVERAGE,
                               (("map_exit_reward", 1.0),), 2000)
    with pytest.raises(ValueError, match="door table"):
        ShapedRewardEnv(ScriptEnv([mk(map=LUMP), mk(map=LUMP)]), DOORS_FULL, FULL_LEVEL_COVERAGE)


def test_run_doom_preset_wiring() -> None:
    from configs.a0_env import a0_train_env_config
    from train import run_doom
    assert run_doom.parse_args(["--maps", "generated", "--shaping", "full_level_doors"]).shaping == "full_level_doors"
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--shaping", "full_level_doors"])          # my_way_home has no doors
    r, c = run_doom.shaping_configs("full_level")
    assert r is FULL_LEVEL_REWARD and c == FULL_LEVEL_COVERAGE
    _, cfg = run_doom.build_generated_maps(a0_train_env_config())
    plain = run_doom.build_env_factory(cfg, "full_level", 0, 1)
    assert plain.doors is None and plain.reward is FULL_LEVEL_REWARD
    fac = run_doom.build_env_factory(cfg, "full_level_doors", 0, 1)     # no engine starts
    assert fac.reward == FULL_LEVEL_DOORS_REWARD
    table = dict(fac.doors)
    assert set(table) == set(cfg.maps) and sum(len(v) for v in table.values()) > 0
    assert pickle.loads(pickle.dumps(fac)) == fac
    with pytest.raises(ValueError, match="door table"):
        dataclasses.replace(fac, doors=None)
