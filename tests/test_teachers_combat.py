"""Combat teacher (teachers/combat.py + the controller's combat branch), no engine:
S0-S2 labels pinned bit-identical to the pre-combat teacher, engagement labels on
synthetic sightings, visibility rule on a fake label list, S3 job support."""
import json
import math
import os

import numpy as np
import pytest

from teacher_golden import MAP_IDS, synthetic_labels
from teacher_helpers import DOOR_SEC, TURN_BINS, FakeDoomEnv, corridor_route
from teachers.collect import CollectConfig, Job, make_jobs, run_episode
from teachers.combat import CombatConfig, Sighting, choose_target, incoming, sightings_of
from teachers.controller import (H_ATTACK, H_MOVE, H_STRAFE, H_TURN, MOVE_BACK, MOVE_NONE,
                                 STRAFE_NONE, ControllerConfig, TeacherController)

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# ---------------------------------------------------------------- S0-S2 unchanged
def test_fake_env_episodes_bit_identical_to_pre_combat_teacher() -> None:
    with open(os.path.join(FIX, "teacher_labels_pre_combat.json"), encoding="utf-8") as f:
        gold = json.load(f)["episodes"]
    assert len(gold) == 24
    for g in gold:
        env = FakeDoomEnv(corridor_route(), start_angle=g["start_angle"])
        arrays, summ = run_episode(env, env.route, Job(0, "DD0001", "synthetic", "S1", g["seed"], g["noise"]), 3)
        assert arrays["action"].tolist() == g["action"], (g["seed"], g["start_angle"], g["noise"])
        assert arrays["executed_action"].tolist() == g["executed_action"]
        assert not arrays["label_combat"].any()


@pytest.mark.parametrize("map_id", MAP_IDS)
@pytest.mark.parametrize("with_empty_monster_list", [False, True])
def test_real_route_labels_bit_identical_to_pre_combat_teacher(map_id, with_empty_monster_list) -> None:
    with open(os.path.join(FIX, "teacher_labels_pre_combat_maps.json"), encoding="utf-8") as f:
        gold = json.load(f)["labels"][map_id]
    assert synthetic_labels(map_id, monsters=with_empty_monster_list) == gold


# ---------------------------------------------------------------- engagement
def _mon(x, y, name="Zombieman", mid=1, vx=0.0, vy=0.0) -> Sighting:
    return Sighting(mid, name, float(x), float(y), vx, vy, 50, 10, 20)


SELF_PISTOL = np.zeros(34, np.float32)
SELF_PISTOL[2] = 50 / 200.0
SELF_PISTOL[4 + 2] = 1.0


def _obs(self_vec=SELF_PISTOL) -> dict:
    return {"prev_action": np.zeros(6, np.int64), "self": self_vec.copy()}


def _info(x, y, ang, monsters=(), projectiles=()) -> dict:
    return {"pos_x": x, "pos_y": y, "angle": ang, "keys_held": {},
            "door_gaps": {0: 128.0, DOOR_SEC: 0.0, 2: 128.0},
            "monsters": list(monsters), "projectiles": list(projectiles)}


def _ctrl() -> TeacherController:
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(corridor_route())
    return c


def test_aligned_monster_at_range_fire_and_stand() -> None:
    c = _ctrl()
    a = c.act(_obs(), _info(48.0, 48.0, 0.0, [_mon(300.0, 48.0)]))
    assert a[H_ATTACK] == 1 and a[H_MOVE] == MOVE_NONE and TURN_BINS[a[H_TURN]] == 0.0
    assert c.label_source == "combat"


def test_monster_to_the_left_turn_without_firing() -> None:
    c = _ctrl()
    a = c.act(_obs(), _info(48.0, 48.0, 0.0, [_mon(48.0, 80.0 + 48.0)]))   # 90 deg left
    assert a[H_ATTACK] == 0 and TURN_BINS[a[H_TURN]] == -10.0, "coarse left turn, no shot"


def test_fine_correction_fires_with_the_turn() -> None:
    c = _ctrl()
    # 4 degrees left at 400 units: tolerance 2.5, the 4-degree bin cancels it
    th = math.radians(4.0)
    a = c.act(_obs(), _info(48.0, 48.0, 0.0, [_mon(48.0 + 400 * math.cos(th), 48.0 + 400 * math.sin(th))]))
    assert TURN_BINS[a[H_TURN]] == -1.0 and a[H_ATTACK] == 1


def test_no_ammo_or_fist_keeps_the_route() -> None:
    for sv in (np.zeros(34, np.float32), ):
        c, ref = _ctrl(), _ctrl()
        info = _info(48.0, 48.0, 0.0, [_mon(300.0, 48.0)])
        a = c.act(_obs(sv), info)
        b = ref.act({"prev_action": np.zeros(6, np.int64)}, dict(info, monsters=[]))
        assert a.tolist() == b.tolist() and c.label_source != "combat"


def test_close_or_melee_monster_back_off() -> None:
    c = _ctrl()
    a = c.act(_obs(), _info(280.0, 48.0, 0.0, [_mon(400.0, 48.0, "Demon")]))
    assert a[H_MOVE] == MOVE_BACK and a[H_ATTACK] == 1
    c = _ctrl()
    a = c.act(_obs(), _info(280.0, 48.0, 0.0, [_mon(400.0, 48.0, "Zombieman")]))
    assert a[H_MOVE] == MOVE_NONE, "a hitscanner at 120 is not close enough to back off"


def test_back_off_only_with_room_behind() -> None:
    c = _ctrl()
    a = c.act(_obs(), _info(20.0, 48.0, 0.0, [_mon(80.0, 48.0, "Demon")]))   # wall right behind
    assert a[H_MOVE] == MOVE_NONE


def test_incoming_projectile_strafe_away() -> None:
    c = _ctrl()
    # projectile coming from the east, passing 20 units to the player's LEFT (north)
    ball = Sighting(9, "DoomImpBall", 250.0, 68.0, -10.0, 0.0, 10, 4, 4)
    a = c.act(_obs(), _info(48.0, 48.0, 0.0, [_mon(300.0, 48.0, "DoomImp")], [ball]))
    assert a[H_STRAFE] == 2, "strafe right, away from the side it passes on"
    miss = Sighting(9, "DoomImpBall", 250.0, 48.0, 0.0, 10.0, 10, 4, 4)   # moving away north
    assert incoming([miss], 48.0, 48.0, CombatConfig()) is None


def test_target_choice_nearest_with_melee_priority() -> None:
    cfg = CombatConfig()
    near, far = _mon(100.0, 0.0, mid=1), _mon(200.0, 0.0, mid=2)
    assert choose_target([far, near], 0.0, 0.0, cfg).id == 1
    demon = _mon(150.0, 0.0, "Demon", mid=3)
    assert choose_target([near, demon], 0.0, 0.0, cfg).id == 3, "150 * 0.6 < 100"
    assert choose_target([_mon(5000.0, 0.0)], 0.0, 0.0, cfg) is None, "out of engage range"


def test_route_state_frozen_during_combat_and_resumes() -> None:
    c, ref = _ctrl(), _ctrl()
    info0 = _info(48.0, 48.0, 0.0)
    c.act(_obs(), info0)
    ref.act(_obs(), info0)
    k, phase, steps = c.k, c.phase, c.phase_steps
    for _ in range(5):
        c.act(_obs(), _info(48.0, 48.0, 0.0, [_mon(300.0, 48.0)]))
    assert (c.k, c.phase, c.phase_steps) == (k, phase, steps)
    a = c.act(_obs(), _info(48.0, 48.0, 0.0))       # monster gone from view: route again
    b = ref.act(_obs(), _info(48.0, 48.0, 0.0))
    assert c.label_source == "nav" and a[H_ATTACK] == 0
    assert a.tolist() == b.tolist()


def test_give_up_on_a_target_engaged_too_long() -> None:
    cfg = ControllerConfig(combat=CombatConfig(max_engage_decisions=3))
    c = TeacherController(TURN_BINS, cfg, seed=0)
    c.reset(corridor_route())
    m = _mon(300.0, 48.0)
    for _ in range(3):
        c.act(_obs(), _info(48.0, 48.0, 0.0, [m]))
        assert c.label_source == "combat"
    c.act(_obs(), _info(48.0, 48.0, 0.0, [m]))
    assert c.label_source == "nav" and m.id in c.given_up


def test_combat_never_draws_from_the_controller_rng() -> None:
    c = _ctrl()
    state = c.rng.getstate()
    for ang in (0.0, 45.0, 180.0):
        c.act(_obs(), _info(48.0, 48.0, ang, [_mon(300.0, 48.0)]))
    assert c.rng.getstate() == state


# ---------------------------------------------------------------- visibility rule
class _Lab:
    def __init__(self, oid, name, cat, value, box, pos):
        self.object_id, self.object_name, self.object_category, self.value = oid, name, cat, value
        self.x, self.y, self.width, self.height = box
        self.object_position_x, self.object_position_y = pos
        self.object_velocity_x = self.object_velocity_y = 0.0


class _St:
    def __init__(self, labels, buf):
        self.labels, self.labels_buffer = labels, buf


class _G:
    def __init__(self, st):
        self.st = st

    def get_state(self):
        return self.st


def test_sightings_need_an_own_pixel_and_the_monster_category() -> None:
    buf = np.zeros((120, 160), np.uint8)
    buf[10:20, 10:20] = 7                       # monster 1 visible
    buf[30:40, 30:40] = 9                       # projectile visible
    labs = [_Lab(1, "Zombieman", "Monster", 7, (10, 10, 10, 10), (5.0, 6.0)),
            _Lab(2, "DoomImp", "Monster", 8, (50, 50, 10, 10), (7.0, 8.0)),       # hidden: no pixel
            _Lab(3, "Zombieman", "Dead", 11, (10, 10, 10, 10), (1.0, 1.0)),        # not Monster
            _Lab(4, "DoomImpBall", "Explosive", 9, (30, 30, 10, 10), (2.0, 2.0))]
    mons, proj = sightings_of(_G(_St(labs, buf)))
    assert [m.id for m in mons] == [1] and (mons[0].x, mons[0].y) == (5.0, 6.0)
    assert [p.id for p in proj] == [4]


def test_fake_game_without_labels_gives_no_sightings() -> None:
    env = FakeDoomEnv(corridor_route())
    env.reset()
    assert sightings_of(env.game) == ([], [])


def test_make_jobs_accepts_s3() -> None:
    lumps = {"DD0001": "S0-train-100001", "DD0003": "S3-train-103001"}
    stages = {"DD0001": "S0", "DD0003": "S3"}
    jobs = make_jobs(CollectConfig(stages=("S3",), episodes_per_map=2, out=None), lumps, stages)
    assert [j.stage for j in jobs] == ["S3", "S3"]


# ---------------------------------------------------------------- engine replay (S0-S2)
def test_engine_replay_s0_s2_labels_identical_to_saved_dataset() -> None:
    """Re-run the teacher on saved pre-combat bc_v4map episodes (one per stage, one
    engine): labels must be bit-identical. Skips without the dataset or memory."""
    psutil = pytest.importorskip("psutil")
    root = os.path.dirname(FIX.rstrip(os.sep))
    data = os.path.join(os.path.dirname(root), "data", "teacher", "bc_v4map")
    if not os.path.exists(os.path.join(data, "index.jsonl")):
        pytest.skip("data/teacher/bc_v4map not present")
    if psutil.virtual_memory().available < 2.5 * 2 ** 30:
        pytest.skip("under 2.5 GiB available: no engine start")
    from teachers.replay_check import replay
    res = replay(data, ("S0", "S1", "S2"), 1)
    assert res["n"] == 3 and res["labels_identical"], res["episodes"]
