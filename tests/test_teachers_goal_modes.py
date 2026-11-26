"""Goal-mode labels (teachers/goal_modes.py): the table is identical to the model-side copy
in configs/goal_modes.py; the route and explore teachers report the right id per
situation on synthetic states; collect.py saves `goal_mode` int64 (T,) per episode and
per-stage shares. No engine."""
import ast
import math
import os

import numpy as np
import pytest

import configs.goal_modes as CGM
import teachers.goal_modes as TGM
from teacher_helpers import DOOR_SEC, TURN_BINS, FakeDoomEnv, corridor_route
from teachers.collect import Job, _mode_share, run_episode
from teachers.combat import Sighting
from teachers.controller import MOVE_FWD, ControllerConfig, TeacherController
from teachers.navigator import Anchor, NavConfig, Route, distance_field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H_USE = 4


# ---------------------------------------------------------------- the table
def _int_constants(mod) -> dict:
    return {k: v for k, v in vars(mod).items() if k.isupper() and isinstance(v, int)}


def test_tables_identical() -> None:
    assert TGM.GOAL_MODES == CGM.GOAL_MODES
    assert TGM.N_GOAL_MODES == CGM.N_GOAL_MODES == 7
    assert _int_constants(TGM) == _int_constants(CGM)
    expect = {"EXPLORE": 0, "GET_KEY": 1, "GO_LOCKED_DOOR": 2, "GO_EXIT": 3, "FIGHT": 4,
              "OPEN_DOOR": 5, "RECOVER": 6}
    for name, i in expect.items():
        assert getattr(TGM, name) == i
        assert TGM.GOAL_MODES[i] == name.lower()


@pytest.mark.parametrize("rel", ["configs/goal_modes.py", "teachers/goal_modes.py"])
def test_tables_import_nothing_heavy(rel: str) -> None:
    """Torch-free, numpy-free, and the configs copy never reaches teachers/."""
    tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read())
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names.append(n.module or "")
    assert set(names) <= {"__future__"}, names


# ---------------------------------------------------------------- route teacher
def _obs(prev_use: int = 0, prev=None) -> dict:
    p = np.zeros(6, np.int64) if prev is None else np.asarray(prev, np.int64).copy()
    p[H_USE] = prev_use
    return {"prev_action": p}


def _info(x, y, ang, keys=None, gap=0.0, monsters=None) -> dict:
    return {"pos_x": x, "pos_y": y, "angle": ang, "keys_held": keys or {},
            "door_gaps": {0: 128.0, DOOR_SEC: gap, 2: 128.0}, "monsters": monsters or [],
            "projectiles": []}


def test_route_exit_leg_door_handling_and_after() -> None:
    r = corridor_route()
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(r)
    c.act(_obs(), _info(48.0, 48.0, 0.0))                 # approaching an ordinary door
    assert c.phase == "approach" and c.goal_mode == TGM.GO_EXIT
    d = r.anchors[0]
    c.reset(r)
    a = c.act(_obs(), _info(d.x, d.y, d.face))            # facing it in reach: press
    assert a[H_USE] == 1 and c.goal_mode == TGM.OPEN_DOOR
    c.act(_obs(prev_use=1), _info(d.x, d.y, d.face, gap=32.0))   # opening: push through
    assert c.phase == "push" and c.goal_mode == TGM.OPEN_DOOR
    (cx, cy), (dx, _) = d.door_center, d.door_dir
    c.act(_obs(), _info(cx + dx * 40.0, cy, d.face, gap=72.0))   # crossed: exit leg
    assert c.k == 1 and c.goal_mode == TGM.GO_EXIT


def _key_route() -> Route:
    """corridor_route with a blue lock and a blue key anchor first (in the start room)."""
    r = corridor_route(door_color="blue")
    g, cfg = r.grid, NavConfig()
    key = Anchor("key", 112.0, 16.0, key_color="blue")
    key.dist = distance_field(g, [g.cell_of(key.x, key.y)], cfg)
    return Route(r.map_id, r.stage, g, [key] + r.anchors, r.start, [], cfg, r.use_lines)


def test_route_key_leg_then_lock_leg() -> None:
    r = _key_route()
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(r)
    c.act(_obs(), _info(48.0, 48.0, 0.0, keys={"blue": False}))
    assert c.k == 0 and c.goal_mode == TGM.GET_KEY
    c.act(_obs(), _info(48.0, 48.0, 0.0, keys={"blue": True}))   # picked up: lock leg
    assert c.k == 1 and c.phase == "approach" and c.goal_mode == TGM.GO_LOCKED_DOOR
    d = r.anchors[1]
    a = c.act(_obs(), _info(d.x, d.y, d.face, keys={"blue": True}))
    assert a[H_USE] == 1 and c.goal_mode == TGM.OPEN_DOOR


def test_route_never_reports_explore_and_unstick_is_recover() -> None:
    r = corridor_route()
    cfg = ControllerConfig()
    c = TeacherController(TURN_BINS, cfg, seed=0)
    c.reset(r)
    fwd = np.array([MOVE_FWD, 0, 3, 0, 0, 0])
    modes = []
    for _ in range(cfg.stuck_window + 3):
        c.act(_obs(prev=fwd), _info(48.0, 48.0, 0.0))
        modes.append((c.label_source, c.goal_mode))
    assert ("unstick", TGM.RECOVER) in modes
    assert all(m != TGM.EXPLORE for _, m in modes)
    assert all((s == "unstick") == (m == TGM.RECOVER) for s, m in modes)


def test_combat_is_fight_and_route_resumes() -> None:
    r = corridor_route()
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(r)
    imp = Sighting(7, "DoomImp", 300.0, 48.0, 0.0, 0.0, 40, 10, 20)
    c.act(_obs(), _info(48.0, 48.0, 0.0, monsters=[imp]))
    assert c.label_source == "combat" and c.goal_mode == TGM.FIGHT
    c.act(_obs(), _info(48.0, 48.0, 0.0))
    assert c.label_source == "nav" and c.goal_mode == TGM.GO_EXIT


def test_goal_mode_is_deterministic() -> None:
    """The mode is deterministic with the episode seed, like the labels."""
    r = corridor_route()
    a1, _ = run_episode(FakeDoomEnv(r), r, Job(0, "DD0001", "synthetic", "S1", 7, 0.3), burst_max=3)
    a2, _ = run_episode(FakeDoomEnv(r), r, Job(0, "DD0001", "synthetic", "S1", 7, 0.3), burst_max=3)
    assert np.array_equal(a1["action"], a2["action"])
    assert np.array_equal(a1["goal_mode"], a2["goal_mode"])


# ---------------------------------------------------------------- collector
def test_npz_carries_goal_mode(tmp_path) -> None:
    r = corridor_route()
    arrays, summ = run_episode(FakeDoomEnv(r, start_angle=180.0), r,
                               Job(3, "DD0001", "synthetic", "S1", 11, 0.0), burst_max=3)
    T = summ["length"]
    p = tmp_path / "ep_000003.npz"
    np.savez_compressed(p, **arrays)
    with np.load(p) as z:
        gm = z["goal_mode"]
        assert gm.shape == (T,) and gm.dtype == np.int64 and z["action"].shape[0] == T
    assert set(np.unique(gm)) <= {TGM.GO_EXIT, TGM.OPEN_DOOR, TGM.RECOVER}
    assert TGM.OPEN_DOOR in gm and TGM.GO_EXIT in gm
    assert sum(summ["goal_mode_counts"]) == T and len(summ["goal_mode_counts"]) == TGM.N_GOAL_MODES
    # an older dataset without goal_mode still loads (the key is simply absent)
    old = {k: v for k, v in arrays.items() if k != "goal_mode"}
    np.savez_compressed(tmp_path / "old.npz", **old)
    with np.load(tmp_path / "old.npz") as z:
        assert "goal_mode" not in z.files and z["action"].shape[0] == T


def test_mode_share_per_stage() -> None:
    eps = [{"stage": "S1", "goal_mode_counts": [0, 0, 0, 6, 0, 2, 0]},
           {"stage": "S1", "goal_mode_counts": [0, 0, 0, 2, 0, 0, 0]},
           {"stage": "S1"}]                                  # pre-goal-mode episode: skipped
    s = _mode_share(eps)
    assert s["go_exit"] == 0.8 and s["open_door"] == 0.2 and set(s) == set(TGM.GOAL_MODES)


# ---------------------------------------------------------------- explore teacher
def _xm(map_id):
    from maps.loader import resolve
    from teachers.explore import load_explore_map
    return load_explore_map(resolve(map_id, "train"), map_id=map_id)


def _xinfo(x, y, ang, xm, keys=None):
    return {"pos_x": x, "pos_y": y, "angle": ang, "keys_held": keys or {},
            "door_gaps": {s: 0.0 for s in xm.door_sectors}, "monsters": [], "projectiles": []}


def _xobs(prev=None):
    return {"prev_action": np.zeros(6, np.int64) if prev is None else np.asarray(prev, np.int64),
            "self": np.zeros(20, np.float32)}


def test_explore_frontier_is_explore_then_exit_is_go_exit() -> None:
    from teachers.explore import ExploreController
    from teachers.navigator import VOID
    xm = _xm("S0-train-100003")
    c = ExploreController(TURN_BINS, ControllerConfig())
    c.reset(xm)
    x, y = xm.start
    c.act(_xobs(), _xinfo(x, y, 0.0, xm))
    assert c.goal is not None
    if c.goal[0] in ("frontier", "doorf", "spin"):
        assert c.goal_mode in (TGM.EXPLORE, TGM.OPEN_DOOR, TGM.RECOVER)
        if c.press_target is None and c.anchor.kind != "door":
            assert c.goal_mode == TGM.EXPLORE
    c.kn.seen[:] = xm.grid.sector != VOID
    c.kn.exit_seen = True
    c.kn.version += 1
    c.act(_xobs(), _xinfo(x, y, 0.0, xm))
    assert c.goal == ("exit",)
    if c.press_target is None and not (c.anchor.kind == "door" and c.phase in ("align", "push")):
        assert c.goal_mode == TGM.GO_EXIT


@pytest.mark.parametrize("goal,expect", [(None, TGM.EXPLORE), (("spin",), TGM.EXPLORE),
                                         (("frontier", (3, 4)), TGM.EXPLORE),
                                         (("doorf", 5), TGM.EXPLORE), (("key", 0), TGM.GET_KEY),
                                         (("lockdoor", 5), TGM.GO_LOCKED_DOOR),
                                         (("exit",), TGM.GO_EXIT)])
def test_explore_goal_kinds_map_to_ids(goal, expect) -> None:
    from teachers.explore import ExploreController
    xm = _xm("S0-train-100003")
    c = ExploreController(TURN_BINS, ControllerConfig())
    c.reset(xm)
    c.goal = goal
    c.label_source, c.press_target, c.phase = "nav", None, "approach"
    assert c._goal_mode() == expect
    c.label_source = "combat"
    assert c._goal_mode() == TGM.FIGHT
    c.label_source = "unstick"
    assert c._goal_mode() == TGM.RECOVER
    c.label_source, c.press_target = "nav", ("door", 5)
    assert c._goal_mode() == TGM.OPEN_DOOR


def test_explore_unknown_locked_door_press_is_open_door() -> None:
    from maps.loader import resolve
    from teachers.explore import ExploreController
    from teachers.navigator import load_route
    map_id = "S2-train-102001"
    xm = _xm(map_id)
    door = next(a for a in load_route(resolve(map_id, "train")).anchors
                if a.kind == "door" and a.door_color)
    c = ExploreController(TURN_BINS, ControllerConfig())
    c.reset(xm)
    a = c.act(_xobs(), _xinfo(door.x, door.y, door.face, xm))
    assert a[H_USE] == 1 and c.goal_mode == TGM.OPEN_DOOR
