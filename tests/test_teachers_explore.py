"""Exploration teacher (teachers/explore.py) on real TRAIN maps, no engine: it plans only
over what the player has seen, ignores the exit until it is seen, learns locks only from
the needs-key pulse, and `--teacher` defaults to the route teacher."""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from maps.loader import resolve
from teachers.collect import parse_args
from teachers.controller import ControllerConfig
from teachers.explore import (KEY_COLOURS, ExploreConfig, ExploreController, Knowledge,
                              load_explore_map)
from teachers.navigator import VOID, load_route

TURN_BINS = (-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0)
H_USE = 4


def _xm(map_id):
    return load_explore_map(resolve(map_id, "train"), map_id=map_id)


def _info(x, y, ang, xm, keys=None, gaps=None):
    return {"pos_x": x, "pos_y": y, "angle": ang, "keys_held": keys or {},
            "door_gaps": gaps if gaps is not None else {s: 0.0 for s in xm.door_sectors},
            "monsters": [], "projectiles": []}


def _obs(prev=None, needs=None):
    sv = np.zeros(20, np.float32)
    if needs is not None:
        sv[-3 + KEY_COLOURS.index(needs)] = 1.0
    return {"prev_action": np.zeros(6, np.int64) if prev is None else np.asarray(prev, np.int64),
            "self": sv}


@pytest.mark.parametrize("map_id", ["S1-train-101002", "S2-train-102001", "S3-train-103002"])
def test_plans_only_over_seen_cells(map_id: str) -> None:
    xm = _xm(map_id)
    ctrl = ExploreController(TURN_BINS, ControllerConfig())
    ctrl.reset(xm)
    x, y = xm.start
    for ang in (0.0, 30.0, 60.0):
        ctrl.act(_obs(), _info(x, y, ang, xm))
    kn = ctrl.kn
    assert 0 < kn.seen.sum() < np.count_nonzero(xm.grid.sector != VOID)
    kg = ctrl.route.grid
    unseen_open = (~kn.seen) & (xm.grid.door_of == VOID)
    assert (kg.sector[unseen_open] == VOID).all(), "an unseen cell is walkable in the plan"
    for a in ctrl.route.anchors:
        if a.dist is not None:
            assert not np.isfinite(a.dist[unseen_open]).any(), "a distance field reaches unseen space"


def test_rays_stop_at_closed_doors() -> None:
    xm = _xm("S1-train-101002")
    r = load_route(resolve("S1-train-101002", "train"))
    door = next(a for a in r.anchors if a.kind == "door")
    beyond = xm.grid.cell_of(door.pass_x + door.door_dir[0] * 64, door.pass_y + door.door_dir[1] * 64)
    closed = Knowledge(xm)
    closed.observe(door.x, door.y, door.face, {s: 0.0 for s in xm.door_sectors})
    assert door.door_sector in closed.doors_seen
    assert not closed.seen[beyond]
    opened = Knowledge(xm)
    opened.observe(door.x, door.y, door.face, {s: 72.0 for s in xm.door_sectors})
    assert opened.seen[beyond]


def test_exit_ignored_until_seen() -> None:
    xm = _xm("S0-train-100001")
    ctrl = ExploreController(TURN_BINS, ControllerConfig())
    ctrl.reset(xm)
    x, y = xm.start
    ctrl.act(_obs(), _info(x, y, 0.0, xm))
    if not ctrl.kn.exit_seen:
        assert ctrl.goal is None or ctrl.goal[0] != "exit"
        assert all(a.kind not in ("exit_pad", "exit_switch") for a in ctrl.route.anchors)
    # reveal the whole map: the exit becomes the goal
    ctrl.kn.seen[:] = xm.grid.sector != VOID
    ctrl.kn.exit_seen = True
    ctrl.kn.version += 1
    ctrl.act(_obs(), _info(x, y, 0.0, xm))
    assert ctrl.goal == ("exit",)
    assert ctrl.route.anchors[-1].kind in ("exit_pad", "exit_switch")


def test_lock_learned_only_from_pulse() -> None:
    map_id = "S2-train-102001"
    xm = _xm(map_id)
    r = load_route(resolve(map_id, "train"))
    door = next(a for a in r.anchors if a.kind == "door" and a.door_color)
    ctrl = ExploreController(TURN_BINS, ControllerConfig())
    ctrl.reset(xm)
    info = _info(door.x, door.y, door.face, xm)
    a1 = ctrl.act(_obs(), info)
    assert ctrl.kn.locked == {}
    assert a1[H_USE] == 1, "an unknown locked door in reach is pressed once"
    pressed = np.zeros(6, np.int64)
    pressed[H_USE] = 1
    ctrl.act(_obs(prev=pressed), info)                 # pressed, but no pulse: nothing learned
    assert ctrl.kn.locked == {}
    ctrl.act(_obs(prev=pressed, needs=door.door_color), info)
    assert ctrl.kn.locked == {door.door_sector: door.door_color}
    a4 = ctrl.act(_obs(), info)
    assert a4[H_USE] == 0, "a door known locked without its key is not pressed again"
    a5 = ctrl.act(_obs(), _info(door.x, door.y, door.face, xm, keys={door.door_color: True}))
    assert a5[H_USE] == 1, "with the key held it is pressed"


def test_pulse_without_press_learns_nothing() -> None:
    map_id = "S2-train-102001"
    xm = _xm(map_id)
    r = load_route(resolve(map_id, "train"))
    door = next(a for a in r.anchors if a.kind == "door" and a.door_color)
    ctrl = ExploreController(TURN_BINS, ControllerConfig())
    ctrl.reset(xm)
    info = _info(door.x, door.y, door.face, xm)
    ctrl.act(_obs(), info)
    ctrl.act(_obs(needs=door.door_color), info)       # previous executed USE was up
    assert ctrl.kn.locked == {}


def test_frontier_goal_is_seen_and_ahead() -> None:
    xm = _xm("S0-train-100003")
    ctrl = ExploreController(TURN_BINS, ControllerConfig())
    ctrl.reset(xm)
    x, y = xm.start
    ctrl.act(_obs(), _info(x, y, 0.0, xm))
    assert ctrl.goal is not None and ctrl.goal[0] in ("frontier", "doorf", "exit", "spin")
    if ctrl.goal[0] == "frontier":
        assert ctrl.kn.seen[ctrl.goal[1]]


def test_dagger_relabels_from_any_pose() -> None:
    """student_driven: the executed action differs from the label and the pose jumps;
    the teacher re-plans from the new pose over seen cells only, and a student-executed
    press followed by a pulse still teaches the lock."""
    map_id = "S2-train-102001"
    xm = _xm(map_id)
    r = load_route(resolve(map_id, "train"))
    door = next(a for a in r.anchors if a.kind == "door" and a.door_color)
    ctrl = ExploreController(TURN_BINS, ControllerConfig(student_driven=True))
    ctrl.reset(xm)
    rng = np.random.default_rng(0)
    open_cells = np.argwhere(xm.grid.sector != VOID)
    poses = [xm.start] + [xm.grid.center(*open_cells[i]) for i in rng.choice(len(open_cells), 6)]
    for (x, y) in poses:
        ex = np.array([2, 1, 0, 0, 0, 1], np.int64)       # never what the teacher labels
        a = ctrl.act(_obs(prev=ex), _info(x, y, float(rng.uniform(0, 360)), xm))
        assert a.shape == (6,)
        unseen_open = (~ctrl.kn.seen) & (xm.grid.door_of == VOID)
        assert (ctrl.route.grid.sector[unseen_open] == VOID).all()
    info = _info(door.x, door.y, door.face, xm)
    ctrl.act(_obs(), info)
    student_press = np.array([0, 0, 3, 0, 1, 0], np.int64)
    ctrl.act(_obs(prev=student_press, needs=door.door_color), info)
    assert ctrl.kn.locked == {door.door_sector: door.door_color}


FIXTURE = Path(__file__).parent / "fixtures" / "explore_nonloop_states.json.gz"


def _replay(ep: dict, xcfg) -> list[tuple[int, bool, bool]]:
    """Replay recorded controller inputs open-loop; per step (t, label equal, mode equal)."""
    from teachers.combat import Sighting
    xm = _xm(ep["map_id"])
    from env.config import TURN_BINS as ENV_TURN_BINS
    ctrl = ExploreController(tuple(ENV_TURN_BINS), ControllerConfig(), seed=ep["seed"], xcfg=xcfg)
    ctrl.reset(xm)
    out = []
    for t, s in enumerate(ep["steps"]):
        obs = {"prev_action": np.asarray(s["prev_action"], np.int64), "self": np.asarray(s["self"], np.float32)}
        info = {"pos_x": s["pos_x"], "pos_y": s["pos_y"], "angle": s["angle"], "keys_held": s["keys_held"],
                "door_gaps": {int(k): v for k, v in s["door_gaps"].items()},
                "monsters": [Sighting(*m) for m in s["monsters"]],
                "projectiles": [Sighting(*p) for p in s["projectiles"]]}
        a = ctrl.act(obs, info)
        out.append((t, bool(np.array_equal(a, s["label"])), ctrl.goal_mode == s["goal_mode"]))
    return out


def test_door_loop_fix_keeps_labels_on_non_looping_episodes() -> None:
    """Recorded in the engine (noise 0.2) with the bc_v6_explore behaviour on 4 episodes
    with no door loop: the fixed teacher gives the same label and goal mode at every
    step, and so does the fix switched off (the replay itself is faithful)."""
    data = json.load(gzip.open(FIXTURE, "rt", encoding="utf-8"))
    assert {e["stage"] for e in data["episodes"]} >= {"S1", "S2", "S3"}
    for ep in data["episodes"]:
        for xcfg in (ExploreConfig(), ExploreConfig(crossed_max_across=0.0)):
            bad = [t for t, lab, mode in _replay(ep, xcfg) if not (lab and mode)]
            assert not bad, (ep["map_id"], xcfg.crossed_max_across, bad[:5])


def test_crossed_needs_to_be_near_the_door() -> None:
    """The loop cause: a door anchor planned from far off the door axis counted as crossed."""
    from teacher_helpers import corridor_route
    door = corridor_route().anchors[0]
    xm = _xm("S0-train-100003")
    (cx, cy), (dx, dy) = door.door_center, door.door_dir
    far_side = (cx + dx * 60.0 - dy * 500.0, cy + dy * 60.0 + dx * 500.0)   # beyond, 500 across
    through = (cx + dx * 60.0, cy + dy * 60.0)                              # beyond, on the axis
    for lim, expect_far in ((ExploreConfig().crossed_max_across, False), (0.0, True)):
        c = ExploreController(TURN_BINS, ControllerConfig(), xcfg=ExploreConfig(crossed_max_across=lim))
        c.reset(xm)                                  # same 32-unit cell as the corridor
        assert c._crossed(*through, door)
        assert c._crossed(*far_side, door) == expect_far


def test_teacher_flag_defaults_to_route() -> None:
    assert parse_args(["--episodes-per-map", "1", "--no-save"]).teacher == "route"
    assert parse_args(["--episodes-per-map", "1", "--no-save", "--teacher", "explore"]).teacher == "explore"
    with pytest.raises(SystemExit):
        parse_args(["--episodes-per-map", "1", "--teacher", "oracle"])
