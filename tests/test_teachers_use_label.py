"""Change C3: the teacher labels USE from what the player is looking at (synthetic poses
and a kinematic fake engine; one geometry check on real generated maps)."""
import math

import numpy as np
import pytest

from teacher_helpers import DOOR_COL, DOOR_SEC, TURN_BINS, FakeDoomEnv, corridor_route
from teachers.collect import Job, run_episode
from teachers.controller import H_TURN, H_USE, ControllerConfig, TeacherController
from teachers.use_label import (KIND_DOOR, KIND_SWITCH, KIND_WALL, USE_RANGE_ENGINE,
                                USE_RANGE_LABEL, facing_hit, make_use_lines, press_target,
                                wants_press)

FACE_X = DOOR_COL * 32.0          # the near door face of the corridor is the line x = 160
GAPS_CLOSED = {0: 128.0, DOOR_SEC: 0.0, 2: 128.0}


def _gaps(g: float) -> dict:
    return {0: 128.0, DOOR_SEC: g, 2: 128.0}


def _obs(prev_use: int = 0) -> dict:
    p = np.zeros(6, np.int64)
    p[H_USE] = prev_use
    return {"prev_action": p}


def _info(x, y, ang, gap=0.0, keys=None) -> dict:
    return {"pos_x": x, "pos_y": y, "angle": ang, "keys_held": keys or {}, "door_gaps": _gaps(gap)}


# ---------------------------------------------------------------- the press condition
def test_label_range_is_inside_the_engine_use_range() -> None:
    assert USE_RANGE_ENGINE == 64.0 and 48.0 <= USE_RANGE_LABEL < USE_RANGE_ENGINE


@pytest.mark.parametrize("dist,expect", [(20.0, True), (40.0, True), (USE_RANGE_LABEL - 1, True),
                                         (USE_RANGE_LABEL + 2, False), (90.0, False)])
def test_press_only_within_range_of_a_closed_door_face(dist, expect) -> None:
    L = corridor_route().use_lines
    assert wants_press(L, FACE_X - dist, 48.0, 0.0, GAPS_CLOSED, {}, 0) is expect


@pytest.mark.parametrize("ang,expect", [(0.0, True), (20.0, True), (60.0, False), (90.0, False),
                                        (180.0, False)])
def test_press_only_when_the_view_ray_hits_the_door_first(ang, expect) -> None:
    """From 30 units in front of the door's middle row: at 20 degrees the ray still meets
    the face inside range; at 60 it meets the corridor wall first; at 90/180 a wall."""
    L = corridor_route().use_lines
    assert wants_press(L, FACE_X - 30.0, 48.0, ang, GAPS_CLOSED, {}, 0) is expect


@pytest.mark.parametrize("gap", [1.0, 8.0, 32.0, 72.0])
def test_never_press_a_moving_or_open_door(gap) -> None:
    L = corridor_route().use_lines
    assert not wants_press(L, FACE_X - 30.0, 48.0, 0.0, _gaps(gap), {}, 0)


def test_never_two_presses_in_a_row() -> None:
    L = corridor_route().use_lines
    assert wants_press(L, FACE_X - 30.0, 48.0, 0.0, GAPS_CLOSED, {}, 0)
    assert not wants_press(L, FACE_X - 30.0, 48.0, 0.0, GAPS_CLOSED, {}, 1)


def test_no_press_from_inside_the_door_or_from_its_back_side() -> None:
    L = corridor_route().use_lines
    far_face_x = FACE_X + 32.0
    # standing beyond the door facing back west at its far face: front side, so a press is fine
    assert wants_press(L, far_face_x + 30.0, 48.0, 180.0, GAPS_CLOSED, {}, 0)
    # a door-face segment seen from its back side never counts
    back = make_use_lines([(FACE_X, 96.0, FACE_X, 0.0, KIND_DOOR, DOOR_SEC, None)])
    # v1 -> v2 runs south, so the front (right) side is WEST; the player stands east of it
    assert not wants_press(back, FACE_X + 30.0, 48.0, 180.0, GAPS_CLOSED, {}, 0)
    assert wants_press(back, FACE_X - 30.0, 48.0, 0.0, GAPS_CLOSED, {}, 0)


def test_occluding_wall_blocks_the_press() -> None:
    rows = [(FACE_X, 96.0, FACE_X, 0.0, KIND_DOOR, DOOR_SEC, None),
            (FACE_X - 10.0, 96.0, FACE_X - 10.0, 0.0, KIND_WALL, -1, None)]
    assert not wants_press(make_use_lines(rows), FACE_X - 30.0, 48.0, 0.0, GAPS_CLOSED, {}, 0)


def test_exit_switch_pressed_when_faced_in_range() -> None:
    sw = make_use_lines([(100.0, 96.0, 100.0, 0.0, KIND_SWITCH, -1, None)])
    assert press_target(sw, 60.0, 48.0, 0.0, {}, {}) == ("switch", -1)
    assert press_target(sw, 20.0, 48.0, 0.0, {}, {}) is None        # 80 units: out of range
    assert press_target(sw, 60.0, 48.0, 90.0, {}, {}) is None       # looking along it
    assert press_target(sw, 140.0, 48.0, 180.0, {}, {}) is None     # behind the switch wall


def test_locked_door_needs_its_key() -> None:
    L = corridor_route(door_color="red").use_lines
    assert not wants_press(L, FACE_X - 30.0, 48.0, 0.0, GAPS_CLOSED, {"red": False}, 0)
    assert wants_press(L, FACE_X - 30.0, 48.0, 0.0, GAPS_CLOSED, {"red": True}, 0)


def test_missing_door_gap_fails_loudly() -> None:
    L = corridor_route().use_lines
    with pytest.raises(KeyError):
        wants_press(L, FACE_X - 30.0, 48.0, 0.0, {0: 128.0}, {}, 0)


def test_facing_hit_distance() -> None:
    L = corridor_route().use_lines
    i, d = facing_hit(L, FACE_X - 25.0, 48.0, 0.0, 100.0)
    assert int(L.kind[i]) == KIND_DOOR and d == pytest.approx(25.0)


# ---------------------------------------------------------------- controller overlay
def test_label_depends_on_state_not_route_progress() -> None:
    """DAgger: a student facing a closed door in range gets the press label even when the
    teacher's route bookkeeping has moved on (here: already on the exit-pad anchor)."""
    r = corridor_route()
    for student in (False, True):
        c = TeacherController(TURN_BINS, ControllerConfig(student_driven=student), seed=0)
        c.reset(r)
        c.k = 1
        a = c.act(_obs(), _info(FACE_X - 30.0, 48.0, 0.0))
        assert a[H_USE] == 1, student
        # same pose with the door open: no press whatever the route says
        c.reset(r)
        assert c.act(_obs(), _info(FACE_X - 30.0, 48.0, 0.0, gap=40.0))[H_USE] == 0


def test_press_label_drops_a_turn_that_would_leave_the_target() -> None:
    """The engine turns (one tic) before it runs the use trace. From 30 units in front of
    the face, heading 55 degrees, the ray meets the face near its top edge (y ~ 91 of 96).
    A label turning left by 10 deg/tic would put the ray on the corridor wall on the first
    tic, so the press keeps the position and drops the turn; a right turn (toward the
    door's middle) keeps both."""
    r = corridor_route()
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(r)
    x, y, ang = FACE_X - 30.0, 48.0, 55.0
    info = _info(x, y, ang)
    left, right = TURN_BINS.index(-10.0), TURN_BINS.index(10.0)
    assert press_target(r.use_lines, x, y, ang, GAPS_CLOSED, {}) == ("door", DOOR_SEC)
    a = c._use_overlay(np.array([0, 0, left, 0, 0, 0]), x, y, ang, info, 0)
    assert a[H_USE] == 1 and a[H_TURN] == c.zero_turn
    a = c._use_overlay(np.array([0, 0, right, 0, 0, 0]), x, y, ang, info, 0)
    assert a[H_USE] == 1 and a[H_TURN] == right


def test_overlay_applies_to_unstick_labels() -> None:
    """Whatever produced the rest of the label, USE comes from the state."""
    r = corridor_route()
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(r)
    c.unstick_left = 4
    a = c.act(_obs(), _info(FACE_X - 30.0, 48.0, 0.0))
    assert c.label_source == "unstick" and a[H_USE] == 1


# ---------------------------------------------------------------- whole episodes (fake engine)
@pytest.mark.parametrize("start_angle", [0.0, 90.0, 180.0, 270.0])
def test_teacher_episode_presses_once_and_never_reverses(start_angle) -> None:
    r = corridor_route()
    env = FakeDoomEnv(r, start_angle=start_angle)
    arrays, summ = run_episode(env, r, Job(0, "DD0001", "synthetic", "S1", 4, 0.0), burst_max=3)
    assert summ["success"], summ
    assert env.presses == 1 and env.reversals == 0
    assert summ["use_labels"] == 1 and summ["door_presses"] == 1
    assert summ["press_open_target"] == 0 and summ["press_into_moving"] == 0
    assert summ["ineffective_presses"] == 0 and summ["max_presses_per_closed_spell"] == 1
    assert summ["reversals_label"] == 0


@pytest.mark.parametrize("seed", range(6))
def test_noisy_teacher_never_labels_a_press_at_a_moving_door(seed) -> None:
    r = corridor_route()
    env = FakeDoomEnv(r, max_steps=400)
    arrays, summ = run_episode(env, r, Job(0, "DD0001", "synthetic", "S1", seed, 0.3), burst_max=3)
    assert summ["success"], summ
    assert summ["press_open_target"] == 0 and summ["press_into_moving"] == 0
    assert summ["reversals_label"] == 0
    lab, ex = arrays["action"][:, H_USE], arrays["executed_action"][:, H_USE]
    # a press label is always a fresh edge against what actually ran
    assert all(ex[t - 1] == 0 for t in range(1, len(lab)) if lab[t] == 1)


def test_scripted_press_release_press_counts_one_reversal() -> None:
    """UseStats: a student that presses, releases, then presses again while the door is
    rising reverses it once (attributed to the executor, not the label)."""
    r = corridor_route()
    d = r.anchors[0]

    class _Script:
        def __init__(self):
            self.t = 0

        def begin(self, seed):
            self.t = 0

        def act(self, obs, start):
            a = np.array([0, 0, 3, 0, [1, 0, 1, 0, 0, 0][min(self.t, 5)], 0], np.int64)
            self.t += 1
            return a

    env = FakeDoomEnv(r, max_steps=8)
    env.route = r
    # start at the stand-off facing the door
    r2 = type(r)(r.map_id, r.stage, r.grid, r.anchors, (d.x, d.y), r.cell_path, r.cfg, r.use_lines)
    env.route = r2
    env.start_angle = d.face
    _, summ = run_episode(env, r2, Job(0, "DD0001", "synthetic", "S1", 1, 0.0), burst_max=3,
                          student=_Script(), beta=0.0)
    assert env.reversals == 1
    assert summ["reversals_other"] == 1 and summ["reversals_label"] == 0


# ---------------------------------------------------------------- real map geometry
@pytest.mark.parametrize("map_id", ["S1-train-101001", "S2-train-102001"])
def test_real_map_lines_have_two_faces_per_door(map_id) -> None:
    from maps.loader import resolve
    from teachers.navigator import load_route
    try:
        r = load_route(resolve(map_id, "train"), map_id=map_id)
    except FileNotFoundError:
        pytest.skip("generated maps not built")
    L = r.use_lines
    doors = [a for a in r.anchors if a.kind == "door"]
    assert doors
    for a in doors:
        faces = [i for i in range(len(L)) if L.kind[i] == KIND_DOOR and L.sector[i] == a.door_sector]
        assert len(faces) == 2, (a.door_sector, faces)
        # from the stand-off point, facing the door, the label would press it
        gaps = {int(s): 0.0 for s in set(L.sector.tolist()) if s >= 0}
        keys = {a.door_color: True} if a.door_color else {}
        assert press_target(L, a.x, a.y, a.face, gaps, keys) == ("door", a.door_sector)
    sw = [a for a in r.anchors if a.kind == "exit_switch"]
    for a in sw:
        assert press_target(L, a.x, a.y, a.face, {}, {}) == ("switch", -1)
