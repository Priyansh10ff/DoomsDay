"""Controller logic on synthetic poses (no engine): turn bins, USE edges against the
executed action, re-press, door crossing, unstick, locked doors, determinism; and the
collector's dataset contract on a kinematic fake env."""
import math

import numpy as np

from teacher_helpers import DOOR_SEC, TURN_BINS, FakeDoomEnv, corridor_route
from teachers.collect import CollectConfig, Job, make_jobs, run_episode
from teachers.controller import (H_MOVE, H_SPEED, H_TURN, H_USE, MOVE_BACK, MOVE_FWD,
                                 ControllerConfig, TeacherController, turn_bin)

HEADS = (3, 3, 7, 2, 2, 2)


def _obs(prev_use: int = 0, prev=None) -> dict:
    p = np.zeros(6, np.int64) if prev is None else np.asarray(prev, np.int64).copy()
    p[H_USE] = prev_use
    return {"prev_action": p}


def _info(x: float, y: float, ang: float, keys=None, gap: float = 0.0) -> dict:
    return {"pos_x": x, "pos_y": y, "angle": ang, "keys_held": keys or {},
            "door_gaps": {0: 128.0, DOOR_SEC: gap, 2: 128.0}}


def test_turn_bin_mapping() -> None:
    # bin value v changes the heading by -4 v degrees (positive value turns right)
    assert TURN_BINS[turn_bin(40.0, TURN_BINS, 4)] == -10.0
    assert TURN_BINS[turn_bin(-16.0, TURN_BINS, 4)] == 4.0
    assert TURN_BINS[turn_bin(0.0, TURN_BINS, 4)] == 0.0
    assert TURN_BINS[turn_bin(1.5, TURN_BINS, 4)] == 0.0      # dead band: no 4-degree jitter
    assert TURN_BINS[turn_bin(170.0, TURN_BINS, 4)] == -10.0


def test_action_shape_and_ranges() -> None:
    r = corridor_route()
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(r)
    for ang in (0.0, 90.0, 200.0):
        a = c.act(_obs(), _info(48.0, 48.0, ang))
        assert a.shape == (6,) and a.dtype == np.int64, "action must be int64 (6,)"
        assert all(0 <= a[i] < HEADS[i] for i in range(6))
        assert a[3] == 0, "attack head stays 0"


def test_turns_toward_target_and_moves_when_aligned() -> None:
    r = corridor_route()
    c = TeacherController(TURN_BINS, seed=0)
    c.reset(r)
    a = c.act(_obs(), _info(48.0, 48.0, 90.0))      # target is east, facing north
    assert a[H_MOVE] == 0 and TURN_BINS[a[H_TURN]] > 0, "turn right (clockwise), do not move"
    c.reset(r)
    a = c.act(_obs(), _info(48.0, 48.0, 0.0))       # facing the target
    assert a[H_MOVE] == MOVE_FWD and TURN_BINS[a[H_TURN]] == 0.0


def _at_door(c: TeacherController, r):
    """At the door's stand-off point (inside USE range), facing it squarely."""
    d = r.anchors[0]
    c.reset(r)
    a = c.act(_obs(), _info(d.x, d.y, d.face))
    return d, a


def test_use_is_a_fresh_edge_against_the_executed_action() -> None:
    r = corridor_route()
    c = TeacherController(TURN_BINS, seed=0)
    d, a = _at_door(c, r)
    assert a[H_USE] == 1 and a[H_MOVE] == 0, "facing a closed door in range with USE up: press"
    # the press did NOT execute (noise held USE up) and the door is still closed: ask again
    a = c.act(_obs(prev_use=0), _info(d.x, d.y, d.face))
    assert a[H_USE] == 1
    # USE held down by the executed action: release first, then press
    a = c.act(_obs(prev_use=1), _info(d.x, d.y, d.face))
    assert a[H_USE] == 0, "USE held down by the executed action: release before pressing"
    a = c.act(_obs(prev_use=0), _info(d.x, d.y, d.face))
    assert a[H_USE] == 1


def test_press_then_push_while_open_then_reapproach_when_shut_then_cross() -> None:
    r = corridor_route()
    c = TeacherController(TURN_BINS, ControllerConfig(), seed=0)
    d, a = _at_door(c, r)
    assert a[H_USE] == 1
    # the press executed and the door is rising: push, never press while gap > 0
    for gap in (32.0, 64.0, 72.0, 72.0, 40.0, 8.0):
        a = c.act(_obs(prev_use=1 if gap == 32.0 else 0), _info(d.x, d.y, d.face, gap=gap))
        assert c.phase == "push" and a[H_USE] == 0 and a[H_MOVE] == MOVE_FWD, gap
    # it shut before we crossed: face it again and press once (a re-press on a closed door)
    a = c.act(_obs(), _info(d.x, d.y, d.face, gap=0.0))
    assert c.phase == "align" and a[H_USE] == 1
    # crossing the door advances to the exit anchor
    (cx, cy), (dx, _) = d.door_center, d.door_dir
    c.act(_obs(), _info(cx + dx * 40.0, cy, d.face, gap=72.0))
    assert c.k == 1


def test_locked_door_not_pressed_without_key() -> None:
    r = corridor_route(door_color="blue")
    c = TeacherController(TURN_BINS, seed=0)
    d = r.anchors[0]
    c.reset(r)
    for _ in range(5):
        a = c.act(_obs(), _info(d.x, d.y, d.face, keys={"blue": False}))
        assert a[H_USE] == 0
    a = c.act(_obs(), _info(d.x, d.y, d.face, keys={"blue": True}))
    assert a[H_USE] == 1


def test_unstick_when_forward_makes_no_progress() -> None:
    r = corridor_route()
    cfg = ControllerConfig()
    c = TeacherController(TURN_BINS, cfg, seed=0)
    c.reset(r)
    # facing the door stand-off 76 units ahead: the label is plain forward (no run), and the
    # executed action equals it, so this is the teacher's own command failing to move us
    fwd = np.array([MOVE_FWD, 0, 3, 0, 0, 0])
    acts = [c.act(_obs(prev=fwd), _info(48.0, 48.0, 0.0)) for _ in range(cfg.stuck_window + 1)]
    assert np.array_equal(acts[0], fwd)
    assert any(a[H_MOVE] == MOVE_BACK for a in acts), "stuck against nothing: back off"


def test_deterministic_with_seed() -> None:
    r = corridor_route()
    out = []
    for _ in range(2):
        env = FakeDoomEnv(r)
        arrays, summ = run_episode(env, r, Job(0, "DD0001", "synthetic", "S1", 7, 0.3), burst_max=3)
        out.append((arrays, summ))
    for k in out[0][0]:
        assert np.array_equal(out[0][0][k], out[1][0][k]), k


def test_episode_contract_on_fake_env() -> None:
    r = corridor_route()
    env = FakeDoomEnv(r, start_angle=180.0)
    arrays, summ = run_episode(env, r, Job(3, "DD0001", "synthetic", "S1", 11, 0.0), burst_max=3)
    assert summ["success"], summ
    T = summ["length"]
    assert set(arrays) == {"rays", "entity_mask", "prev_action", "action", "executed_action",
                           "engine_reward", "done", "label_unstick", "student_driven"}
    assert not arrays["student_driven"].any()
    assert arrays["rays"].shape == (T, 2, 4) and arrays["rays"].dtype == np.float32
    assert arrays["entity_mask"].dtype == np.bool_ and arrays["prev_action"].dtype == np.int64
    assert arrays["action"].shape == (T, 6) and arrays["action"].dtype == np.int64
    assert arrays["executed_action"].shape == (T, 6) and arrays["executed_action"].dtype == np.int64
    assert arrays["engine_reward"].shape == (T,) and arrays["engine_reward"].dtype == np.float32
    assert arrays["done"].dtype == np.bool_ and arrays["done"][-1] and not arrays["done"][:-1].any()
    assert np.array_equal(arrays["action"], arrays["executed_action"]), "noise 0: label == executed"
    # obs[t] is the observation action[t] was chosen from: prev_action[t+1] == executed[t]
    assert np.array_equal(arrays["prev_action"][1:], arrays["executed_action"][:-1])
    # the door was opened with a USE edge
    use = arrays["executed_action"][:, H_USE]
    assert any(use[t] == 1 and (t == 0 or use[t - 1] == 0) for t in range(T))


def test_noise_executes_random_actions_but_records_teacher_labels() -> None:
    r = corridor_route()
    env = FakeDoomEnv(r)
    arrays, summ = run_episode(env, r, Job(0, "DD0001", "synthetic", "S1", 5, 0.5), burst_max=3)
    diff = np.any(arrays["action"] != arrays["executed_action"], axis=1)
    assert summ["noisy_decisions"] > 0 and diff.any()
    assert (arrays["executed_action"][:, 3] == 0).all(), "noise never fires"
    assert (arrays["action"][:, H_SPEED] <= 1).all()


def test_make_jobs_seeds_and_clean_episodes() -> None:
    lumps = {"DD0001": "S0-train-100001", "DD0002": "S1-train-101001", "DD0003": "S3-train-103001"}
    stages = {"DD0001": "S0", "DD0002": "S1", "DD0003": "S3"}
    cfg = CollectConfig(stages=("S0", "S1"), episodes_per_map=5, out=None, noise=0.2, clean_every=5, seed=2)
    jobs = make_jobs(cfg, lumps, stages)
    assert len(jobs) == 10 and {j.stage for j in jobs} == {"S0", "S1"}
    assert [j.episode for j in jobs] == list(range(10))
    assert len({j.seed for j in jobs}) == 10
    assert sum(j.noise == 0.0 for j in jobs) == 2, "every 5th episode of each map is clean"
    assert make_jobs(cfg, lumps, stages) == jobs, "job list is deterministic"


# ---------------------------------------------------------------- DAgger (student drives)
class _ConstStudent:
    """Stand-in for teachers.collect.StudentDriver: always the same action."""

    def __init__(self, action) -> None:
        self.action = np.asarray(action, np.int64)
        self.starts: list[bool] = []

    def begin(self, seed: int) -> None:
        self.starts = []

    def act(self, obs: dict, start: bool) -> np.ndarray:
        self.starts.append(start)
        return self.action.copy()


def test_student_driven_unstick_only_when_teacher_would_push_forward() -> None:
    r = corridor_route()
    cfg = ControllerConfig(student_driven=True)
    fwd = np.array([MOVE_FWD, 0, 3, 0, 0, 0])
    # facing AWAY from the route (west): the teacher's label is "turn", so a student that
    # pushes forward into the wall gets the turn label, never a scripted unstick
    c = TeacherController(TURN_BINS, cfg, seed=0)
    c.reset(r)
    acts = [c.act(_obs(prev=fwd), _info(48.0, 48.0, 180.0)) for _ in range(3 * cfg.stuck_window)]
    assert all(a[H_MOVE] != MOVE_BACK for a in acts) and c.label_source == "nav"
    assert all(a[H_TURN] != 3 for a in acts), "label keeps turning toward the route"
    # facing along the route in the top row, executed forward, no progress: unstick,
    # strafing to the side with more room (right = toward rows 1 and 0), never a coin flip
    sides = set()
    for seed in range(4):
        c = TeacherController(TURN_BINS, cfg, seed=seed)
        c.reset(r)
        acts = [c.act(_obs(prev=fwd), _info(48.0, 80.0, 0.0)) for _ in range(cfg.stuck_window + 1)]
        assert any(a[H_MOVE] == MOVE_BACK for a in acts) and c.label_source == "unstick"
        sides |= {int(a[1]) for a in acts if a[H_MOVE] == MOVE_BACK}
    assert sides == {2}, sides   # STRAFE_RIGHT


def test_student_driven_keeps_stuck_history_on_override() -> None:
    """Teacher mode drops the stuck history when the executed action differs from the label
    (a noise burst); student mode must not, or no student state is ever judged stuck."""
    r = corridor_route()
    fwd_run = np.array([MOVE_FWD, 0, 3, 0, 0, 1])     # differs from the teacher's label (no run)
    for student, expect in ((False, False), (True, True)):
        c = TeacherController(TURN_BINS, ControllerConfig(student_driven=student), seed=0)
        c.reset(r)
        acts = [c.act(_obs(prev=fwd_run), _info(48.0, 80.0, 0.0)) for _ in range(10)]
        assert any(a[H_MOVE] == MOVE_BACK for a in acts) == expect


def test_dagger_episode_records_teacher_labels_and_student_actions() -> None:
    r = corridor_route()
    still = [0, 0, 3, 0, 0, 0]                        # a student that never moves
    st = _ConstStudent(still)
    arrays, summ = run_episode(FakeDoomEnv(r, max_steps=50), r, Job(0, "DD0001", "synthetic", "S1", 3, 0.0),
                               burst_max=3, student=st, beta=0.0)
    assert not summ["success"] and summ["student_decisions"] == summ["length"] == 50
    assert (arrays["executed_action"] == np.asarray(still)).all()
    assert arrays["student_driven"].all()
    assert (arrays["action"][:, H_MOVE] == MOVE_FWD).any(), "labels are the teacher's, not the student's"
    assert np.array_equal(arrays["prev_action"][1:], arrays["executed_action"][:-1])
    assert st.starts[0] and not any(st.starts[1:]), "episode_start only on the first decision"
    # beta = 1: the teacher's label is always executed, so the teacher solves it
    arrays1, summ1 = run_episode(FakeDoomEnv(r), r, Job(1, "DD0001", "synthetic", "S1", 3, 0.0),
                                 burst_max=3, student=_ConstStudent(still), beta=1.0)
    assert summ1["success"] and summ1["student_decisions"] == 0
    assert np.array_equal(arrays1["action"], arrays1["executed_action"])
    # the cap ends the episode and is never a success
    _, summ2 = run_episode(FakeDoomEnv(r), r, Job(2, "DD0001", "synthetic", "S1", 3, 0.0),
                           burst_max=3, student=_ConstStudent(still), beta=0.0, max_steps=7)
    assert summ2["length"] == 7 and summ2["capped"] and not summ2["success"]


def test_dagger_cli_arguments() -> None:
    import pytest
    from teachers.collect import parse_args
    c = parse_args(["--episodes-per-map", "1", "--student", "m.pt", "--beta", "0.3", "--max-steps", "400"])
    assert c.student == "m.pt" and c.beta == 0.3 and c.max_steps == 400
    for bad in (["--beta", "0.5"], ["--student", "m.pt", "--noise", "0.2"], ["--student", "m.pt", "--beta", "2"]):
        with pytest.raises(SystemExit):
            parse_args(["--episodes-per-map", "1", *bad])
