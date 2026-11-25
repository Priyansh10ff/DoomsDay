"""Observation v2 through the real engine on generated TRAIN maps: line-of-sight door
channel, key bits, and the v1 contract. One engine at a time; the scripted teacher
(test-side only) drives the player. Expected door extents come from the WAD geometry
(maps.doors) with a pinhole projection, independent of env/door_cue's ray cast."""
import contextlib

import numpy as np
import pytest

from env import DoomEnv
from env import door_cue as D
from env.config import TRAIN_CONFIG

pytestmark = pytest.mark.filterwarnings("ignore")


def _bundle():
    from maps.loader import training_bundle
    bundle, lump_to_id, _ = training_bundle()
    return bundle, lump_to_id


def _cfg(v):
    from configs.a0_env import generated_maps_env_config
    bundle, lump_to_id = _bundle()
    base = TRAIN_CONFIG.replace(screen_enabled=False, obs_version=v, win_timer_ms=None)
    return generated_maps_env_config(base, bundle, tuple(sorted(lump_to_id)))


def _lump(map_id):
    return {v: k for k, v in _bundle()[1].items()}[map_id]


@contextlib.contextmanager
def make(v):
    env = DoomEnv(_cfg(v))
    try:
        yield env
    finally:
        env.close()


def _doors(map_id):
    from maps.doors import AXIS_X, door_specs
    from maps.loader import resolve
    from maps.qa.validate import load_map
    out = {}
    for d in door_specs(load_map(resolve(map_id, "train"), "MAP01")):
        if d.axis == AXIS_X:
            faces = [(d.a0, d.lo, d.a0, d.hi), (d.a1, d.lo, d.a1, d.hi)]
        else:
            faces = [(d.lo, d.a0, d.hi, d.a0), (d.lo, d.a1, d.hi, d.a1)]
        out[d.sector] = faces
    return out


def _project(face, x, y, ang):
    """Pinhole projection of a face's endpoints: (bin_lo, bin_hi, min forward dist) or None."""
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    pts = []
    for px, py in ((face[0], face[1]), (face[2], face[3])):
        dx, dy = px - x, py - y
        pts.append((dx * c + dy * s, -dx * s + dy * c))
    (f1, l1), (f2, l2) = pts
    near = 1.0
    if f1 <= near and f2 <= near:
        return None
    if f1 <= near or f2 <= near:     # clip the part behind the view plane
        u = (near - f1) / (f2 - f1)
        fc, lc = near, l1 + u * (l2 - l1)
        (f1, l1), (f2, l2) = ((fc, lc), (f2, l2)) if f1 <= near else ((f1, l1), (fc, lc))
    cols = sorted([80.0 - 80.0 * l1 / f1, 80.0 - 80.0 * l2 / f2])
    lo, hi = (int(np.floor(np.clip(v, -1e6, 1e6) / 2)) for v in cols)
    return lo, hi, min(f1, f2)


def _gaps(env):
    st = env.game.get_state()
    return {s.id: s.ceiling_height - s.floor_height for s in st.sectors}


def _drive(env, map_id, max_steps, on_step, drop_key=False, press_locked=False, seed=1):
    from maps.loader import resolve
    from teachers.controller import ControllerConfig, TeacherController
    from teachers.navigator import load_route
    from teachers.use_label import door_gaps_of
    route = load_route(resolve(map_id, "train"), map_id=map_id)
    if drop_key:
        route.anchors = [a for a in route.anchors if a.kind != "key"]
    ctrl = TeacherController(env.cfg.turn_bins, ControllerConfig(), seed=seed)
    obs, info = env.reset(seed=seed, options={"map": _lump(map_id)})
    ctrl.reset(route)
    on_step(-1, obs, info, ctrl, None)
    for t in range(max_steps):
        # the teacher's USE label reads live door gaps (as teachers/collect.py supplies them);
        # only the controller's copy gets them, `info` stays exactly what the env returned
        a = ctrl.act(obs, dict(info, door_gaps=door_gaps_of(env.game)))
        # aligned at a locked door (the teacher never labels a press there without the key)
        if press_locked and ctrl.phase == "align" and a[2] == ctrl.zero_turn and ctrl.anchor.door_color:
            a = a.copy()
            a[4] = t % 2
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            return obs, info, True
        if on_step(t, obs, info, ctrl, a) is False:
            break
    return obs, info, False


def _extent_bins(doors, gaps, x, y, ang, closed_only=True):
    bins = set()
    for sid, faces in doors.items():
        if closed_only and gaps.get(sid, 999) > D.DOOR_CAST_MAX_GAP:
            continue
        for f in faces:
            p = _project(f, x, y, ang)
            if p is None:
                continue
            lo, hi, _ = p
            bins.update(range(max(lo - 1, 0), min(hi + 1, 79) + 1))
    return bins


@pytest.mark.parametrize("map_id,need_hidden", [("S1-train-101004", True), ("S1-train-101001", False)])
def test_door_channel_matches_closed_door_extents(map_id, need_hidden):
    doors = _doors(map_id)
    stats = {"frames": 0, "facing_frames": 0, "covered": [], "hidden_checked": 0, "behind_zero": 0,
             "open_checked": 0}
    with make(2) as env:
        space = env.observation_space

        def check(t, obs, info, ctrl, a):
            assert space.contains(obs), "observation outside its declared space"
            assert set(obs) == set(space.spaces)
            assert not any("sector" in k or "door" in k for k in info)
            door = obs["rays"][3]
            assert set(np.unique(door)) <= {0.0, 1.0}
            x, y, ang = info["pos_x"], info["pos_y"], info["angle"]
            gaps = _gaps(env)
            ext = _extent_bins(doors, gaps, x, y, ang)
            lit = set(np.where(door > 0)[0].tolist())
            stats["frames"] += 1
            # never lit outside the projected extent of a closed / partly closed door
            assert lit <= ext, (t, sorted(lit - ext))
            if not ext:
                assert not lit
                stats["behind_zero"] += 1
            mid = obs["rays"][1] * 255.0
            for sid, faces in doors.items():
                for f in faces:
                    p = _project(f, x, y, ang)
                    if p is None or gaps[sid] > 0:
                        continue
                    lo, hi, fmin = p
                    if lo < 0 or hi > 79:
                        continue
                    cb = (lo + hi) // 2
                    pred = D.DEPTH_PER_UNIT * fmin + D.DEPTH_OFFSET
                    if 0 < mid[cb] < pred - 15:            # a nearer surface hides it here
                        others = _extent_bins({k: v for k, v in doors.items() if k != sid}, gaps, x, y, ang)
                        if cb not in others:
                            stats["hidden_checked"] += 1
                            assert door[cb] == 0, (t, sid, cb)
                    elif hi - lo >= 3 and fmin < 600 and abs(mid[cb] - pred) <= 2:
                        # visible face straight in view: most of its columns light up
                        inner = set(range(lo + 1, hi))
                        stats["covered"].append((round(fmin), len(inner & lit) / max(len(inner), 1)))
            # open doors: columns inside an open door's extent and no closed door's -> 0
            for sid, faces in doors.items():
                if gaps[sid] < 64:
                    continue
                own = _extent_bins({sid: faces}, gaps, x, y, ang, closed_only=False)
                closed = _extent_bins(doors, gaps, x, y, ang)
                free = own - closed
                if free:
                    stats["open_checked"] += 1
                    assert not (free & lit), (t, sid)
            return True

        _, info, ended = _drive(env, map_id, 600, check)
        assert ended and info["level_exit"]
    cov = stats["covered"]
    assert len(cov) >= 5, stats
    near = [c for d, c in cov if d < 150]
    far = [c for d, c in cov if d >= 150]
    assert near and far, cov
    assert np.mean([c for _, c in cov]) >= 0.8, cov
    assert stats["behind_zero"] > 0 and stats["open_checked"] > 0, stats
    if need_hidden:   # a closed door in the field of view behind a nearer wall was checked
        assert stats["hidden_checked"] > 0, stats


def test_door_channel_turning_in_front_of_closed_door():
    """At the door, turn left and right: the lit block shifts across the screen the other
    way and disappears once the door leaves the field of view."""
    map_id = "S1-train-101001"
    centres = []
    with make(2) as env:
        state = {"stop": False}

        def until_press(t, obs, info, ctrl, a):
            return not (a is not None and a[4] == 1)     # stop once the teacher pressed

        obs, info, ended = _drive(env, map_id, 200, until_press)
        assert not ended
        lit0 = np.where(obs["rays"][3] > 0)[0]
        assert len(lit0) >= 20
        turn_bins = env.cfg.turn_bins
        right = turn_bins.index(4.0)
        left = turn_bins.index(-4.0)
        for b in [right] * 3 + [left] * 6:
            obs, *_ = env.step(np.array([0, 0, b, 0, 0, 0]))
            lit = np.where(obs["rays"][3] > 0)[0]
            centres.append(lit.mean() if len(lit) else None)
        # turning right (positive delta = clockwise) moves the door left on screen
        assert centres[0] is not None and centres[2] is not None and centres[2] < centres[0] < lit0.mean() + 1
        later = [c for c in centres[3:] if c is not None]
        assert later and later[-1] > centres[2]
        for _ in range(12):   # keep turning away: the door leaves the field of view
            obs, *_ = env.step(np.array([0, 0, turn_bins.index(-10.0), 0, 0, 0]))
        assert obs["rays"][3].sum() == 0


def test_no_door_map_is_all_zero():
    map_id = "S0-train-100001"
    lit_total = [0]
    with make(2) as env:
        def check(t, obs, info, ctrl, a):
            lit_total[0] += int(obs["rays"][3].sum())
            return True
        _drive(env, map_id, 400, check)
        for _ in range(40):   # spin in place too
            obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0]))
            lit_total[0] += int(obs["rays"][3].sum())
    assert lit_total[0] == 0


def test_key_held_bit_set_on_pickup_and_cleared_on_reset():
    map_id = "S2-train-102001"   # blue key
    seen = {"before": 0, "after": 0}
    with make(2) as env:
        def check(t, obs, info, ctrl, a):
            held = info["keys_held"]["blue"]
            assert obs["self"][28] == (1.0 if held else 0.0)
            assert obs["self"][29] == obs["self"][30] == 0.0
            seen["after" if held else "before"] += 1
            return True
        _, info, ended = _drive(env, map_id, 900, check)
        assert seen["before"] > 0 and seen["after"] > 0, seen
        obs, info = env.reset(seed=3, options={"map": _lump(map_id)})
        assert obs["self"][28:].sum() == 0 and info["keys_held"]["blue"] is False


def test_needs_key_is_a_one_decision_pulse():
    map_id = "S2-train-102001"   # blue lock; key anchor dropped so the teacher goes to the door
    rows = []
    with make(2) as env:
        def check(t, obs, info, ctrl, a):
            txt = env.game.get_state().notifications_buffer or ""
            rows.append(("key needed" in txt, obs["self"][31:34].copy(), obs["self"][28]))
            return len(rows) < 140
        _drive(env, map_id, 400, check, drop_key=True, press_locked=True)
    pulses = [r for r in rows if r[0]]
    assert len(pulses) >= 3, "the locked-door message never appeared"
    for msg, needs, held in rows:
        assert needs.tolist() == ([1.0, 0.0, 0.0] if msg else [0.0, 0.0, 0.0])
        assert held == 0.0
    # the decision after a pulse reads 0 (no decay)
    idx = [i for i, r in enumerate(rows) if r[0]]
    assert any(i + 1 < len(rows) and not rows[i + 1][0] for i in idx)
    assert all(rows[i + 1][1].sum() == 0 for i in idx if i + 1 < len(rows) and not rows[i + 1][0])


def test_v1_contract_identical_and_v2_is_a_strict_extension():
    map_id = "S1-train-101004"
    rec = {"actions": [], "obs": []}
    with make(2) as env:
        assert env.observation_space["rays"].shape == (4, 80)
        assert env.observation_space["self"].shape == (34,)

        def record(t, obs, info, ctrl, a):
            if a is not None:
                rec["actions"].append(a.copy())
            rec["obs"].append({k: v.copy() for k, v in obs.items()})
            return len(rec["actions"]) < 50
        _drive(env, map_id, 60, record, seed=7)
    with make(1) as env:
        assert env.observation_space["rays"].shape == (3, 80)
        assert env.observation_space["self"].shape == (28,)
        assert env.cfg.engine_sectors_enabled is False
        obs, info = env.reset(seed=7, options={"map": _lump(map_id)})
        assert not env.game.get_state().sectors   # sector info really off under v1
        v1 = [obs]
        for a in rec["actions"]:
            obs, *_ = env.step(a)
            v1.append(obs)
    assert len(v1) == len(rec["obs"])
    for o1, o2 in zip(v1, rec["obs"]):
        assert set(o1) == set(o2)
        assert np.array_equal(o1["rays"], o2["rays"][:3])
        assert np.array_equal(o1["self"], o2["self"][:28])
        for k in o1:
            if k not in ("rays", "self"):
                assert np.array_equal(o1[k], o2[k]), k


def test_vanilla_map01_door_lights_under_v2():
    """Vanilla map (freedoom2 MAP01, engine-test config): the probed door sector 27 is
    closed at start and lights the columns it fills when faced from 34 units."""
    from env_test_helpers import FD2_ENGINE_TEST_CONFIG as FD2
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",), episode_timeout_tics=100000,
                      obs_version=2, screen_enabled=False)
    env = DoomEnv(cfg)
    try:
        env.reset(seed=0)
        gaps = _gaps(env)
        assert gaps[27] == 0
        for _ in range(3):
            env.step([0, 0, 3, 0, 0, 0])
        env.game.send_game_command("warp 670 -160")  # 34 units west of door 27, facing east
        obs, *_ = env.step([0, 0, 3, 0, 0, 0])
        assert obs["rays"][3][20:60].sum() == 40, obs["rays"][3]
    finally:
        env.close()


def test_viz_nocheat_v2_door_row_zero_and_modality_as_v1():
    from env.config import EVAL_CONFIG
    rows = {}
    for v in (1, 2):
        cfg = EVAL_CONFIG.replace(obs_version=v, game_args=("+viz_nocheat 1",), screen_enabled=False)
        cfg.assert_vanilla()
        env = DoomEnv(cfg)
        try:
            obs, _ = env.reset(seed=0)
            seq = [obs]
            for _ in range(10):
                obs, *_ = env.step(np.array([1, 0, 5, 0, 0, 0]))
                seq.append(obs)
            assert all(env.observation_space.contains(o) for o in seq)
            rows[v] = seq
        finally:
            env.close()
    for o1, o2 in zip(rows[1], rows[2]):
        assert o2["rays"][3].sum() == 0
        assert np.array_equal(o1["modality"], o2["modality"])
        assert np.array_equal(o1["rays"], o2["rays"][:3])
