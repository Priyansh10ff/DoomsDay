"""Observation v3 `egomap` (env/egomap.py): unit tests (no engine) and engine tests on a
synthetic corridor (tests/egomap_maps.py). One engine at a time, each closed."""
import contextlib
import inspect
import os

import numpy as np
import pytest

from env import egomap as E
from env import observation as O

pytestmark = pytest.mark.filterwarnings("ignore")

FWD = [1, 0, 3, 0, 0, 0]       # move forward, no turn
BACK = [2, 0, 3, 0, 0, 0]
STILL = [0, 0, 3, 0, 0, 0]
R40 = [0, 0, 6, 0, 0, 0]       # +10 deg/tic right = 40 deg per decision (frame skip 4)
R16 = [0, 0, 5, 0, 0, 0]
R4 = [0, 0, 4, 0, 0, 0]
L40 = [0, 0, 0, 0, 0, 0]
TURN_180 = [R40] * 4 + [R16, R4]


# ------------------------------------------------------------------ unit
def _levels_for_corridor(half_w=64.0, ahead=None):
    """(80,) levels of a straight corridor seen from its axis (flat walls at +-half_w,
    optional end wall `ahead` units in front), using the env's own column geometry."""
    t = np.abs(E._COL_TAN)
    fwd = np.where(t > 1e-9, half_w / np.maximum(t, 1e-9), np.inf)
    if ahead is not None:
        fwd = np.minimum(fwd, ahead)
    lv = np.rint(E.DEPTH_PER_UNIT * fwd + E.DEPTH_OFFSET)
    return np.clip(lv, 1, E.SATURATED_LEVEL).astype(np.float64)


def test_contract_shape_and_zero_before_update():
    m = E.EgoMap()
    c = m.crop()
    assert E.EGOMAP_SHAPE == (5, 32, 32) and c.shape == (5, 32, 32) and c.dtype == np.uint8
    assert c.sum() == 0
    assert E.EGOMAP_CELL_UNITS == 48.0 and E.EGOMAP_AGENT_CELL == (16, 16)


def test_unit_corridor_walls_and_free():
    m = E.EgoMap()
    m.update(0.0, 0.0, 0.0, _levels_for_corridor(ahead=300.0), None)
    c = m.crop()
    assert set(np.unique(c)) <= {0, 1}
    assert c[0, 11:16, 16].all()                 # free straight ahead
    assert c[1, 12:16, 15].all() and c[1, 12:16, 17].all()   # walls one cell either side
    assert c[1, 9:16, :14].sum() == 0 and c[1, 9:16, 19:].sum() == 0
    assert c[1, 9:11, 15:18].any()               # end wall ~300 ahead = 6.25 cells -> row 9/10
    assert c[2, 16, 16] == 1 and c[2].sum() == 1  # visited: own cell only


def test_unit_rotation_is_exact_for_quarter_turn():
    """Turning 90 degrees left in place with no new data rotates the crop: a world point
    seen at (row r, col c) moves to (row c, col 32 - r)."""
    m = E.EgoMap()
    m.update(100.0, -40.0, 30.0, _levels_for_corridor(ahead=300.0), None)
    a = m.crop()
    m.update(100.0, -40.0, 120.0, np.zeros(80), None)   # turn 90 left, no readings
    b = m.crop()
    for r in range(1, 32):
        for c in range(32):
            assert np.array_equal(b[:, c, 32 - r], a[:, r, c]), (r, c)


def test_unit_translation_and_spawn_heading_invariance():
    """Same relative scene and motion, different absolute spawn pose -> identical crops."""
    lv = _levels_for_corridor(ahead=250.0)
    outs = []
    for x0, y0, h0 in ((0.0, 0.0, 0.0), (1234.0, -987.0, 0.0), (-512.0, 2048.0, 0.0)):
        m = E.EgoMap()
        seq = []
        for k in range(5):
            m.update(x0 + 20.0 * k, y0, h0 + 7.0 * k, lv, None)
            seq.append(m.crop())
        outs.append(seq)
    for seq in outs[1:]:
        for a, b in zip(outs[0], seq):
            assert np.array_equal(a, b)


def test_unit_saturated_and_missing_readings():
    m = E.EgoMap()
    lv = np.zeros(80)
    lv[40] = E.SATURATED_LEVEL          # saturated: free out to ~455 units, no wall
    lv[41] = 255
    m.update(0.0, 0.0, 0.0, lv, None)
    c = m.crop()
    assert c[1].sum() == 0
    assert c[0, 8:16, 16].all()          # 455 units = 9.5 cells
    assert c[0, :6].sum() == 0


def test_unit_door_cells_and_open_door_clears():
    m = E.EgoMap()
    lv = _levels_for_corridor(ahead=200.0)
    door = (np.abs(E._COL_TAN) < 0.3).astype(np.float32)
    m.update(0.0, 0.0, 0.0, lv, door)
    c = m.crop()
    assert c[3, 11:13, 16].any() and c[1, 11:13, 16].sum() == 0
    lv2 = _levels_for_corridor(ahead=None)              # door opened: rays pass through
    m.update(0.0, 0.0, 0.0, lv2, np.zeros(80))
    assert m.crop()[3].sum() == 0


def test_unit_key_seen_persists_and_clears_on_pickup():
    T = O.T
    m = E.EgoMap()
    ents = np.zeros((24, 8), np.float32)
    types = np.zeros(24, np.int64)
    mask = np.zeros(24, bool)
    ents[0, E.U_COL] = 0.0
    ents[0, E.D_COL] = (E.DEPTH_PER_UNIT * 240.0 + E.DEPTH_OFFSET) / 255.0   # 240 ahead
    types[0], mask[0] = T["key-blue"], True
    ents[1, E.U_COL], ents[1, E.D_COL] = 0.5, (E.DEPTH_PER_UNIT * 200 + E.DEPTH_OFFSET) / 255.0
    types[1], mask[1] = T["key-red"], True
    m.update(0.0, 0.0, 0.0, np.zeros(80), None, ents, types, mask, [False, False, False])
    c = m.crop()
    assert c[4, 11, 16] == 1 and c[4].sum() == 2       # 240 units = 5 cells ahead
    m.update(-300.0, 0.0, 0.0, np.zeros(80), None, None, None, None, [False, False, False])
    assert m.crop()[4].sum() == 2                       # walked away: still remembered
    m.update(-300.0, 0.0, 0.0, np.zeros(80), None, None, None, None, [True, False, False])
    c = m.crop()
    assert c[4].sum() == 1                              # blue cleared, red kept
    m.update(0.0, 0.0, 0.0, np.zeros(80), None, ents, types, mask, [True, False, False])
    c = m.crop()                                        # back at the first pose, both visible
    assert c[4].sum() == 1 and c[4, 11, 16] == 0        # held colour is never re-marked


def test_unit_reset_clears():
    m = E.EgoMap()
    m.update(5.0, 5.0, 10.0, _levels_for_corridor(), None)
    m.reset()
    assert m.crop().sum() == 0 and m.grid.sum() == 0


def test_source_reads_no_god_view_and_outputs_crop_only():
    import ast
    tree = ast.parse(inspect.getsource(E))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for bad in ("sectors", "objects", "automap_buffer", "get_state", "game", "labels",
                "object_position_x", "position_x"):
        assert bad not in names, bad
    from env import doom_env
    body = inspect.getsource(doom_env.DoomEnv._add_v3_egomap)
    assert 'obs["egomap"] = self._egomap.crop()' in body
    assert "info" not in body.replace("InfoVars", "")


def test_config_versions_and_signature(monkeypatch):
    from configs import a0_env
    from env.config import EVAL_CONFIG
    from train.bc_spec import EGOMAP_STEP_SHAPE, a0_training_spec_and_signature, egomap_step_shape
    assert a0_env.OBS_VERSION == 3
    assert EGOMAP_STEP_SHAPE == E.EGOMAP_SHAPE and egomap_step_shape(2) is None
    cfg3 = EVAL_CONFIG.replace(obs_version=3)
    cfg3.assert_vanilla()
    assert ("obs_version", 3) in cfg3.sensor_signature() and cfg3.egomap and cfg3.door_channel
    with pytest.raises(ValueError):
        EVAL_CONFIG.replace(obs_version=4).validate()
    for v, ego in (("1", None), ("2", None), ("3", (5, 32, 32)), ("", (5, 32, 32))):
        monkeypatch.setenv(a0_env.OBS_VERSION_ENV_VAR, v)
        spec, sig = a0_training_spec_and_signature()
        assert getattr(spec, "egomap", None) == ego, v
        want = int(v or 3)
        assert spec.ray_channels == (3 if want == 1 else 4) and spec.self_dim == (28 if want == 1 else 34)
        assert dict(sig).get("obs_version", 1) == want


# ------------------------------------------------------------------ engine
@contextlib.contextmanager
def corridor_env(tmp_path, v=3, name="c.wad", **kw):
    from egomap_maps import corridor_config, corridor_wad
    from env import DoomEnv
    wad = corridor_wad(os.path.join(str(tmp_path), name), **kw)
    env = DoomEnv(corridor_config(wad, obs_version=v))
    try:
        yield env
    finally:
        env.close()


def _run(env, actions, seed=0):
    obs, info = env.reset(seed=seed)
    out = [(obs, info)]
    for a in actions:
        obs, r, te, tr, info = env.step(np.array(a))
        out.append((obs, info))
        assert not (te or tr)
    return out


def test_engine_corridor_free_ahead_walls_both_sides(tmp_path):
    with corridor_env(tmp_path, with_door=False) as env:
        assert env.observation_space["egomap"].shape == (5, 32, 32)
        (obs, _), = _run(env, [])
        assert env.observation_space.contains(obs)
        e = obs["egomap"]
        assert e[0, 8:16, 16].all()                      # free ahead out to ~450 units
        # corridor half-width 64: walls in the cells one to the left and right (cols 15, 17)
        assert e[1, 8:16, 15].sum() >= 7 and e[1, 8:16, 17].sum() >= 7
        assert e[1, 8:16, :14].sum() == 0 and e[1, 8:16, 19:].sum() == 0
        assert e[1, 8:16, 14].sum() + e[1, 8:16, 18].sum() <= 2
        assert e[1, :16, 16].sum() == 0                  # nothing blocks the axis
        assert e[3].sum() == 0 and e[4].sum() == 0


def _cell_world(info, r, c):
    """Test-side only: world (x, y) of crop cell (r, c) centre from the privileged pose."""
    f, l = (16 - r) * 48.0, (16 - c) * 48.0
    t = np.radians(info["angle"])
    return (info["pos_x"] + f * np.cos(t) - l * np.sin(t), info["pos_y"] + f * np.sin(t) + l * np.cos(t))


def _ego_cell(info, wx, wy):
    t = np.radians(info["angle"])
    dx, dy = wx - info["pos_x"], wy - info["pos_y"]
    f, l = dx * np.cos(t) + dy * np.sin(t), -dx * np.sin(t) + dy * np.cos(t)
    return 16 - f / 48.0, 16 - l / 48.0


def test_engine_turning_rotates_crop_and_trail_stays_in_world(tmp_path):
    with corridor_env(tmp_path, with_door=False) as env:
        traj = _run(env, [FWD] * 10 + TURN_180 + [R40, R40, R16])
        x0 = traj[0][1]["pos_x"]
        x1 = max(i["pos_x"] for _, i in traj)          # momentum carries on after the last FWD
        assert x1 - x0 > 150
        checked = set()
        for obs, info in traj[10:]:
            e = obs["egomap"]
            vis = np.argwhere(e[2])
            assert len(vis) >= (x1 - x0) / 48 - 1
            for r, c in vis:        # every visited cell lies on the walked segment, in the world
                wx, wy = _cell_world(info, r, c)
                assert abs(wy) <= 48 and x0 - 60 <= wx <= x1 + 60, (r, c, wx, wy, info["angle"])
            fr = np.argwhere(e[0])
            assert len(fr) > 10
            for r, c in fr:         # free cells lie inside the corridor (half-width 64)
                wx, wy = _cell_world(info, r, c)
                assert abs(wy) <= 64 + 36, (r, c, wx, wy, info["angle"])
            checked.add(round(info["angle"]) % 360)
        assert len(checked) >= 5           # several headings, incl. ~180 and ~90 off the start
        # the crop really rotates: facing back, the trail is ahead (row < 16)
        obs, info = traj[10 + len(TURN_180)]
        assert 150 < info["angle"] < 210, info["angle"]
        assert np.argwhere(obs["egomap"][2])[:, 0].min() < 14


def test_engine_closed_door_in_channel_3(tmp_path):
    with corridor_env(tmp_path, with_door=True) as env:
        traj = _run(env, [FWD] * 14)
        obs, info = traj[-1]
        dist = 640 - info["pos_x"]
        assert 60 < dist < 400, dist
        assert obs["rays"][3].sum() > 10                   # v2 door row lights
        e = obs["egomap"]
        row = int(16 - np.floor(dist / 48 + 0.5))
        assert e[3, row - 1:row + 2, 15:18].sum() >= 2, (row, np.argwhere(e[3]))
        assert e[3, :row - 1].sum() == 0 and e[3, row + 2:].sum() == 0
        assert e[0, row + 2:16, 16].all()


def test_engine_reset_clears_and_reproduces(tmp_path):
    with corridor_env(tmp_path, with_door=False) as env:
        first = _run(env, [FWD] * 5 + [R40] * 2)
        again = _run(env, [])
        assert np.array_equal(again[0][0]["egomap"], first[0][0]["egomap"])
        assert again[0][0]["egomap"][2].sum() == 1


def _face(env, info, target):
    """Test-side steering: turn in place until the heading is within 2 degrees of target."""
    obs = None
    for _ in range(30):
        err = (target - info["angle"] + 180) % 360 - 180      # + = need to turn left (CCW)
        if abs(err) <= 2:
            break
        bins = np.array(env.cfg.turn_bins) * env.cfg.frame_skip   # + = right (clockwise)
        k = int(np.argmin(np.abs(-bins - err)))
        obs, r, te, tr, info = env.step(np.array([0, 0, k, 0, 0, 0]))
    assert abs((target - info["angle"] + 180) % 360 - 180) <= 2
    return obs, info


def test_engine_key_seen_persists_then_cleared_on_pickup(tmp_path):
    with corridor_env(tmp_path, with_door=False, player_x=300, key_x=600) as env:
        obs, info = env.reset(seed=0)
        k0 = obs["egomap"][4]
        assert k0.sum() >= 1 and k0[:16, 15:18].sum() >= 1, np.argwhere(k0)
        assert np.argwhere(k0)[:, 0].min() >= 16 - 8            # ~300 units ahead
        obs, info = _face(env, info, 180.0)          # turn round, then walk away along the axis
        for a in [FWD] * 4:
            obs, r, te, tr, info = env.step(np.array(a))
        assert not obs["entity_mask"][obs["entity_type"] == O.T["key-blue"]].any()
        k1 = obs["egomap"][4]
        er, ec = _ego_cell(info, 600.0, 0.0)                    # where the key really is
        assert er > 16                                          # behind the agent now
        assert k1.sum() >= 1 and k1[int(er) - 1:int(er) + 3, int(ec) - 1:int(ec) + 3].sum() >= 1,             (np.argwhere(k1), er, ec)
        obs, info = _face(env, info, 0.0)
        held = False
        for _ in range(40):
            obs, *_ = env.step(np.array(FWD))
            held = obs["self"][-6] > 0.5
            if held:
                break
        assert held, "key not picked up"
        assert obs["egomap"][4].sum() == 0


def test_engine_same_observation_under_map_translation(tmp_path):
    """The same corridor (door, key) translated by (+1000, -704) on the map: every
    observation key, the egomap included, is identical step by step."""
    acts = [FWD] * 5 + [R40, R16, FWD, FWD, L40, L40, FWD, BACK, R4]
    runs = []
    for i, shift in enumerate(((0, 0), (1000, -704))):
        with corridor_env(tmp_path, name=f"t{i}.wad", shift=shift, key_x=500) as env:
            runs.append(_run(env, acts))
    assert runs[0][0][1]["pos_x"] != runs[1][0][1]["pos_x"]
    for (o0, _), (o1, _) in zip(*runs):
        assert set(o0) == set(o1)
        for k in o0:
            assert np.array_equal(o0[k], o1[k]), k


def test_engine_v1_v2_unchanged_and_v3_strict_extension(tmp_path):
    acts = [FWD] * 6 + [R40, FWD, R16, FWD, L40]
    runs = {}
    for v in (1, 2, 3):
        with corridor_env(tmp_path, v=v, name=f"v{v}.wad", key_x=400) as env:
            assert ("egomap" in env.observation_space.spaces) == (v == 3)
            runs[v] = [o for o, _ in _run(env, acts)]
    for o1, o2, o3 in zip(runs[1], runs[2], runs[3]):
        assert set(o3) == set(o2) | {"egomap"}
        for k in o2:
            assert np.array_equal(o2[k], o3[k]), k
        assert np.array_equal(o1["rays"], o2["rays"][:3])
        assert np.array_equal(o1["self"], o2["self"][:28])
    assert any(o["egomap"].any() for o in runs[3])


def test_engine_viz_nocheat_v3_map_zero_and_modality_as_v2():
    from env import DoomEnv
    from env.config import EVAL_CONFIG
    rows = {}
    for v in (2, 3):
        cfg = EVAL_CONFIG.replace(obs_version=v, game_args=("+viz_nocheat 1",), screen_enabled=False)
        cfg.assert_vanilla()
        env = DoomEnv(cfg)
        try:
            obs, _ = env.reset(seed=0)
            seq = [obs]
            for _ in range(8):
                obs, *_ = env.step(np.array([1, 0, 5, 0, 0, 0]))
                seq.append(obs)
            assert all(env.observation_space.contains(o) for o in seq)
            rows[v] = seq
        finally:
            env.close()
    for o2, o3 in zip(rows[2], rows[3]):
        assert o3["egomap"].sum() == 0
        for k in o2:
            assert np.array_equal(o2[k], o3[k]), k
