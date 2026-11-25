"""Observation v2 (door channel + key bits), engine-free: door_cue maths, key parsing,
the OBS_VERSION switch, sensor signatures and source-level isolation."""
import json
import math
import os

import numpy as np
import pytest

from env import door_cue as D
from env import observation as O
from env.config import EVAL_CONFIG, TRAIN_CONFIG, assert_same_sensors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The v1 sensor signature exactly as checkpoints stored it before obs_version existed.
V1_SIGNATURE_JSON = ('[["frame_skip", 4], ["render_hud", true], ["turn_bins", [-10.0, -4.0, -1.0, 0.0, '
                     '1.0, 4.0, 10.0]], ["max_entities", 24], ["screen_enabled", false], ["ray_bands", '
                     '[[30, 46], [46, 54], [54, 70]]]]')


def canon(sig):
    return json.dumps(sig, sort_keys=True)


def _pred(perp):
    return D.DEPTH_PER_UNIT * perp + D.DEPTH_OFFSET


def _wall_reading(perp_per_col):
    return np.array([max(1.0, round(_pred(p))) for p in perp_per_col])


# ---------------------------------------------------------------- door_channel maths
def test_face_straight_ahead_lights_its_screen_extent():
    # player at origin facing +x; door face x = 200, y in [-40, 40] -> spans
    # tan = +-0.2 -> pixel x in [64, 96] -> bins 32..47
    face = np.array([[200.0, -40.0, 200.0, 40.0]])
    mid = np.full(80, round(_pred(200.0)))
    out = D.door_channel(face, 0.0, 0.0, 0.0, mid)
    lit = np.where(out > 0)[0]
    assert out.dtype == np.float32 and out.shape == (80,)
    assert lit.min() >= 31 and lit.max() <= 48 and len(lit) >= 15
    assert np.all(np.diff(lit) == 1)  # contiguous


def test_occluded_face_is_zero():
    face = np.array([[200.0, -40.0, 200.0, 40.0]])
    mid = np.full(80, round(_pred(100.0)))  # a wall at 100 units in front of it
    assert D.door_channel(face, 0.0, 0.0, 0.0, mid).sum() == 0


def test_face_behind_player_or_outside_fov_is_zero():
    mid = np.full(80, round(_pred(200.0)))
    behind = np.array([[-200.0, -40.0, -200.0, 40.0]])
    assert D.door_channel(behind, 0.0, 0.0, 0.0, mid).sum() == 0
    side = np.array([[10.0, 200.0, 50.0, 200.0]])   # off to the left, outside 90-degree FOV
    assert D.door_channel(side, 0.0, 0.0, 0.0, np.full(80, 30.0)).sum() == 0


def test_no_reading_columns_are_zero_and_open_passage_is_zero():
    face = np.array([[200.0, -40.0, 200.0, 40.0]])
    assert D.door_channel(face, 0.0, 0.0, 0.0, np.zeros(80)).sum() == 0      # masked / no reading
    far = np.full(80, round(_pred(600.0)))                                    # sees past the face
    assert D.door_channel(face, 0.0, 0.0, 0.0, far).sum() == 0


def test_screen_right_is_clockwise():
    # facing +y (90 deg); a face at x > 0 is to the player's right -> right half of the screen
    face = np.array([[60.0, 200.0, 140.0, 200.0]])
    out = D.door_channel(face, 0.0, 0.0, 90.0, np.full(80, round(_pred(200.0))))
    lit = np.where(out > 0)[0]
    assert len(lit) and lit.min() >= 40


@pytest.mark.parametrize("dx,dy,rot", [(0, 0, 0), (1234.5, -987.0, 0), (-3000, 512, 37.0),
                                       (777, 777, 180.0), (-50, 4000, 271.3)])
def test_rigid_motion_invariance(dx, dy, rot):
    """Same local scene at a different absolute place and heading -> same output, so the
    channel cannot carry absolute position or heading."""
    faces = np.array([[200.0, -40.0, 200.0, 40.0], [150.0, 60.0, 150.0, 120.0]])
    mid = np.concatenate([np.full(40, round(_pred(150.0))), np.full(40, round(_pred(200.0)))])
    ref = D.door_channel(faces, 0.0, 0.0, 0.0, mid)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    moved = faces.copy()
    for j in (0, 2):
        x, y = faces[:, j], faces[:, j + 1]
        moved[:, j], moved[:, j + 1] = x * c - y * s + dx, x * s + y * c + dy
    out = D.door_channel(moved, dx, dy, rot, mid)
    assert np.array_equal(out, ref)
    assert ref.sum() > 0


def test_tolerance_bounds():
    face = np.array([[300.0, -10.0, 300.0, 10.0]])
    p = _pred(300.0)
    lit = lambda r: D.door_channel(face, 0.0, 0.0, 0.0, np.full(80, r)).sum() > 0
    assert lit(round(p)) and lit(p + D.DOOR_TOL_FAR - 0.01)
    assert not lit(p + D.DOOR_TOL_FAR + 0.6)
    assert lit(p - D.DOOR_TOL_NEAR_ABS - D.DOOR_TOL_NEAR_REL * p + 0.01)
    assert not lit(p - D.DOOR_TOL_NEAR_ABS - D.DOOR_TOL_NEAR_REL * p - 0.6)


# ---------------------------------------------------------------- occlusion gate (audit 02 F1)
FACE = np.array([[200.0, -40.0, 200.0, 40.0]])
EYE = 41.0                                      # floor 0 + EYE_HEIGHT


def _occ(lines):
    """lines: [(x1, y1, x2, y2, lo, hi)] -> occlusion table as DoorTracker.scene_now gives."""
    a = np.array(lines, dtype=np.float64).reshape(-1, 6)
    return a[:, :4], a[:, 4], a[:, 5]


ONE_SIDED = (np.inf, -np.inf)
# the scene's own geometry: the closed door face (two-sided, opening 0) and its jambs
SCENE = [(200.0, -40.0, 200.0, 40.0, 0.0, 0.0), (200.0, -40.0, 216.0, -40.0, *ONE_SIDED),
         (200.0, 40.0, 216.0, 40.0, *ONE_SIDED), (216.0, -40.0, 216.0, 40.0, 0.0, 0.0)]


@pytest.mark.parametrize("gap", [8.0, 16.0, 24.0, 30.0])
def test_face_behind_nearer_wall_is_zero_with_gate(gap):
    """Audit 02 case: a one-sided wall `gap` units in front of the face, reading = the wall's
    predicted level. The depth check alone lights it; the gate must not."""
    wall = (200.0 - gap, -60.0, 200.0 - gap, 60.0, *ONE_SIDED)
    mid = np.full(80, round(_pred(200.0 - gap)))
    assert D.door_channel(FACE, 0.0, 0.0, 0.0, mid, _occ(SCENE + [wall]), EYE).sum() == 0
    # a closed door in front of the face blocks the same way (two-sided, opening 0)
    door2 = (200.0 - gap, -60.0, 200.0 - gap, 60.0, 0.0, 0.0)
    assert D.door_channel(FACE, 0.0, 0.0, 0.0, mid, _occ(SCENE + [door2]), EYE).sum() == 0


def test_old_leak_reproduced_without_gate():
    for gap in (8.0, 16.0, 24.0, 30.0):
        face = np.array([[400.0, -40.0, 400.0, 40.0]])
        mid = np.full(80, round(_pred(400.0 - gap)))
        assert D.door_channel(face, 0.0, 0.0, 0.0, mid).sum() > 0, gap


def test_visible_face_unchanged_by_gate():
    mid = np.full(80, round(_pred(200.0)))
    ref = D.door_channel(FACE, 0.0, 0.0, 0.0, mid)
    assert ref.sum() >= 15
    assert np.array_equal(D.door_channel(FACE, 0.0, 0.0, 0.0, mid, _occ(SCENE), EYE), ref)
    # lines beyond the face, behind the player, or beside the ray do not matter
    extra = [(300.0, -100.0, 300.0, 100.0, *ONE_SIDED), (-50.0, -100.0, -50.0, 100.0, *ONE_SIDED),
             (100.0, 150.0, 190.0, 150.0, *ONE_SIDED)]
    assert np.array_equal(D.door_channel(FACE, 0.0, 0.0, 0.0, mid, _occ(SCENE + extra), EYE), ref)


@pytest.mark.parametrize("dist", [40.0, 120.0, 200.0, 320.0, 400.0])
@pytest.mark.parametrize("heading", [-35.0, 0.0, 30.0])
def test_visible_face_lights_at_distances_and_angles(dist, heading):
    face = np.array([[dist, -40.0, dist, 40.0]])
    scene = [(dist, -40.0, dist, 40.0, 0.0, 0.0), (dist, -40.0, dist + 16, -40.0, *ONE_SIDED),
             (dist, 40.0, dist + 16, 40.0, *ONE_SIDED),
             (0.5 * dist, -200.0, 0.5 * dist, 200.0, 0.0, 128.0)]   # open archway in between
    f, ok = D._cast(*D._to_player_frame(face, 0.0, 0.0, heading), np.arange(80))
    mid = np.where(ok[:, 0], np.rint(np.clip(_pred(np.where(ok, f, 1.0)[:, 0]), 1, 255)), 0.0)
    ref = D.door_channel(face, 0.0, 0.0, heading, mid)
    gated = D.door_channel(face, 0.0, 0.0, heading, mid, _occ(scene), EYE)
    assert ref.sum() > 0 and np.array_equal(gated, ref)


def test_two_sided_openings():
    mid = np.full(80, round(_pred(190.0)))           # reading consistent with the face (-10 u)
    lit = lambda line: D.door_channel(FACE, 0.0, 0.0, 0.0, mid, _occ(SCENE + [line]), EYE).sum()
    x = 190.0
    assert lit((x, -60.0, x, 60.0, 0.0, 128.0)) > 0     # open doorway: no block
    assert lit((x, -60.0, x, 60.0, 16.0, 128.0)) > 0    # 16-unit step: band clears
    assert lit((x, -60.0, x, 60.0, 0.0, 48.0)) == 0     # lintel 7 above eye: inside the band at 190
    assert lit((x, -60.0, x, 60.0, 0.0, 40.0)) == 0     # lintel below eye: blocks
    assert lit((x, -60.0, x, 60.0, 40.0, 128.0)) == 0   # floor at eye level: blocks
    # far in front of the face (depth check carries it): a lintel 7 above eye does not block
    mid_face = np.full(80, round(_pred(200.0)))
    far = (20.0, -60.0, 20.0, 60.0, 0.0, 48.0)
    assert D.door_channel(FACE, 0.0, 0.0, 0.0, mid_face, _occ(SCENE + [far]), EYE).sum() > 0


@pytest.mark.parametrize("dx,dy,dz,rot", [(0, 0, 0, 0), (1234.5, -987.0, 64.0, 37.0),
                                          (-3000, 512, -200.0, 271.3)])
def test_gate_rigid_motion_and_height_invariance(dx, dy, dz, rot):
    gap = 20.0
    scene = SCENE + [(200.0 - gap, -60.0, 200.0 - gap, -10.0, *ONE_SIDED),   # hides the left half
                     (100.0, -300.0, 100.0, 300.0, 24.0, 100.0)]
    mid = np.concatenate([np.full(40, round(_pred(200.0))), np.full(40, round(_pred(200.0 - gap)))])
    seg, lo, hi = _occ(scene)
    ref = D.door_channel(FACE, 0.0, 0.0, 0.0, mid, (seg, lo, hi), EYE)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))

    def move(a):
        b = a.copy()
        for j in (0, 2):
            x, y = a[:, j], a[:, j + 1]
            b[:, j], b[:, j + 1] = x * c - y * s + dx, x * s + y * c + dy
        return b
    out = D.door_channel(move(FACE), dx, dy, rot, mid, (move(seg), lo + dz, hi + dz), EYE + dz)
    assert np.array_equal(out, ref) and 0 < ref.sum() < D.door_channel(FACE, 0.0, 0.0, 0.0, mid).sum()


def test_gated_columns_give_no_egomap_door_cells():
    from env import egomap as E
    gap = 20.0
    mid = np.full(80, round(_pred(200.0 - gap)))
    occ = _occ(SCENE + [(200.0 - gap, -60.0, 200.0 - gap, 60.0, *ONE_SIDED)])
    door = D.door_channel(FACE, 0.0, 0.0, 0.0, mid, occ, EYE)
    m = E.EgoMap()
    m.update(0.0, 0.0, 0.0, mid, door)
    assert m.crop()[E.CH_DOOR].sum() == 0 and m.crop()[E.CH_WALL].sum() > 0
    old = D.door_channel(FACE, 0.0, 0.0, 0.0, mid)       # ungated: the leak reached channel 3
    m2 = E.EgoMap()
    m2.update(0.0, 0.0, 0.0, mid, old)
    assert m2.crop()[E.CH_DOOR].sum() > 0


class _L:
    def __init__(self, x1, y1, x2, y2, b):
        self.x1, self.y1, self.x2, self.y2, self.is_blocking = x1, y1, x2, y2, b


class _S:
    def __init__(self, sid, floor, ceil, lines):
        self.id, self.floor_height, self.ceiling_height, self.lines = sid, floor, ceil, lines


def test_tracker_remembers_closed_doors_uses_faces_only_and_drops_open_ones():
    lines = [_L(0, 0, 0, 64, False), _L(16, 0, 16, 64, False), _L(0, 0, 16, 0, True), _L(0, 64, 16, 64, True)]
    t = D.DoorTracker()
    # a low-ceiling room never seen closed is not a door
    assert t.faces_now([_S(1, 0, 30, lines)]).shape == (0, 4)
    assert t.faces_now([_S(2, 0, 0, lines)]).shape == (2, 4)                # closed: 2 faces, no jambs
    assert t.faces_now([_S(2, 0, 40, lines)]).shape == (2, 4)               # rising: still a door
    assert t.faces_now([_S(2, 0, 48, lines)]).shape == (0, 4)               # passage visible
    t.reset()
    assert t.faces_now([_S(2, 0, 40, lines)]).shape == (0, 4)               # memory cleared


def test_tracker_scene_openings_from_two_sectors_not_is_blocking():
    # room 1 (floor 0, ceil 128) | door 2 (closed) | room 3 (floor 16, ceil 96); an
    # impassable window line (is_blocking True) between rooms 1 and 3 still counts as open.
    face_a, face_b = _L(0, 0, 0, 64, False), _L(16, 0, 16, 64, False)
    jamb1, jamb2 = _L(0, 0, 16, 0, True), _L(0, 64, 16, 64, True)
    window = _L(-100, 200, 100, 200, True)
    wall = _L(-100, -50, 100, -50, True)
    secs = [_S(1, 0, 128, [face_a, window, wall]), _S(2, 0, 0, [face_a, face_b, jamb1, jamb2]),
            _S(3, 16, 96, [_L(16, 64, 16, 0, False), window])]   # face_b listed reversed
    t = D.DoorTracker()
    faces, (seg, lo, hi) = t.scene_now(secs)
    assert faces.shape == (2, 4)
    op = {tuple(s): (l, h) for s, l, h in zip(seg.tolist(), lo.tolist(), hi.tolist())}
    assert len(op) == 6
    assert op[(0.0, 0.0, 0.0, 64.0)] == (0.0, 0.0)          # face A: closed door
    assert op[(16.0, 0.0, 16.0, 64.0)] == (16.0, 0.0)       # face B: empty opening
    assert op[(-100.0, 200.0, 100.0, 200.0)] == (16.0, 96.0)  # window: open, despite is_blocking
    assert op[(-100.0, -50.0, 100.0, -50.0)] == (np.inf, -np.inf)   # one-sided
    secs[1] = _S(2, 0, 72, secs[1].lines)                   # door opened: only heights change
    _, (_, lo2, hi2) = t.scene_now(secs)
    op2 = {tuple(s): (l, h) for s, l, h in zip(seg.tolist(), lo2.tolist(), hi2.tolist())}
    assert op2[(0.0, 0.0, 0.0, 64.0)] == (0.0, 72.0)


# ---------------------------------------------------------------- keys
def test_keys_needed_parses_locked_text_only():
    captured = ("G\n\x1d\x1e\x1e\x1f\x1cB\nBlue key needed for this door.\n\x1cG\n\x1d\x1e\x1f\x1c-")
    assert D.keys_needed(captured) == {"blue"}
    assert D.keys_needed("Yellow key needed for this door.") == {"yellow"}
    assert D.keys_needed("Blue key needed.") == {"blue"}
    assert D.keys_needed("You need a red key to open this door") == {"red"}
    assert D.keys_needed("Blue passcard secured!") == set()
    assert D.keys_needed("Red skeleton key secured!\nPicked up a clip.") == set()
    assert D.keys_needed("") == set() and D.keys_needed(None) == set()


def test_key_bits_layout():
    b = D.key_bits({"blue": True, "yellow": False, "red": True}, {"yellow"})
    assert b.dtype == np.float32 and b.tolist() == [1, 0, 1, 0, 1, 0]


# ---------------------------------------------------------------- switch and signatures
def test_v1_signature_is_byte_identical_to_before():
    cfg = TRAIN_CONFIG.replace(screen_enabled=False)
    assert cfg.obs_version == 1
    assert canon(cfg.sensor_signature()) == V1_SIGNATURE_JSON


def test_versions_have_different_signatures_and_do_not_mix():
    v1 = TRAIN_CONFIG.replace(screen_enabled=False)
    v2 = v1.replace(obs_version=2)
    assert canon(v2.sensor_signature()) != canon(v1.sensor_signature())
    assert_same_sensors(v1, EVAL_CONFIG.replace(screen_enabled=False))
    assert_same_sensors(v2, EVAL_CONFIG.replace(screen_enabled=False, obs_version=2))
    with pytest.raises(AssertionError):
        assert_same_sensors(v1, EVAL_CONFIG.replace(screen_enabled=False, obs_version=2))
    with pytest.raises(AssertionError):
        assert_same_sensors(v2, EVAL_CONFIG.replace(screen_enabled=False))


def test_obs_version_switch_applies_to_every_builder(monkeypatch, tmp_path):
    from configs import a0_env as A
    assert A.OBS_VERSION == 3            # v3 (egomap) is the default since 2026-11-26
    for raw, want in (("", 3), ("1", 1), ("2", 2), ("3", 3)):
        monkeypatch.setenv(A.OBS_VERSION_ENV_VAR, raw)
        tr, ev = A.a0_train_env_config(), A.a0_eval_env_config()
        assert tr.obs_version == ev.obs_version == want
        assert_same_sensors(tr, ev)
        ev.assert_vanilla()
        wad = tmp_path / "b.wad"
        wad.write_bytes(b"PWAD" + (2).to_bytes(4, "little") + (12).to_bytes(4, "little")
                        + b"\0" * 8 + b"DD0001\0\0" + b"\0" * 8 + b"THINGS\0\0")
        gen = A.generated_maps_env_config(tr, str(wad), ("DD0001",))
        assert gen.obs_version == want and gen.sensor_signature() == tr.sensor_signature()
    monkeypatch.setenv(A.OBS_VERSION_ENV_VAR, "4")
    with pytest.raises(ValueError):
        A.a0_train_env_config()


def test_eval_v2_is_vanilla_but_raw_sector_field_is_not():
    ev2 = EVAL_CONFIG.replace(obs_version=2)
    ev2.assert_vanilla()
    assert ev2.engine_sectors_enabled and not EVAL_CONFIG.engine_sectors_enabled
    with pytest.raises(AssertionError):
        ev2.replace(sectors_info=True).assert_vanilla()
    EVAL_CONFIG.replace(obs_version=3).assert_vanilla()   # v3 egomap: same engine inputs as v2
    with pytest.raises(AssertionError):
        EVAL_CONFIG.replace(obs_version=4).assert_vanilla()


def test_evaluate_reads_v1_and_v2_checkpoint_signatures():
    from evaluate.sensors import SensorSignatureError, trained_config_from_signature
    base = EVAL_CONFIG.replace(screen_enabled=False)
    t1 = trained_config_from_signature(V1_SIGNATURE_JSON, base.replace(obs_version=2))
    assert t1.obs_version == 1
    v2_json = canon(base.replace(obs_version=2).sensor_signature())
    t2 = trained_config_from_signature(v2_json, base)
    assert t2.obs_version == 2
    with pytest.raises(AssertionError):
        assert_same_sensors(t1, base.replace(obs_version=2))
    with pytest.raises(SensorSignatureError):
        trained_config_from_signature(V1_SIGNATURE_JSON.replace('["max_entities", 24], ', ''), base)


@pytest.mark.parametrize("v,rays,selfd", [(1, 3, 28), (2, 4, 34)])
def test_bc_spec_follows_the_switch(monkeypatch, v, rays, selfd):
    from configs.a0_env import OBS_VERSION_ENV_VAR
    from train.bc_spec import a0_training_spec_and_signature
    monkeypatch.setenv(OBS_VERSION_ENV_VAR, str(v))
    spec, sig = a0_training_spec_and_signature()
    assert (spec.ray_channels, spec.self_dim) == (rays, selfd)
    assert ("obs_version", 2) in sig if v == 2 else canon(sig) == V1_SIGNATURE_JSON


# ---------------------------------------------------------------- source isolation
def _method(src, name):
    return src.split(f"def {name}(")[1].split("\n    def ")[0]


def test_v2_cues_add_only_door_and_key_arrays():
    src = open(os.path.join(ROOT, "env", "doom_env.py"), encoding="utf-8").read()
    body = _method(src, "_add_v2_cues")
    # what is written into obs: only the door_channel result and key_bits
    writes = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("obs[")]
    assert writes == ['obs["rays"] = np.concatenate([obs["rays"], door[None, :]], axis=0)',
                      'obs["self"] = np.concatenate([obs["self"], D.key_bits(self._keys_held, self._keys_needed)])']
    assert "notifications" not in body and "_map" not in body
    # raw sector data reaches only DoorTracker.faces_now / door_channel
    code = [ln.split("#")[0].strip() for ln in body.splitlines()]
    used = [ln for ln in code if "sectors" in ln]
    assert used == ["sectors = state.sectors if telemetry_ok else None",
                    "if sectors:",
                    "faces, occ = self._doors.scene_now(sectors)"]
    # the gate is always on in the env: faces, occlusion table and eye height are passed
    calls = [ln for ln in code if "D.door_channel(" in ln]
    assert calls == ["door = D.door_channel(faces, v.pos_x, v.pos_y, v.angle, mid, occ, v.pos_z + D.EYE_HEIGHT)"]
    info_body = _method(src, "_build_info")
    assert "sector" not in info_body and "door" not in info_body


def test_door_cue_returns_only_arrays_and_no_atan2():
    src = open(os.path.join(ROOT, "env", "door_cue.py"), encoding="utf-8").read()
    assert "atan2" not in src and "arctan2" not in src
    assert "position_x" not in src.lower() and "POSITION_" not in src
