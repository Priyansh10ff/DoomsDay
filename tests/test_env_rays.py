"""Ray and entity-depth CONTENT tests (own-weapon pixels must never reach them).

Engine tests open one DoomGame each, sequentially, and always close it."""
import contextlib

import numpy as np

from env import TRAIN_CONFIG, DoomEnv
from env_test_helpers import FD2_ENGINE_TEST_CONFIG as FD2
from env import observation as O

LEVEL = 1.0 / 255.0


@contextlib.contextmanager
def make(cfg):
    env = DoomEnv(cfg)
    try:
        yield env
    finally:
        env.close()


# ---------------------------------------------------------------- engine-free
class _Lab:
    def __init__(self, oid, name, cat, x, y, w, h, value):
        self.object_id, self.object_name, self.object_category = oid, name, cat
        self.x, self.y, self.width, self.height, self.value = x, y, w, h, value


def test_rays_exclude_own_pixels_and_encode_no_reading():
    depth = np.full((120, 160), 40, np.uint8)
    lbuf = np.zeros((120, 160), np.uint8)
    lbuf[30:70, 60:100] = 255          # weapon, Self label
    depth[30:70, 60:100] = 0
    depth[30:70, 0:4] = 0              # unclaimed zero (firing overlay)
    depth[30:46, 120:122] = 1          # a real, very near world reading
    labels = [_Lab(1, "DoomPlayer", "Self", 60, 30, 40, 40, 255)]
    bad = O.invalid_depth_mask(labels, lbuf, depth)
    r = O.rays(depth, ((30, 46), (46, 54), (54, 70)), bad)
    assert (r[:, 30:50] == O.NO_READING).all()      # cells entirely own: no reading
    assert (r[:, 0:2] == O.NO_READING).all()
    assert np.isclose(r[0, 60], LEVEL)              # depth 1 stays a (non-zero) reading
    assert np.allclose(r[:, 10:20], 40 * LEVEL)     # world elsewhere untouched
    # without the mask the weapon reads "contact" (clamped 1/255) = the old bug
    assert np.allclose(O.rays(depth, ((30, 46),))[0, 30:50], LEVEL)


def test_depth0_pixels_of_a_labelled_thing_are_invalid_but_the_rest_counts():
    """BulletPuff (translucent) reports some depth-0 pixels while sitting on a wall."""
    depth = np.full((120, 160), 40, np.uint8)
    lbuf = np.zeros((120, 160), np.uint8)
    lbuf[30:46, 10:14] = 127
    depth[30:46, 10:12] = 0            # artefact pixels of the puff
    depth[30:46, 12:14] = 4            # puff pixels with a real depth
    labels = [_Lab(5, "BulletPuff", "Explosive", 10, 30, 4, 16, 127)]
    bad = O.invalid_depth_mask(labels, lbuf, depth)
    r = O.rays(depth, ((30, 46),), bad)
    assert r[0, 5] == O.NO_READING and np.isclose(r[0, 6], 4 * LEVEL)
    f, t, m = O.entities(labels, lbuf, depth, O.EntityTracker(), 24)
    assert np.isclose(f[0, 4], 4 * LEVEL)  # mean over valid pixels only


def test_entity_fully_covered_by_weapon_gives_no_row():
    """Probe R4.2: a label with no pixel of its own value is not visible -> no row."""
    depth = np.full((120, 160), 50, np.uint8)
    lbuf = np.zeros((120, 160), np.uint8)
    lbuf[60:90, 60:100] = 255
    depth[60:90, 60:100] = 0
    labels = [_Lab(1, "DoomPlayer", "Self", 60, 60, 40, 30, 255),
              _Lab(9, "Zombieman", "Monster", 70, 65, 10, 20, 33)]  # box fully under weapon
    tr = O.EntityTracker()
    tr.prev = {9: (0.0, 20 * LEVEL)}
    f, t, m = O.entities(labels, lbuf, depth, tr, 24)
    assert m.sum() == 0 and not f.any() and not t.any()
    assert tr.prev == {}


def test_entity_hidden_behind_other_sprite_gives_no_row_partly_visible_does():
    depth = np.full((120, 160), 50, np.uint8)
    lbuf = np.zeros((120, 160), np.uint8)
    lbuf[40:60, 20:40] = 77            # a monster in front ...
    depth[40:60, 20:40] = 10
    lbuf[40:60, 80:85] = 90            # a pickup partly behind another sprite
    lbuf[40:60, 85:95] = 91
    depth[40:60, 80:95] = 12
    labels = [_Lab(3, "Zombieman", "Monster", 20, 40, 20, 20, 77),
              _Lab(4, "Clip", "Ammo", 25, 50, 4, 4, 66),        # ... fully hides this clip
              _Lab(5, "Medikit", "Health", 80, 40, 12, 20, 90),  # partly visible
              _Lab(6, "Stimpack", "Health", 85, 40, 10, 20, 91)]
    f, t, m = O.entities(labels, lbuf, depth, O.EntityTracker(), 24)
    assert m.sum() == 3
    assert list(t[:3]) == [O.T["monster"], O.T["health"], O.T["health"]]
    assert np.isclose(f[1, 4], 12 * LEVEL) and np.isclose(f[1, 2], 12 / 160)  # box kept as listed


def test_own_pixels_all_at_depth0_keep_a_no_reading_row():
    """Own pixels present but none with a valid depth (translucent puff): visible, so
    the row stays, with d = NO_READING and dd = 0."""
    depth = np.full((120, 160), 50, np.uint8)
    lbuf = np.zeros((120, 160), np.uint8)
    lbuf[30:34, 10:12] = 127
    depth[30:34, 10:12] = 0
    labels = [_Lab(5, "BulletPuff", "Explosive", 10, 30, 2, 4, 127)]
    tr = O.EntityTracker()
    tr.prev = {5: (0.0, 20 * LEVEL)}
    f, t, m = O.entities(labels, lbuf, depth, tr, 24)
    assert m.sum() == 1 and f[0, 4] == O.NO_READING and f[0, 6] == 0.0 and f[0, 7] == 0.0


def test_entity_hidden_then_back_returns_as_new():
    depth = np.full((120, 160), 50, np.uint8)
    lab = _Lab(9, "Zombieman", "Monster", 70, 40, 10, 20, 33)
    vis = np.zeros((120, 160), np.uint8)
    vis[40:60, 70:80] = 33
    hid = np.zeros((120, 160), np.uint8)
    tr = O.EntityTracker()
    f, _, m = O.entities([lab], vis, depth, tr, 24)
    assert m[0] and f[0, 7] == 1.0
    f, _, m = O.entities([lab], vis, depth, tr, 24)
    assert m[0] and f[0, 7] == 0.0
    f, _, m = O.entities([lab], hid, depth, tr, 24)   # hidden behind a sprite: no row
    assert not m.any()
    f, _, m = O.entities([lab], vis, depth, tr, 24)   # back: new, zero deltas
    assert m[0] and f[0, 7] == 1.0 and f[0, 5] == 0.0 and f[0, 6] == 0.0


# ---------------------------------------------------------------- real engine
def _face_mapo1_door(env):
    for _ in range(3):  # past the 7-tic turn dead time
        env.step([0, 0, 3, 0, 0, 0])
    env.game.send_game_command("warp 670 -160")  # 34 units west of door sector 27, facing east
    obs, *_ = env.step([0, 0, 3, 0, 0, 0])
    return obs


def test_rays_facing_wall_at_known_distance_are_plausible():
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=0)
        obs = _face_mapo1_door(env)
        r = obs["rays"]
        centre = r[:2, 36:44]  # bands above and at the horizon, centre columns
        # door face ~34 map units ahead at ~8 units per depth level -> about 4 levels
        assert (centre > 0).all(), r[:, 36:44]
        assert (centre >= 2 * LEVEL).all() and (centre <= 8 * LEVEL).all(), centre * 255


def test_own_weapon_never_leaks_into_rays_while_attacking():
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=0)
        _face_mapo1_door(env)
        weapon_in_bands = zero_depth_seen = 0
        bands = env.cfg.ray_bands
        rmin, rmax = bands[0][0], bands[-1][1]
        for i in range(24):
            a = [0, 0, (3, 2, 4)[i % 3], 1, 0, 0]  # attack held, small turns, near the door
            obs, r, te, tr, info = env.step(a)
            assert not (te or tr)
            s = env.game.get_state()
            d, lb = s.depth_buffer, s.labels_buffer
            weapon_in_bands += int((lb[rmin:rmax] == 255).any())
            zero_depth_seen += int(((d[rmin:rmax] == 0) & (lb[rmin:rmax] != 255)).any())
            rays = obs["rays"]
            # A leaked invalid pixel reads depth 0 -> 1/255 after clamping. World
            # surfaces here are >= 2 levels away, so any value in (0, 1.5/255] is a leak.
            assert not ((rays > 0) & (rays <= 1.5 * LEVEL)).any(), (i, np.argwhere(
                (rays > 0) & (rays <= 1.5 * LEVEL)))
            # Reference from the raw buffers: valid = depth > 0 and not the Self label
            # (255). A cell reads 0 only if it has no valid pixel, else the valid min.
            # The original bug (plain min over all pixels) fails this.
            for bi, (r0, r1) in enumerate(bands):
                dd = d[r0:r1].astype(int)
                valid = (dd > 0) & (lb[r0:r1] != 255)
                ref = np.where(valid, dd, 10**6).reshape(r1 - r0, 80, 2).min(axis=(0, 2))
                exp = np.where(ref == 10**6, 0.0, ref / 255.0)
                np.testing.assert_allclose(rays[bi], exp, atol=1e-6, err_msg=f"band {bi} step {i}")
            # bands 0-1 see the door face: each cell is a real reading (>= 2 levels) or
            # NO_READING where a bullet puff's depth-0 pixels cover the whole cell
            c = rays[:2, 30:50]
            assert ((c == O.NO_READING) | (c >= 2 * LEVEL)).all(), c * 255
            assert (c >= 2 * LEVEL).mean() >= 0.75, c * 255
        assert weapon_in_bands > 0, "weapon never reached the ray bands: test is vacuous"
        assert zero_depth_seen > 0, "no non-weapon depth-0 pixels while firing: test is vacuous"
