"""Observation v4 (egomap channel 5, locked door known), the v3 pin, and the day-run
training knobs (DOOMSDAY_EPISODE_TICS, per-episode skill / DOOMSDAY_SKILL_MIX).
Engine tests: one engine at a time, each closed (env-engineer, 2026-11-26)."""
import contextlib
import json
import os
import tempfile

import numpy as np
import pytest

from env import egomap as E
from env import observation as O

pytestmark = pytest.mark.filterwarnings("ignore")

FWD = [1, 0, 3, 0, 0, 0]
BACK = [2, 0, 3, 0, 0, 0]
STILL = [0, 0, 3, 0, 0, 0]
USE = [0, 0, 3, 0, 1, 0]
NEEDS_BLUE = [True, False, False]
NO_NEED = [False, False, False]


# ------------------------------------------------------------------ unit (no engine)
def _levels(half_w=64.0, ahead=None):
    t = np.abs(E._COL_TAN)
    fwd = np.where(t > 1e-9, half_w / np.maximum(t, 1e-9), np.inf)
    if ahead is not None:
        fwd = np.minimum(fwd, ahead)
    return np.clip(np.rint(E.DEPTH_PER_UNIT * fwd + E.DEPTH_OFFSET), 1, E.SATURATED_LEVEL).astype(np.float64)


def _wall_levels(dist_along_x, x, heading_deg):
    """(80,) levels of an infinite wall x = dist_along_x seen from (x, 0) at a heading
    (perpendicular-to-view-plane depth, as the engine's depth buffer reports it)."""
    h = np.radians(heading_deg)
    ang = h - E._COL_ANG
    c = np.cos(ang)
    rng = np.where(c > 1e-6, (dist_along_x - x) / np.maximum(c, 1e-6), np.inf)
    perp = rng / E._COL_INV_COS
    return np.clip(np.rint(E.DEPTH_PER_UNIT * perp + E.DEPTH_OFFSET), 1, E.SATURATED_LEVEL).astype(np.float64)


def _near(lock, door, r=1):
    """Every lock cell lies within r cells of a door cell (channel 3)."""
    d = np.argwhere(door)
    return all(len(d) and np.abs(d - p).max(axis=1).min() <= r for p in np.argwhere(lock))


def test_v4_shape_helpers_and_v3_unchanged_contract():
    assert E.egomap_shape(2) is None and E.egomap_shape(3) == (5, 32, 32) == E.EGOMAP_SHAPE
    assert E.egomap_shape(4) == (6, 32, 32) == E.EGOMAP_SHAPE_V4
    assert E.EgoMap().crop().shape == (5, 32, 32)
    assert E.EgoMap(locked_channel=True).crop().shape == (6, 32, 32)
    # a v3 map ignores the pulse entirely
    m = E.EgoMap()
    m.update(0.0, 0.0, 0.0, _levels(ahead=60.0), np.ones(80, np.float32), needs_key=NEEDS_BLUE)
    assert m.crop().shape[0] == 5 and not (m.grid & 128).any()


def test_unit_locked_mark_needs_pulse_door_and_use_range():
    door = (np.abs(E._COL_TAN) < 0.4).astype(np.float32)
    # pulse, door at 60 units ahead on the centre columns -> marked at channel 3's door cells
    # (a 48-unit block around each centre hit, so it never drops out of the rotated crop)
    m = E.EgoMap(locked_channel=True)
    m.update(0.0, 0.0, 0.0, _levels(ahead=60.0), door, needs_key=NEEDS_BLUE)
    c = m.crop()
    assert c[5].sum() >= 1 and c[5, 14:16, 15:18].sum() == c[5].sum(), np.argwhere(c[5])
    assert _near(c[5], c[3])
    # no pulse -> nothing; door too far for USE -> nothing; centre not a door -> nothing
    for lv, dc, need in ((_levels(ahead=60.0), door, NO_NEED),
                         (_levels(ahead=300.0), door, NEEDS_BLUE),
                         (_levels(ahead=60.0), np.zeros(80, np.float32), NEEDS_BLUE)):
        m = E.EgoMap(locked_channel=True)
        m.update(0.0, 0.0, 0.0, lv, dc, needs_key=need)
        assert m.crop()[5].sum() == 0


def test_unit_locked_persists_and_survives_grazing_and_closed_views():
    """Marked at the door face x = 60, then: walk back, turn, view the CLOSED door at
    15-25 degrees from close range (grazing), turn away: the mark stays."""
    door_all = np.ones(80, np.float32)
    m = E.EgoMap(locked_channel=True)
    m.update(0.0, 0.0, 0.0, _wall_levels(60.0, 0.0, 0.0), door_all, needs_key=NEEDS_BLUE)
    n0 = int((m.grid & 128).astype(bool).sum())
    assert n0 >= 1
    for x, hdg in ((0.0, 15.0), (10.0, 20.0), (20.0, 25.0), (-5.0, -20.0), (0.0, 0.0)):
        m.update(x, 0.0, hdg, _wall_levels(60.0, x, hdg), door_all, needs_key=NO_NEED)
        assert int((m.grid & 128).astype(bool).sum()) == n0, (x, hdg)
    # the same grazing views with the face lit as WALL (door channel off) still do not clear:
    # no ray gets past the plane
    for x, hdg in ((0.0, 15.0), (20.0, 25.0)):
        m.update(x, 0.0, hdg, _wall_levels(60.0, x, hdg), np.zeros(80, np.float32), needs_key=NO_NEED)
        assert int((m.grid & 128).astype(bool).sum()) == n0
    for x, hdg in ((-200.0, 0.0), (-200.0, 90.0), (-200.0, 180.0)):
        m.update(x, 0.0, hdg, np.zeros(80), None, needs_key=NO_NEED)
    assert int((m.grid & 128).astype(bool).sum()) == n0
    assert m.crop()[5].sum() >= 1


def test_unit_locked_cleared_when_seen_open_or_passed():
    door_all = np.ones(80, np.float32)
    # seen open: rays run far past the door plane
    m = E.EgoMap(locked_channel=True)
    m.update(0.0, 0.0, 0.0, _wall_levels(60.0, 0.0, 0.0), door_all, needs_key=NEEDS_BLUE)
    m.update(-100.0, 0.0, 0.0, _levels(ahead=None), np.zeros(80, np.float32))
    assert not (m.grid & 128).any() and m.crop()[5].sum() == 0
    # passed: the trail goes through the plane, no readings at all
    m = E.EgoMap(locked_channel=True)
    m.update(0.0, 0.0, 0.0, _wall_levels(60.0, 0.0, 0.0), door_all, needs_key=NEEDS_BLUE)
    m.update(100.0, 0.0, 0.0, np.zeros(80), None)
    assert not (m.grid & 128).any()
    # a second locked door elsewhere is kept when the first clears
    m = E.EgoMap(locked_channel=True)
    m.update(0.0, 0.0, 0.0, _wall_levels(60.0, 0.0, 0.0), door_all, needs_key=NEEDS_BLUE)
    m.update(0.0, 500.0, 180.0, _levels(ahead=60.0), door_all, needs_key=[False, True, False])
    m.update(100.0, 0.0, 0.0, np.zeros(80), None)
    assert (m.grid & 128).any()


def test_unit_v4_translation_invariance():
    door_all = np.ones(80, np.float32)
    outs = []
    for x0, y0 in ((0.0, 0.0), (1234.0, -987.0)):
        m = E.EgoMap(locked_channel=True)
        seq = []
        for k in range(4):
            m.update(x0, y0, 0.0, _wall_levels(60.0, 0.0, 0.0), door_all,
                     needs_key=NEEDS_BLUE if k == 1 else NO_NEED)
            seq.append(m.crop())
        outs.append(seq)
    for a, b in zip(*outs):
        assert np.array_equal(a, b)
    assert outs[0][-1][5].any()


# ------------------------------------------------------------------ v3 pin
def _golden():
    from obs_v3_golden import GOLDEN
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def test_v3_unit_egomap_byte_identical_to_pre_v4_golden():
    from obs_v3_golden import unit_hashes
    assert unit_hashes() == _golden()["unit"]


def test_v3_engine_observation_byte_identical_to_pre_v4_golden(tmp_path):
    from obs_v3_golden import engine_run
    g = _golden()
    _, steps = engine_run(str(tmp_path), g["engine_actions"])
    assert len(steps) == len(g["engine"])
    for i, (a, b) in enumerate(zip(steps, g["engine"])):
        assert a == b, (i, {k for k in a if a[k] != b.get(k)})


# ------------------------------------------------------------------ config / spec
def test_default_version_3_and_v4_opt_in(monkeypatch):
    from configs import a0_env
    from env.config import EVAL_CONFIG
    from train.bc_spec import a0_training_spec_and_signature, egomap_step_shape
    assert a0_env.OBS_VERSION == 3 and 4 in a0_env.OBS_VERSION_CHOICES
    monkeypatch.delenv(a0_env.OBS_VERSION_ENV_VAR, raising=False)
    assert a0_env.obs_version() == 3
    assert egomap_step_shape(4) == E.EGOMAP_SHAPE_V4 and egomap_step_shape(3) == E.EGOMAP_SHAPE
    sigs = {}
    for v in ("3", "4"):
        monkeypatch.setenv(a0_env.OBS_VERSION_ENV_VAR, v)
        spec, sig = a0_training_spec_and_signature()
        assert spec.egomap == E.egomap_shape(int(v))
        sigs[v] = dict(sig)
    # v4 differs from v3 ONLY in obs_version (init-from growth relies on it)
    assert {k for k in sigs["3"] if sigs["3"][k] != sigs["4"].get(k)} == {"obs_version"}
    assert set(sigs["3"]) == set(sigs["4"])
    cfg4 = EVAL_CONFIG.replace(obs_version=4)
    cfg4.assert_vanilla()
    assert cfg4.locked_door_channel and cfg4.egomap and not EVAL_CONFIG.replace(obs_version=3).locked_door_channel
    with pytest.raises(ValueError):
        EVAL_CONFIG.replace(obs_version=5).validate()


def test_eval_rejects_skill_mix_and_train_validates_it():
    from env.config import EVAL_CONFIG, RULE_FIELDS, TRAIN_CONFIG
    assert "skill_mix" in RULE_FIELDS and EVAL_CONFIG.skill_mix == ()
    with pytest.raises(AssertionError, match="skill_mix"):
        EVAL_CONFIG.replace(skill_mix=(1, 3)).assert_vanilla()
    with pytest.raises(ValueError):
        TRAIN_CONFIG.replace(skill_mix=(0, 3)).validate()
    TRAIN_CONFIG.replace(skill_mix=(1, 2, 3)).validate()
    # skill_mix is not a sensor field: the signature is unchanged
    assert TRAIN_CONFIG.replace(skill_mix=(1, 4)).sensor_signature() == TRAIN_CONFIG.sensor_signature()


def _bundle_cfg():
    from configs.a0_env import a0_train_env_config, generated_maps_env_config
    from maps.loader import training_bundle
    bundle, lump_to_id, _ = training_bundle()
    return generated_maps_env_config(a0_train_env_config(), bundle, tuple(sorted(lump_to_id)))


def test_episode_tics_and_skill_mix_env_vars(monkeypatch):
    from configs import a0_env
    monkeypatch.delenv(a0_env.EPISODE_TICS_ENV_VAR, raising=False)
    monkeypatch.delenv(a0_env.SKILL_MIX_ENV_VAR, raising=False)
    cfg = _bundle_cfg()
    assert cfg.episode_timeout_tics == 4200 and cfg.skill == 3 and cfg.skill_mix == ()
    assert a0_env.GENERATED_MAPS.episode_timeout_tics == 4200   # live-run default untouched
    monkeypatch.setenv(a0_env.EPISODE_TICS_ENV_VAR, "8400")
    monkeypatch.setenv(a0_env.SKILL_MIX_ENV_VAR, "1,2,3,3,4")
    cfg = _bundle_cfg()
    assert cfg.episode_timeout_tics == 8400 and cfg.skill_mix == (1, 2, 3, 3, 4) and cfg.skill == 3
    s = a0_env.generated_maps_settings()
    assert s.episode_timeout_tics == 8400 and s.skill_mix == (1, 2, 3, 3, 4)
    for bad_t, bad_m in (("0", ""), ("", "0,3"), ("", "x")):
        monkeypatch.setenv(a0_env.EPISODE_TICS_ENV_VAR, bad_t)
        monkeypatch.setenv(a0_env.SKILL_MIX_ENV_VAR, bad_m)
        with pytest.raises(ValueError):
            a0_env.generated_maps_settings()


def test_eval_configs_never_carry_the_skill_mix(monkeypatch):
    """Evaluation configs read only GENERATED_MAPS.skill; the mix never reaches them."""
    import inspect
    from evaluate import map_jobs, record
    for mod in (map_jobs, record):
        assert "skill_mix" not in inspect.getsource(mod)


def _fake_env(cfg, seed=0):
    from env.doom_env import DoomEnv
    import gymnasium as gym
    env = DoomEnv.__new__(DoomEnv)
    env.cfg = cfg
    gym.Env.reset(env, seed=seed)
    return env


def test_skill_choice_option_mix_and_default():
    from env.config import EVAL_CONFIG, TRAIN_CONFIG
    mix = (1, 2, 3, 3, 4)
    env = _fake_env(TRAIN_CONFIG.replace(skill=3, skill_mix=mix), seed=5)
    draws = [env._episode_skill({}) for _ in range(5000)]
    counts = {k: draws.count(k) for k in set(draws)}
    assert set(counts) == {1, 2, 3, 4}
    assert abs(counts[3] / 5000 - 0.4) < 0.03 and abs(counts[1] / 5000 - 0.2) < 0.03
    env2 = _fake_env(TRAIN_CONFIG.replace(skill=3, skill_mix=mix), seed=5)
    assert [env2._episode_skill({}) for _ in range(50)] == draws[:50]   # seeded from the env RNG
    assert env._episode_skill({"skill": 5}) == 5
    with pytest.raises(ValueError):
        env._episode_skill({"skill": 6})
    plain = _fake_env(TRAIN_CONFIG.replace(skill=3))
    state = plain.np_random.bit_generator.state
    assert plain._episode_skill({}) == 3
    assert plain.np_random.bit_generator.state == state      # no draw without a mix
    ev = _fake_env(EVAL_CONFIG)
    assert ev._episode_skill({}) == 3
    with pytest.raises(ValueError, match="EVAL"):
        ev._episode_skill({"skill": 1})


# ------------------------------------------------------------------ engine
@contextlib.contextmanager
def _corridor_env(tmp_path, v=4, name="locked.wad", **kw):
    from egomap_maps import corridor_config, corridor_wad
    from env import DoomEnv
    cfg_kw = kw.pop("cfg", {})
    wad = corridor_wad(os.path.join(str(tmp_path), name), **kw)
    env = DoomEnv(corridor_config(wad, obs_version=v).replace(**cfg_kw))
    try:
        yield env
    finally:
        env.close()


def _face(env, info, target):
    obs = None
    for _ in range(30):
        err = (target - info["angle"] + 180) % 360 - 180
        if abs(err) <= 2:
            break
        bins = np.array(env.cfg.turn_bins) * env.cfg.frame_skip
        k = int(np.argmin(np.abs(-bins - err)))
        obs, r, te, tr, info = env.step(np.array([0, 0, k, 0, 0, 0]))
    assert abs((target - info["angle"] + 180) % 360 - 180) <= 2
    return obs, info


def _walk_to_door(env, obs, info):
    n = 0
    while info["pos_x"] < 580 and n < 60:
        obs, r, te, tr, info = env.step(np.array(FWD))
        n += 1
    for a in (STILL, STILL):
        obs, r, te, tr, info = env.step(np.array(a))
    return obs, info


def test_engine_corridor_locked_door_marked_persists_and_clears(tmp_path):
    """Blue-locked DR door (special 26) at x = 640, blue key behind the player. One
    episode: USE without the key -> channel 5 set; back off and fetch the key -> still
    set; USE with the key, door opens, walk through -> cleared."""
    with _corridor_env(tmp_path, door_special=26, player_x=300, key_x=120) as env:
        assert env.observation_space["egomap"].shape == (6, 32, 32)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs) and obs["egomap"][5].sum() == 0
        obs, info = _walk_to_door(env, obs, info)
        assert obs["egomap"][5].sum() == 0 and obs["egomap"][3].sum() > 0     # closed door, not yet tried
        pulses = 0
        for a in (USE, STILL, USE, STILL):
            obs, r, te, tr, info = env.step(np.array(a))
            pulses += int(obs["self"][31] > 0.5)
            assert env.observation_space.contains(obs)
        assert pulses >= 1, "no needs-key pulse at the locked door"
        lock = obs["egomap"][5]
        assert lock.sum() >= 1 and lock[10:17, 13:20].sum() == lock.sum(), np.argwhere(lock)
        assert _near(lock, obs["egomap"][3])
        # fetch the key behind: the mark stays the whole way
        obs, info = _face(env, info, 180.0)
        held = False
        for _ in range(60):
            obs, r, te, tr, info = env.step(np.array(FWD))
            held = bool(obs["self"][28] > 0.5)
            assert not (te or tr)
            if held:
                break
        assert held, "key not picked up"
        er = (640 - info["pos_x"])                   # door now ~ this far behind the agent
        assert er < 768 - 48                         # still inside the crop
        assert obs["egomap"][5].sum() >= 1, "locked mark lost while fetching the key"
        obs, info = _face(env, info, 0.0)
        assert obs["egomap"][5].sum() >= 1
        obs, info = _walk_to_door(env, obs, info)
        assert obs["egomap"][5].sum() >= 1
        for _ in range(3):
            obs, r, te, tr, info = env.step(np.array(USE))
        for _ in range(12):                          # door opening; rays pass the plane
            obs, r, te, tr, info = env.step(np.array(STILL))
        for _ in range(8):
            obs, r, te, tr, info = env.step(np.array(FWD))
        assert info["pos_x"] > 660, info["pos_x"]
        assert obs["egomap"][5].sum() == 0


def test_engine_v4_equals_v3_except_channel_5_and_translation(tmp_path):
    acts = [FWD] * 16 + [USE, STILL, USE, STILL, [0, 0, 5, 0, 0, 0], BACK, BACK]
    runs = {}
    for v, shift in ((3, (0, 0)), (4, (0, 0)), (4, (1000, -704))):
        with _corridor_env(tmp_path, v=v, name=f"v{v}_{shift[0]}.wad", door_special=26, player_x=300,
                           shift=shift) as env:
            obs, info = env.reset(seed=0)
            seq = [obs]
            for a in acts:
                obs, *_ = env.step(np.array(a))
                seq.append(obs)
            runs[(v, shift)] = seq
    for o3, o4, o4s in zip(runs[(3, (0, 0))], runs[(4, (0, 0))], runs[(4, (1000, -704))]):
        assert set(o3) == set(o4)
        for k in o3:
            if k == "egomap":
                assert np.array_equal(o3[k], o4[k][:5])
            else:
                assert np.array_equal(o3[k], o4[k]), k
        for k in o4:
            assert np.array_equal(o4[k], o4s[k]), k
    assert any(o["egomap"][5].any() for o in runs[(4, (0, 0))])


def test_engine_s2_teacher_locked_door_marked_and_persists():
    """Generated S2 TRAIN map (blue lock), scripted teacher (test side only) sent to the
    door without the key and pressing USE there: channel 5 is set on the pulse and stays
    set for the rest of the drive (no reset)."""
    import test_env_obs_v2_engine as V2
    from configs.a0_env import generated_maps_env_config
    from env import DoomEnv
    from env.config import TRAIN_CONFIG
    bundle, lump_to_id = V2._bundle()
    base = TRAIN_CONFIG.replace(screen_enabled=False, obs_version=4, win_timer_ms=None)
    env = DoomEnv(generated_maps_env_config(base, bundle, tuple(sorted(lump_to_id))))
    rows = []
    try:
        assert env.observation_space["egomap"].shape == (6, 32, 32)

        def check(t, obs, info, ctrl, a):
            rows.append((bool(obs["self"][31] > 0.5), int(obs["egomap"][5].sum()),
                         bool(env.observation_space.contains(obs))))
            return len(rows) < 160
        V2._drive(env, "S2-train-102001", 400, check, drop_key=True, press_locked=True)
    finally:
        env.close()
    assert all(r[2] for r in rows)
    first = next((i for i, r in enumerate(rows) if r[0]), None)
    assert first is not None, "the locked-door message never appeared"
    assert all(r[1] == 0 for r in rows[:first]), "marked before any needs-key pulse"
    marked = [i for i, r in enumerate(rows) if r[1] > 0]
    assert marked and marked[0] == first, (first, marked[:3])
    assert all(r[1] > 0 for r in rows[first:]), "locked mark not persistent"


def _zombies(env):
    return sum(1 for o in env.game.get_state().objects if o.name == "Zombieman")


SKILL_THINGS = tuple((1000 + 100 * j, -40 + 40 * i, 3004, fl | 8)
                     for i, fl in enumerate((1, 2, 4)) for j in range(1 + i))   # 1 easy, 2 medium, 3 hard


def test_engine_skill_option_and_mix_change_the_engine_skill(tmp_path):
    """Probe R6.1 as a test: zombiemen flagged easy-only (1), medium-only (2), hard-only (3)."""
    with _corridor_env(tmp_path, v=3, name="skill.wad", with_door=False, extra_things=SKILL_THINGS,
                       cfg={"objects_info": True, "skill": 3}) as env:
        obs, info = env.reset(seed=0)
        assert _zombies(env) == 2 and info["skill"] == 3          # plain reset after init: config skill
        for k, want in ((1, 1), (4, 3), (2, 1), (3, 2), (5, 3), (4, 3)):
            obs, info = env.reset(seed=1, options={"skill": k})
            assert (_zombies(env), info["skill"]) == (want, k), k
        obs, info = env.reset(seed=2)                             # back to the config skill
        assert _zombies(env) == 2 and info["skill"] == 3
    with _corridor_env(tmp_path, v=3, name="skill2.wad", with_door=False, extra_things=SKILL_THINGS,
                       cfg={"objects_info": True, "skill": 3, "skill_mix": (1, 4)}) as env:
        seen = {}
        env.reset(seed=11)
        for _ in range(12):
            obs, info = env.reset()
            seen.setdefault(info["skill"], set()).add(_zombies(env))
        assert seen == {1: {1}, 4: {3}}, seen


def test_engine_viz_nocheat_v4_map_zero():
    from env import DoomEnv
    from env.config import EVAL_CONFIG
    cfg = EVAL_CONFIG.replace(obs_version=4, game_args=("+viz_nocheat 1",), screen_enabled=False)
    cfg.assert_vanilla()
    env = DoomEnv(cfg)
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(6):
            obs, *_ = env.step(np.array([1, 0, 5, 0, 1, 0]))
            assert obs["egomap"].shape == (6, 32, 32) and obs["egomap"].sum() == 0
    finally:
        env.close()


def _door_lit_near_centre(dist_x, x, heading_deg, half=64.0):
    """Door channel lit only for columns hitting the wall x = dist_x within `half` units
    (along the wall) of the centre ray's hit: a door in a flush wall."""
    h = np.radians(heading_deg)
    ang = h - E._COL_ANG
    yc = (dist_x - x) * np.tan(h)
    y = (dist_x - x) * np.tan(ang)
    return ((np.cos(ang) > 1e-6) & (np.abs(y - yc) <= half)).astype(np.float32)


@pytest.mark.parametrize("hdg", [0.0, 20.0, 30.0, 45.0, -45.0])
def test_unit_oblique_press_keeps_its_mark(hdg):
    """USE pressed at a door in a flush wall at an angle: the wall beside the door must not
    count as 'seen through the door' (the face normal comes from the lit door columns)."""
    x = 0.0
    lv, dc = _wall_levels(40.0, x, hdg), _door_lit_near_centre(40.0, x, hdg)
    m = E.EgoMap(locked_channel=True)
    m.update(x, 0.0, hdg, lv, dc, needs_key=NEEDS_BLUE)
    assert len(m._locks) == 1
    nx, ny = m._locks[0][3:]
    # the wall's normal (world +x), expressed in the spawn frame (x axis = heading at spawn)
    want = (np.cos(np.radians(-hdg)), np.sin(np.radians(-hdg)))
    assert np.hypot(nx - want[0], ny - want[1]) < 0.05, (nx, ny, want)
    for k in range(3):
        m.update(x, 0.0, hdg, lv, dc, needs_key=NO_NEED)
        m.update(x - 20.0, 0.0, -hdg, _wall_levels(40.0, x - 20.0, -hdg), np.zeros(80, np.float32))
    assert len(m._locks) == 1 and m.crop()[5].any()


def test_unit_mark_visible_in_crop_at_every_heading_and_offset():
    door_all = np.ones(80, np.float32)
    m = E.EgoMap(locked_channel=True)
    m.update(0.0, 0.0, 0.0, _wall_levels(60.0, 0.0, 0.0), door_all, needs_key=NEEDS_BLUE)
    for hdg in np.arange(0.0, 360.0, 7.5):
        for dx, dy in ((0.0, 0.0), (-13.0, 7.0), (-31.0, -19.0), (-200.0, 150.0)):
            m.update(dx, dy, float(hdg), np.zeros(80), None, needs_key=NO_NEED)
            assert m.crop()[5].any(), (hdg, dx, dy)
