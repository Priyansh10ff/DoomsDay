"""Engine tests on the EVAL config (one DoomGame for the whole module, always closed)."""
import numpy as np
import pytest

from env import EVAL_CONFIG, DoomEnv

N_STEPS = 400


@pytest.fixture(scope="module")
def rollout():
    env = DoomEnv(EVAL_CONFIG)
    try:
        rng = np.random.default_rng(0)
        env.action_space.seed(0)
        obs, info = env.reset(seed=0)
        traj = [(obs, info)]
        game_args = env.game.get_game_args()
        flags = (env.game.is_objects_info_enabled(), env.game.is_sectors_info_enabled(),
                 env.game.is_automap_buffer_enabled())
        for _ in range(N_STEPS):
            obs, r, te, tr, info = env.step(env.action_space.sample())
            traj.append((obs, info))
            if te or tr:
                obs, info = env.reset(seed=int(rng.integers(1000)))
                traj.append((obs, info))
        yield {"env_space": env.observation_space, "traj": traj, "game_args": game_args,
               "flags": flags}
    finally:
        env.close()


def test_obs_shapes_and_dtypes_match_space(rollout):
    space = rollout["env_space"]
    expected = {
        "rays": ((3, 80), np.float32), "entities": ((24, 8), np.float32),
        "entity_type": ((24,), np.int64), "entity_mask": ((24,), np.bool_),
        "self": ((28,), np.float32), "prev_action": ((6,), np.int64),
        "screen": ((3, 60, 80), np.uint8), "modality": ((3,), np.float32),
    }
    assert set(space.spaces) == set(expected)
    for obs, _ in rollout["traj"]:
        assert set(obs) == set(expected)
        for k, (shape, dt) in expected.items():
            assert obs[k].shape == shape, k
            assert obs[k].dtype == dt, k
            assert space[k].shape == shape and space[k].dtype == dt, k
        assert space.contains(obs)


def test_telemetry_present_under_eval(rollout):
    obs, _ = rollout["traj"][-1]
    assert obs["modality"].tolist() == [1.0, 1.0, 1.0]
    assert any(o["entity_mask"].any() for o, _ in rollout["traj"]), "no entity ever seen"


def test_eval_engine_has_no_mods(rollout):
    assert rollout["game_args"].strip() == ""
    assert rollout["flags"] == (False, False, False)


INFO_PRIVILEGED = ("pos_x", "pos_y", "pos_z", "angle", "episode_seed")


def test_no_info_key_is_an_obs_key(rollout):
    obs, info = rollout["traj"][0]
    assert not set(obs) & set(info)
    for k in obs:
        for bad in ("pos", "angle", "map", "sector", "line", "seed", "automap", "x_abs", "y_abs"):
            assert bad not in k


def test_obs_values_are_numeric_arrays_only(rollout):
    for obs, _ in rollout["traj"]:
        for k, v in obs.items():
            assert isinstance(v, np.ndarray) and v.dtype.kind in "fiub", k


def test_no_absolute_value_appears_in_obs(rollout):
    """Every non-trivial privileged number (raw, or as a plausible scaling) must be
    absent from every non-pixel observation key of the same step."""
    hits = []
    n_checked = 0
    prev_angle = None
    for obs, info in rollout["traj"]:
        vals = []
        # The legit turn delta equals the absolute heading when the previous heading
        # was 0 (mod 360), e.g. right after a spawn facing east; skip angle then.
        skip_angle = prev_angle is None or abs(((prev_angle + 180) % 360) - 180) < 1e-6
        prev_angle = float(info["angle"])
        for k in INFO_PRIVILEGED:
            v = float(info[k])
            if abs(v) < 2.0 or (k == "angle" and skip_angle):
                continue
            for s in (1.0, 1 / 8, 1 / 100, 1 / 180, 1 / 255, 1 / 360, 1 / 1000, 1 / 2048, 1 / 4096):
                x = v * s
                # Values on the sensor quantization grids (k/160, k/120 screen boxes,
                # k/255 depth, k/180 turn) collide with legitimate features by
                # construction, so a match there carries no evidence; skip them.
                on_grid = any(abs(x * q - round(x * q)) < 1e-6 for q in (160, 120, 255, 180))
                if abs(x) >= 0.02 and not on_grid:
                    vals.append((k, s, x))
        for key, arr in obs.items():
            if key == "screen":
                continue
            flat = np.asarray(arr, dtype=np.float64).ravel()
            for k, s, x in vals:
                n_checked += 1
                if np.any(np.isclose(flat, x, rtol=0, atol=1e-6)):
                    hits.append((key, k, s, x))
    assert n_checked > 0
    assert not hits, hits[:10]


@pytest.mark.skip(reason="needs a custom map with the same local scene at two absolute "
                         "locations (owned by map-forge, maps/); does not exist yet")
def test_same_observation_under_map_translation():
    pass
