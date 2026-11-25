"""Reward inputs in `info` (ADR 0010): health, armor, ammo (per slot), damage_taken_total.

These are privileged reward inputs. The tests check that they reach `info`, have the
shape rewards/terms.py reads, and that nothing about the observation changed.
One EVAL engine for the whole module, always closed.
"""
import os

import numpy as np
import pytest
import vizdoom as vzd

from env import EVAL_CONFIG, DoomEnv
from env import observation as O
from env.config import SENSOR_FIELDS, TRAIN_CONFIG, assert_same_sensors
from env.doom_env import GAME_VARS, split_vars

REWARD_KEYS = ("health", "armor", "ammo", "damage_taken_total")
GV = vzd.GameVariable
N_STEPS = 300


@pytest.fixture(scope="module")
def rollout():
    env = DoomEnv(EVAL_CONFIG)
    try:
        env.action_space.seed(1)
        obs, info = env.reset(seed=1)
        g = env.game
        at_reset = {"selected_weapon": g.get_game_variable(GV.SELECTED_WEAPON),
                    "selected_ammo": g.get_game_variable(GV.SELECTED_WEAPON_AMMO),
                    "health": g.get_game_variable(GV.HEALTH),
                    "armor": g.get_game_variable(GV.ARMOR)}
        episodes = [[(obs, info, False)]]
        for _ in range(N_STEPS):
            obs, r, te, tr, info = env.step(env.action_space.sample())
            episodes[-1].append((obs, info, bool(te or tr)))
            if te or tr:
                obs, info = env.reset()
                episodes.append([(obs, info, False)])
        yield {"space": env.observation_space, "episodes": episodes, "at_reset": at_reset}
    finally:
        env.close()


def test_reset_info_has_reward_inputs_and_zero_damage(rollout):
    for ep in rollout["episodes"]:
        _, info, _ = ep[0]
        for k in REWARD_KEYS:
            assert k in info, k
        assert info["damage_taken_total"] == 0.0


def test_reward_inputs_have_the_shape_rewards_reads(rollout):
    for ep in rollout["episodes"]:
        for _, info, _ in ep:
            assert isinstance(info["health"], float) and isinstance(info["armor"], float)
            a = info["ammo"]
            assert isinstance(a, tuple) and len(a) == 10 and all(isinstance(x, float) for x in a)
            assert isinstance(info["damage_taken_total"], float)


def test_values_match_the_engine_at_reset(rollout):
    _, info, _ = rollout["episodes"][0][0]
    r = rollout["at_reset"]
    assert info["health"] == r["health"] and info["armor"] == r["armor"]
    # Per-slot semantics on the installed engine: the selected weapon's slot holds its ammo.
    assert info["ammo"][int(r["selected_weapon"])] == r["selected_ammo"]


def test_damage_total_is_cumulative_within_an_episode(rollout):
    for ep in rollout["episodes"]:
        d = [info["damage_taken_total"] for _, info, _ in ep]
        assert all(x >= 0.0 for x in d)
        assert all(b >= a for a, b in zip(d, d[1:]))


def test_full_level_reward_terms_run_on_real_infos(rollout):
    # Every term with a non-zero scale finds its field (no MissingInfoField).
    from configs.reward_config import FULL_LEVEL_REWARD
    from rewards.terms import ExtrinsicReward

    ext = ExtrinsicReward(FULL_LEVEL_REWARD)
    for ep in rollout["episodes"]:
        ext.reset(ep[0][1])
        for _, info, ended in ep[1:]:
            ext.step(info, ended)


def test_damage_total_through_real_damage_death_and_reset():
    """Barrels damage then kill the player (TRAIN-side engine test config, cheats on)."""
    from env_test_helpers import FD2_ENGINE_TEST_CONFIG as FD2

    env = DoomEnv(FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                              episode_timeout_tics=100000))
    try:
        _, info = env.reset(seed=1)
        assert info["damage_taken_total"] == 0.0
        for _ in range(3):
            _, _, _, _, info = env.step([0, 0, 3, 0, 0, 0])
        totals, te = [info["damage_taken_total"]], False
        for _ in range(4):
            env.game.send_game_command("summon ExplosiveBarrel 48")
            for _ in range(12):
                _, _, te, tr, info = env.step([0, 0, 3, 1, 0, 0])
                if te or tr:
                    break
                totals.append(info["damage_taken_total"])
            if te or tr:
                break
        assert te and info["dead"], "player did not die from barrels"
        assert info["final_obs_fresh"] is False
        assert max(totals) > 0.0, "no damage before the lethal step: test would be vacuous"
        assert all(b >= a for a, b in zip(totals, totals[1:]))
        assert info["last_step_damage_taken"] > 0.0
        assert info["damage_taken_total"] == totals[-1] + info["last_step_damage_taken"]
        _, info = env.reset(seed=2)
        assert info["damage_taken_total"] == 0.0
    finally:
        env.close()


# ---------------------------------------------------------------- observation unchanged
EXPECTED_SPACE = {
    "rays": ((3, 80), np.float32), "entities": ((24, 8), np.float32),
    "entity_type": ((24,), np.int64), "entity_mask": ((24,), np.bool_),
    "self": ((28,), np.float32), "prev_action": ((6,), np.int64),
    "screen": ((3, 60, 80), np.uint8), "modality": ((3,), np.float32),
}


def test_observation_space_unchanged_and_has_no_reward_inputs(rollout):
    space = rollout["space"]
    assert {k: (s.shape, s.dtype) for k, s in space.spaces.items()} == EXPECTED_SPACE
    for ep in rollout["episodes"]:
        for obs, info, _ in ep:
            assert not set(obs) & set(REWARD_KEYS)
            assert space.contains(obs)
    assert O.SELF_DIM == 28


def test_sensor_signature_unchanged_and_train_eval_agree():
    assert SENSOR_FIELDS == ("frame_skip", "render_hud", "turn_bins", "max_entities",
                             "screen_enabled", "obs_version")
    sig = dict(EVAL_CONFIG.sensor_signature())
    # obs_version is in the signature only when not 1 (v1 signature byte-identical)
    assert set(sig) == (set(SENSOR_FIELDS) - {"obs_version"}) | {"ray_bands"}
    assert not any(k in sig for k in REWARD_KEYS)
    assert_same_sensors(TRAIN_CONFIG, EVAL_CONFIG)


def test_ammo_slots_do_not_reach_obs_vars():
    # Changing only the appended AMMO0..9 values must leave ObsVars identical.
    assert len(GAME_VARS) == 36
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 200, 36)
    other = base.copy()
    other[26:36] = rng.uniform(0, 200, 10)
    a, ia = split_vars(base)
    b, ib = split_vars(other)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert ia.ammo != ib.ammo


def test_build_obs_never_reads_reward_inputs():
    src = open(os.path.join(os.path.dirname(O.__file__), "doom_env.py"), encoding="utf-8").read()
    body = src.split("def _build_obs(")[1].split("\n    def ")[0]
    for word in ("v.ammo", "damage_taken_raw", "_damage_base", "AMMO_VARS", "InfoVars"):
        assert word not in body, f"_build_obs references {word!r}"
