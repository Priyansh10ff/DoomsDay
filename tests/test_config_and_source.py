"""Engine-free tests: config whitelist, action encoding, source audit, ego-motion."""
import dataclasses
import math
import os
import re

import numpy as np
import pytest

from env import config as config_module
from env import doom_env, wintimer
from env import observation as O
from env.actions import BUTTONS, IDX_USE, encode, head_sizes
from env.config import (EVAL_CONFIG, FIELD_GROUPS, TIMER_ENV_VAR, TRAIN_CONFIG, TURN_BINS,
                        EnvConfig, assert_same_sensors, horizon_row)
from env.doom_env import keys_picked_up

ENV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env")


# ---------------------------------------------------------------- TRAIN vs EVAL
def test_eval_is_vanilla_and_named():
    EVAL_CONFIG.assert_vanilla()
    assert EVAL_CONFIG.name == "eval" and TRAIN_CONFIG.name == "train"
    assert EVAL_CONFIG.vanilla_eval and not TRAIN_CONFIG.vanilla_eval


def test_eval_has_no_training_only_fields_set():
    # Every training-only knob of EVAL equals the dataclass default (= vanilla).
    defaults = EnvConfig(name="x", vanilla_eval=True)
    for f in ("scenario_wad", "game_args", "objects_info", "sectors_info", "engine_rewards", "extra"):
        assert getattr(EVAL_CONFIG, f) == getattr(defaults, f), f


@pytest.mark.parametrize("mod", [
    {"game_args": ("+sv_cheats 1",)},
    {"game_args": ("+viz_nocheat 0",)},
    {"scenario_wad": "curriculum.wad"},
    {"objects_info": True},
    {"sectors_info": True},
    {"engine_rewards": (("map_exit_reward", 1.0), ("living_reward", -0.01))},
    {"engine_rewards": (("map_exit_reward", 5.0),)},
    {"iwad": "custom"},
    {"extra": {"infinite_ammo": True}},
    {"vanilla_eval": False},
])
def test_any_mod_added_to_eval_fails(mod):
    modded = EVAL_CONFIG.replace(**mod)
    with pytest.raises(AssertionError):
        modded.assert_vanilla()
    if mod.get("vanilla_eval", True):
        with pytest.raises(AssertionError):
            modded.validate()  # DoomEnv.__init__ calls this before starting the engine


def test_train_may_carry_mods():
    TRAIN_CONFIG.replace(game_args=("+sv_cheats 1",), objects_info=True,
                         engine_rewards=(("living_reward", -1.0),)).validate()


# ---------------------------------------------------------------- item 8: field groups
def test_every_field_is_in_exactly_one_group():
    names = [f.name for f in dataclasses.fields(EnvConfig)]
    grouped = [f for g in FIELD_GROUPS.values() for f in g]
    assert sorted(grouped) == sorted(names), (set(names) ^ set(grouped))
    assert len(grouped) == len(set(grouped))


@pytest.mark.parametrize("change", [
    {"frame_skip": 2}, {"render_hud": False}, {"turn_bins": (-5.0, 0.0, 5.0)},
    {"max_entities": 16}, {"screen_enabled": False}, {"win_timer_ms": 1},
    {"maps": ("MAP07",)}, {"skill": 5}, {"episode_timeout_tics": 2100},
])
def test_sensor_harness_and_host_fields_do_not_break_vanilla(change):
    EVAL_CONFIG.replace(**change).assert_vanilla()


@pytest.mark.parametrize("change", [{"skill": 0}, {"skill": 6}, {"maps": ()},
                                    {"episode_timeout_tics": 0}])
def test_illegal_harness_values_fail(change):
    with pytest.raises(AssertionError):
        EVAL_CONFIG.replace(**change).assert_vanilla()


@pytest.mark.parametrize("change", [{"frame_skip": 2}, {"render_hud": False},
                                    {"turn_bins": (-5.0, 0.0, 5.0)}, {"max_entities": 16},
                                    {"screen_enabled": False}])
def test_sensor_mismatch_between_train_and_eval_fails(change):
    assert_same_sensors(TRAIN_CONFIG, EVAL_CONFIG)  # shipped defaults agree
    with pytest.raises(AssertionError):
        assert_same_sensors(TRAIN_CONFIG.replace(**change), EVAL_CONFIG)


def test_vanilla_docstring_names_every_group():
    doc = EnvConfig.assert_vanilla.__doc__ + config_module.__doc__
    for word in ("VANILLA RULES", "HARNESS", "SENSOR", "HOST"):
        assert word in doc


# ---------------------------------------------------------------- ray bands (0006)
def test_ray_bands_follow_horizon():
    assert horizon_row(True) == 50 and horizon_row(False) == 60
    assert EVAL_CONFIG.ray_bands == ((30, 46), (46, 54), (54, 70))
    assert EVAL_CONFIG.replace(render_hud=False).ray_bands == ((40, 56), (56, 64), (64, 80))
    assert TRAIN_CONFIG.ray_bands == EVAL_CONFIG.ray_bands


# ---------------------------------------------------------------- timer switch (0006)
def test_timer_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv(TIMER_ENV_VAR, raising=False)
    assert TRAIN_CONFIG.effective_timer_ms() == 1
    assert EVAL_CONFIG.effective_timer_ms() is None
    monkeypatch.setenv(TIMER_ENV_VAR, "0")
    assert TRAIN_CONFIG.effective_timer_ms() is None
    monkeypatch.setenv(TIMER_ENV_VAR, "1")
    assert EVAL_CONFIG.effective_timer_ms() == 1


class _FakeWinmm:
    def __init__(self, begin_ret=0):
        self.begin, self.end, self.begin_ret = 0, 0, begin_ret

    def timeBeginPeriod(self, p):
        self.begin += 1
        return self.begin_ret

    def timeEndPeriod(self, p):
        self.end += 1
        return 0


def test_timer_request_pairs_exactly_once(monkeypatch):
    fake = _FakeWinmm()
    monkeypatch.setattr(wintimer, "_winmm", lambda: fake)
    t = wintimer.TimerRequest(1)
    t.release()
    t.release()
    assert (fake.begin, fake.end) == (1, 1)
    failed = _FakeWinmm(begin_ret=97)  # TIMERR_NOCANDO: nothing to release
    monkeypatch.setattr(wintimer, "_winmm", lambda: failed)
    t = wintimer.TimerRequest(1)
    t.release()
    assert (failed.begin, failed.end) == (1, 0)
    off = _FakeWinmm()
    monkeypatch.setattr(wintimer, "_winmm", lambda: off)
    wintimer.TimerRequest(None).release()
    assert (off.begin, off.end) == (0, 0)


def test_env_init_failure_takes_no_timer(monkeypatch):
    fake = _FakeWinmm()
    monkeypatch.setattr(wintimer, "_winmm", lambda: fake)

    def boom(self):
        raise RuntimeError("engine failed to start")

    monkeypatch.setattr(doom_env.DoomEnv, "_make_game", boom)
    with pytest.raises(RuntimeError):
        doom_env.DoomEnv(TRAIN_CONFIG)
    with pytest.raises(AssertionError):  # invalid EVAL config: fails before anything starts
        doom_env.DoomEnv(EVAL_CONFIG.replace(game_args=("+sv_cheats 1",)))
    assert (fake.begin, fake.end) == (0, 0)


# ---------------------------------------------------------------- actions
def test_action_heads_match_adr0002():
    assert head_sizes(len(TURN_BINS)) == (3, 3, 7, 2, 2, 2)
    assert TURN_BINS == (-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0)
    assert len(BUTTONS) == 8


def test_use_is_plain_button_state():
    for a in ([0, 0, 3, 0, 1, 0], [1, 2, 6, 1, 1, 1]):
        assert encode(a, TURN_BINS)[IDX_USE] == 1.0
    assert encode([0, 0, 3, 0, 0, 0], TURN_BINS)[IDX_USE] == 0.0
    src = open(os.path.join(ENV_DIR, "doom_env.py"), encoding="utf-8").read()
    assert src.count("advance_action(") == 1  # one round trip per decision, no split


# ---------------------------------------------------------------- source audit
FORBIDDEN = ["atan2", "arctan2", "object_position", "object_angle", "object_velocity",
             "position_x", "position_y", "position_z", "POSITION_", "automap", ".sectors", ".objects"]


def test_observation_module_has_no_forbidden_reads():
    src = open(os.path.join(ENV_DIR, "observation.py"), encoding="utf-8").read()
    for word in FORBIDDEN:
        assert word not in src, f"observation.py contains {word!r}"


def test_policy_path_receives_no_info_vars():
    src = open(os.path.join(ENV_DIR, "doom_env.py"), encoding="utf-8").read()
    body = src.split("def _build_obs(")[1].split("\n    def ")[0]
    for word in ("InfoVars", "pos_x", "pos_y", "pos_z", "killcount", "_map", "episode_seed",
                 "keys_held", "notifications", "objects", "sectors", "object_position"):
        assert word not in body, f"_build_obs references {word!r}"
    assert not re.search(r"atan2|arctan2", src)


# ---------------------------------------------------------------- ego-motion
@pytest.mark.parametrize("offset", [0.0, 37.0, 90.0, 180.0, 271.5])
def test_ego_motion_invariant_to_world_rotation(offset):
    rng = np.random.default_rng(1)
    for _ in range(20):
        vx, vy, vz = rng.uniform(-15, 15, 3)
        h0 = rng.uniform(0, 360)
        dh = rng.uniform(-40, 40)
        base = O.ego_motion(vx, vy, vz, h0 + dh, h0)
        r = math.radians(offset)
        rvx = vx * math.cos(r) - vy * math.sin(r)
        rvy = vx * math.sin(r) + vy * math.cos(r)
        rot = O.ego_motion(rvx, rvy, vz, (h0 + dh + offset) % 360, (h0 + offset) % 360)
        np.testing.assert_allclose(base, rot, atol=1e-5)


def test_ego_motion_forward_is_forward():
    e = O.ego_motion(10.0, 0.0, 0.0, 0.0, 0.0)
    assert e[0] > 0 and abs(e[1]) < 1e-6
    e = O.ego_motion(0.0, 10.0, 0.0, 90.0, 90.0)
    assert e[0] > 0 and abs(e[1]) < 1e-6


# ---------------------------------------------------------------- entity table
def test_entity_type_table():
    t = O.T
    assert O.entity_type_id("BlueCard", "Key") == t["key-blue"]
    assert O.entity_type_id("YellowSkull", "Key") == t["key-yellow"]
    assert O.entity_type_id("RedCard", "Key") == t["key-red"]
    assert O.entity_type_id("Zombieman", "Monster") == t["monster"]
    assert O.entity_type_id("DoomImpBall", "Explosive") == t["projectile"]
    assert O.entity_type_id("Medikit", "Health") == t["health"]
    assert O.entity_type_id("SomethingNew", "WeirdCategory") == t["unknown"]
    assert O.entity_type_id("ExplosiveBarrel", "Hazard") == t["hazard"] == len(O.ENTITY_TYPES) - 1
    assert t["monster"] == 1 and t["unknown"] == 12  # appending hazard moved no id


def test_item6_bullet_puff_is_not_a_projectile():
    # Probe `projectiles`: our own shots make BulletPuff, category "Explosive".
    t = O.T
    assert O.entity_type_id("BulletPuff", "Explosive") == t["decoration"]
    assert O.entity_type_id("RevenantTracerSmoke", "Explosive") == t["decoration"]
    for name in ("DoomImpBall", "CacodemonBall", "BaronBall", "Rocket", "PlasmaBall",
                 "ArachnotronPlasma", "FatShot", "RevenantTracer", "BFGBall"):
        assert O.entity_type_id(name, "Explosive") == t["projectile"], name
    assert O.entity_type_id("SomeModMissile", "Explosive") == t["unknown"]


# ---------------------------------------------------------------- item 7: key messages
# Strings copied from the engine (probes `keys` and `locked`, freedoom2).
LOCKED_TEXT = ("\x1cG\n\x1d\x1e\x1e\x1f\x1cB\nYellow key needed for this door.\n\x1cG\n\x1c-\n",
               "\x1cB\nBlue key needed.\n", "Red key needed for this door.")
PICKUP_TEXT = ("Red skeleton key secured!\nYellow skeleton key secured!\nBlue skeleton key secured!\n"
               "Red passcard secured!\nYellow passcard secured!\nBlue passcard secured!\n")


def test_item7_locked_door_text_is_not_a_pickup():
    for txt in LOCKED_TEXT:
        assert keys_picked_up(txt) == set(), txt


def test_item7_pickup_text_is_parsed():
    assert keys_picked_up(PICKUP_TEXT) == {"blue", "yellow", "red"}
    assert keys_picked_up("Blue passcard secured!\n") == {"blue"}
    # Doom IWAD wording (string table, not probed here)
    assert keys_picked_up("Picked up a yellow skull key.") == {"yellow"}
    assert keys_picked_up("Picked up a red keycard.") == {"red"}
    assert keys_picked_up("map01 - Hydroelectric Plant\nPicked up a clip.") == set()


class _Lab:
    def __init__(self, oid, name, cat, x, y, w, h, value):
        self.object_id, self.object_name, self.object_category = oid, name, cat
        self.x, self.y, self.width, self.height, self.value = x, y, w, h, value


def test_entities_excludes_self_keeps_nearest_and_tracks_motion():
    depth = np.full((120, 160), 200, np.uint8)
    lbuf = np.zeros((120, 160), np.uint8)
    labs = [_Lab(1, "DoomPlayer", "Self", 70, 80, 20, 20, 255)]
    for i in range(30):
        x = 2 + 5 * i
        lbuf[10:14, x:x + 4] = 10 + i
        depth[10:14, x:x + 4] = 10 + 3 * i
        labs.append(_Lab(100 + i, "Zombieman", "Monster", x, 10, 4, 4, 10 + i))
    tr = O.EntityTracker()
    f, t, m = O.entities(labs, lbuf, depth, tr, 24)
    assert m.sum() == 24 and (t[m] == O.T["monster"]).all()
    assert f[:, 4].max() <= (10 + 3 * 23) / 255 + 1e-6  # nearest 24 kept
    assert (f[m, 7] == 1.0).all()  # all new
    f2, _, m2 = O.entities(labs, lbuf, depth, tr, 24)
    assert (f2[m2, 7] == 0.0).all() and np.allclose(f2[m2, 5:7], 0.0)


# ---------------------------------------------------------------- ADR 0007 ruling 6
def _tmp_wad(tmp_path, rel="harness/maps.wad"):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"PWAD\x00\x00\x00\x00\x0c\x00\x00\x00")
    return str(p)


def test_eval_with_harness_map_wad_passes(tmp_path):
    EVAL_CONFIG.replace(map_wad=_tmp_wad(tmp_path), maps=("MAP01", "MAP02")).assert_vanilla()
    EVAL_CONFIG.replace(iwad_file=_tmp_wad(tmp_path, "iwad/doom2.wad")).assert_vanilla()


def test_eval_map_wad_must_exist():
    with pytest.raises(AssertionError):
        EVAL_CONFIG.replace(map_wad="no/such/file.wad").assert_vanilla()
    with pytest.raises(AssertionError):
        EVAL_CONFIG.replace(iwad_file="no/such/iwad.wad").assert_vanilla()


def test_eval_with_scenario_wad_still_fails(tmp_path):
    with pytest.raises(AssertionError):
        EVAL_CONFIG.replace(scenario_wad=_tmp_wad(tmp_path)).assert_vanilla()


def test_map_wad_and_scenario_wad_together_rejected(tmp_path):
    w = _tmp_wad(tmp_path)
    with pytest.raises(ValueError):
        TRAIN_CONFIG.replace(map_wad=w).validate()  # TRAIN already has a scenario_wad


def test_eval_game_args_only_exact_viz_nocheat():
    EVAL_CONFIG.replace(game_args=("+viz_nocheat 1",)).assert_vanilla()
    for bad in [("+viz_nocheat 1", "+sv_cheats 1"), ("+viz_nocheat  1",), ("+viz_nocheat 0",),
                ("+viz_nocheat",), ("+sv_cheats 1",), ("+viz_nocheat 1 +sv_cheats 1",),
                ("+sv_cheats 1", "+viz_nocheat 1"), ("-deathmatch",)]:
        with pytest.raises(AssertionError):
            EVAL_CONFIG.replace(game_args=bad).assert_vanilla()


def test_training_folder_path_as_map_wad_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "REPO_ROOT", str(tmp_path))
    for d in config_module.TRAINING_MAP_DIRS:
        w = _tmp_wad(tmp_path, f"{d}/gen/seed1.wad")
        with pytest.raises(AssertionError):
            EVAL_CONFIG.replace(map_wad=w).assert_vanilla()
    EVAL_CONFIG.replace(map_wad=_tmp_wad(tmp_path, "judges/maps.wad")).assert_vanilla()


# ---------------------------------------------------------------- ADR 0007 ruling 7 / ADR 0005
def test_train_is_bundled_my_way_home_and_no_freedoom2_map():
    assert os.path.basename(TRAIN_CONFIG.scenario_wad) == "my_way_home.wad"
    assert TRAIN_CONFIG.pwad_path() == TRAIN_CONFIG.scenario_wad  # maps come from the PWAD
    # reward = exactly what the bundled cfg sets: no map_exit_reward added on top
    assert TRAIN_CONFIG.engine_rewards == (("living_reward", -0.0001),)
    assert (TRAIN_CONFIG.skill, TRAIN_CONFIG.episode_timeout_tics) == (5, 2100)
    # EVAL loads no PWAD at all, so no TRAIN map can reach it
    assert EVAL_CONFIG.pwad_path() is None
    assert_same_sensors(TRAIN_CONFIG, EVAL_CONFIG)


# ---------------------------------------------------------------- ADR 0008: held-out split, training side
@pytest.mark.parametrize("sub", ["generated", "external"])
def test_train_config_refuses_heldout_maps_paths(tmp_path, monkeypatch, sub):
    monkeypatch.setattr(config_module, "REPO_ROOT", str(tmp_path))
    w = _tmp_wad(tmp_path, f"heldout_maps/{sub}/m.wad")
    for field in ("scenario_wad", "iwad_file"):
        with pytest.raises(ValueError):
            TRAIN_CONFIG.replace(**{field: w}).validate()
    with pytest.raises(ValueError):
        TRAIN_CONFIG.replace(scenario_wad=None, map_wad=w).validate()
    # the engine-test opt-in does not lift the held-out refusal
    with pytest.raises(ValueError):
        TRAIN_CONFIG.replace(scenario_wad=w, phase2_engine_test=True).validate()


def test_eval_may_load_from_heldout_maps(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "REPO_ROOT", str(tmp_path))
    EVAL_CONFIG.replace(map_wad=_tmp_wad(tmp_path, "heldout_maps/generated/m.wad")).validate()
    EVAL_CONFIG.replace(map_wad=_tmp_wad(tmp_path, "heldout_maps/external/m.wad")).validate()


def test_train_config_refuses_phase2_campaign_maps(tmp_path):
    for maps in (("MAP01",), ("MAP02", "MAP03")):
        with pytest.raises(ValueError):
            TRAIN_CONFIG.replace(scenario_wad=None, maps=maps).validate()
    fake_fd2 = _tmp_wad(tmp_path, "elsewhere/freedoom2.wad")
    with pytest.raises(ValueError):
        TRAIN_CONFIG.replace(scenario_wad=None, iwad_file=fake_fd2).validate()
    # allowed: a scenario PWAD on the freedoom2 IWAD (my_way_home), Phase 1 maps,
    # and the explicit engine-test opt-in
    TRAIN_CONFIG.validate()
    TRAIN_CONFIG.replace(scenario_wad=None, iwad="freedoom1", maps=("E1M1",)).validate()
    TRAIN_CONFIG.replace(scenario_wad=None, phase2_engine_test=True).validate()


def test_refusal_happens_before_engine_starts(monkeypatch):
    started = []
    monkeypatch.setattr(doom_env.DoomEnv, "_make_game", lambda self: started.append(1))
    with pytest.raises(ValueError):
        doom_env.DoomEnv(TRAIN_CONFIG.replace(scenario_wad=None))
    assert started == []
