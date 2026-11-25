"""Audit 01 item E1 (a training reset may only load maps of its own PWAD) and the
per-instance engine config file. Probes: `pwad_map_fallback`, `ini_path`
(docs/reference/engine-probes.md R4.1, R4.3). Engine tests open one DoomGame at a
time and close it."""
import contextlib
import os

import pytest

from env import EVAL_CONFIG, TRAIN_CONFIG, DoomEnv
from env.config import REPO_ROOT, iwad_path, wad_map_names
from env_test_helpers import FD2_ENGINE_TEST_CONFIG as FD2

BUNDLE = os.path.join(REPO_ROOT, "maps", "bundles", "train-all.wad")
needs_bundle = pytest.mark.skipif(not os.path.isfile(BUNDLE), reason="generated bundle not built")


@contextlib.contextmanager
def make(cfg):
    env = DoomEnv(cfg.replace(win_timer_ms=None))
    try:
        yield env
    finally:
        env.close()


def bundle_config():
    lumps = tuple(sorted(wad_map_names(BUNDLE)))
    return TRAIN_CONFIG.replace(name="bundle-test", scenario_wad=None, map_wad=BUNDLE, maps=lumps,
                                engine_rewards=EVAL_CONFIG.engine_rewards)


# ---------------------------------------------------------------- WAD directory
def test_wad_map_names_reads_markers_only():
    assert wad_map_names(TRAIN_CONFIG.scenario_wad) == frozenset({"MAP01"})
    fd2 = wad_map_names(iwad_path("freedoom2"))
    assert {"MAP01", "MAP02", "MAP32"} <= fd2 and "THINGS" not in fd2


@needs_bundle
def test_bundle_has_no_iwad_map_names():
    names = wad_map_names(BUNDLE)
    assert names and all(n.startswith("DD") for n in names)
    assert not names & wad_map_names(iwad_path("freedoom2"))


# ---------------------------------------------------------------- config guard
@pytest.mark.parametrize("bad", ["MAP02", "MAP32", "E1M1"])
def test_training_config_refuses_map_absent_from_pwad(bad):
    with pytest.raises(ValueError, match="not in"):
        TRAIN_CONFIG.replace(maps=(bad,)).validate()
    with pytest.raises(ValueError):
        TRAIN_CONFIG.check_reset_map(bad)


def test_training_config_accepts_its_own_map_any_case():
    TRAIN_CONFIG.validate()
    TRAIN_CONFIG.check_reset_map("MAP01")
    TRAIN_CONFIG.check_reset_map("map01")


@pytest.mark.parametrize("bad", [None, "", 2, "MAP02"])
def test_check_reset_map_rejects_non_names(bad):
    with pytest.raises(ValueError):
        TRAIN_CONFIG.check_reset_map(bad)


@needs_bundle
def test_bundle_config_refuses_iwad_map_names():
    cfg = bundle_config()
    cfg.validate()
    cfg.check_reset_map(sorted(cfg.maps)[0])
    for bad in ("MAP01", "MAP02"):
        with pytest.raises(ValueError):
            cfg.check_reset_map(bad)
        with pytest.raises(ValueError):
            cfg.replace(maps=cfg.maps + (bad,)).validate()


def test_no_pwad_training_config_limited_to_its_maps_tuple():
    FD2.check_reset_map("MAP01")
    with pytest.raises(ValueError):
        FD2.check_reset_map("MAP02")


def test_eval_may_load_any_map():
    assert EVAL_CONFIG.allowed_maps() is None
    for m in ("MAP02", "MAP32", "anything"):
        EVAL_CONFIG.check_reset_map(m)
    EVAL_CONFIG.replace(maps=("MAP02", "MAP30")).validate()


# ---------------------------------------------------------------- engine
def test_training_reset_refuses_iwad_map_and_keeps_reset_contract():
    with make(TRAIN_CONFIG) as env:
        _, info = env.reset(seed=1, options={"map": "MAP01"})
        assert info["map"] == "MAP01"
        with pytest.raises(ValueError, match="audit E1"):
            env.reset(seed=2, options={"map": "MAP02"})
        _, info = env.reset(seed=3, options={"map": "MAP01"})  # env still usable
        assert info["map"] == "MAP01"


@needs_bundle
def test_bundle_reset_refuses_iwad_map():
    cfg = bundle_config()
    lump = sorted(cfg.maps)[1]
    with make(cfg) as env:
        _, info = env.reset(seed=1, options={"map": lump})
        assert info["map"] == lump  # train/curriculum_vec_env.py checks exactly this
        with pytest.raises(ValueError):
            env.reset(seed=2, options={"map": "MAP02"})


def test_eval_reset_loads_iwad_map_by_option():
    with make(EVAL_CONFIG) as env:
        _, info = env.reset(seed=1, options={"map": "MAP02"})
        assert info["map"] == "MAP02"


def test_engine_config_file_is_per_instance_and_removed():
    # (No mtime check on the cwd _vizdoom.ini: engines started by other processes
    # with older code may still write it.)
    env = DoomEnv(EVAL_CONFIG)
    try:
        d = env._ini_dir
        path = env.game.get_doom_config_path()
        assert d is not None and os.path.isdir(d)
        assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(d)
        assert os.path.abspath(path) != os.path.abspath("_vizdoom.ini")
        env.reset(seed=0)
        env.step([0, 0, 3, 0, 0, 0])
    finally:
        env.close()
    assert not os.path.exists(d)
