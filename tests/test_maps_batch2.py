"""Batch-2 train maps and the `train-all-b2` bundle: the old bundle is unchanged, the
new one holds every train map, lump names are valid, seeds are disjoint, the bundle is
chosen by $DOOMSDAY_TRAIN_BUNDLE (unset = train-all), and the engine loads the right
map at the start, middle and end of the large PWAD."""
import json
import math
import os

import pytest

from maps import loader as L

B2 = "train-all-b2"


def _splits():
    return L.load_splits()


def _has_b2(sp):
    return any(b["id"] == B2 for b in sp.data["bundles"])


pytestmark = pytest.mark.skipif(not os.path.isfile(L.SPLITS_PATH) or not _has_b2(_splits()),
                                reason="batch 2 not built (python -m maps.build_batch2 finalize)")


@pytest.fixture(scope="module")
def sp():
    return _splits()


def _bundle(sp, bid):
    return next(b for b in sp.data["bundles"] if b["id"] == bid)


def test_old_bundle_is_batch1_only_and_unchanged(sp):
    old = _bundle(sp, "train-all")
    b1 = [e["id"] for _, e in sp.entries("train") if e.get("batch", 1) == 1]
    assert old["members"] == b1 and len(b1) == 80
    assert old["lumps"] == [f"DD{i:04d}" for i in range(80)]
    assert L.sha256_file(L._abs(old["file"])) == old["sha256"]


def test_b2_bundle_holds_every_train_map_with_batch1_prefix(sp):
    old, new = _bundle(sp, "train-all"), _bundle(sp, B2)
    assert new["members"][:80] == old["members"] and new["lumps"][:80] == old["lumps"]
    assert set(new["members"]) == {e["id"] for _, e in sp.entries("train")}
    assert len(new["members"]) == len(set(new["members"]))
    path, lumps, stages = L.training_bundle(B2, sp)
    assert os.path.normcase(path).endswith(os.path.normcase(os.path.join("bundles", B2 + ".wad")))
    assert set(stages.values()) >= {"S0", "S1", "S2", "S3"}


def test_b2_lump_names_are_valid_and_match_the_wad_directory(sp):
    from env.config import wad_map_names
    new = _bundle(sp, B2)
    assert new["lumps"] == [f"DD{i:04d}" for i in range(len(new["lumps"]))]
    assert all(len(n) <= 8 and n.isascii() and n.isalnum() and n == n.upper() for n in new["lumps"])
    assert wad_map_names(L._abs(new["file"])) == frozenset(new["lumps"])


def test_batch2_seeds_disjoint_from_batch1_val_heldout(sp):
    b2 = [e for _, e in sp.entries("train") if e.get("batch") == 2]
    assert b2 and all(e["config_file"] == "maps/gen_config_b2.json" and e["mods"] == [] for e in b2)
    other = {e["seed"] for s, e in sp.entries() if not (s == "train" and e.get("batch") == 2)}
    assert not {e["seed"] for e in b2} & other
    cfg = json.load(open(L._abs("maps/gen_config_b2.json"), encoding="utf-8"))
    for e in b2:   # each stage in its own seed block
        base = cfg["batch"]["seed_base"][e["stage"]]
        assert base < e["seed"] < base + 10000 and e["id"] == f"{e['stage']}-train-{e['seed']}"


def test_env_var_selects_bundle(sp, monkeypatch):
    from curriculum.scheduler import CurriculumScheduler, load_config
    monkeypatch.delenv(L.BUNDLE_ENV, raising=False)
    path, lumps, _ = L.training_bundle(splits=sp)
    assert len(lumps) == 80 and path.endswith("train-all.wad")
    sch = CurriculumScheduler.from_bundle(load_config(), splits=sp)
    assert sch.bundle_id == "train-all" and sch.bundle_path.endswith("train-all.wad")
    monkeypatch.setenv(L.BUNDLE_ENV, B2)
    path, lumps, _ = L.training_bundle(splits=sp)
    assert len(lumps) == len(_bundle(sp, B2)["members"]) and path.endswith(B2 + ".wad")
    sch = CurriculumScheduler.from_bundle(load_config(), splits=sp)
    assert sch.bundle_id == B2 and sum(len(v) for v in sch.pool.values()) == len(lumps)
    monkeypatch.setenv(L.BUNDLE_ENV, "no-such-bundle")
    with pytest.raises(KeyError, match="no-such-bundle"):
        L.training_bundle(splits=sp)


def test_b2_bundle_never_holds_heldout_or_val(sp):
    new = _bundle(sp, B2)
    hh = sp.heldout_hashes() | {e["sha256"] for _, e in sp.entries("val")}
    for mid in new["members"]:
        split, e = sp.by_id(mid)
        assert split == "train" and e["sha256"] not in hh and not L.under_heldout_dir(e["file"])


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_engine_loads_the_right_map_from_the_large_bundle(sp, where):
    """Through DoomEnv on the generated-maps training config: the lump loads and is the
    member it should be (sector count and spawn point match that member's WAD), so the
    engine did not fall back to another map."""
    from configs.a0_env import a0_train_env_config, generated_maps_env_config
    from env import DoomEnv
    from maps.qa.validate import load_map
    import vizdoom as vzd
    path, lumps, _ = L.training_bundle(B2, sp)
    names = sorted(lumps)
    lump = {"first": names[0], "middle": names[len(names) // 2], "last": names[-1]}[where]
    m = load_map(L.resolve(lumps[lump], "train", sp))
    start = next(t for t in m["things"] if t.type == 1)
    cfg = generated_maps_env_config(a0_train_env_config(), path, tuple(names))
    env = DoomEnv(cfg.replace(win_timer_ms=None))
    try:
        _, info = env.reset(seed=0, options={"map": lump})
        assert info["map"] == lump
        st = env.game.get_state()
        assert len(st.sectors) == len(m["sectors"])
        x = env.game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = env.game.get_game_variable(vzd.GameVariable.POSITION_Y)
        assert math.hypot(x - start.x, y - start.y) < 2.0
    finally:
        env.close()


def test_full_rebuild_refuses_while_batch2_is_listed(tmp_path, monkeypatch):
    """`python -m maps.build` deletes every split; with batch 2 listed it must stop first."""
    import shutil
    from maps import build
    (tmp_path / "maps").mkdir()
    shutil.copy(L.SPLITS_PATH, tmp_path / "maps" / "splits.json")
    before = L.sha256_file(str(tmp_path / "maps" / "splits.json"))
    monkeypatch.setattr(build, "REPO_ROOT", str(tmp_path))
    assert build.main(["--no-engine"]) == 2
    assert L.sha256_file(str(tmp_path / "maps" / "splits.json")) == before
    assert os.listdir(tmp_path / "maps") == ["splits.json"]
