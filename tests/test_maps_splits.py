"""Splits: held-out assert in the loader, disjointness by id and hash, hash
verification, agreement with env.config's own held-out refusal."""
import copy
import json
import os
import shutil

import pytest

from maps import loader as L

pytestmark = pytest.mark.skipif(not os.path.isfile(L.SPLITS_PATH), reason="run python -m maps.build")


@pytest.fixture(scope="module")
def sp():
    return L.load_splits()


def test_counts_and_stage_distribution(sp):
    for split in L.SPLITS:
        stages = {e["stage"] for _, e in sp.entries(split)}
        assert stages == {"S0", "S1", "S2", "S3"}
        assert all(e["mods"] == [] for _, e in sp.entries(split))


def test_disjoint_by_id_and_hash(sp):
    ids, hashes = [], []
    for _, e in sp.entries():
        ids.append(e["id"])
        hashes.append(e["sha256"])
    assert len(ids) == len(set(ids)) and len(hashes) == len(set(hashes))
    seeds = {s: {e["seed"] for _, e in sp.entries(s)} for s in L.SPLITS}
    assert not seeds["train"] & seeds["heldout"] and not seeds["train"] & seeds["val"]


def test_heldout_files_live_under_heldout_dir(sp):
    for s, e in sp.entries():
        assert L.under_heldout_dir(e["file"]) == (s == "heldout")


def test_loader_assert_fires_for_heldout_in_training(sp):
    for _, e in sp.entries("heldout"):
        with pytest.raises(L.HeldOutAccessError, match="HELD-OUT"):
            L.resolve(e["id"], "train", sp)
        assert os.path.isfile(L.resolve(e["id"], "eval", sp))
    with pytest.raises(L.HeldOutAccessError):
        L.resolve(next(e for _, e in sp.entries("heldout"))["id"], "val", sp)


def test_val_and_freedoom_refused_in_training(sp):
    v = next(e for _, e in sp.entries("val"))
    with pytest.raises(PermissionError):
        L.resolve(v["id"], "train", sp)
    assert os.path.isfile(L.resolve(v["id"], "val", sp))
    for e in sp.data["external_eval"]:
        assert e["split"] in ("val", "heldout")
        with pytest.raises(L.HeldOutAccessError):
            L.resolve(e["id"], "train", sp)


def test_heldout_hash_under_another_name_refused(sp):
    """A train entry whose file hash equals a held-out map's is refused in training."""
    data = copy.deepcopy(sp.data)
    h = next(e for _, e in sp.entries("heldout"))
    t = data["splits"]["train"][0]
    t["sha256"] = h["sha256"]
    with pytest.raises(L.SplitIntegrityError):
        L.check_integrity(data)          # duplicate hash across splits
    bad = L.Splits(data)
    with pytest.raises(L.HeldOutAccessError):
        L.resolve(t["id"], "train", bad)


def test_duplicate_id_rejected(sp):
    data = copy.deepcopy(sp.data)
    dup = copy.deepcopy(data["splits"]["train"][0])
    dup["sha256"] = "0" * 64
    dup["file"] = data["splits"]["val"][0]["file"]
    data["splits"]["val"].append(dup)
    with pytest.raises(L.SplitIntegrityError):
        L.check_integrity(data)


def test_hash_verification_catches_changed_file(sp, tmp_path):
    e = copy.deepcopy(next(e for _, e in sp.entries("train")))
    p = tmp_path / "copy.wad"
    shutil.copy(L._abs(e["file"]), p)
    e["file"] = str(p)
    data = copy.deepcopy(sp.data)
    data["splits"]["train"][0] = e
    s2 = L.Splits(data)
    assert L.resolve(e["id"], "train", s2) == str(p)
    with open(p, "r+b") as f:
        f.seek(40)
        b = f.read(1)
        f.seek(40)
        f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(L.SplitIntegrityError, match="sha256"):
        L.resolve(e["id"], "train", s2)


def test_every_listed_file_matches_its_hash(sp):
    for _, e in sp.entries():
        assert L.sha256_file(L._abs(e["file"])) == e["sha256"], e["id"]


def test_loader_and_env_config_agree(sp):
    from env.config import HELDOUT_MAP_DIR, TRAIN_CONFIG, VANILLA_ENGINE_REWARDS
    assert HELDOUT_MAP_DIR == L.HELDOUT_DIR
    base = TRAIN_CONFIG.replace(scenario_wad=None, engine_rewards=VANILLA_ENGINE_REWARDS)
    for split, e in sp.entries():
        cfg = base.replace(map_wad=L._abs(e["file"]))
        if split == "heldout":
            with pytest.raises(ValueError):
                cfg.validate()
            with pytest.raises(L.HeldOutAccessError):
                L.resolve(e["id"], "train", sp)
        elif split == "train":
            cfg.validate()
            L.resolve(e["id"], "train", sp)


def test_training_bundle_is_train_only(sp):
    path, lumps, stages = L.training_bundle("train-all", sp)
    batch1_ids = {e["id"] for _, e in sp.entries("train") if e.get("batch", 1) == 1}
    assert set(lumps.values()) == batch1_ids      # train-all: batch 1 only, unchanged
    if any(b["id"] == "train-all-b2" for b in sp.data["bundles"]):
        _, lumps2, _ = L.training_bundle("train-all-b2", sp)
        assert set(lumps2.values()) == {e["id"] for _, e in sp.entries("train")}
    assert set(stages.values()) == {"S0", "S1", "S2", "S3"}
    assert not L.under_heldout_dir(path)


def test_bundle_with_heldout_member_rejected(sp):
    data = copy.deepcopy(sp.data)
    data["bundles"][0]["members"][0] = next(e for _, e in sp.entries("heldout"))["id"]
    with pytest.raises(L.SplitIntegrityError):
        L.check_integrity(data)


def test_manifest_matches_splits(sp):
    man = json.load(open(os.path.join(L.REPO_ROOT, L.HELDOUT_DIR, "MANIFEST.json")))
    assert {(f["file"], f["sha256"]) for f in man["files"]} == \
        {(e["file"], e["sha256"]) for _, e in sp.entries("heldout")}


def test_every_map_regenerates_byte_identically_from_its_record(sp, tmp_path):
    """Held-out WADs are git-ignored on the promise that (split, stage, seed) rebuilds
    them exactly; check that for every entry of every split."""
    from maps import generator as G
    from maps.wad import write_wad
    cfgs = {}
    for split, e in sp.entries():
        assert e["generator_version"] == G.GEN_VERSION
        cf = e.get("config_file", "maps/gen_config.json")   # batch 2: maps/gen_config_b2.json
        if cf not in cfgs:
            cfgs[cf] = json.load(open(os.path.join(L.REPO_ROOT, cf), encoding="utf-8"))
            assert L.sha256_file(os.path.join(L.REPO_ROOT, cf)) == e["config_sha256"], cf
        lumps, rec = G.generate(cfgs[cf], split, e["stage"], e["seed"])
        assert rec["attempt"] == e["attempt"]
        p = tmp_path / "r.wad"
        write_wad(str(p), lumps)
        assert L.sha256_file(str(p)) == e["sha256"], e["id"]
