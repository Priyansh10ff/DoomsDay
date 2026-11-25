"""Generator: determinism, vanilla resources, per-stage features, static QA."""
import json
import os
import subprocess
import sys

import pytest

from maps import generator as G
from maps import themes as T
from maps.qa.validate import check_record, structure, wad_search
from maps.wad import iwad_resources, unpack_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, "maps", "gen_config.json"), encoding="utf-8"))


def _bytes(lumps):
    import io
    from maps.wad import write_wad
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.wad")
        write_wad(p, lumps)
        return open(p, "rb").read()


@pytest.fixture(scope="module")
def resources():
    from env.config import iwad_path
    return iwad_resources(iwad_path("freedoom2"))


def test_theme_names_exist_in_freedoom2(resources):
    assert not T.all_texture_names() - resources["textures"]
    assert not T.all_flat_names() - resources["flats"]


def test_split_themes_are_disjoint():
    sets = [set(v) for v in T.THEMES_BY_SPLIT.values()]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not sets[i] & sets[j]
    for split, th in T.THEMES_BY_SPLIT.items():
        assert tuple(CFG["splits"][split]["themes"]) == th


def test_seed_ranges_are_disjoint():
    r = sorted(tuple(CFG["splits"][s]["seed_range"]) for s in ("train", "val", "heldout"))
    for (a0, a1), (b0, b1) in zip(r, r[1:]):
        assert a1 < b0


SCRIPT = ("import json,sys,hashlib;sys.path.insert(0,{root!r});from maps.generator import generate;"
          "cfg=json.load(open({cfg!r}));from maps.wad import write_wad;import tempfile,os;"
          "d=tempfile.mkdtemp();out=[]\n"
          "for st in ('S0','S1','S2','S3'):\n"
          " l,r=generate(cfg,'train',st,123456);p=os.path.join(d,st+'.wad');write_wad(p,l);"
          "out.append(hashlib.sha256(open(p,'rb').read()).hexdigest())\n"
          "print(','.join(out))")


def test_same_seed_byte_identical_across_processes():
    code = SCRIPT.format(root=ROOT, cfg=os.path.join(ROOT, "maps", "gen_config.json"))
    outs = []
    for hs in ("0", "12345"):  # different hash seeds: set/dict-order bugs would show
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                   env=env, check=True).stdout.strip())
    assert outs[0] == outs[1] and len(outs[0].split(",")) == 4


def test_different_seeds_differ():
    a = _bytes(G.generate(CFG, "train", "S1", 100001)[0])
    b = _bytes(G.generate(CFG, "train", "S1", 100002)[0])
    assert a != b


@pytest.mark.parametrize("stage", ["S0", "S1", "S2", "S3"])
def test_stage_features_and_static_qa(stage, resources):
    for seed in range(200100, 200106):
        lumps, rec = G.generate(CFG, "train", stage, seed)
        m = unpack_map(lumps, "MAP01")
        assert check_record(m, rec, resources) == []
        specials = {ld.special for ld in m["linedefs"]}
        types = [t.type for t in m["things"]]
        assert types.count(1) == 1
        assert specials & {11, 52}
        assert rec["mods"] == []
        assert "BEHAVIOR" not in m["lump_names"]
        if stage == "S0":
            assert not specials & {1, 26, 27, 28}
            assert types == [1]
        if stage in ("S1", "S2", "S3"):
            assert 1 in specials
        if stage in ("S2", "S3"):
            assert specials & {26, 27, 28}
            assert set(types) & {5, 6, 13, 38, 39, 40}
            assert not wad_search(m, use_keys=False)["exit_reachable"]
        if stage == "S3":
            assert set(types) & set(T.MONSTERS.values())
        else:
            assert not set(types) & set(T.MONSTERS.values())


def test_qa_catches_a_bypassed_lock(resources):
    """Turning the locked door into a normal door must fail the keyed-stage checks."""
    lumps, rec = G.generate(CFG, "train", "S2", 200200)
    m = unpack_map(lumps, "MAP01")
    for ld in m["linedefs"]:
        if ld.special in (26, 27, 28):
            ld.special = 1
    probs = check_record(m, rec, resources)
    assert any("WITHOUT the key" in p for p in probs)


def test_qa_catches_open_sector():
    lumps, rec = G.generate(CFG, "train", "S0", 200300)
    m = unpack_map(lumps, "MAP01")
    del m["linedefs"][0]
    assert any("open sector" in p for p in structure(m))
