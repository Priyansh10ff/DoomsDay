"""Engine tests for generated maps. One ViZDoom instance at a time."""
import json
import os

import pytest

from maps import generator as G
from maps.qa.validate import engine_check, load_map
from maps.wad import write_wad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, "maps", "gen_config.json"), encoding="utf-8"))


def test_keyed_map_loads_lock_refuses_and_exit_works(tmp_path):
    from env.config import iwad_path
    lumps, rec = G.generate(CFG, "train", "S2", 200400)
    p = str(tmp_path / "s2.wad")
    write_wad(p, lumps)   # empty SEGS/SSECTORS/NODES/REJECT/BLOCKMAP: engine builds them
    r = engine_check(p, load_map(p), rec, iwad_path("freedoom2"))
    assert r["ok"], r["problems"]
    assert r["locked_without_key"]["ceil_after"] == r["locked_without_key"]["ceil_before"]
    assert "key needed" in r["locked_without_key"]["message"].lower()
    assert r["exit_finished"] and r["exit_reward"] >= 1.0
