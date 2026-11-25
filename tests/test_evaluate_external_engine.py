"""Engine smokes for `evaluate.run --wad` and `evaluate.record`: one engine instance at a
time, short time limits, a val map copied outside maps/ (the judges' situation: a map
file we did not place). Skipped without the BC checkpoint, the val WADs, or enough free
memory (a training run may share the machine)."""
from __future__ import annotations

import json
import os
import shutil

import pytest

import a0_helpers  # noqa: F401  (caps torch threads)
from configs.a0_env import OBS_VERSION_ENV_VAR
from evaluate import record as R
from evaluate import run as RUN

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "reports", "runs", "20261125-bc-dagger-r1", "bc_model.pt")
VAL_ID = "S0-val-500001"
VAL_WAD = os.path.join(REPO, "maps", "val", f"{VAL_ID}.wad")
SMOKE_TIMEOUT_TICS = 140          # 35 decisions at frame skip 4
MIN_FREE_GIB = 2.5                # one engine + this process must leave > 2 GiB free


def _free_gib() -> float:
    try:
        import psutil
    except ImportError:
        return float("inf")
    return psutil.virtual_memory().available / 2**30


@pytest.fixture
def ready(monkeypatch):
    if not (os.path.isfile(CKPT) and os.path.isfile(VAL_WAD)):
        pytest.skip("BC checkpoint or val WADs missing")
    if _free_gib() < MIN_FREE_GIB:
        pytest.skip(f"less than {MIN_FREE_GIB} GiB free")
    monkeypatch.setenv(OBS_VERSION_ENV_VAR, "")   # the checkpoint's version must be picked automatically


def test_run_wad_smoke(ready, tmp_path) -> None:
    wad = shutil.copyfile(VAL_WAD, str(tmp_path / "judge_map.wad"))
    out = str(tmp_path / "eval")
    assert RUN.main(["--checkpoint", CKPT, "--wad", wad, "--episodes-per-map", "1",
                     "--timeout-tics", str(SMOKE_TIMEOUT_TICS), "--out", out]) == 0
    rows = [json.loads(x) for x in open(os.path.join(out, "results.jsonl"), encoding="utf-8")]
    assert len(rows) == 1 and rows[0]["split"] == "external" and rows[0]["map_id"] == "judge_map.wad:MAP01"
    assert rows[0]["length"] >= 1
    s = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    assert "assert_vanilla" in s["verification"] and s["meta"]["maps"][0]["lumps"] == ["MAP01"]
    assert os.environ[OBS_VERSION_ENV_VAR] == "1"   # pinned from the v1 checkpoint


def test_record_smoke(ready, tmp_path) -> None:
    out = str(tmp_path / "rec")
    assert R.main(["--checkpoint", CKPT, "--split", "val", "--map-id", VAL_ID, "--episodes", "2",
                   "--timeout-tics", str(SMOKE_TIMEOUT_TICS), "--out", out]) == 0
    s = json.load(open(os.path.join(out, "record_summary.json"), encoding="utf-8"))
    assert len(s["episodes"]) == 2
    for e in s["episodes"]:
        assert e["frames"] >= 1 and e["frame_size"] == [120, 160, 3] and e["videos"]
        for p in e["videos"].values():
            assert os.path.exists(p)
        assert os.path.isfile(os.path.join(out, f"episode_{e['episode_index']:03d}.json"))
