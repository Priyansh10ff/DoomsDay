"""evaluate --wad/--iwad (external maps) and evaluate.record: argument handling,
held-out refusal, lump discovery, obs-version pinning, recorder proxy and video
writers. No engine is started in this file."""
from __future__ import annotations

import glob
import os
import shutil
import struct

import numpy as np
import pytest

import a0_helpers  # noqa: F401  (caps torch threads)
from configs.a0_env import OBS_VERSION_ENV_VAR, a0_eval_env_config
from env.config import iwad_path
from evaluate import record as R
from evaluate import run as RUN
from evaluate.external import (EXTERNAL_HELDOUT_STAGE, EXTERNAL_SPLIT, EXTERNAL_STAGE, ExternalEvalRunConfig,
                               ExternalMapError, ExternalTarget, discover_lumps, plan_external, resolve_iwad,
                               select_lumps)
from evaluate.obs_select import checkpoint_obs_version, pin_obs_version
from evaluate.sensors import canonical_json, check_sensors, trained_config_from_signature
from evaluate.video import VideoConfig, upscale, write_video
from maps import loader as L

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_WAD = os.path.join(REPO, "maps", "val", "S0-val-500001.wad")
HELDOUT_WADS = sorted(glob.glob(os.path.join(REPO, "heldout_maps", "generated", "*.wad")))
V1_SIG = canonical_json(a0_eval_env_config().replace(obs_version=1).sensor_signature())


def make_wad(path: str, lumps: list[str], magic: bytes = b"PWAD") -> str:
    """Directory-only WAD: every lump is empty; only the names matter."""
    data_off = 12
    dir_entries = b"".join(struct.pack("<ii8s", data_off, 0, n.encode("ascii")) for n in lumps)
    with open(path, "wb") as f:
        f.write(magic + struct.pack("<ii", len(lumps), data_off) + dir_entries)
    return path


@pytest.fixture
def val_copy(tmp_path) -> str:
    if not os.path.isfile(VAL_WAD):
        pytest.skip("val WADs not built (python -m maps.build)")
    return shutil.copyfile(VAL_WAD, str(tmp_path / "judge_map.wad"))


# ---------------------------------------------------------------- argument handling
def _args(*argv: str):
    return RUN.parse_args(["--checkpoint", "x.pt", *argv])


def test_split_or_wad_required() -> None:
    with pytest.raises(SystemExit):
        _args()


@pytest.mark.parametrize("split", ["val", "my_way_home"])
def test_split_and_wad_exclusive(split: str) -> None:
    with pytest.raises(SystemExit):
        _args("--split", split, "--wad", "a.wad")
    with pytest.raises(SystemExit):
        _args("--split", split, "--iwad", "freedoom2")


@pytest.mark.parametrize("flag", [["--map", "MAP01"], ["--skill", "4"], ["--timeout-tics", "100"]])
def test_external_flags_need_wad(flag: list[str]) -> None:
    with pytest.raises(SystemExit):
        _args("--split", "val", *flag)


def test_external_args_parse() -> None:
    a = _args("--wad", "a.wad", "--map", "MAP01", "--map", "map02", "--skill", "4", "--timeout-tics", "99")
    t = RUN.external_target(a)
    assert t.wad == "a.wad" and t.maps == ("MAP01", "map02") and t.skill == 4 and t.timeout_tics == 99
    assert not t.heldout_ack
    t2 = RUN.external_target(_args("--split", "heldout", "--iwad", "freedoom2", "--map", "MAP05"))
    assert t2.heldout_ack and t2.wad is None
    d = RUN.external_target(_args("--wad", "a.wad"))
    assert d.maps == () and d.skill == 3 and d.timeout_tics == 21_000


def test_external_run_config() -> None:
    r = ExternalEvalRunConfig(episodes_per_map=2)
    assert r.split == EXTERNAL_SPLIT and r.episode_seed(1, 1) == r.base_seed + r.seed_stride_map + 1
    with pytest.raises(ValueError):
        ExternalEvalRunConfig(episodes_per_map=0)
    with pytest.raises(ValueError):
        ExternalEvalRunConfig(policy="bogus")
    with pytest.raises(ValueError):
        ExternalTarget(wad=None, iwad=None)
    with pytest.raises(ValueError):
        ExternalTarget(wad="a.wad", iwad=None, skill=6)


# ---------------------------------------------------------------- lump discovery
def test_discover_lumps_sorted_doom_and_udmf(tmp_path) -> None:
    p = make_wad(str(tmp_path / "m.wad"), ["MAP02", "THINGS", "LINEDEFS", "E1M1", "THINGS", "PLAYPAL",
                                           "MAP01", "TEXTMAP", "ENDMAP"])
    assert discover_lumps(p) == ("E1M1", "MAP01", "MAP02")


def test_select_lumps(tmp_path) -> None:
    p = make_wad(str(tmp_path / "m.wad"), ["MAP03", "THINGS", "MAP01", "THINGS"])
    assert select_lumps(p, ()) == ("MAP01", "MAP03")
    assert select_lumps(p, ("map03", "MAP03")) == ("MAP03",)
    with pytest.raises(ExternalMapError, match="MAP05"):
        select_lumps(p, ("MAP05",))
    empty = make_wad(str(tmp_path / "e.wad"), ["PLAYPAL", "COLORMAP"])
    with pytest.raises(ExternalMapError, match="no map lumps"):
        select_lumps(empty, ())
    bad = tmp_path / "bad.wad"
    bad.write_bytes(b"NOTAWAD-----")
    with pytest.raises(ExternalMapError):
        discover_lumps(str(bad))


def test_resolve_iwad(tmp_path) -> None:
    assert resolve_iwad(None) == (a0_eval_env_config().iwad, None)
    assert resolve_iwad("freedoom1") == ("freedoom1", None)
    assert resolve_iwad("freedoom2.wad")[0] == "freedoom2"
    custom = make_wad(str(tmp_path / "doom2.wad"), ["MAP01", "THINGS"], magic=b"IWAD")
    assert resolve_iwad(custom) == (a0_eval_env_config().iwad, os.path.abspath(custom))
    with pytest.raises(ExternalMapError, match="not an IWAD"):
        resolve_iwad(make_wad(str(tmp_path / "p.wad"), ["MAP01", "THINGS"]))
    with pytest.raises(ExternalMapError):
        resolve_iwad(str(tmp_path / "missing.wad"))


# ---------------------------------------------------------------- held-out refusal
@pytest.mark.skipif(not HELDOUT_WADS, reason="held-out WADs not built")
def test_heldout_file_refused_by_path_and_by_hash(tmp_path) -> None:
    src = HELDOUT_WADS[0]
    with pytest.raises(L.HeldOutAccessError, match="--split heldout"):
        plan_external(ExternalTarget(wad=src, iwad=None))
    renamed = shutil.copyfile(src, str(tmp_path / "innocent.wad"))
    with pytest.raises(L.HeldOutAccessError):
        plan_external(ExternalTarget(wad=renamed, iwad=None))
    plan = plan_external(ExternalTarget(wad=renamed, iwad=None, heldout_ack=True))
    assert plan.jobs and all(j.stage == EXTERNAL_HELDOUT_STAGE for j in plan.jobs)


def test_freedoom2_iwad_only_needs_ack() -> None:
    with pytest.raises(L.HeldOutAccessError):
        plan_external(ExternalTarget(wad=None, iwad="freedoom2", maps=("MAP05",)))
    plan = plan_external(ExternalTarget(wad=None, iwad="freedoom2", maps=("MAP05",), heldout_ack=True))
    (job,) = plan.jobs
    c = job.env_config
    assert c.map_wad is None and c.maps == ("MAP05",) and c.game_path() == iwad_path("freedoom2")
    assert job.stage == EXTERNAL_HELDOUT_STAGE and job.split == EXTERNAL_SPLIT
    c.assert_vanilla()


def test_loader_guard_unchanged() -> None:
    """The training-mode loader still refuses every held-out map (not weakened)."""
    splits = L.load_splits()
    for _, e in splits.entries("heldout"):
        with pytest.raises(L.HeldOutAccessError):
            L.resolve(e["id"], "train", splits)


# ---------------------------------------------------------------- plan on an unseen WAD
def test_plan_external_val_copy(val_copy, monkeypatch) -> None:
    monkeypatch.setenv(OBS_VERSION_ENV_VAR, "1")
    plan = plan_external(ExternalTarget(wad=val_copy, iwad=None, timeout_tics=140, skill=2))
    (job,) = plan.jobs
    c = job.env_config
    assert job.split == EXTERNAL_SPLIT and job.stage == EXTERNAL_STAGE
    assert c.map_wad == os.path.abspath(val_copy) and c.maps == ("MAP01",)
    assert c.skill == 2 and c.episode_timeout_tics == 140 and c.vanilla_eval
    assert any("val-split" in n for n in job.notes)           # hash recognised, noted
    check_sensors(trained_config_from_signature(V1_SIG, a0_eval_env_config()), c)
    with pytest.raises(ExternalMapError):
        plan_external(ExternalTarget(wad=val_copy, iwad=None, maps=("MAP02",)))


def test_wad_under_maps_refused_by_vanilla_check(monkeypatch) -> None:
    if not os.path.isfile(VAL_WAD):
        pytest.skip("val WADs not built")
    monkeypatch.setenv(OBS_VERSION_ENV_VAR, "1")
    plan = plan_external(ExternalTarget(wad=VAL_WAD, iwad=None))
    with pytest.raises(AssertionError, match="training-map folder"):
        plan.jobs[0].env_config.assert_vanilla()


# ---------------------------------------------------------------- obs-version pinning
def test_pin_obs_version(monkeypatch) -> None:
    assert checkpoint_obs_version(V1_SIG) == 1 and checkpoint_obs_version(None) is None
    monkeypatch.setenv(OBS_VERSION_ENV_VAR, "")
    assert pin_obs_version(V1_SIG) == 1 and os.environ[OBS_VERSION_ENV_VAR] == "1"
    assert a0_eval_env_config().obs_version == 1
    monkeypatch.setenv(OBS_VERSION_ENV_VAR, "2")
    with pytest.raises(ValueError, match="observation version 1"):
        pin_obs_version(V1_SIG)


# ---------------------------------------------------------------- recorder
def _rec_args(*argv: str):
    return R.parse_args(["--checkpoint", "x.pt", "--out", "o", *argv])


@pytest.mark.parametrize("argv", [[], ["--split", "val"], ["--split", "val", "--map-id", "a", "--wad", "w"],
                                  ["--wad", "w", "--map-id", "a"], ["--split", "val", "--map-id", "a",
                                                                    "--map", "MAP01"],
                                  ["--wad", "w", "--episodes", "0"], ["--split", "heldout", "--map-id", "a"]])
def test_record_args_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _rec_args(*argv)


def test_record_args_ok() -> None:
    a = _rec_args("--split", "val", "--map-id", "S0-val-500001", "--episodes", "2", "--seed", "7")
    assert a.episodes == 2 and a.seed == 7 and a.policy == "sample"
    b = _rec_args("--wad", "w.wad", "--map", "MAP01", "--argmax")
    assert b.maps == ["MAP01"] and b.policy == "argmax"


def test_record_val_plan_refuses_non_val() -> None:
    splits = L.load_splits()
    for s in ("train", "heldout"):
        mid = next(e["id"] for _, e in splits.entries(s))
        with pytest.raises(L.HeldOutAccessError):
            R.val_plan(mid, None, None)


class _FakeEnv:
    render_mode = "rgb_array"

    def __init__(self, n_steps: int) -> None:
        self.n, self.t = n_steps, 0

    def _obs(self) -> dict:
        return {"rays": np.full((3, 80), self.t, np.float32)}

    def reset(self, **kw):
        self.t = 0
        return self._obs(), {"seed": kw.get("seed")}

    def step(self, action):
        self.t += 1
        done = self.t >= self.n
        return self._obs(), 0.0, done, False, {}

    def render(self):
        return None if self.t >= self.n else np.full((120, 160, 3), self.t, np.uint8)


def test_recording_env_is_pass_through() -> None:
    fake = _FakeEnv(3)
    rec = R.RecordingEnv(fake)  # type: ignore[arg-type]
    obs, info = rec.reset(seed=5)
    assert info == {"seed": 5} and obs["rays"][0, 0] == 0
    outs = [rec.step(np.zeros(6, np.int64)) for _ in range(3)]
    assert [o[0]["rays"][0, 0] for o in outs] == [1, 2, 3] and outs[-1][2] is True
    # reset frame + 2 live steps; the terminal step has no screen
    assert [int(f[0, 0, 0]) for f in rec.frames] == [0, 1, 2]
    rec.reset(seed=6)
    assert len(rec.frames) == 1


def test_video_writers(tmp_path) -> None:
    frames = [np.full((120, 160, 3), i * 20, np.uint8) for i in range(6)]
    assert upscale(frames[0], 3).shape == (360, 480, 3)
    out = write_video(frames, str(tmp_path / "ep"), 8.75, VideoConfig())
    assert out, "no writer produced output"
    for p in out.values():
        assert os.path.exists(p)
    only_png = write_video(frames, str(tmp_path / "p"), 8.75, VideoConfig(write_mp4=False, write_gif=False))
    assert "png_frames" in only_png and len(os.listdir(only_png["png_frames"])) == 6


def test_evaluate_never_imports_teachers() -> None:
    for p in glob.glob(os.path.join(REPO, "evaluate", "*.py")):
        with open(p, encoding="utf-8") as f:
            src = f.read()
        assert "import teachers" not in src and "from teachers" not in src, p
