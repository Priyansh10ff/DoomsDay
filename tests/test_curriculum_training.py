"""Training on generated maps through the curriculum (train/curriculum_vec_env.py,
--maps generated). No engine is started: fake envs only (plus the real maps/ bundle
and curriculum config, read but never loaded into an engine)."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from a0_helpers import ROOT
from configs.a0_env import GENERATED_MAPS, a0_train_env_config, generated_maps_env_config
from configs.reward_config import FULL_LEVEL_COVERAGE, FULL_LEVEL_REWARD
from configs.train_config import PPOConfig, RunConfig, SafetyConfig
from curriculum.scheduler import CurriculumScheduler
from curriculum_helpers import TINY_CURRICULUM, TINY_POOL, FakeMapEnv, fake_map_env
from test_train_ppo import SMALL
from train import checkpoint as CK
from train import run_doom
from train.curriculum_vec_env import CURRICULUM_INFO_KEY, CurriculumMismatch, CurriculumVecEnv
from train.run_loop import CHECKPOINT_NAME, EPISODES_NAME, METRICS_NAME, run_training
from train.subproc_vec_env import SubprocVecEnv
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SIG = (("fake", True),)
QUIET = dict(log=lambda m: None, install_sigint=False)


def tiny_scheduler() -> CurriculumScheduler:
    return CurriculumScheduler(TINY_CURRICULUM, TINY_POOL)


def run_cfg(n: int, maps: str = "generated", t_len: int = 6) -> RunConfig:
    return replace(RunConfig(), seed=0, num_workers=1, envs_per_worker=n, maps=maps,
                   ppo=PPOConfig(rollout_len=t_len, num_minibatches=2, epochs=1))


def sync_envs(n: int) -> SyncVecEnv:
    return SyncVecEnv([FakeMapEnv for _ in range(n)])


# ---------------------------------------------------------------- vec env plumbing
def test_sync_vec_env_passes_reset_options_at_reset_and_auto_reset() -> None:
    v = sync_envs(2)
    _, infos = v.reset(1, [{"map": "DD0001"}, None])
    assert [i["map"] for i in infos] == ["DD0001", "MAP01"]
    assert v.envs[1].resets == [None]           # None -> reset() called without options
    acts = np.zeros((2, 6), np.int64)
    for _ in range(2):
        v.step(acts, [{"map": "DD0002"}, {"map": "DD0003"}])
    st = v.step(acts, [{"map": "DD0002"}, {"map": "DD0003"}])   # step 3 ends both episodes
    assert (st.terminated | st.truncated).all()
    assert [i["map"] for i in st.final_infos] == ["DD0001", "MAP01"]
    assert [i["map"] for i in st.infos] == ["DD0002", "DD0003"]
    # default call (no options) is unchanged
    st2 = [v.step(acts) for _ in range(3)][-1]
    assert [i["map"] for i in st2.infos] == ["MAP01", "MAP01"]


def test_subproc_vec_env_passes_reset_options_like_sync() -> None:
    v = SubprocVecEnv([[fake_map_env, fake_map_env], [fake_map_env]], SafetyConfig(worker_join_timeout_s=5.0))
    try:
        ref = sync_envs(3)
        opts = [{"map": f"DD000{i}"} for i in range(3)]
        (o1, i1), (o2, i2) = v.reset(4, opts), ref.reset(4, opts)
        assert [x["map"] for x in i1] == [x["map"] for x in i2] == ["DD0000", "DD0001", "DD0002"]
        for k in o1:
            assert np.array_equal(o1[k], o2[k])
        acts = np.zeros((3, 6), np.int64)
        nxt = [{"map": "DD0003"}, None, {"map": "DD0001"}]
        for _ in range(3):
            s1, s2 = v.step(acts, nxt), ref.step(acts, nxt)
        assert [x["map"] for x in s1.infos] == [x["map"] for x in s2.infos] == ["DD0003", "MAP01", "DD0001"]
        assert [x["map"] for x in s1.final_infos] == ["DD0000", "DD0001", "DD0002"]
        assert np.array_equal(s1.terminated, s2.terminated)
    finally:
        v.close()


# ---------------------------------------------------------------- curriculum wrapper
def test_curriculum_drives_maps_records_outcomes_and_promotes() -> None:
    cv = CurriculumVecEnv(sync_envs(2), tiny_scheduler(), recent_window=10)
    _, infos = cv.reset(0)
    assert all(i["map"] in ("DD0000", "DD0001") for i in infos)   # stage S0 maps
    acts = np.zeros((2, 6), np.int64)
    ends: list[dict] = []
    for _ in range(12):
        st = cv.step(acts)
        for i in np.flatnonzero(st.terminated | st.truncated):
            c = st.final_infos[i][CURRICULUM_INFO_KEY]
            assert c["lump"] == st.final_infos[i]["map"]           # played what was chosen
            ends.append(c)
    assert ends[0]["stage"] == "S0" and ends[0]["map_id"].startswith("S0-")
    assert cv.sch.stage == "S1" and len(cv.sch.history) == 1      # S0 maps exit -> promoted
    assert any(e["promoted"] for e in ends)
    assert ends[-1]["stage"] == "S1"
    m = cv.metrics()
    assert m["curriculum_stage_idx"] == 1.0 and m["curriculum_mix_S1"] == 1.0
    assert m["curriculum_recent_success_S0"] == 1.0 and m["curriculum_recent_success_S1"] == 0.0
    assert len(cv.drain_promotions()) == 1 and cv.drain_promotions() == []


def test_reset_replays_held_choices_so_pending_does_not_grow() -> None:
    cv = CurriculumVecEnv(sync_envs(3), tiny_scheduler(), recent_window=10)
    cv.reset(0)
    issued, pending = cv.sch.issued, len(cv.sch.state_dict()["pending"])
    assert issued == pending == 6                               # current + upcoming per env
    cur = list(cv.current)
    for s in (1, 2):
        _, infos = cv.reset(s)
        assert cv.current == cur and [i["map"] for i in infos] == [c.lump for c in cur]
    assert cv.sch.issued == issued and len(cv.sch.state_dict()["pending"]) == pending


def test_env_that_ignores_the_chosen_map_fails_loudly() -> None:
    class Deaf(FakeMapEnv):
        def reset(self, *, seed=None, options=None):
            return super().reset(seed=seed, options=None)
    cv = CurriculumVecEnv(SyncVecEnv([Deaf]), tiny_scheduler(), recent_window=10)
    with pytest.raises(CurriculumMismatch, match="did not reach the env"):
        cv.reset(0)


def _drive(cv: CurriculumVecEnv, steps: int) -> list[tuple]:
    out = []
    acts = np.zeros((cv.num_envs, 6), np.int64)
    for _ in range(steps):
        st = cv.step(acts)
        out.append(tuple(None if f is None else (f[CURRICULUM_INFO_KEY]["choice"], f["map"]) for f in st.final_infos))
    return out


def test_state_round_trip_continues_the_curriculum_exactly() -> None:
    a = CurriculumVecEnv(sync_envs(2), tiny_scheduler(), recent_window=10)
    a.reset(0)
    _drive(a, 5)                                   # mid-curriculum, episodes in flight
    text = json.loads(json.dumps(a.state_json()))  # survives JSON (checkpoint text)
    b = CurriculumVecEnv(sync_envs(2), tiny_scheduler(), recent_window=10)
    b.load_state_json(text)
    a.reset(9), b.reset(9)
    assert b.current == a.current and b.upcoming == a.upcoming and b.dangling_restored == 0
    assert _drive(a, 9) == _drive(b, 9)
    assert a.sch.state_dict() == b.sch.state_dict() and a.metrics() == b.metrics()


def test_state_from_a_different_bundle_is_refused() -> None:
    a = CurriculumVecEnv(sync_envs(1), tiny_scheduler(), recent_window=10, bundle_sha256="aa")
    a.reset(0)
    b = CurriculumVecEnv(sync_envs(1), tiny_scheduler(), recent_window=10, bundle_sha256="bb")
    with pytest.raises(ValueError, match="bundle changed"):
        b.load_state_json(a.state_json())


def test_restore_with_fewer_envs_counts_dangling_choices() -> None:
    a = CurriculumVecEnv(sync_envs(2), tiny_scheduler(), recent_window=10)
    a.reset(0)
    b = CurriculumVecEnv(sync_envs(1), tiny_scheduler(), recent_window=10)
    b.load_state_json(a.state_json())
    b.reset(0)
    assert b.dangling_restored == 2 and b.current[0] == a.current[0]


# ---------------------------------------------------------------- trainer + run loop
def _recording_trainer(curriculum: bool) -> tuple[Trainer, list[dict[str, torch.Tensor]]]:
    # no exit maps: episode outcomes (and so the trainer's calls) do not depend on the map
    venv: Any = SyncVecEnv([lambda: FakeMapEnv(exit_maps=()) for _ in range(2)])
    if curriculum:
        venv = CurriculumVecEnv(venv, tiny_scheduler(), recent_window=10)
    tr = Trainer(venv, SMALL, run_cfg(2, "generated" if curriculum else "my_way_home"))
    seen: list[dict[str, torch.Tensor]] = []
    real = tr.model.step

    def spy(inp, h, starts):
        seen.append({k: v.clone() for k, v in inp.items()})
        return real(inp, h, starts)
    tr.model.step = spy  # type: ignore[method-assign]
    return tr, seen


def test_model_inputs_are_unchanged_by_the_curriculum_and_carry_no_map_id() -> None:
    a, seen_a = _recording_trainer(False)
    b, seen_b = _recording_trainer(True)
    a.train_iteration(), b.train_iteration()
    assert len(seen_a) == len(seen_b) > 0
    for x, y in zip(seen_a, seen_b):
        assert set(x) == set(y)
        assert not any(w in k for k in y for w in ("map", "stage", "curriculum", "lump"))
        for k in x:
            assert torch.equal(x[k], y[k]), k
    assert all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters()))
    recs = b.log.pending
    assert recs and all(r["stage"] in ("S0", "S1") and r["map_id"].startswith(r["stage"]) for r in recs)
    assert all("stage" not in r and "map_id" not in r for r in a.log.pending)   # unchanged without it


def test_run_loop_logs_stages_and_checkpoints_curriculum_state(tmp_path: "os.PathLike[str]") -> None:
    venv = CurriculumVecEnv(sync_envs(2), tiny_scheduler(), recent_window=10)
    tr = Trainer(venv, SMALL, run_cfg(2))
    d = str(tmp_path)
    run_training(tr, d, SIG, SafetyConfig(), max_updates=3, max_seconds=60, curriculum=venv, **QUIET)
    with open(os.path.join(d, METRICS_NAME), encoding="utf-8") as f:
        m = [json.loads(x) for x in f]
    assert "curriculum_mix_S0" in m[0] and "curriculum_recent_success_S0" in m[-1]
    assert any(k.startswith("health") for k in m[-1])   # the ADR 0011 monitor is still fed
    with open(os.path.join(d, EPISODES_NAME), encoding="utf-8") as f:
        eps = [json.loads(x) for x in f]
    assert eps and all({"stage", "map_id", "replay"} <= set(e) for e in eps)
    info = CK.checkpoint_resume_info(os.path.join(d, CHECKPOINT_NAME))
    assert info["maps"] == "generated" and json.loads(info["curriculum_json"]) == json.loads(venv.state_json())


def test_resume_refuses_a_different_map_source(tmp_path: "os.PathLike[str]") -> None:
    tr = Trainer(sync_envs(2), SMALL, run_cfg(2, "my_way_home"))
    path = os.path.join(str(tmp_path), "c.pt")
    CK.save_checkpoint(path, tr, SIG)
    other = Trainer(sync_envs(2), SMALL, run_cfg(2, "generated"))
    with pytest.raises(CK.CheckpointMismatch, match="maps"):
        CK.resume(other, path, SIG)
    blob = torch.load(path, map_location="cpu", weights_only=True)   # legacy: no maps field
    rc = json.loads(blob["run_config_json"])
    del rc["maps"]
    blob["run_config_json"] = json.dumps(rc)
    assert CK.maps_of(blob) == "my_way_home"


# ---------------------------------------------------------------- CLI / configs (no engine)
def test_cli_default_is_unchanged_and_shaping_must_fit_the_maps(capsys: pytest.CaptureFixture) -> None:
    ns = run_doom.parse_args([])
    assert ns.maps == "my_way_home" and run_doom.build_run_config(ns).maps == "my_way_home"
    assert run_doom.parse_args(["--maps", "generated", "--shaping", "full_level"]).shaping == "full_level"
    for bad in (["--maps", "generated", "--shaping", "my_way_home"], ["--maps", "generated", "--shaping", "passthrough"],
                ["--shaping", "full_level"]):
        with pytest.raises(SystemExit):
            run_doom.parse_args(bad)
    assert "does not fit" in capsys.readouterr().err


def test_generated_config_uses_the_train_bundle_and_keeps_sensors() -> None:
    sch, cfg = run_doom.build_generated_maps(a0_train_env_config())
    base = a0_train_env_config()
    assert cfg.sensor_signature() == base.sensor_signature()
    assert cfg.scenario_wad is None and os.path.normcase(cfg.map_wad).endswith(os.path.normcase(os.path.join("bundles", "train-all.wad")))
    assert "heldout" not in cfg.map_wad and set(cfg.maps) == {l for s in sch.stages for l, _ in sch.pool[s]}
    assert all(mid.split("-")[1] == "train" for s in sch.stages for _, mid in sch.pool[s])
    assert cfg.skill == GENERATED_MAPS.skill and cfg.episode_timeout_tics == GENERATED_MAPS.episode_timeout_tics
    assert dict(cfg.engine_rewards) == {"map_exit_reward": 1.0} and not cfg.vanilla_eval
    cfg.validate()
    fac = run_doom.build_env_factory(cfg, "full_level", 0, 1)   # validates the reward setup, no engine
    assert fac.reward == FULL_LEVEL_REWARD and fac.coverage == FULL_LEVEL_COVERAGE
    with pytest.raises(ValueError, match="heldout"):
        generated_maps_env_config(base, os.path.join(ROOT, "heldout_maps", "x.wad"), ("DD0000",))
