"""Reward shaping wired into the DOOM entry point (ADR 0010, worker changes 1 and 2) and
the checkpoint double-rename fix. No engine started: the vec env is faked or a SyncVecEnv
over fake envs."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from a0_helpers import ROOT, FakeDoomEnv
from configs.a0_env import a0_train_env_config
from configs.model_config import ModelConfig
from configs.reward_config import MY_WAY_HOME_COVERAGE, MY_WAY_HOME_REWARD
from configs.train_config import PPOConfig, RunConfig, SafetyConfig
from rewards.factory import ShapedEnvFactory
from test_train_ppo import make_trainer
from train import checkpoint as CK
from train import run_doom
from train.doom_factory import DoomEnvFactory
from train.run_loop import CHECKPOINT_NAME, EPISODES_NAME, run_training
from train.trainer import Trainer, episode_record
from train.vec_env import SyncVecEnv

SIG = (("toy", True),)
SMALL = ModelConfig(core_hidden=16, fusion_dim=16, ray_dim=8, entity_token_dim=8, entity_mlp_hidden=8,
                    self_dim=8, self_hidden=8)
CELL = MY_WAY_HOME_COVERAGE.cell_size
UNSHAPED_FIELDS = {"update_idx", "decision", "env_index", "return", "length", "terminated", "truncated",
                   "level_exit", "dead", "timeout", "success"}


# ---------------------------------------------------------------- 1. the flag
def test_default_factory_is_exactly_todays_doom_factory() -> None:
    cfg = a0_train_env_config()
    assert run_doom.parse_args([]).shaping is None
    f = run_doom.build_env_factory(cfg, None, decisions_done=12_345, num_envs=8)
    assert type(f) is DoomEnvFactory and f == DoomEnvFactory(cfg)


def test_shaping_flag_wraps_the_doom_factory_with_the_anneal_offset() -> None:
    cfg = a0_train_env_config()
    assert run_doom.parse_args(["--shaping", "my_way_home"]).shaping == "my_way_home"
    f = run_doom.build_env_factory(cfg, "my_way_home", decisions_done=64_000, num_envs=32)
    assert isinstance(f, ShapedEnvFactory) and f.inner == DoomEnvFactory(cfg)
    assert (f.reward, f.coverage, f.start_decisions) == (MY_WAY_HOME_REWARD, MY_WAY_HOME_COVERAGE, 2000)
    assert f.is_engine_instance
    with pytest.raises(ValueError, match="unknown --shaping"):
        run_doom.build_env_factory(cfg, "no_such_preset", 0, 1)
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--shaping", "nope"])


def test_entry_point_imports_no_reward_code_or_torch_by_default() -> None:
    code = ("import sys, train.run_doom; "
            "print(int('torch' in sys.modules), int(any(m == 'rewards' or m.startswith(('rewards.', 'intrinsic')) "
            "for m in sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["0", "0"], out.stderr


class _VecEnvBuilt(Exception):
    pass


@pytest.mark.parametrize("shaping", [None, "my_way_home"])
def test_resume_reads_decisions_before_the_vec_env_is_built(tmp_path: "os.PathLike[str]",
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            shaping: str | None) -> None:
    tr = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.0)
    tr.decisions_done = 6_400
    # ADR 0011 F6: the checkpoint must have been trained with the same preset (a
    # mismatch is refused; tests/test_adr0011_fixes.py). Its env count (2) sets the clock.
    tr.run = replace(tr.run, shaping=shaping)
    ckpt = os.path.join(str(tmp_path), "ckpt.pt")
    CK.save_checkpoint(ckpt, tr, SIG)
    seen: dict[str, Any] = {}

    def fake_vec_env(layout: list, *a: Any, **k: Any) -> None:
        seen["factory"] = layout[0][0]
        raise _VecEnvBuilt

    monkeypatch.setattr(run_doom, "SubprocVecEnv", fake_vec_env)
    monkeypatch.setattr(run_doom.PC, "engine_processes", lambda names: [])
    monkeypatch.setattr(run_doom.PC, "set_below_normal", lambda *a: None)
    argv = ["--resume", ckpt, "--run-dir", str(tmp_path)] + (["--shaping", shaping] if shaping else [])
    with pytest.raises(_VecEnvBuilt):
        run_doom.main(argv)
    if shaping is None:
        assert seen["factory"] == DoomEnvFactory(a0_train_env_config())
    else:  # per-env decisions on the checkpoint's own clock: 6,400 / its 2 envs
        assert isinstance(seen["factory"], ShapedEnvFactory) and seen["factory"].start_decisions == 3_200


# ---------------------------------------------------------------- 2. episode records
class PosEnv(FakeDoomEnv):
    """FakeDoomEnv (engine reward 1 per step) with a pose that enters a new cell every step."""

    def _pose(self) -> dict[str, float]:
        return {"pos_x": (self._t + 0.5) * CELL, "pos_y": 0.5 * CELL}

    def reset(self, **kw: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = super().reset(**kw)
        return obs, info | self._pose()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, r, te, tr, info = super().step(action)
        return obs, r, te, tr, info | self._pose()


class _Cfg:  # passes check_reward_setup for the my_way_home preset (as in test_rewards_and_coverage)
    engine_rewards = (("living_reward", -0.0001),)
    episode_timeout_tics, frame_skip, vanilla_eval = 2100, 4, False


class _PosFactory:
    config, is_engine_instance = _Cfg(), False

    def __init__(self, exit_env: bool) -> None:
        self.exit_env = exit_env

    def __call__(self) -> PosEnv:
        if not self.exit_env:
            return PosEnv(time_limit=5)
        e = PosEnv(time_limit=50, die_at=3)
        e.exit_instead_of_death = True
        return e


def test_shaped_episodes_log_per_term_totals_and_the_engine_return(tmp_path: "os.PathLike[str]") -> None:
    factories = [ShapedEnvFactory(_PosFactory(i == 1), MY_WAY_HOME_REWARD, MY_WAY_HOME_COVERAGE) for i in range(2)]
    venv = SyncVecEnv(factories)
    run = replace(RunConfig(), num_workers=1, envs_per_worker=2, torch_threads=2,
                  ppo=PPOConfig(rollout_len=10, num_minibatches=1, epochs=1))
    tr = Trainer(venv, SMALL, run)
    d = str(tmp_path)
    run_training(tr, d, SIG, SafetyConfig(), max_updates=1, max_seconds=60, log=lambda m: None,
                 install_sigint=False)
    with open(os.path.join(d, EPISODES_NAME), encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == 2 + 3   # env 0 times out every 5, env 1 exits every 3 (10 decisions)
    for r in recs:
        assert set(r) == UNSHAPED_FIELDS | {"engine_return", "shaping_episode"}
        sh = r["shaping_episode"]
        assert r["engine_return"] == pytest.approx(r["length"] * 1.0) == pytest.approx(sh["engine_return_unscaled"])
        assert sh["reward_config"] == "my_way_home" and sh["cells_visited"] == r["length"] + 1
        assert sh["intrinsic"] > 0 and r["return"] != r["engine_return"]
        assert r["return"] - r["engine_return"] == pytest.approx(sh["intrinsic"], abs=1e-5)
        assert all(v == 0.0 for k, v in sh.items() if k.startswith("paid/"))
        assert r["success"] == r["level_exit"] == (r["env_index"] == 1)


def test_unshaped_record_is_unchanged() -> None:
    rec = episode_record(0, 1, 0, 2.0, 2, True, False, {"level_exit": True})
    assert set(rec) == UNSHAPED_FIELDS and rec["success"]


# ---------------------------------------------------------------- 3. checkpoint double rename
def test_crash_while_saving_never_leaves_no_main_checkpoint(tmp_path: "os.PathLike[str]",
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    a = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.0)
    path = os.path.join(str(tmp_path), CHECKPOINT_NAME)
    a.decisions_done = 111
    CK.save_checkpoint(path, a, SIG)
    a.decisions_done = 222
    real_replace = os.replace
    calls = {"n": 0}

    def crashing_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:   # the second rename of the save: simulated crash
            raise OSError("simulated crash between renames")
        real_replace(src, dst)

    monkeypatch.setattr(CK.os, "replace", crashing_replace)
    with pytest.raises(OSError, match="simulated crash"):
        CK.save_checkpoint(path, a, SIG)
    monkeypatch.setattr(CK.os, "replace", real_replace)
    assert os.path.exists(path)
    blob = CK.load_checkpoint(path)
    assert blob["loaded_from"] == path and int(blob["trainer"]["decisions_done"]) == 111
    CK.save_checkpoint(path, a, SIG)   # a later save succeeds and both generations load
    assert CK.checkpoint_decisions_done(path) == 222
    assert int(CK.load_checkpoint(path + CK.PREV_SUFFIX)["trainer"]["decisions_done"]) == 111


def test_load_falls_back_to_prev_when_main_is_missing_but_never_to_tmp(tmp_path: "os.PathLike[str]") -> None:
    a = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.0)
    path = os.path.join(str(tmp_path), CHECKPOINT_NAME)
    a.decisions_done = 5
    CK.save_checkpoint(path, a, SIG)
    a.decisions_done = 9
    CK.save_checkpoint(path, a, SIG)
    os.replace(path, path + CK.TMP_SUFFIX)   # the state the old two-rename save could leave
    blob = CK.load_checkpoint(path)
    assert blob["loaded_from"] == path + CK.PREV_SUFFIX and int(blob["trainer"]["decisions_done"]) == 5
    b = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.0)
    assert CK.resume(b, path, SIG)["loaded_from"] == path + CK.PREV_SUFFIX and b.decisions_done == 5
    os.remove(path + CK.PREV_SUFFIX)
    with pytest.raises(FileNotFoundError):
        CK.load_checkpoint(path)            # .tmp alone is never trusted
    assert torch.load(path + CK.TMP_SUFFIX, weights_only=True)  # (it exists; it is just not used)
