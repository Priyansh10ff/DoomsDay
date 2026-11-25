"""episodes.jsonl (ADR 0009 Decision 5 item 2) and the Phase 2 refusal. No engine, no subprocess."""
import json
import os
from dataclasses import replace

import pytest

from a0_helpers import FakeDoomEnv
from configs.model_config import ModelConfig
from configs.train_config import PPOConfig, RunConfig, SafetyConfig
from train.checkpoint import resume
from train.doom_factory import DoomEnvFactory
from train.run_loop import CHECKPOINT_NAME, EPISODES_NAME, METRICS_NAME, run_training
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SIG = (("toy", True),)
SMALL = ModelConfig(core_hidden=16, fusion_dim=16, ray_dim=8, entity_token_dim=8, entity_mlp_hidden=8,
                    self_dim=8, self_hidden=8)
FIELDS = {"update_idx", "decision", "env_index", "return", "length", "terminated", "truncated",
          "level_exit", "dead", "timeout", "success"}


def make_env(i: int) -> FakeDoomEnv:
    # env 0 times out after 5 decisions; env 1 reaches the exit after 3
    if i == 0:
        return FakeDoomEnv(time_limit=5)
    e = FakeDoomEnv(time_limit=50, die_at=3)
    e.exit_instead_of_death = True
    return e


def trainer() -> Trainer:
    venv = SyncVecEnv([lambda i=i: make_env(i) for i in range(2)])
    run = replace(RunConfig(), num_workers=1, envs_per_worker=2, torch_threads=2,
                  ppo=PPOConfig(rollout_len=10, num_minibatches=1, epochs=1))
    return Trainer(venv, SMALL, run)


def read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_one_line_per_episode_and_resume_appends(tmp_path: "os.PathLike[str]") -> None:
    d = str(tmp_path)
    tr = trainer()
    run_training(tr, d, SIG, SafetyConfig(), max_updates=2, max_seconds=60, log=lambda m: None,
                 install_sigint=False)
    first = read(os.path.join(d, EPISODES_NAME))
    # 20 decisions per env: env 0 ends 4 times (every 5), env 1 ends 6 times (every 3)
    assert len(first) == len(tr.log.returns) == 10
    assert all(set(r) == FIELDS for r in first)
    timeouts = [r for r in first if r["env_index"] == 0]
    exits = [r for r in first if r["env_index"] == 1]
    assert len(timeouts) == 4 and all(r["timeout"] and r["truncated"] and not r["success"] for r in timeouts)
    assert all(r["length"] == 5 and r["return"] == 5.0 for r in timeouts)
    assert len(exits) == 6 and all(r["level_exit"] and r["success"] and not r["dead"] for r in exits)
    assert all(r["length"] == 3 for r in exits)
    assert [r["update_idx"] for r in first] == sorted(r["update_idx"] for r in first)
    assert {r["update_idx"] for r in first} == {0, 1}
    assert first[0]["decision"] == 3 * 2   # env 1 ends at rollout step 3, 2 envs per step
    m = read(os.path.join(d, METRICS_NAME))[-1]
    assert m["recent_success"] == pytest.approx(0.6) and m["recent_length"] == pytest.approx(3.8)
    assert m["episodes_done"] == 10.0

    tr2 = trainer()
    resume(tr2, os.path.join(d, CHECKPOINT_NAME), SIG)
    run_training(tr2, d, SIG, SafetyConfig(), max_updates=1, max_seconds=60, log=lambda m: None,
                 install_sigint=False)
    both = read(os.path.join(d, EPISODES_NAME))
    assert both[:len(first)] == first                   # appended, not overwritten
    assert len(both) > len(first) and all(r["update_idx"] == 2 for r in both[len(first):])
    assert all(r["decision"] > 40 for r in both[len(first):])
    assert len(read(os.path.join(d, METRICS_NAME))) == 3


def test_trainer_refuses_phase2_engine_test_config() -> None:
    class Cfg:
        phase2_engine_test = True

    with pytest.raises(ValueError):
        DoomEnvFactory(Cfg())
    env_config = pytest.importorskip("env.config")  # imports vizdoom, starts no engine
    if "phase2_engine_test" not in {f.name for f in env_config.EnvConfig.__dataclass_fields__.values()}:
        pytest.skip("env.config has no phase2_engine_test field")
    with pytest.raises(ValueError):
        DoomEnvFactory(env_config.TRAIN_CONFIG.replace(phase2_engine_test=True))
    DoomEnvFactory(env_config.TRAIN_CONFIG)  # the normal config is accepted (engine not started)
