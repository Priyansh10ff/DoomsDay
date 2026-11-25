"""Checkpoint/resume (weights_only=True, sensor signature, sessions, generations),
learning-rate schedule over decisions, A0 env config. No engine started."""
import dataclasses
import os

import pytest
import torch

import a0_helpers  # noqa: F401
from configs.a0_env import a0_eval_env_config, a0_train_env_config
from configs.train_config import PPOConfig
from test_train_ppo import make_trainer
from train.checkpoint import PREV_SUFFIX, CheckpointMismatch, load_checkpoint, resume, save_checkpoint
from train.trainer import session_env_seed

# Shaped like EnvConfig.sensor_signature(): nested tuples, floats, bools.
SIG = (("frame_skip", 4), ("render_hud", True), ("turn_bins", (-10.0, -4.0, 0.0, 4.0, 10.0)),
       ("max_entities", 24), ("screen_enabled", False), ("ray_bands", ((30, 46), (46, 54), (54, 70))))


def test_weights_only_round_trip(tmp_path: "os.PathLike[str]") -> None:
    """Decides which stored types pass torch.load(weights_only=True) (ADR 0008 point 7)."""
    a = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.5)
    a.train_iteration()
    path = os.path.join(str(tmp_path), "ckpt.pt")
    save_checkpoint(path, a, SIG)
    raw = torch.load(path, map_location="cpu", weights_only=True)  # must not raise
    assert set(raw) == {"trainer", "model_config_json", "obs_spec_json", "run_config_json", "sensor_signature_json",
                        "shaping_json"}  # ADR 0011 F6
    blob = load_checkpoint(path)
    assert blob["model_config"] == a.model.cfg
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, blob["trainer"]["model"][k])


def test_resume_restores_state_and_reseeds_envs(tmp_path: "os.PathLike[str]") -> None:
    a = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.5)
    a.train_iteration()
    path = os.path.join(str(tmp_path), "ckpt.pt")
    save_checkpoint(path, a, SIG)
    b = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.5)
    blob = resume(b, path, SIG)
    assert b.update_idx == 1 and b.decisions_done == a.decisions_done == 8
    assert b.scale.state_dict() == a.scale.state_dict()
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(x, y)
    for x, y in zip(a.optimizer.state_dict()["state"].values(), b.optimizer.state_dict()["state"].values()):
        assert torch.equal(x["exp_avg"], y["exp_avg"])
    assert torch.equal(a.act_gen.get_state(), b.act_gen.get_state())
    # (resume then starts new episodes, which draws from the dropout RNG; check the raw restore)
    raw = make_trainer(n_envs=2, t_len=4, seed=5, drop_prob=0.5)
    raw.load_state_dict(load_checkpoint(path)["trainer"])
    assert a.dropout.rng.bit_generator.state == raw.dropout.rng.bit_generator.state
    # new session, env seed derived from (run seed, session), not the run seed again
    assert b.session == 1 and blob["session_env_seed"] == session_env_seed(0, 1) != 0
    assert session_env_seed(0, 0) == 0 and session_env_seed(0, 1) != session_env_seed(0, 2)
    fresh = make_trainer(n_envs=2, t_len=4, seed=0)
    assert not all((fresh.obs[k] == b.obs[k]).all() for k in ("rays", "entities"))
    b.train_iteration()
    assert b.update_idx == 2
    save_checkpoint(path, b, SIG)
    c = make_trainer(n_envs=2, t_len=4, seed=0, drop_prob=0.5)
    resume(c, path, SIG)
    assert c.session == 2


def test_previous_generation_kept(tmp_path: "os.PathLike[str]") -> None:
    a = make_trainer(n_envs=2, t_len=4)
    path = os.path.join(str(tmp_path), "ckpt.pt")
    save_checkpoint(path, a, SIG)
    a.train_iteration()
    save_checkpoint(path, a, SIG)
    assert os.path.exists(path + PREV_SUFFIX)
    old = load_checkpoint(path + PREV_SUFFIX)
    new = load_checkpoint(path)
    assert old["trainer"]["update_idx"] == 0 and new["trainer"]["update_idx"] == 1


def test_resume_refuses_different_sensors_or_model(tmp_path: "os.PathLike[str]") -> None:
    a = make_trainer(n_envs=2, t_len=4)
    path = os.path.join(str(tmp_path), "ckpt.pt")
    save_checkpoint(path, a, SIG)
    other = tuple((k, 2) if k == "frame_skip" else (k, v) for k, v in SIG)
    with pytest.raises(CheckpointMismatch):
        resume(make_trainer(n_envs=2, t_len=4), path, other)
    from configs.model_config import ModelConfig
    with pytest.raises(CheckpointMismatch):
        resume(make_trainer(n_envs=2, t_len=4, model_cfg=ModelConfig()), path, SIG)
    with pytest.raises(ValueError):
        save_checkpoint(path, a, ())


def test_lr_constant_by_default_and_decay_over_decisions() -> None:
    tr = make_trainer(n_envs=2, t_len=4)
    for _ in range(3):
        assert tr.train_iteration()["lr"] == pytest.approx(PPOConfig().lr)
    with pytest.raises(ValueError):
        PPOConfig(lr_final_fraction=0.0, lr_decay_decisions=100)   # below the 0.1 floor
    with pytest.raises(ValueError):
        PPOConfig(lr_final_fraction=0.5)                          # decay needs a decision budget
    tr2 = make_trainer(n_envs=2, t_len=4)
    tr2.ppo = dataclasses.replace(tr2.ppo, lr_final_fraction=0.1, lr_decay_decisions=16)
    lrs = [tr2.train_iteration()["lr"] for _ in range(4)]  # 8 decisions per update
    base = PPOConfig().lr
    assert lrs == pytest.approx([base, base * (1 - 0.5 * 0.9), base * 0.1, base * 0.1])
    assert tr2.decisions_done == 32 and tr2.state_dict()["decisions_done"] == 32


def test_a0_env_configs_have_screen_off_and_same_sensors() -> None:
    cfg_mod = pytest.importorskip("env.config")  # imports vizdoom, does not start it
    train_cfg, eval_cfg = a0_train_env_config(), a0_eval_env_config()
    assert train_cfg.screen_enabled is False and eval_cfg.screen_enabled is False
    cfg_mod.assert_same_sensors(train_cfg, eval_cfg)
    eval_cfg.assert_vanilla()


def test_real_sensor_signature_round_trips(tmp_path: "os.PathLike[str]") -> None:
    pytest.importorskip("env.config")
    sig = a0_train_env_config().sensor_signature()
    a = make_trainer(n_envs=2, t_len=4)
    path = os.path.join(str(tmp_path), "ckpt.pt")
    save_checkpoint(path, a, sig)
    resume(make_trainer(n_envs=2, t_len=4), path, sig)


def test_entity_type_table_size_matches_helper_default() -> None:
    obs_mod = pytest.importorskip("env.observation")
    from a0_helpers import FakeDoomEnv
    space = FakeDoomEnv().observation_space
    assert int(space["entity_type"].high.max()) + 1 == len(obs_mod.ENTITY_TYPES)
