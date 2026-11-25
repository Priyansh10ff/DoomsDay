"""Trainer: ratio == 1 before any update, determinism, stale time-out handling,
hidden state carried across rollouts and reset per env, torch-free vec_env."""
import subprocess
import sys
from dataclasses import replace

import numpy as np
import torch

from a0_helpers import ROOT, FakeDoomEnv
from configs.model_config import ModelConfig
from configs.train_config import PPOConfig, RunConfig, TelemetryMaskConfig
from train.ppo import evaluate_minibatch
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SMALL = ModelConfig(core_hidden=32, fusion_dim=32, ray_dim=16, entity_token_dim=16, entity_mlp_hidden=16,
                    self_dim=16, self_hidden=16)


def make_trainer(n_envs: int = 4, t_len: int = 10, seed: int = 0, stale: bool = False, time_limit: int = 7,
                 die_at: int | None = None, drop_prob: float = 0.0, model_cfg: ModelConfig = SMALL,
                 omit_fresh_key: bool = False, last_damage: float = 0.0) -> Trainer:
    venv = SyncVecEnv([lambda: FakeDoomEnv(time_limit, stale, die_at, omit_fresh_key=omit_fresh_key,
                                           last_damage=last_damage) for _ in range(n_envs)])
    run = replace(RunConfig(), seed=seed, num_workers=1, envs_per_worker=n_envs,
                  ppo=PPOConfig(rollout_len=t_len, num_minibatches=2, epochs=2),
                  telemetry_mask=TelemetryMaskConfig(episode_prob=drop_prob))
    return Trainer(venv, model_cfg, run)


def test_ratio_is_one_before_first_update() -> None:
    for drop in (0.0, 0.5):
        tr = make_trainer(drop_prob=drop)
        ro = tr.collect()
        adv, ret = tr.advantages(ro)
        idx = torch.arange(ro.shape[1])
        with torch.no_grad():
            res = evaluate_minibatch(tr.model, ro, idx, adv, ret, tr.ppo, tr.run.aux)
        torch.testing.assert_close(res.logp, ro.logp, atol=1e-5, rtol=0)
        assert float((res.ratio - 1).abs().max()) < 1e-5


def test_ratio_one_on_first_minibatch_of_full_update() -> None:
    tr = make_trainer()
    stats = tr.train_iteration()
    assert stats["first_minibatch_ratio_max_dev"] < 1e-5
    assert np.isfinite(stats["loss"])


def test_determinism_same_seed_same_params() -> None:
    def run(seed: int) -> list[torch.Tensor]:
        tr = make_trainer(seed=seed, drop_prob=0.3)
        for _ in range(2):
            tr.train_iteration()
        return [p.detach().clone() for p in tr.model.parameters()]

    a, b, c = run(0), run(0), run(1)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_stale_timeout_is_dropped_not_bootstrapped() -> None:
    tr = make_trainer(n_envs=2, t_len=10, stale=True, time_limit=4)
    ro = tr.collect()
    # episodes of 4 decisions: time-outs at t = 3 and 7
    for t in range(10):
        is_to = t in (3, 7)
        assert bool(ro.truncated[t].all()) == is_to
        assert bool(ro.valid[t].all()) == (not is_to)
    assert torch.all(ro.bootstrap_value == 0)
    assert tr.last_stale_timeouts == 4
    adv, ret = tr.advantages(ro)
    g = tr.ppo.gamma
    assert torch.all(adv[3] == 0) and torch.all(adv[7] == 0)
    torch.testing.assert_close(adv[2], ro.rewards[2] + g * ro.values[3] - ro.values[2])
    # hidden reset right after each time-out
    assert torch.all(ro.episode_start[4] == 1) and torch.all(ro.episode_start[8] == 1)
    assert torch.all(ro.episode_start[0] == 1) and torch.all(ro.episode_start[5] == 0)


def test_fresh_timeout_bootstraps_from_final_obs_value() -> None:
    tr = make_trainer(n_envs=2, t_len=6, stale=False, time_limit=4)
    ro = tr.collect()
    assert torch.all(ro.valid)
    assert torch.all(ro.truncated[3]) and torch.all(ro.bootstrap_value[3] != 0)
    assert torch.all(ro.bootstrap_value[[0, 1, 2, 4, 5]] == 0)


def test_termination_is_not_bootstrapped() -> None:
    tr = make_trainer(n_envs=2, t_len=6, stale=True, time_limit=50, die_at=3)
    ro = tr.collect()
    assert torch.all(ro.terminated[2]) and not bool(ro.truncated.any()) and torch.all(ro.valid)
    adv, _ = tr.advantages(ro)
    torch.testing.assert_close(adv[2], ro.rewards[2] - ro.values[2])


def test_hidden_state_carried_across_rollouts() -> None:
    tr = make_trainer(n_envs=4, t_len=5, time_limit=7)
    ro1 = tr.collect()
    h_end = tr.h.clone()
    ro2 = tr.collect()
    assert torch.equal(ro2.h0, h_end)
    assert not torch.all(ro2.h0 == 0)
    # the recomputed sequence with stored h0 reproduces the stored log-probs (no reset at t=0 of ro2)
    assert torch.all(ro2.episode_start[0] == 0)
    adv, ret = tr.advantages(ro2)
    with torch.no_grad():
        res = evaluate_minibatch(tr.model, ro2, torch.arange(4), adv, ret, tr.ppo, tr.run.aux)
    torch.testing.assert_close(res.logp, ro2.logp, atol=1e-5, rtol=0)
    del ro1


def test_telemetry_masking_keeps_true_obs_in_storage() -> None:
    tr = make_trainer(n_envs=4, t_len=6, drop_prob=1.0)
    ro = tr.collect()
    assert bool(ro.telemetry_dropped.all())
    assert float(ro.obs["rays"].abs().sum()) > 0          # true rays kept for aux targets
    assert bool(ro.obs["entity_mask"].any())


def test_aux_labels_flow_into_rollout() -> None:
    tr = make_trainer(n_envs=2, t_len=6, stale=True, time_limit=4)
    ro = tr.collect()
    assert bool(ro.keys_valid.all())
    assert torch.all(ro.keys[0, :, 0] == 0)               # reset info: no key
    assert torch.all(ro.keys[1, :, 0] == 1)               # step info: blue key
    # the time-out's damage comes from the final info (0.0 here): window valid, negative
    assert bool(ro.damage_valid[3].all()) and torch.all(ro.damage[3] == 0)


def test_missing_fresh_key_is_treated_as_stale_and_counted() -> None:
    tr = make_trainer(n_envs=2, t_len=10, time_limit=4, omit_fresh_key=True)
    stats = tr.train_iteration()
    assert stats["stale_timeouts"] == 4 and stats["stale_timeouts_total"] == 4  # 2 envs x 2 time-outs


def test_fresh_timeouts_are_not_counted() -> None:
    tr = make_trainer(n_envs=2, t_len=10, time_limit=4)
    assert tr.train_iteration()["stale_timeouts"] == 0


def test_fresh_bootstrap_uses_unreset_hidden_state_and_final_obs() -> None:
    tr = make_trainer(n_envs=2, t_len=6, time_limit=4)
    captured: list = []
    inner = tr.env.step

    def spy(actions: np.ndarray):  # record the final observations the env returned
        st = inner(actions)
        captured.append(st)
        return st

    tr.env.step = spy  # type: ignore[method-assign]
    ro = tr.collect()
    with torch.no_grad():
        out, h = tr.model.forward_sequence(ro.obs, ro.h0, ro.episode_start)
        h_hist = []
        hh = ro.h0
        for t in range(4):
            _, hh = tr.model.step({k: v[t] for k, v in ro.obs.items()}, hh, ro.episode_start[t])
            h_hist.append(hh)
        from agent.tensorize import to_tensors
        from train.vec_env import stack_obs
        fo = to_tensors(stack_obs(captured[3].final_obs))
        expected, _ = tr.model.step(fo, h_hist[3], torch.zeros(2))  # no reset
        wrong, _ = tr.model.step(fo, h_hist[3], torch.ones(2))      # with reset
    scale = tr.scale.scale
    torch.testing.assert_close(ro.bootstrap_value[3], expected.value_ext * scale, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(ro.bootstrap_value[3], wrong.value_ext * scale)
    del out, h


def test_death_window_is_a_valid_positive() -> None:
    tr = make_trainer(n_envs=2, t_len=8, time_limit=50, die_at=3, last_damage=0.0)
    ro = tr.collect()
    # windows starting at t=0..2 end in the death at transition 2: valid and positive
    assert bool(ro.damage_valid[:3].all())
    assert torch.all(ro.damage[:3] == 1)


def test_value_int_receives_no_gradient_in_update() -> None:
    tr = make_trainer(n_envs=2, t_len=6)
    before = [p.detach().clone() for p in tr.model.value_int.parameters()]
    tr.train_iteration()
    for p, b in zip(tr.model.value_int.parameters(), before):
        assert p.grad is None or torch.all(p.grad == 0)
        assert torch.equal(p.detach(), b)
    stats = tr.train_iteration()
    assert not any("int" in k for k in stats), "value_int must not be logged before the intrinsic stream"


def test_vec_env_module_does_not_import_torch() -> None:
    code = ("import sys; sys.path.insert(0, r'%s'); import train.vec_env, train.aux_labels, configs; "
            "assert 'torch' not in sys.modules, 'torch imported'" % ROOT)
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
