"""RND exploration bonus (train/rnd.py) and the inverse-dynamics auxiliary loss
(train/inverse_dynamics.py) in our PPO trainer. No engine is started."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from a0_helpers import ROOT, FakeDoomEnv, doom_like_spaces, random_obs
from agent.model import build_agent
from agent.obs_spec import spec_from_spaces
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig
from configs.train_config import InvDynConfig, PPOConfig, RNDConfig, RunConfig
from train.rnd import RND, ReturnFilter, RunningMeanStd, feature_layout, rnd_features
from train import checkpoint as CK
from train import run_doom
from train.inverse_dynamics import build_head, inverse_dynamics_loss, pair_mask
from train.rollout import rollout_to
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SMALL = ModelConfig(core_hidden=32, fusion_dim=32, ray_dim=16, entity_token_dim=16, entity_mlp_hidden=16,
                    self_dim=16, self_hidden=16)
SIG = ("test-signature",)
SPACE, ACT = doom_like_spaces()
SHAPES = {k: tuple(s.shape) for k, s in SPACE.spaces.items()}
RND_ON = RNDConfig(int_coef=0.5, hidden_dim=32, out_dim=16)
INV_ON = InvDynConfig(coef=0.1, hidden_dim=32)


def make_trainer(rnd: RNDConfig | None = None, inv: InvDynConfig | None = None, seed: int = 0,
                 time_limit: int = 9, die_at: int | None = None, **ppo_kw) -> Trainer:
    venv = SyncVecEnv([lambda: FakeDoomEnv(time_limit, die_at=die_at) for _ in range(2)])
    ppo = PPOConfig(**{"rollout_len": 8, "num_minibatches": 1, "epochs": 2, **ppo_kw})
    run = replace(RunConfig(), seed=seed, num_workers=1, envs_per_worker=2, ppo=ppo,
                  rnd=rnd or RNDConfig(), inv_dyn=inv or InvDynConfig())
    return Trainer(venv, SMALL, run)


def params(tr: Trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in tr.learner.parameters()]


# ---------------------------------------------------------------- RND input
def test_rnd_input_ignores_buttons_ammo_damage_weapon_and_prev_action() -> None:
    lay = feature_layout(SHAPES, RNDConfig())
    assert lay.dim == 3 * 80 + (28 - 12)
    rng = np.random.default_rng(0)
    obs = random_obs(SPACE, 5, rng)
    base = rnd_features(obs, lay)
    assert base.shape == (5, lay.dim) and base.dtype == np.float32
    farmed = {k: v.copy() for k, v in obs.items()}
    farmed["prev_action"] = (farmed["prev_action"] + 1) % 2
    farmed["self"][:, 2:14] = rng.uniform(-1, 1, (5, 12))
    np.testing.assert_array_equal(rnd_features(farmed, lay), base)
    moved = {k: v.copy() for k, v in obs.items()}
    moved["rays"][:, 0, 3] += 0.5
    assert not np.array_equal(rnd_features(moved, lay), base)
    keyed = {k: v.copy() for k, v in obs.items()}
    keyed["self"][:, 0] += 0.1               # health stays in
    assert not np.array_equal(rnd_features(keyed, lay), base)


def test_egomap_is_pooled_and_included_when_present() -> None:
    shapes = dict(SHAPES, egomap=(4, 32, 32))
    lay = feature_layout(shapes, RNDConfig())
    assert lay.extra == (("egomap", 4 * 16 * 16),) and lay.dim == 3 * 80 + 16 + 1024
    obs = random_obs(SPACE, 2, np.random.default_rng(1))
    obs["egomap"] = np.zeros((2, 4, 32, 32), np.uint8)
    obs["egomap"][0, 1, 4:6, 8:10] = 200
    f = rnd_features(obs, lay)
    pooled = f[:, -1024:].reshape(2, 4, 16, 16)
    assert pooled[0, 1, 2, 4] == pytest.approx(200.0) and pooled[1].sum() == 0
    with pytest.raises(ValueError, match="max_extra_dim"):
        feature_layout(shapes, RNDConfig(extra_pool=1, max_extra_dim=1000))


# ---------------------------------------------------------------- normalisers
def test_running_mean_std_matches_numpy_and_round_trips() -> None:
    g = np.random.default_rng(2)
    rms = RunningMeanStd((3,), eps=1e-4)
    chunks = [g.normal(5, 2, (n, 3)) for n in (7, 1, 30, 0, 12)]
    for c in chunks:
        rms.update(torch.as_tensor(c))
    allx = np.concatenate(chunks)
    np.testing.assert_allclose(rms.mean.numpy(), allx.mean(0), rtol=1e-3)
    np.testing.assert_allclose(rms.var.numpy(), allx.var(0), rtol=1e-3)
    assert torch.all(rms.std(10.0) == 10.0)   # floor
    other = RunningMeanStd((3,), eps=1e-4)
    other.load_state_dict(rms.state_dict())
    assert torch.equal(other.mean, rms.mean) and torch.equal(other.var, rms.var) and other.count == rms.count


def test_return_filter_is_non_episodic_and_skips_unusable_rows() -> None:
    f = ReturnFilter(2, gamma=0.5, eps=1e-8)
    r = torch.tensor([[1.0, 2.0], [0.0, 0.0], [4.0, 0.0]])
    ok = torch.tensor([[True, True], [True, False], [True, True]])
    f.update(r, ok)
    torch.testing.assert_close(f.ret, torch.tensor([4.25, 0.5], dtype=torch.float64))
    seen = np.array([1.0, 2.0, 0.5, 4.25, 0.5])          # (t1, env1) left out
    assert f.rms.mean.item() == pytest.approx(seen.mean(), rel=1e-6)
    assert f.std(1e-2) == pytest.approx(seen.std(), rel=1e-5)


# ---------------------------------------------------------------- the bonus
def test_rnd_reward_high_on_novel_inputs_and_decays_on_repeats() -> None:
    cfg = replace(RND_ON, predictor_lr=1e-3, predictor_fraction=1.0, predictor_epochs=4)
    lay = feature_layout(SHAPES, cfg)
    rnd = RND(lay, cfg, n_envs=8, net_seed=0, sample_seed=1)
    g = torch.Generator().manual_seed(3)
    seen = torch.rand(16, 8, lay.dim, generator=g)
    ok = torch.ones(16, 8, dtype=torch.bool)
    first, _ = rnd.intrinsic_rewards(seen, ok)
    raw = lambda x: rnd.nets.error(rnd.normalise(x.reshape(-1, lay.dim))).mean().item()  # noqa: E731
    before = raw(seen)
    for _ in range(30):
        rnd.train_predictor(seen, ok)
    after = raw(seen)
    novel = torch.rand(16, 8, lay.dim, generator=g) * 3.0 - 1.0      # outside the seen range
    assert after < 0.3 * before, (before, after)
    assert raw(novel) > 3.0 * after, (raw(novel), after)
    again, stats = rnd.intrinsic_rewards(seen, ok)
    assert again.mean() < first.mean() and stats["rnd_int_reward_mean"] >= 0.0
    # unusable rows: reward exactly 0
    ok2 = ok.clone()
    ok2[3, 2] = False
    r2, _ = rnd.intrinsic_rewards(novel, ok2)
    assert r2[3, 2] == 0.0 and r2[3, 3] > 0.0


def test_target_is_frozen_and_only_the_predictor_trains() -> None:
    lay = feature_layout(SHAPES, RND_ON)
    rnd = RND(lay, RND_ON, n_envs=2, net_seed=0, sample_seed=1)
    tgt = [p.clone() for p in rnd.nets.target.parameters()]
    pred = [p.clone() for p in rnd.nets.predictor.parameters()]
    x = torch.rand(8, 2, lay.dim)
    rnd.intrinsic_rewards(x, torch.ones(8, 2, dtype=torch.bool))
    stats = rnd.train_predictor(x, torch.ones(8, 2, dtype=torch.bool))
    assert stats["rnd_predictor_rows"] == 4.0          # 25% of 16 rows
    assert all(torch.equal(a, b) for a, b in zip(tgt, rnd.nets.target.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(pred, rnd.nets.predictor.parameters()))


# ---------------------------------------------------------------- trainer: off is unchanged
def test_off_builds_nothing_and_on_does_not_disturb_model_init_or_sampling() -> None:
    off = make_trainer()
    assert off.rnd is None and off.inv_dyn is None
    # non-default sizes but coefficients 0: still nothing built, identical update
    off2 = make_trainer(replace(RNDConfig(), hidden_dim=8), replace(InvDynConfig(), hidden_dim=8))
    s1, s2 = off.train_iteration(), off2.train_iteration()
    assert all(torch.equal(a, b) for a, b in zip(params(off), params(off2)))
    assert s1.keys() == s2.keys() and not any(k.startswith(("rnd", "inv_dyn", "v_int", "value_int")) for k in s1)
    assert torch.equal(off.optimizer.state_dict()["state"][0]["exp_avg"], off2.optimizer.state_dict()["state"][0]["exp_avg"])
    # switching on leaves the model's init and the action sampling untouched
    a, b = make_trainer(), make_trainer(RND_ON, INV_ON)
    assert all(torch.equal(x, y) for x, y in zip(params(a), params(b)))
    ra, rb = a.collect(), b.collect()
    assert torch.equal(ra.actions, rb.actions) and torch.equal(ra.logp, rb.logp)
    assert ra.rnd_feat.numel() == 0 and rb.rnd_feat.shape == (8, 2, b.rnd.layout.dim)
    rollout_to(ra, "cpu")      # empty RND fields move fine (hybrid path)
    rollout_to(rb, "cpu")


# ---------------------------------------------------------------- trainer: on
def test_rnd_on_trains_value_int_and_logs() -> None:
    tr = make_trainer(RND_ON, lr=1e-3)
    vi = [p.detach().clone() for p in tr.learner.value_int.parameters()]
    s = tr.train_iteration()
    for k in ("rnd_int_reward_mean", "rnd_int_reward_std", "rnd_predictor_loss", "value_int_explained_variance",
              "v_int_loss", "rnd_int_return_std"):
        assert k in s and (np.isfinite(s[k]) or k == "value_int_explained_variance"), (k, s.get(k))
    assert not all(torch.equal(a, b) for a, b in zip(vi, tr.learner.value_int.parameters()))


def test_stale_final_obs_gets_no_intrinsic_reward() -> None:
    tr = make_trainer(RND_ON, time_limit=50, die_at=3)
    ro = tr.collect()
    # FakeDoomEnv dies at its 3rd step (t=2): the repeated final observation is unusable
    assert not bool(ro.rnd_ok[2].any()) and bool(ro.rnd_ok[0].all()) and bool(ro.rnd_ok[3].all())
    r, _ = tr.rnd.intrinsic_rewards(ro.rnd_feat, ro.rnd_ok)
    assert torch.all(r[2] == 0)


def test_non_episodic_intrinsic_advantage_crosses_episode_ends() -> None:
    tr = make_trainer(RND_ON, time_limit=50, die_at=3)
    ro = tr.collect()
    adv_ext, _ = tr.advantages(ro)
    adv, ret_int, _ = tr._rnd_advantages(ro, adv_ext)
    # at the death step the extrinsic stream bootstraps 0; the intrinsic stream bootstraps
    # from the next episode's first value: ret_int[2] = 0 (reward) + gamma * (A_int[3] + V_int[3]) ...
    lam_ret = ro.values_int[3] + (ret_int[3] - ro.values_int[3]) * RND_ON.gae_lambda_int
    torch.testing.assert_close(ret_int[2], RND_ON.gamma_int * lam_ret, rtol=1e-5, atol=1e-6)
    assert adv.shape == adv_ext.shape


def test_runs_with_rnd_and_inverse_dynamics_are_deterministic() -> None:
    a, b = make_trainer(RND_ON, INV_ON), make_trainer(RND_ON, INV_ON)
    for _ in range(2):
        sa, sb = a.train_iteration(), b.train_iteration()
    assert all(torch.equal(x, y) for x, y in zip(params(a), params(b)))
    assert all((sa[k] == sb[k]) or (np.isnan(sa[k]) and np.isnan(sb[k])) for k in sa if not k.endswith("_dps"))
    for x, y in zip(a.inv_dyn.head.parameters(), b.inv_dyn.head.parameters()):
        assert torch.equal(x, y)


# ---------------------------------------------------------------- checkpoints
def test_checkpoint_round_trip_restores_rnd_and_head(tmp_path: "os.PathLike[str]") -> None:
    tr = make_trainer(RND_ON, INV_ON)
    tr.train_iteration()
    path = os.path.join(str(tmp_path), "checkpoint.pt")
    CK.save_checkpoint(path, tr, SIG)
    new = make_trainer(RND_ON, INV_ON, seed=0)
    CK.resume(new, path, SIG)
    for x, y in zip(tr.rnd.nets.state_dict().values(), new.rnd.nets.state_dict().values()):
        assert torch.equal(x, y)
    assert torch.equal(tr.rnd.obs_rms.mean, new.rnd.obs_rms.mean) and tr.rnd.obs_rms.count == new.rnd.obs_rms.count
    assert torch.equal(tr.rnd.ret_filter.ret, new.rnd.ret_filter.ret)
    assert torch.equal(tr.rnd.gen.get_state(), new.rnd.gen.get_state())
    for x, y in zip(tr.inv_dyn.head.parameters(), new.inv_dyn.head.parameters()):
        assert torch.equal(x, y)
    assert new.inv_dyn.optimizer.state_dict()["state"].keys() == tr.inv_dyn.optimizer.state_dict()["state"].keys()
    new.train_iteration()                                     # continues fine
    assert CK.checkpoint_resume_info(path)["switches"] == {"rnd": True, "inv_dyn": True}


def test_resume_refuses_a_change_of_either_switch(tmp_path: "os.PathLike[str]") -> None:
    on = make_trainer(RND_ON, INV_ON)
    p_on = os.path.join(str(tmp_path), "on.pt")
    CK.save_checkpoint(p_on, on, SIG)
    for rnd, inv, what in ((None, INV_ON, "rnd"), (RND_ON, None, "inv_dyn"), (None, None, "rnd")):
        with pytest.raises(CK.CheckpointMismatch, match=what):
            CK.resume(make_trainer(rnd, inv), p_on, SIG)
    # a coefficient change with the switch unchanged is allowed
    CK.resume(make_trainer(replace(RND_ON, int_coef=0.1), replace(INV_ON, coef=0.3)), p_on, SIG)
    # an older checkpoint without the fields reads as off
    plain = make_trainer()
    p_old = os.path.join(str(tmp_path), "old.pt")
    CK.save_checkpoint(p_old, plain, SIG)
    blob = torch.load(p_old, weights_only=True)
    rc = json.loads(blob["run_config_json"])
    del rc["rnd"], rc["inv_dyn"]
    blob["run_config_json"] = json.dumps(rc)
    torch.save(blob, p_old)
    assert CK.checkpoint_resume_info(p_old)["switches"] == {"rnd": False, "inv_dyn": False}
    CK.resume(make_trainer(), p_old, SIG)
    with pytest.raises(CK.CheckpointMismatch, match="rnd"):
        CK.resume(make_trainer(RND_ON), p_old, SIG)


def test_init_from_copies_model_weights_only(tmp_path: "os.PathLike[str]") -> None:
    from train.init_from import apply_init, read_init_source
    src = make_trainer(RND_ON, INV_ON, seed=5)
    src.train_iteration()
    path = os.path.join(str(tmp_path), "src.pt")
    CK.save_checkpoint(path, src, SIG)
    fresh = make_trainer(RND_ON, INV_ON, seed=0)
    ref = make_trainer(RND_ON, INV_ON, seed=0)
    apply_init(fresh, read_init_source(path, SIG))
    assert all(torch.equal(x, y) for x, y in zip(params(fresh), params(src)))
    for x, y in zip(fresh.rnd.nets.parameters(), ref.rnd.nets.parameters()):
        assert torch.equal(x, y)                              # RND starts fresh
    for x, y in zip(fresh.inv_dyn.head.parameters(), ref.inv_dyn.head.parameters()):
        assert torch.equal(x, y)


# ---------------------------------------------------------------- inverse dynamics
def test_pair_mask_cuts_episode_boundaries_and_invalid_steps() -> None:
    start = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0]])
    valid = torch.tensor([[True], [True], [True], [False], [True]])
    assert pair_mask(start, valid)[:, 0].tolist() == [True, False, False, False]


def _synthetic(spec_space, t: int, b: int, seed: int, rays_carry_action: bool) -> tuple[dict, torch.Tensor]:
    """Random observations; a_t is visible in o_{t+1} (rays, if asked) and ALWAYS in
    o_{t+1}.prev_action (the leak the loss must ignore)."""
    rng = np.random.default_rng(seed)
    obs = random_obs(spec_space, b, rng, lead=(t,))
    acts = np.stack([rng.integers(0, n, (t, b)) for n in ACT.nvec], axis=-1).astype(np.int64)
    obs["prev_action"][1:] = acts[:-1]
    if rays_carry_action:
        for i, n in enumerate(ACT.nvec):
            obs["rays"][1:, :, 0, i * 10:(i + 1) * 10] = (acts[:-1, :, i] / max(n - 1, 1))[..., None]
    flat = to_tensors({k: v.reshape(t * b, *v.shape[2:]) for k, v in obs.items()})
    return {k: v.reshape(t, b, *v.shape[1:]) for k, v in flat.items()}, torch.as_tensor(acts)


def _train_inv(rays_carry_action: bool, steps: int = 200, b: int = 64) -> tuple[float, float, dict]:
    spec = spec_from_spaces(SPACE, ACT)
    model = build_agent(spec, SMALL, seed=0)
    head = build_head(SMALL.fusion_dim, spec.head_sizes, INV_ON, seed=1, device="cpu")
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    obs, acts = _synthetic(SPACE, 16, b, seed=0, rays_carry_action=rays_carry_action)
    start = torch.zeros(16, b)
    start[0] = 1.0
    valid = torch.ones(16, b, dtype=torch.bool)
    first = float(inverse_dynamics_loss(model, head, obs, acts, start, valid)[0].detach())
    for _ in range(steps):
        loss, _ = inverse_dynamics_loss(model, head, obs, acts, start, valid)
        opt.zero_grad()
        loss.backward()
        opt.step()
    test_obs, test_acts = _synthetic(SPACE, 16, b, seed=1, rays_carry_action=rays_carry_action)
    with torch.no_grad():
        held, stats = inverse_dynamics_loss(model, head, test_obs, test_acts, start, valid)
    return first, float(held), stats


def test_inverse_dynamics_loss_falls_when_the_action_is_inferable() -> None:
    first, held, stats = _train_inv(rays_carry_action=True)
    assert held < 0.4 * first, (first, held)
    assert stats["inv_dyn_acc_head2"] > 0.5 and stats["inv_dyn_pairs_frac"] == 1.0


def test_inverse_dynamics_cannot_read_the_label_from_prev_action() -> None:
    first, held, stats = _train_inv(rays_carry_action=False)
    # only prev_action carries a_t, and it is zeroed: held-out turn accuracy stays near chance (1/7)
    assert stats["inv_dyn_acc_head2"] < 0.35, stats
    assert held > 0.8 * first, (first, held)


def test_inverse_dynamics_gradient_reaches_the_encoders() -> None:
    tr = make_trainer(inv=INV_ON)
    ro = tr.collect()
    loss, _ = inverse_dynamics_loss(tr.learner, tr.inv_dyn.head, ro.obs, ro.actions, ro.episode_start, ro.valid)
    loss.backward()
    assert tr.learner.rays.parameters().__next__().grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tr.learner.fusion.parameters())
    assert all(p.grad is None for p in tr.learner.value_ext.parameters())


# ---------------------------------------------------------------- flags and isolation
def test_run_doom_flags() -> None:
    run = run_doom.build_run_config(run_doom.parse_args(["--rnd-int-coef", "0.5", "--inv-dyn-coef", "0.1",
                                                         "--rnd-ext-coef", "2"]))
    assert run.rnd.enabled and (run.rnd.int_coef, run.rnd.ext_coef) == (0.5, 2.0)
    assert run.inv_dyn.enabled and run.inv_dyn.coef == 0.1
    d = run_doom.build_run_config(run_doom.parse_args([]))
    assert not d.rnd.enabled and not d.inv_dyn.enabled and d.rnd == RNDConfig() and d.inv_dyn == InvDynConfig()
    assert "rnd_int_coef=0.5" in run_doom.run_settings_line(run, "cpu")
    for bad in (["--rnd-int-coef", "-1"], ["--inv-dyn-coef", "-0.1"], ["--rnd-ext-coef", "2"]):
        with pytest.raises(SystemExit):
            run_doom.parse_args(bad)


def test_intrinsic_package_and_configs_stay_torch_free() -> None:
    code = "import sys, intrinsic, configs.train_config; print(int('torch' in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["0"], out.stderr
