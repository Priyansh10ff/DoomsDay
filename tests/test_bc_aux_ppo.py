"""Imitation auxiliary loss inside PPO (train/bc_aux.py), the PPO knobs added with it
(--lr/--clip/--ppo-epochs, LR warm-up, KL early stop, update-count checkpoints), the
resume refusal of changed BC settings, and train/bc.py's --use-pos-weight.
No engine is started."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from a0_helpers import FakeDoomEnv, doom_like_spaces
from agent.obs_spec import A0_KEYS, ObsSpec, spec_from_spaces
from configs.bc_config import USE_HEAD_INDEX
from configs.model_config import ModelConfig
from configs.train_config import BCAuxConfig, PPOConfig, RunConfig
from train import bc as BC
from train import bc_aux as BA
from train import checkpoint as CK
from train import run_doom
from train.bc_spec import expected_step_shapes
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SMALL = ModelConfig(core_hidden=32, fusion_dim=32, ray_dim=16, entity_token_dim=16, entity_mlp_hidden=16,
                    self_dim=16, self_hidden=16)
SPEC_V1 = spec_from_spaces(*doom_like_spaces())
SIG = ("test-signature",)
BURN_TOL = 1e-5


# ---------------------------------------------------------------- synthetic data
def episode_arrays(spec: ObsSpec, length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random observations shaped by `spec`; labels: move 1, turn 3, everything else 0."""
    sh = expected_step_shapes(spec)
    a = {
        "rays": rng.random((length, *sh["rays"]), dtype=np.float32),
        "entities": rng.uniform(-1, 1, (length, *sh["entities"])).astype(np.float32),
        "entity_type": rng.integers(0, spec.n_entity_types, (length, *sh["entity_type"])).astype(np.int64),
        "entity_mask": rng.random((length, *sh["entity_mask"])) < 0.5,
        "self": rng.uniform(-1, 1, (length, *sh["self"])).astype(np.float32),
        "prev_action": rng.integers(0, 2, (length, spec.n_heads)).astype(np.int64),
        "modality": np.ones((length, *sh["modality"]), np.float32),
    }
    a["entity_mask"][:, 0] = True
    action = np.zeros((length, spec.n_heads), np.int64)
    action[:, 0], action[:, 2] = 1, 3
    a["action"] = action
    a["executed_action"] = action.copy()
    a["engine_reward"] = np.zeros(length, np.float32)
    a["done"] = np.eye(1, length, length - 1, dtype=bool)[0]
    return a


def write_dataset(root: str, spec: ObsSpec, lengths: list[int], seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "index.jsonl"), "w", encoding="utf-8") as f:
        for i, n in enumerate(lengths):
            name = f"ep_{i:06d}.npz"
            np.savez_compressed(os.path.join(root, name), **episode_arrays(spec, n, rng))
            f.write(json.dumps({"file": name, "map_id": f"DD{i:04d}", "stage": "S0", "length": n,
                                "success": True, "seed": i, "noise": 0.0}) + "\n")
    return root


@pytest.fixture(scope="module")
def v1_data(tmp_path_factory: pytest.TempPathFactory) -> str:
    return write_dataset(str(tmp_path_factory.mktemp("bc_v1")), SPEC_V1, [30, 90, 12, 150])


def bc_cfg(data: str, **kw) -> BCAuxConfig:
    base = dict(data_dirs=(data,), coef=1.0, coef_final=1.0, batch_seqs=4, seq_len=8)
    base.update(kw)
    return BCAuxConfig(**base)


def make_trainer(bc: BCAuxConfig | None = None, data: str | None = None, seed: int = 0, **ppo_kw) -> Trainer:
    venv = SyncVecEnv([lambda: FakeDoomEnv(9) for _ in range(2)])
    ppo = PPOConfig(**{"rollout_len": 8, "num_minibatches": 1, "epochs": 2, **ppo_kw})
    run = replace(RunConfig(), seed=seed, num_workers=1, envs_per_worker=2, ppo=ppo, bc_aux=bc or BCAuxConfig())
    tr = Trainer(venv, SMALL, run)
    if bc is not None and bc.enabled:
        tr.attach_bc(BA.load_bc_data(bc.data_dirs, tr.spec, bc.max_bytes, 1))
    return tr


def params(tr: Trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in tr.learner.parameters()]


# ---------------------------------------------------------------- sampler
def test_sampler_shapes_padding_and_determinism(v1_data: str) -> None:
    eps = BA.load_bc_data((v1_data,), SPEC_V1, 2**30)
    s1, s2 = BA.BCSampler(eps, 8, 5, seed=3), BA.BCSampler(eps, 8, 5, seed=3)
    b1, b2 = s1.sample(), s2.sample()
    assert b1.labels.shape == (8, 5, 6) and b1.valid.shape == (8, 5) and b1.burn_start.shape == (8, 5)
    for k in A0_KEYS:
        assert b1.obs[k].shape[:2] == (8, 5) and b1.burn_obs[k].shape[:2] == (8, 5), k
        assert torch.equal(b1.obs[k], b2.obs[k]) and torch.equal(b1.burn_obs[k], b2.burn_obs[k])
    assert torch.equal(b1.labels, b2.labels)
    # a start 3 steps before an episode end: 3 valid steps, then masked padding
    b = s1.build([(2, 9)])            # episode 2 has 12 steps
    assert b.valid[:, 0].tolist() == [True] * 3 + [False] * 5
    assert b.burn_start[:, 0].tolist() == [0.0] * 0 + [1.0] + [0.0] * 7   # burn starts at step 1, row 0
    b0 = s1.build([(1, 0)])           # episode start: no burn-in, graded window resets itself
    assert b0.start[0, 0] == 1.0 and b0.burn_start.sum() == 0


def test_burn_in_matches_running_from_the_episode_start(v1_data: str) -> None:
    """Hidden state: offset <= L reproduces the full run from the episode start; offset
    > L starts from a ZERO state at offset - L (reset), not from anything earlier."""
    eps = BA.load_bc_data((v1_data,), SPEC_V1, 2**30)
    from agent.model import build_agent
    model = build_agent(SPEC_V1, SMALL, seed=0)
    l, ep = 8, eps[3]
    smp = BA.BCSampler(eps, l, 1, seed=0)

    def ref(lo: int, hi: int) -> list[torch.Tensor]:
        obs = {k: torch.as_tensor(ep.obs[k][lo:hi]).unsqueeze(1).to(BA._TORCH_DTYPES[k]) for k in A0_KEYS}
        start = torch.zeros(hi - lo, 1)
        start[0] = 1.0
        out, _ = model.forward_sequence(obs, model.initial_state(1), start)
        return [x[-l:] for x in out.logits]

    with torch.no_grad():
        for off, lo in ((5, 0), (8, 0), (20, 12)):
            got = BA.batch_logits(model, smp.build([(3, off)]))
            for g, r in zip(got, ref(lo, off + l)):
                torch.testing.assert_close(g, r, atol=BURN_TOL, rtol=0)


# ---------------------------------------------------------------- trainer
def test_coef_zero_is_bit_identical_and_draws_nothing(v1_data: str) -> None:
    plain = make_trainer()
    # linear schedule from 0: coefficient 0 at the first update, data attached
    bc = make_trainer(bc_cfg(v1_data, coef=0.0, coef_final=1.0, decay_updates=10), v1_data)
    gen_before = bc.bc_sampler.gen.get_state()
    s_plain, s_bc = plain.train_iteration(), bc.train_iteration()
    assert all(torch.equal(a, b) for a, b in zip(params(plain), params(bc)))
    assert s_plain["loss"] == s_bc["loss"] and s_bc["bc_coef"] == 0.0 and "bc_loss" not in s_bc
    assert torch.equal(gen_before, bc.bc_sampler.gen.get_state())
    # value warm-up: the term is off, identical to a plain warm-up run
    w_plain = make_trainer(value_warmup_updates=1)
    w_bc = make_trainer(bc_cfg(v1_data), v1_data, value_warmup_updates=1)
    w_plain.train_iteration()
    s = w_bc.train_iteration()
    assert s["bc_coef"] == 0.0 and all(torch.equal(a, b) for a, b in zip(params(w_plain), params(w_bc)))


def test_bc_term_reduces_bc_loss_and_logs(v1_data: str) -> None:
    tr = make_trainer(bc_cfg(v1_data, coef=5.0, coef_final=5.0), v1_data, lr=1e-3)
    plain = make_trainer(lr=1e-3)
    probe = BA.BCSampler(tr.bc_sampler.eps, 8, 8, seed=123).sample()

    def probe_loss(t: Trainer) -> float:
        with torch.no_grad():
            return float(BA.bc_term(t.learner, probe)[0])

    before = probe_loss(tr)
    for _ in range(8):
        s = tr.train_iteration()
        plain.train_iteration()
    after, after_plain = probe_loss(tr), probe_loss(plain)
    assert after < before - 1.0 and after < after_plain - 1.0, (before, after, after_plain)
    assert s["bc_coef"] == 5.0 and np.isfinite(s["bc_loss"])
    for name in ("move", "strafe", "turn", "attack", "use", "speed"):
        assert 0.0 <= s[f"bc_acc_{name}"] <= 1.0
    assert s["bc_acc_move"] > 0.9


def test_bc_runs_are_deterministic(v1_data: str) -> None:
    a, b = (make_trainer(bc_cfg(v1_data), v1_data) for _ in range(2))
    for _ in range(2):
        a.train_iteration(), b.train_iteration()
    assert all(torch.equal(x, y) for x, y in zip(params(a), params(b)))


def test_coefficient_schedules(v1_data: str) -> None:
    lin = bc_cfg(v1_data, coef=1.0, coef_final=0.2, decay_updates=4)
    assert [round(BA.coef_at(lin, k), 6) for k in range(6)] == [1.0, 0.8, 0.6, 0.4, 0.2, 0.2]
    const = bc_cfg(v1_data, coef=0.5, coef_final=0.0)
    assert BA.coef_at(const, 1000) == 0.5
    ex = bc_cfg(v1_data, coef=1.0, coef_final=0.2, decay_mode="exp", decay_rate=0.5)
    assert [BA.coef_at(ex, k) for k in range(5)] == [1.0, 0.5, 0.25, 0.2, 0.2]
    # trainer: off during the value warm-up, then the schedule counts policy updates
    tr = make_trainer(lin, v1_data, value_warmup_updates=1)
    got = [round(tr.train_iteration()["bc_coef"], 6) for _ in range(4)]
    assert got == [0.0, 1.0, 0.8, 0.6]


def test_lr_warmup_starts_after_the_value_warmup() -> None:
    tr = make_trainer(value_warmup_updates=2, lr_warmup_updates=4, lr=1e-3)
    lrs = [tr.train_iteration()["lr"] for _ in range(7)]
    np.testing.assert_allclose(lrs, [1e-3, 1e-3, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3])


def test_kl_stop() -> None:
    off = make_trainer(epochs=3)
    assert "kl_stopped" not in off.train_iteration()
    tr = make_trainer(epochs=3, kl_stop=1e-12, lr=1e-2)
    s = tr.train_iteration()
    assert s["kl_stopped"] == 1.0 and 1 <= s["minibatches_stepped"] < 3 and s["kl_stop_value"] > 1e-12
    assert s["kl_stop_minibatch"] == s["minibatches_stepped"] and np.isfinite(s["approx_kl"])
    loose = make_trainer(epochs=3, kl_stop=1e9).train_iteration()
    assert loose["kl_stopped"] == 0.0 and loose["kl_stop_minibatch"] == -1.0 and loose["minibatches_stepped"] == 3


# ---------------------------------------------------------------- refusals and resume
def test_v1_dataset_refused_in_a_v2_run(v1_data: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from train.bc_spec import a0_training_spec_and_signature
    monkeypatch.setenv("DOOMSDAY_OBS_VERSION", "2")
    spec2, _ = a0_training_spec_and_signature()
    with pytest.raises(BA.BCDataRefused, match="observation"):
        BA.load_bc_data((v1_data,), spec2, 2**30, 2)
    monkeypatch.setenv("DOOMSDAY_OBS_VERSION", "1")
    spec1, _ = a0_training_spec_and_signature()
    assert len(BA.load_bc_data((v1_data,), spec1, 2**30, 1)) == 4


def test_size_cap_and_missing_dir_refused(v1_data: str, tmp_path: "os.PathLike[str]") -> None:
    with pytest.raises(BA.BCDataRefused, match="GiB"):
        BA.load_bc_data((v1_data,), SPEC_V1, 1000)
    with pytest.raises(BA.BCDataRefused):
        BA.dataset_hashes((os.path.join(str(tmp_path), "nope"),))


def test_resume_refuses_changed_bc_settings(v1_data: str, tmp_path: "os.PathLike[str]") -> None:
    cfg = bc_cfg(v1_data, data_sha256=BA.dataset_hashes((v1_data,)))
    tr = make_trainer(cfg, v1_data)
    tr.train_iteration()
    path = os.path.join(str(tmp_path), "checkpoint.pt")
    CK.save_checkpoint(path, tr, SIG)
    assert CK.checkpoint_resume_info(path)["bc_aux"] == cfg
    with pytest.raises(CK.CheckpointMismatch, match="BC auxiliary"):
        CK.resume(make_trainer(replace(cfg, coef=0.5), v1_data), path, SIG)
    with pytest.raises(CK.CheckpointMismatch, match="BC auxiliary"):
        CK.resume(make_trainer(), path, SIG)
    same = make_trainer(cfg, v1_data)
    CK.resume(same, path, SIG)
    assert torch.equal(same.bc_sampler.gen.get_state(), tr.bc_sampler.gen.get_state())
    # a checkpoint without BC settings reads as off: resumable by a plain run, not by a BC run
    plain = make_trainer()
    p2 = os.path.join(str(tmp_path), "plain.pt")
    CK.save_checkpoint(p2, plain, SIG)
    blob = torch.load(p2, weights_only=True)
    rc = json.loads(blob["run_config_json"])
    del rc["bc_aux"]
    blob["run_config_json"] = json.dumps(rc)
    torch.save(blob, p2)
    assert CK.checkpoint_resume_info(p2)["bc_aux"] == BCAuxConfig()
    CK.resume(make_trainer(), p2, SIG)
    with pytest.raises(CK.CheckpointMismatch):
        CK.resume(make_trainer(cfg, v1_data), p2, SIG)


def test_run_doom_flags() -> None:
    ns = run_doom.parse_args(["--maps", "generated", "--bc-data", "a,b,a", "--bc-coef", "1.0",
                              "--bc-coef-final", "0.2", "--bc-decay-mode", "exp", "--lr", "1e-4",
                              "--clip", "0.1", "--ppo-epochs", "2", "--lr-warmup-updates", "50",
                              "--kl-stop", "0.03", "--checkpoint-every-updates", "50",
                              "--value-warmup-updates", "200", "--entropy-coef", "0", "--start-stage", "S1"])
    run = run_doom.build_run_config(ns)
    assert (run.ppo.lr, run.ppo.clip, run.ppo.epochs, run.ppo.lr_warmup_updates, run.ppo.kl_stop) == \
        (1e-4, 0.1, 2, 50, 0.03)
    assert run.safety.checkpoint_every_updates == 50
    b = run.bc_aux
    assert b.enabled and len(b.data_dirs) == 2 and (b.coef, b.coef_final, b.decay_mode, b.decay_rate) == \
        (1.0, 0.2, "exp", 0.999) and (b.batch_seqs, b.seq_len) == (16, 64)
    d = run_doom.build_run_config(run_doom.parse_args([]))
    assert d.ppo == replace(PPOConfig(), num_minibatches=1) and not d.bc_aux.enabled
    for bad in (["--bc-coef", "1"], ["--bc-data", "x"], ["--kl-stop", "-1"], ["--bc-decay-rate", "0"],
                ["--bc-data", "x", "--bc-coef", "1", "--bc-coef-final", "0.2"],     # final ignored without decay
                ["--bc-data", "x", "--bc-coef", "0", "--bc-coef-final", "0.5"]):
        with pytest.raises(SystemExit):
            run_doom.parse_args(bad)


# ---------------------------------------------------------------- train/bc.py --use-pos-weight
def _use_logits(lin: torch.nn.Linear, x: torch.Tensor) -> list[torch.Tensor]:
    t, b = x.shape[:2]
    heads = [torch.zeros(t, b, n) for n in SPEC_V1.head_sizes]
    heads[USE_HEAD_INDEX] = lin(x)
    return heads


def test_use_pos_weight_one_is_the_plain_loss_and_eight_raises_recall() -> None:
    g = torch.Generator().manual_seed(0)
    t, b = 50, 40
    press = torch.rand(t, b, generator=g) < 0.1
    x = (press.float() * 1.0 + torch.randn(t, b, generator=g)).unsqueeze(-1)
    labels = torch.zeros(t, b, 6, dtype=torch.long)
    labels[..., USE_HEAD_INDEX] = press.long()
    valid = torch.ones(t, b, dtype=torch.bool)
    valid[-3:] = False
    weights = (1.0,) * 6
    lin0 = torch.nn.Linear(1, 2)
    a = BC.bc_loss(_use_logits(lin0, x), labels, valid, weights)
    c = BC.bc_loss(_use_logits(lin0, x), labels, valid, weights, use_pos_weight=1.0)
    assert torch.equal(a.total, c.total) and a.head_loss_sum == c.head_loss_sum
    lg = _use_logits(lin0, x)
    manual = sum(torch.nn.functional.cross_entropy(h[valid], labels[..., i][valid], reduction="sum")
                 for i, h in enumerate(lg)) / int(valid.sum())
    torch.testing.assert_close(a.total, manual)

    def recall(w: float) -> float:
        torch.manual_seed(0)
        lin = torch.nn.Linear(1, 2)
        opt = torch.optim.Adam(lin.parameters(), lr=0.05)
        for _ in range(300):
            loss = BC.bc_loss(_use_logits(lin, x), labels, valid, weights, use_pos_weight=w)
            opt.zero_grad()
            loss.total.backward()
            opt.step()
        with torch.no_grad():
            r = BC.bc_loss(_use_logits(lin, x), labels, valid, weights)
        return r.use_tp / r.use_pos

    r1, r8 = recall(1.0), recall(8.0)
    assert r8 > r1 + 0.3, (r1, r8)
