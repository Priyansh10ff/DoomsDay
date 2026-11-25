"""--device hybrid: rollout inference on CPU, PPO update on CUDA (train/trainer.py,
train/device.py). The CUDA tests run whenever torch sees a GPU. No engine."""
from __future__ import annotations

import copy
import os
from dataclasses import fields, replace

import pytest
import torch

from a0_helpers import FakeDoomEnv
from configs.train_config import PPOConfig, RunConfig
from test_train_ppo import SMALL
from train import checkpoint as CK
from train import run_doom
from train.device import DeviceUnavailable, device_roles, resolve_device
from train.rollout import Rollout, rollout_to
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SIG = (("fake", True),)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# float tolerance between a CPU and a CUDA update of the same rollout (2 epochs x 2 minibatches)
ATOL, RTOL = 1e-5, 1e-4
SYNC_RATIO_TOL = 1e-5     # first minibatch's |ratio - 1| when the actor is in sync


def trainer(device: str, n_envs: int = 4, t_len: int = 10, seed: int = 0) -> Trainer:
    venv = SyncVecEnv([lambda: FakeDoomEnv(7) for _ in range(n_envs)])
    run = replace(RunConfig(), seed=seed, device=device, num_workers=1, envs_per_worker=n_envs,
                  ppo=PPOConfig(rollout_len=t_len, num_minibatches=2, epochs=2))
    return Trainer(venv, SMALL, run)


def rollout_equal(a: Rollout, b: Rollout) -> bool:
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, dict):
            if any(not torch.equal(x[k], y[k].to(x[k].device)) for k in x):
                return False
        elif not torch.equal(x, y.to(x.device)):
            return False
    return True


def test_device_choice_and_roles() -> None:
    assert resolve_device("hybrid", lambda: True) == "hybrid"
    with pytest.raises(DeviceUnavailable, match="--device hybrid"):
        resolve_device("hybrid", lambda: False)
    assert device_roles("hybrid") == ("cpu", "cuda") and device_roles("cpu") == ("cpu", "cpu")
    assert run_doom.parse_args(["--device", "hybrid"]).device == "hybrid"
    assert run_doom.parse_args([]).device == "cpu"
    cpu = trainer("cpu")
    assert cpu.learner is cpu.model and not cpu.split_devices


@CUDA
def test_rollout_to_moves_every_tensor() -> None:
    ro = trainer("cpu").collect()
    g = rollout_to(ro, "cuda")
    assert all((v.is_cuda if not isinstance(v, dict) else all(t.is_cuda for t in v.values()))
               for v in (getattr(g, f.name) for f in fields(g)))
    assert rollout_equal(ro, g)


@CUDA
def test_hybrid_update_matches_cpu_update_on_fixed_data() -> None:
    c, h = trainer("cpu", seed=5), trainer("hybrid", seed=5)
    assert h.split_devices and h.device.type == "cpu" and next(h.learner.parameters()).is_cuda
    assert next(h.model.parameters()).device.type == "cpu"
    ro_c, ro_h = c.collect(), h.collect()
    assert rollout_equal(ro_c, ro_h)            # same CPU inference: same rollout, same hidden states
    assert ro_h.h0.device.type == "cpu"
    fixed = copy.deepcopy(ro_c)
    sc, sh = c.update(ro_c), h.update(fixed)
    for k in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "return_scale"):
        assert sh[k] == pytest.approx(sc[k], rel=RTOL, abs=ATOL), k
    for (name, pc), pl, pa in zip(c.model.named_parameters(), h.learner.parameters(), h.model.parameters()):
        assert torch.allclose(pl.detach().cpu(), pc.detach(), atol=ATOL, rtol=RTOL), name
        assert torch.equal(pa.detach(), pl.detach().cpu()), f"actor not synced: {name}"
    for _ in range(3):                           # a missed sync would show as ratio != 1 at the first minibatch
        st = h.train_iteration()
        assert st["first_minibatch_ratio_max_dev"] < SYNC_RATIO_TOL


@CUDA
def test_hybrid_checkpoints_load_into_cpu_and_back(tmp_path: "os.PathLike[str]") -> None:
    h = trainer("hybrid")
    h.train_iteration()
    p1 = os.path.join(str(tmp_path), "h.pt")
    CK.save_checkpoint(p1, h, SIG)
    c = trainer("cpu")
    CK.resume(c, p1, SIG)
    for x, y in zip(h.learner.state_dict().values(), c.model.state_dict().values()):
        assert torch.equal(x.cpu(), y)
    c.train_iteration()
    p2 = os.path.join(str(tmp_path), "c.pt")
    CK.save_checkpoint(p2, c, SIG)
    h2 = trainer("hybrid")
    CK.resume(h2, p2, SIG)
    for x, y, z in zip(c.model.state_dict().values(), h2.learner.state_dict().values(), h2.model.state_dict().values()):
        assert torch.equal(x, y.cpu()) and torch.equal(x, z)    # learner loaded, actor synced
    st = h2.train_iteration()                   # optimizer state moved to CUDA with the params
    assert st["first_minibatch_ratio_max_dev"] < SYNC_RATIO_TOL
