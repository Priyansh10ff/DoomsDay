"""BC previous-action dropout (train/bc.py --prev-action-dropout): P=0 reproduces the
pre-change training bit for bit (golden tests/fixtures/bc_epoch_golden.torch, captured from
the code before the change), P=1 hides prev_action completely, 0<P<1 is seeded and
leaves validation untouched. No engine is started."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from a0_helpers import ROOT, TORCH_THREADS
from agent.model import build_agent
from configs.bc_config import BCConfig
from test_bc import SMALL, SPEC, episode
from train import bc as BC
from train.bc_data import ChunkStream, Episode

GOLDEN = os.path.join(ROOT, "tests", "fixtures", "bc_epoch_golden.torch")
LENGTHS = (17, 9, 30, 12, 5)


def _episodes() -> list[Episode]:
    rng = np.random.default_rng(5)
    return [episode(n, rng, f"DD{i:04d}") for i, n in enumerate(LENGTHS)]


def _two_epochs(eps: list[Episode], p: float, drop_seed: int = 1, **kw: object) -> tuple[float, float, dict]:
    """Same setup as the golden capture; returns both epochs' summed loss and the weights."""
    model = build_agent(SPEC, SMALL, seed=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    stream = ChunkStream(eps, 8, 3, torch.Generator().manual_seed(0), shuffle=True)
    dg = torch.Generator().manual_seed(drop_seed)
    a1, _ = BC.run_epoch(model, stream, (1.0,) * 6, opt, 1.0, prev_action_dropout=p, dropout_generator=dg, **kw)
    a2, _ = BC.run_epoch(model, stream, (1.0,) * 6, opt, 1.0, prev_action_dropout=p, dropout_generator=dg, **kw)
    return a1.weighted_loss_sum, a2.weighted_loss_sum, model.state_dict()


def test_reset_value_is_what_the_env_gives_at_episode_start() -> None:
    assert BC.NO_PREV_ACTION_VALUE == 0
    with open(os.path.join(ROOT, "env", "doom_env.py"), encoding="utf-8") as f:
        src = f.read()
    assert "self._prev_action = np.zeros(6, dtype=np.int64)" in src, \
        "env reset value of prev_action changed: update NO_PREV_ACTION_VALUE"


@pytest.fixture
def golden_threads():
    """The golden was captured with TORCH_THREADS intra-op threads; another test (train.bc.main)
    may have changed the count, which changes float reduction order in the backward pass."""
    before = torch.get_num_threads()
    torch.set_num_threads(TORCH_THREADS)
    yield
    torch.set_num_threads(before)


def test_p0_is_bit_identical_to_the_pre_change_training(golden_threads: None) -> None:
    g = torch.load(GOLDEN, weights_only=False)
    l1, l2, state = _two_epochs(_episodes(), 0.0)
    assert (l1, l2) == (g["loss1"], g["loss2"])
    for k, v in g["state"].items():
        assert torch.equal(state[k], v), k
    # defaults (no dropout args at all) give the same
    model = build_agent(SPEC, SMALL, seed=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    acc, _ = BC.run_epoch(model, ChunkStream(_episodes(), 8, 3, torch.Generator().manual_seed(0)), (1.0,) * 6, opt, 1.0)
    assert acc.weighted_loss_sum == g["loss1"]


def test_drop_prev_action_rows_and_rate() -> None:
    prev = torch.randint(1, 3, (50, 40, 6), generator=torch.Generator().manual_seed(0))
    out = BC.drop_prev_action(prev, 0.3, torch.Generator().manual_seed(1))
    assert out.shape == prev.shape and out.dtype == prev.dtype
    dropped = (out == BC.NO_PREV_ACTION_VALUE).all(-1)
    kept = (out == prev).all(-1)
    assert bool((dropped ^ kept).all()), "each (t, b) row is either whole or wholly reset"
    assert 0.25 < float(dropped.float().mean()) < 0.35
    again = BC.drop_prev_action(prev, 0.3, torch.Generator().manual_seed(1))
    assert torch.equal(out, again)
    assert bool((BC.drop_prev_action(prev, 1.0, torch.Generator()) == BC.NO_PREV_ACTION_VALUE).all())


def test_p1_model_never_sees_prev_action() -> None:
    eps = _episodes()
    rng = np.random.default_rng(9)
    scrambled = [Episode(dict(e.obs, prev_action=rng.integers(0, 3, e.obs["prev_action"].shape).astype(np.int64)),
                         e.labels, e.map_id, e.stage, e.file) for e in eps]
    zeroed = [Episode(dict(e.obs, prev_action=np.zeros_like(e.obs["prev_action"])), e.labels, e.map_id, e.stage,
                      e.file) for e in eps]
    a = _two_epochs(eps, 1.0)
    b = _two_epochs(scrambled, 1.0, drop_seed=77)
    c = _two_epochs(zeroed, 0.0)            # P=1 equals training on reset-valued prev_action
    assert a[:2] == b[:2] == c[:2]
    for k in a[2]:
        assert torch.equal(a[2][k], b[2][k]) and torch.equal(a[2][k], c[2][k]), k
    assert _two_epochs(eps, 0.0)[:2] != a[:2], "prev_action does matter without dropout"


def test_partial_dropout_is_seeded_and_changes_training() -> None:
    eps = _episodes()
    a, b = _two_epochs(eps, 0.5, drop_seed=4), _two_epochs(eps, 0.5, drop_seed=4)
    assert a[:2] == b[:2]
    assert _two_epochs(eps, 0.5, drop_seed=5)[:2] != a[:2]
    assert _two_epochs(eps, 0.0)[:2] != a[:2]


def test_validation_pass_never_drops() -> None:
    eps = _episodes()
    model = build_agent(SPEC, SMALL, seed=3)

    def val(p: float) -> float:
        stream = ChunkStream(eps, 8, 3, torch.Generator().manual_seed(0), shuffle=False)
        acc, _ = BC.run_epoch(model, stream, (1.0,) * 6, None, 1.0, prev_action_dropout=p,
                              dropout_generator=torch.Generator().manual_seed(0))
        return acc.weighted_loss_sum
    assert val(0.0) == val(1.0)
    with pytest.raises(ValueError, match="dropout_generator"):
        BC.run_epoch(model, ChunkStream(eps, 8, 3, torch.Generator()), (1.0,) * 6,
                     torch.optim.Adam(model.parameters()), 1.0, prev_action_dropout=0.5)


def test_flag_config_and_logging(tmp_path: "os.PathLike[str]") -> None:
    ns = BC.parse_args(["--data", "x", "--out", "y", "--prev-action-dropout", "0.3"])
    assert BC.config_from_args(ns).prev_action_dropout == 0.3
    assert BC.config_from_args(BC.parse_args(["--data", "x", "--out", "y"])).prev_action_dropout == 0.0
    with pytest.raises(ValueError):
        BCConfig(prev_action_dropout=1.5)
    cfg = BCConfig(epochs=1, seq_len=4, batch_seqs=2, lr=1e-3, seed=0, prev_action_dropout=0.3)
    logs: list[str] = []
    out = os.path.join(str(tmp_path), "run")
    summary = BC.train_bc(_episodes()[:3], _episodes()[3:], SPEC, SMALL, cfg, "cpu", out, ("sig",),
                          {"kind": "test"}, log=logs.append)
    assert summary["prev_action_dropout"] == 0.3
    assert any("prev_action dropout P = 0.3" in m for m in logs)
    with open(os.path.join(out, BC.METRICS_NAME), encoding="utf-8") as f:
        assert json.loads(f.readline())["prev_action_dropout"] == 0.3
