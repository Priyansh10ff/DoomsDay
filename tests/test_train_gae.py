"""GAE against hand-computed values: termination and truncation tested separately."""
import pytest
import torch

import a0_helpers  # noqa: F401
from train.gae import compute_gae

G, L = 0.9, 0.8


def run(rewards: list[float], values: list[float], last: float, term: list[int] | None = None,
        trunc: list[int] | None = None, boot: list[float] | None = None,
        valid: list[int] | None = None) -> tuple[list[float], list[float]]:
    r = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
    v = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    t = r.shape[0]
    z = torch.zeros(t, 1)
    term = torch.tensor(term or [0] * t, dtype=torch.bool).reshape(-1, 1)
    trunc = torch.tensor(trunc or [0] * t, dtype=torch.bool).reshape(-1, 1)
    boot = torch.tensor(boot, dtype=torch.float32).reshape(-1, 1) if boot else z
    valid = torch.tensor(valid or [1] * t, dtype=torch.bool).reshape(-1, 1)
    adv, ret = compute_gae(r, v, torch.tensor([last], dtype=torch.float32), term, trunc, boot, valid, G, L)
    return adv[:, 0].tolist(), ret[:, 0].tolist()


def test_no_episode_end() -> None:
    adv, ret = run([1.0, 2.0], [0.5, 1.0], last=3.0)
    d1 = 2.0 + G * 3.0 - 1.0
    d0 = 1.0 + G * 1.0 - 0.5
    assert adv == pytest.approx([d0 + G * L * d1, d1])
    assert ret == pytest.approx([d0 + G * L * d1 + 0.5, d1 + 1.0])


def test_termination_mid_rollout_no_bootstrap_and_chain_cut() -> None:
    # step 1 terminates; step 2 is the first step of a new episode
    adv, _ = run([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], last=4.0, term=[0, 1, 0])
    d2 = 3.0 + G * 4.0 - 1.5
    d1 = 2.0 - 1.0                       # no bootstrap, no chain
    d0 = 1.0 + G * 1.0 - 0.5
    assert adv == pytest.approx([d0 + G * L * d1, d1, d2])


def test_truncation_mid_rollout_bootstraps_from_final_value() -> None:
    adv, _ = run([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], last=4.0, trunc=[0, 1, 0], boot=[0.0, 7.0, 0.0])
    d2 = 3.0 + G * 4.0 - 1.5
    d1 = 2.0 + G * 7.0 - 1.0             # bootstrap from V(true final obs), chain cut
    d0 = 1.0 + G * 1.0 - 0.5
    assert adv == pytest.approx([d0 + G * L * d1, d1, d2])


def test_termination_and_truncation_give_different_advantages() -> None:
    a_term, _ = run([1.0, 2.0], [0.5, 1.0], last=4.0, term=[0, 1], boot=[0.0, 7.0])
    a_trunc, _ = run([1.0, 2.0], [0.5, 1.0], last=4.0, trunc=[0, 1], boot=[0.0, 7.0])
    assert a_term[1] == pytest.approx(2.0 - 1.0)
    assert a_trunc[1] == pytest.approx(2.0 + G * 7.0 - 1.0)
    assert a_term != pytest.approx(a_trunc)


def test_truncation_at_last_rollout_step_ignores_last_value() -> None:
    adv, _ = run([1.0], [0.5], last=100.0, trunc=[1], boot=[2.0])
    assert adv == pytest.approx([1.0 + G * 2.0 - 0.5])


def test_stale_timeout_dropped_and_previous_step_bootstraps_from_real_state() -> None:
    # step 2 is a time-out whose final obs is stale -> valid False.
    r, v = [1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]
    adv, ret = run(r, v, last=5.0, trunc=[0, 0, 1, 0], boot=[0, 0, 999.0, 0], valid=[1, 1, 0, 1])
    assert adv[2] == 0.0
    assert ret[2] == pytest.approx(v[2])
    # exactly A_{t-1} = r_{t-1} + gamma V(s_t) - V(s_{t-1})
    assert adv[1] == pytest.approx(r[1] + G * v[2] - v[1])
    d0 = r[0] + G * v[1] - v[0]
    assert adv[0] == pytest.approx(d0 + G * L * adv[1])
    assert adv[3] == pytest.approx(r[3] + G * 5.0 - v[3])


def test_batched_envs_independent() -> None:
    r = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    v = torch.zeros(2, 2)
    term = torch.tensor([[False, True], [False, False]])
    z = torch.zeros(2, 2)
    adv, _ = compute_gae(r, v, torch.tensor([10.0, 10.0]), term, torch.zeros(2, 2, dtype=torch.bool), z,
                         torch.ones(2, 2, dtype=torch.bool), G, L)
    assert adv[0, 1].item() == pytest.approx(1.0)
    assert adv[1, 1].item() == pytest.approx(1.0 + G * 10.0)
    assert adv[0, 0].item() == pytest.approx(1.0 + G * L * (1.0 + G * 10.0))
