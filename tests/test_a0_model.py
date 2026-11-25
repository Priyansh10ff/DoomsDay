"""A0 agent: shapes from spaces, sequence == stepwise, hidden reset, determinism, inference."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from a0_helpers import DOOM_HEADS, doom_like_spaces, random_obs
from agent.distributions import FactorizedCategorical
from agent.inference import PolicyRunner
from agent.model import A0Agent, build_agent, count_parameters
from agent.obs_spec import ObsSpec, spec_from_spaces
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig

CFG = ModelConfig()


def make(n_types: int = 14, with_screen: bool = False, seed: int = 0,
         ray_bands: int = 3) -> tuple[gym.spaces.Dict, ObsSpec, A0Agent]:
    obs_space, act_space = doom_like_spaces(n_types=n_types, with_screen=with_screen, ray_bands=ray_bands)
    spec = spec_from_spaces(obs_space, act_space)
    return obs_space, spec, build_agent(spec, CFG, seed=seed)


@pytest.mark.parametrize("n_types,with_screen,ray_bands", [(13, True, 3), (14, False, 3), (14, False, 4)])
def test_spec_and_output_shapes(n_types: int, with_screen: bool, ray_bands: int) -> None:
    obs_space, spec, model = make(n_types, with_screen, ray_bands=ray_bands)
    assert spec.n_entity_types == n_types
    assert spec.head_sizes == DOOM_HEADS
    assert model.entities.type_embed.num_embeddings == n_types
    rng = np.random.default_rng(0)
    b = 5
    obs = to_tensors(random_obs(obs_space, b, rng))
    assert "screen" not in obs
    out, h = model.step(obs, model.initial_state(b), torch.zeros(b))
    assert h.shape == (b, CFG.core_hidden)
    assert [lg.shape for lg in out.logits] == [(b, n) for n in DOOM_HEADS]
    assert out.value_ext.shape == out.value_int.shape == out.aux_damage.shape == out.aux_monsters.shape == (b,)
    assert out.aux_keys.shape == (b, CFG.n_key_bits)
    t = 7
    seq = to_tensors(random_obs(obs_space, b, rng, lead=(t,)))
    out_s, h_s = model.forward_sequence(seq, model.initial_state(b), torch.zeros(t, b))
    assert [lg.shape for lg in out_s.logits] == [(t, b, n) for n in DOOM_HEADS]
    assert out_s.value_ext.shape == (t, b) and h_s.shape == (b, CFG.core_hidden)


def test_parameter_count_under_one_million() -> None:
    _, _, model = make(14, False)
    n = count_parameters(model)
    print(f"A0 parameter count (rays 3x80, 24x8 entities, 14 types, self 28, heads {DOOM_HEADS}): {n}")
    assert n < 1_000_000


def test_forward_sequence_matches_stepwise_with_resets() -> None:
    obs_space, _, model = make()
    rng = np.random.default_rng(1)
    t, b = 9, 4
    seq = to_tensors(random_obs(obs_space, b, rng, lead=(t,)))
    starts = torch.zeros(t, b)
    starts[0] = 1
    starts[3, 1] = 1
    starts[6, 2] = 1
    h0 = torch.randn(b, CFG.core_hidden, generator=torch.Generator().manual_seed(0))
    out_s, h_end = model.forward_sequence(seq, h0, starts)
    h = h0
    for i in range(t):
        o, h = model.step({k: v[i] for k, v in seq.items()}, h, starts[i])
        torch.testing.assert_close(o.value_ext, out_s.value_ext[i], atol=1e-5, rtol=1e-5)
        for a, c in zip(o.logits, out_s.logits):
            torch.testing.assert_close(a, c[i], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h, h_end, atol=1e-5, rtol=1e-5)


def test_hidden_reset_only_for_ended_envs() -> None:
    obs_space, _, model = make()
    rng = np.random.default_rng(2)
    b = 4
    obs = to_tensors(random_obs(obs_space, b, rng))
    h_prev = torch.randn(b, CFG.core_hidden, generator=torch.Generator().manual_seed(1))
    starts = torch.tensor([0.0, 1.0, 0.0, 1.0])
    _, h_mixed = model.step(obs, h_prev, starts)
    _, h_keep = model.step(obs, h_prev, torch.zeros(b))
    _, h_fresh = model.step(obs, model.initial_state(b), torch.zeros(b))
    for i in (0, 2):  # continuing envs: state carried
        torch.testing.assert_close(h_mixed[i], h_keep[i])
        assert not torch.allclose(h_mixed[i], h_fresh[i])
    for i in (1, 3):  # ended envs: same as a fresh start
        torch.testing.assert_close(h_mixed[i], h_fresh[i])


def test_memoryless_ablation_zeroes_state_every_step() -> None:
    obs_space, act_space = doom_like_spaces()
    spec = spec_from_spaces(obs_space, act_space)
    model = build_agent(spec, ModelConfig(recurrent=False), seed=0)
    obs = to_tensors(random_obs(obs_space, 3, np.random.default_rng(3)))
    _, h1 = model.step(obs, torch.randn(3, CFG.core_hidden), torch.zeros(3))
    _, h2 = model.step(obs, model.initial_state(3), torch.zeros(3))
    torch.testing.assert_close(h1, h2)


def test_build_agent_deterministic_and_does_not_touch_global_rng() -> None:
    _, _, m1 = make(seed=5)
    state = torch.random.get_rng_state()
    _, _, m2 = make(seed=5)
    assert torch.equal(state, torch.random.get_rng_state())
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p, q)
    _, _, m3 = make(seed=6)
    assert not all(torch.equal(p, q) for p, q in zip(m1.parameters(), m3.parameters()))


def test_factorized_sampling_seeded_and_logprob_sums_heads() -> None:
    g = torch.Generator().manual_seed(0)
    logits = [torch.randn(4, n, generator=g) for n in DOOM_HEADS]
    dist = FactorizedCategorical(logits)
    a1 = dist.sample(torch.Generator().manual_seed(3))
    a2 = dist.sample(torch.Generator().manual_seed(3))
    assert torch.equal(a1, a2) and a1.shape == (4, 6)
    manual = sum(torch.log_softmax(lg, -1).gather(-1, a1[:, i:i + 1]).squeeze(-1) for i, lg in enumerate(logits))
    torch.testing.assert_close(dist.log_prob(a1), manual)
    assert dist.entropy_per_head().shape == (4, 6)


def test_policy_runner_argmax_is_deterministic_and_resets() -> None:
    obs_space, _, model = make()
    obs = random_obs(obs_space, 3, np.random.default_rng(4))
    r1 = PolicyRunner(model, 3, argmax_heads=(True,) * 6, seed=0)
    r2 = PolicyRunner(model, 3, argmax_heads=(True,) * 6, seed=123)
    a1 = r1.act(obs, np.ones(3, np.float32))
    a2 = r2.act(obs, np.ones(3, np.float32))
    assert np.array_equal(a1, a2) and a1.shape == (3, 6) and a1.dtype == np.int64
    s1 = PolicyRunner(model, 3, argmax_heads=(False,) * 6, seed=9).act(obs, np.ones(3, np.float32))
    s2 = PolicyRunner(model, 3, argmax_heads=(False,) * 6, seed=9).act(obs, np.ones(3, np.float32))
    assert np.array_equal(s1, s2)
