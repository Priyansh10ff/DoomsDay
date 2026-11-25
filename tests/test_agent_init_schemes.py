"""ADR 0008 Decision 3 (arm B1) and point 5 (arm I): initialisation variants.

The update-gate chunk of the GRUCell bias is located BY TEST: the debugger's metric
(cue trace in the hidden state at step 20, at init) must rise by orders of magnitude
when the retention bias is put on the chosen chunk, and not on the others."""
import math

import pytest
import torch

import a0_helpers  # noqa: F401
from agent import init_schemes as I
from agent.model import build_agent
from agent.obs_spec import spec_from_spaces
from configs.model_config import ModelConfig
from train.toy_memory import ToyMemoryEnv, cue_trace, toy_model_config


def test_retention_biases_span_the_configured_time_scales() -> None:
    b = I.retention_biases(64, 2.0, 256.0)
    keep = torch.sigmoid(b)
    taus = 1.0 / (1.0 - keep)
    assert taus.min().item() == pytest.approx(2.0, rel=1e-4)
    assert taus.max().item() == pytest.approx(256.0, rel=1e-3)
    assert torch.all(taus[1:] > taus[:-1])
    assert b[0].item() == pytest.approx(0.0, abs=1e-6)            # tau 2 -> z 0.5
    assert b[-1].item() == pytest.approx(math.log(255.0), rel=1e-4)


def _trace_with_chunk(chunk: int, monkeypatch: pytest.MonkeyPatch) -> float:
    import agent.model as M
    orig = I.init_gru_update_bias
    monkeypatch.setattr(M, "init_gru_update_bias", lambda cell, cfg: orig(cell, cfg, chunk))
    return cue_trace(toy_model_config(True, gru_update_bias_init=True), seed=0, gap=20)


def test_update_gate_chunk_is_the_one_that_keeps_state(monkeypatch: pytest.MonkeyPatch) -> None:
    base = cue_trace(toy_model_config(True), seed=0, gap=20)
    traces = {c: _trace_with_chunk(c, monkeypatch) for c in range(3)}
    print(f"cue trace at step 20: base {base:.3g}, bias on chunk {traces}")
    assert base < 1e-4
    keep = traces[I.GRU_UPDATE_GATE_CHUNK]
    assert keep > 100 * base, "retention bias must raise the step-20 cue trace by orders of magnitude"
    for c, t in traces.items():
        if c != I.GRU_UPDATE_GATE_CHUNK:
            assert keep > 10 * t


def test_bias_init_applied_only_when_enabled_and_after_orthogonal() -> None:
    env = ToyMemoryEnv()
    spec = spec_from_spaces(env.observation_space, env.action_space)
    h = 64
    sl = slice(I.GRU_UPDATE_GATE_CHUNK * h, (I.GRU_UPDATE_GATE_CHUNK + 1) * h)
    expected = I.retention_biases(h, 2.0, 256.0)
    for scheme in ("default", "orthogonal"):
        m = build_agent(spec, toy_model_config(True, init_scheme=scheme, gru_update_bias_init=True), 0)
        torch.testing.assert_close(m.core.bias_hh.detach()[sl], expected)
        assert torch.all(m.core.bias_ih.detach()[sl] == 0)
    plain = build_agent(spec, toy_model_config(True), 0)
    assert not torch.allclose(plain.core.bias_hh.detach()[sl], expected)


def test_orthogonal_scheme() -> None:
    env = ToyMemoryEnv()
    spec = spec_from_spaces(env.observation_space, env.action_space)
    m = build_agent(spec, toy_model_config(True, init_scheme="orthogonal"), 0)
    assert torch.all(m.core.bias_hh == 0) and torch.all(m.core.bias_ih == 0)
    assert torch.all(m.fusion[0].bias == 0) and torch.all(m.value_ext.bias == 0)
    w = m.value_ext.weight.detach()
    assert w.norm().item() == pytest.approx(1.0, rel=1e-4)                # gain 1, one row
    w = m.policy_heads[0].weight.detach()
    torch.testing.assert_close(w @ w.T, (0.01 ** 2) * torch.eye(w.shape[0]), atol=1e-6, rtol=1e-4)
    with pytest.raises(ValueError):
        build_agent(spec, ModelConfig(init_scheme="nope"), 0)
