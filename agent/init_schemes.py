"""Initialisation variants measured by ADR 0008 (arms B1 and I). Initialisation only:
every parameter stays trained.

GRU update-gate bias (arm B1). The chunk of the GRUCell bias that acts as the
update gate z in h' = (1 - z) * n + z * h is located by test
(tests/test_agent_init_schemes.py measures the cue trace at step 20), not taken from
documentation. A unit with retention time tau keeps a fraction z = 1 - 1/tau per
step, so its bias is logit(1 - 1/tau); taus are spread log-uniformly.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from configs.model_config import ModelConfig

# Index of the update-gate chunk among the GRUCell's three bias chunks; established
# by test_update_gate_chunk_is_the_one_that_keeps_state.
GRU_UPDATE_GATE_CHUNK = 1


def retention_biases(hidden: int, tau_min: float, tau_max: float) -> torch.Tensor:
    assert hidden > 0 and 1.0 < tau_min <= tau_max, "need hidden > 0 and 1 < tau_min <= tau_max"
    if hidden == 1:
        taus = torch.tensor([tau_min])
    else:
        taus = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), hidden))
    keep = 1.0 - 1.0 / taus
    return torch.log(keep / (1.0 - keep))


@torch.no_grad()
def init_gru_update_bias(cell: nn.GRUCell, cfg: ModelConfig, chunk: int = GRU_UPDATE_GATE_CHUNK) -> None:
    h = cell.hidden_size
    b = retention_biases(h, cfg.gru_retention_min, cfg.gru_retention_max)
    cell.bias_ih[chunk * h:(chunk + 1) * h] = 0.0
    cell.bias_hh[chunk * h:(chunk + 1) * h] = b


@torch.no_grad()
def apply_orthogonal_init(model: nn.Module, cfg: ModelConfig, policy_heads: nn.ModuleList,
                          unit_gain_modules: list[nn.Module], gru: nn.GRUCell) -> None:
    """0007-recipe scheme: hidden Linear/Conv1d orthogonal gain sqrt 2, GRU and the
    value/aux heads gain 1, policy heads policy_head_init_gain; biases 0. Attention
    in-projection weights (not Linear modules) keep torch's default."""
    special = {id(m) for m in [*policy_heads, *unit_gain_modules]}
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)) and id(m) not in special:
            nn.init.orthogonal_(m.weight, gain=cfg.orthogonal_hidden_gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    for m in unit_gain_modules:
        nn.init.orthogonal_(m.weight, gain=cfg.orthogonal_unit_gain)
        nn.init.zeros_(m.bias)
    for m in policy_heads:
        nn.init.orthogonal_(m.weight, gain=cfg.policy_head_init_gain)
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(gru.weight_ih, gain=cfg.orthogonal_unit_gain)
    nn.init.orthogonal_(gru.weight_hh, gain=cfg.orthogonal_unit_gain)
    nn.init.zeros_(gru.bias_ih)
    nn.init.zeros_(gru.bias_hh)
