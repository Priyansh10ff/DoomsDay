"""Generalized advantage estimation with termination and truncation kept apart.

Inputs are (T, N) tensors:
  terminated[t]      episode ended at transition t by death/exit: next value is 0.
  truncated[t]       episode ended by time limit: next value is bootstrap_value[t].
  bootstrap_value[t] V(true next state) for truncated steps (ignored elsewhere).
  valid[t]           False drops transition t: its advantage is 0 and it does not
                     feed the step before it. Used for a time-out whose final
                     observation is stale; step t-1 then bootstraps from values[t],
                     the value of a real state computed during collection.
The lambda-chain is cut at every episode end.
"""
from __future__ import annotations

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, last_value: torch.Tensor,
                terminated: torch.Tensor, truncated: torch.Tensor, bootstrap_value: torch.Tensor,
                valid: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    t_len, n = rewards.shape
    for name, x in (("values", values), ("terminated", terminated), ("truncated", truncated),
                    ("bootstrap_value", bootstrap_value), ("valid", valid)):
        assert x.shape == (t_len, n), f"{name} expected {(t_len, n)}, got {tuple(x.shape)}"
    assert last_value.shape == (n,), f"last_value expected ({n},), got {tuple(last_value.shape)}"
    term, trunc, ok = terminated.bool(), truncated.bool(), valid.bool()
    zero = torch.zeros(n, dtype=values.dtype, device=values.device)
    adv = torch.zeros_like(values)
    next_adv = zero
    next_v = last_value
    for t in reversed(range(t_len)):
        if t < t_len - 1:
            next_v, next_adv = values[t + 1], adv[t + 1]
        ended = term[t] | trunc[t]
        nv = torch.where(term[t], zero, torch.where(trunc[t], bootstrap_value[t], next_v))
        na = torch.where(ended, zero, next_adv)
        delta = rewards[t] + gamma * nv - values[t]
        adv[t] = torch.where(ok[t], delta + gamma * lam * na, zero)
    return adv, adv + values
