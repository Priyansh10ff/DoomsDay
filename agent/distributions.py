"""Factorized categorical over independent action heads (ADR 0002).

Joint log-probability and entropy are sums over heads. Sampling takes an explicit
torch.Generator so no global RNG is used.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class FactorizedCategorical:
    def __init__(self, logits: list[torch.Tensor]) -> None:
        assert len(logits) > 0, "need at least one head"
        lead = logits[0].shape[:-1]
        for lg in logits:
            assert lg.shape[:-1] == lead, f"head leading shapes differ: {lg.shape[:-1]} vs {lead}"
        self.logits = logits
        self.log_probs = [F.log_softmax(lg, dim=-1) for lg in logits]
        self.lead_shape = lead

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert actions.shape == (*self.lead_shape, len(self.logits)), \
            f"actions expected {(*self.lead_shape, len(self.logits))}, got {tuple(actions.shape)}"
        a = actions.long()
        total = torch.zeros(self.lead_shape, dtype=self.log_probs[0].dtype, device=a.device)
        for i, lp in enumerate(self.log_probs):
            total = total + lp.gather(-1, a[..., i:i + 1]).squeeze(-1)
        return total

    def entropy_per_head(self) -> torch.Tensor:
        """(..., n_heads)"""
        ents = [-(lp.exp() * lp).sum(-1) for lp in self.log_probs]
        return torch.stack(ents, dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.entropy_per_head().sum(-1)

    def sample(self, generator: torch.Generator | None) -> torch.Tensor:
        outs = []
        for lp in self.log_probs:
            p = lp.exp().reshape(-1, lp.shape[-1])
            if generator is not None:
                p = p.to(generator.device)  # generator and probs must share a device
            idx = torch.multinomial(p, 1, generator=generator).squeeze(-1).to(lp.device)
            outs.append(idx.reshape(self.lead_shape))
        return torch.stack(outs, dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.stack([lg.argmax(-1) for lg in self.logits], dim=-1)
