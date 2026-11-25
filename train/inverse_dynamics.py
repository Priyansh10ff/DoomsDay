"""Inverse-dynamics auxiliary loss (idea from Pathak et al. 2017, ICM; written here from
the paper's description). Used ONLY as a representation loss on the model's encoder,
not as a curiosity reward. TRAINING ONLY: the head is owned by the trainer, is never
part of `agent/`, and is never used to act.

A small head takes the encoder features (`A0Agent.encode`) of o_t and o_{t+1} and
predicts the six-head action a_t; the loss is the cross-entropy summed over heads,
averaged over usable pairs. Gradients flow into the model's encoders.

Leak guard: o_{t+1}["prev_action"] IS a_t, so `prev_action` is zeroed in both
observations before encoding; otherwise the head would read the label from the input.

Usable pair (t, t+1) inside a rollout sequence: valid[t], valid[t+1] and
episode_start[t+1] == 0 (o_{t+1} belongs to the same episode). The last step of the
sequence has no o_{t+1} in the buffer and is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from configs.train_config import InvDynConfig

PREV_ACTION_KEY = "prev_action"


class InverseDynamicsHead(nn.Module):
    def __init__(self, feat_dim: int, head_sizes: tuple[int, ...], hidden: int) -> None:
        super().__init__()
        self.feat_dim, self.head_sizes = feat_dim, tuple(head_sizes)
        self.body = nn.Sequential(nn.Linear(2 * feat_dim, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, sum(self.head_sizes))

    def forward(self, x_t: torch.Tensor, x_next: torch.Tensor) -> list[torch.Tensor]:
        """(B, F), (B, F) -> one (B, n_i) logit tensor per action head."""
        assert x_t.shape == x_next.shape and x_t.dim() == 2 and x_t.shape[1] == self.feat_dim, \
            f"inverse dynamics expected two (B, {self.feat_dim}), got {tuple(x_t.shape)} / {tuple(x_next.shape)}"
        z = self.out(self.body(torch.cat([x_t, x_next], dim=-1)))
        return list(z.split(self.head_sizes, dim=-1))


def build_head(feat_dim: int, head_sizes: tuple[int, ...], cfg: InvDynConfig, seed: int,
               device: torch.device | str) -> InverseDynamicsHead:
    """Seeded init from a forked RNG (global torch RNG untouched)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = InverseDynamicsHead(feat_dim, head_sizes, cfg.hidden_dim)
    return head.to(device)


@dataclass
class InvDynTerm:
    """What the PPO update needs: the head, its own optimizer, coefficient and clip."""
    head: InverseDynamicsHead
    optimizer: torch.optim.Optimizer
    coef: float
    max_grad_norm: float


def make_term(head: InverseDynamicsHead, cfg: InvDynConfig) -> InvDynTerm:
    return InvDynTerm(head, torch.optim.Adam(head.parameters(), lr=cfg.lr), cfg.coef, cfg.max_grad_norm)


def pair_mask(episode_start: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(T, B) -> (T-1, B) bool: pair (t, t+1) usable."""
    assert episode_start.shape == valid.shape and episode_start.dim() == 2, \
        f"episode_start / valid expected matching (T, B), got {tuple(episode_start.shape)} / {tuple(valid.shape)}"
    v = valid.bool()
    return v[:-1] & v[1:] & (episode_start[1:] == 0)


def inverse_dynamics_loss(model: nn.Module, head: InverseDynamicsHead, obs: dict[str, torch.Tensor],
                          actions: torch.Tensor, episode_start: torch.Tensor,
                          valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """obs tensors (T, B, ...) (the policy input), actions (T, B, K) long,
    episode_start / valid (T, B). Returns (loss, stats)."""
    t, b = episode_start.shape
    assert actions.shape == (t, b, len(head.head_sizes)), \
        f"actions expected ({t}, {b}, {len(head.head_sizes)}), got {tuple(actions.shape)}"
    flat = {k: v.reshape(t * b, *v.shape[2:]) for k, v in obs.items()}
    flat[PREV_ACTION_KEY] = torch.zeros_like(flat[PREV_ACTION_KEY])   # leak guard (module docstring)
    x = model.encode(flat).reshape(t, b, -1)
    m = pair_mask(episode_start, valid)
    n_ok = int(m.sum())
    stats: dict[str, float] = {"inv_dyn_pairs_frac": n_ok / max(m.numel(), 1)}
    if n_ok == 0:
        zero = x.sum() * 0.0   # keeps the graph; contributes nothing
        stats["inv_dyn_loss"] = 0.0
        for i in range(len(head.head_sizes)):   # same keys every minibatch (averaged by ppo_update)
            stats[f"inv_dyn_acc_head{i}"] = 0.0
        return zero, stats
    logits = head(x[:-1][m], x[1:][m])
    a = actions[:-1][m]
    loss = x.new_zeros(())
    for i, lg in enumerate(logits):
        ce = F.cross_entropy(lg, a[:, i])
        loss = loss + ce
        with torch.no_grad():
            stats[f"inv_dyn_acc_head{i}"] = float((lg.argmax(-1) == a[:, i]).float().mean())
    stats["inv_dyn_loss"] = float(loss.detach())
    return loss, stats
