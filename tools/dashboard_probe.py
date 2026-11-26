"""Read-only probe around a trained A0Agent for the demo dashboard.

`ModelProbe` is handed to the unchanged `evaluate.policies.NetworkPolicy` /
`agent.inference.PolicyRunner` in place of the model. Its `step` calls the model's own
`step` with the observation tensors exactly as given and returns the model's own output,
so actions come from the same forward pass, distribution and seeded sampler as in
`evaluate.record`. Afterwards it copies, for display only: the input arrays, the value,
the per-head probabilities and the optional goal head.

Entity attention: a forward hook on the entity encoder's pooling MultiheadAttention
(`model.entities.pool`) receives the exact (query, tokens, tokens, key_padding_mask) of
that call and calls THE SAME module again with `need_weights=True` (head-averaged
weights). The recomputed pooled vector must equal the model's own output to within
`ATTN_MATCH_TOL`; a decision that fails the check shows no weights (counted in
`attn_mismatches`; `strict=True` raises instead). Nothing in `agent/` is edited; the hook
returns None, so the model's output is untouched.

Temperature is applied by the evaluation path itself (`NetworkPolicy(temperature=T)` ->
`PolicyRunner` -> `agent.inference.tempered_logits`); the probe never changes the output
object. `display_temperature` is used only to show the probabilities that were sampled
from (softmax(logits / T)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from agent.inference import DEFAULT_TEMPERATURE, check_temperature, tempered_logits
from agent.model import A0Agent, AgentOutput

ATTN_MATCH_TOL = 1e-6


@dataclass
class StepCapture:
    """One decision, batch row 0, as numpy (display only)."""
    obs: dict[str, np.ndarray]               # model input tensors as given (batch dim removed)
    value: float                             # value_ext
    probs: list[np.ndarray]                  # per-head probabilities of the SAMPLED distribution
    attn: np.ndarray | None                  # (N,) pooling weight per entity row (0 on padding)
    attn_empty: float | None                 # weight on the learned "no entity" token
    attn_max_abs_diff: float | None          # |recomputed pooled - model pooled|_inf
    goal_logits: np.ndarray | None           # (n_goals,) when the model has a goal head


class ModelProbe:
    """Duck-typed stand-in for A0Agent inside PolicyRunner (uses .spec, .initial_state, .step)."""

    def __init__(self, model: A0Agent, display_temperature: float = DEFAULT_TEMPERATURE,
                 strict: bool = False) -> None:
        self.model, self.spec, self.strict = model, model.spec, bool(strict)
        self.attn_mismatches = 0
        self.display_temperature = check_temperature(display_temperature)
        self.last: StepCapture | None = None
        self.max_attn_diff = 0.0
        self._pool: nn.MultiheadAttention = model.entities.pool
        self._in_hook = False
        self._attn: tuple[np.ndarray, float, float] | None = None
        self._handle = self._pool.register_forward_hook(self._pool_hook, with_kwargs=True)

    def close(self) -> None:
        self._handle.remove()

    # ------------------------------------------------------------ runner interface
    def initial_state(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return self.model.initial_state(batch, device)

    def step(self, obs: dict[str, torch.Tensor], h: torch.Tensor,
             episode_start: torch.Tensor) -> tuple[AgentOutput, torch.Tensor]:
        self._attn = None
        out, h_new = self.model.step(obs, h, episode_start)
        self.last = self._capture(obs, out)
        return out, h_new

    # ------------------------------------------------------------ internals
    def _pool_hook(self, module: nn.MultiheadAttention, args: tuple, kwargs: dict[str, Any],
                   output: tuple[torch.Tensor, torch.Tensor | None]) -> None:
        if self._in_hook:
            return None
        q, k, v = args[:3]
        pad = kwargs.get("key_padding_mask")
        self._in_hook = True
        try:
            with torch.no_grad():
                pooled, w = module(q, k, v, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
        finally:
            self._in_hook = False
        b, n_tok = k.shape[0], k.shape[1]
        assert w is not None and w.shape == (b, 1, n_tok), f"pool weights expected (B, 1, {n_tok}), got {None if w is None else tuple(w.shape)}"
        diff = float((pooled - output[0]).abs().max())
        self.max_attn_diff = max(self.max_attn_diff, diff)
        if diff > ATTN_MATCH_TOL:
            # Not verified: show no weights for this decision rather than unverified ones
            # (strict=True makes it an error instead).
            self.attn_mismatches += 1
            if self.strict:
                raise AssertionError(f"recomputed pooled entity vector differs from the model's own by "
                                     f"{diff:.3g} (> {ATTN_MATCH_TOL}); attention weights would not be the model's")
            return None
        row = w[0, 0].detach().float().cpu().numpy()
        self._attn = (row[1:].copy(), float(row[0]), diff)     # token 0 = learned "no entity"
        return None

    def _capture(self, obs: dict[str, torch.Tensor], out: AgentOutput) -> StepCapture:
        o = {k: t[0].detach().cpu().numpy().copy() for k, t in obs.items()}
        probs = [torch.softmax(lg[0].float(), dim=-1).cpu().numpy()
                 for lg in tempered_logits(out.logits, self.display_temperature)]
        goal = getattr(out, "goal_logits", None)       # None when the model has no goal head
        goal_np = None if goal is None else goal[0].detach().float().cpu().numpy().reshape(-1)
        attn, empty, diff = self._attn if self._attn is not None else (None, None, None)
        if attn is not None:
            n = o["entity_mask"].shape[0]
            assert attn.shape == (n,), f"entity weights expected ({n},), got {attn.shape}"
        return StepCapture(obs=o, value=float(out.value_ext[0]), probs=probs, attn=attn, attn_empty=empty,
                           attn_max_abs_diff=diff, goal_logits=goal_np)
