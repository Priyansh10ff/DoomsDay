"""Rollout storage for synchronous recurrent PPO: (T, N) per field.

Stores the TRUE observation (A0 keys) plus a per-step `telemetry_dropped` flag;
the policy input is rebuilt by `mask_telemetry` both in collection and in the
update, so masked training and unmasked auxiliary targets share one buffer.
`h0` is the core state before step 0 of this rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch

from agent.obs_spec import EGOMAP_KEY, ObsSpec

TensorDict = dict[str, torch.Tensor]


def _obs_shapes(spec: ObsSpec) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """The model's observation keys; `egomap` (uint8, observation v3) only when the spec
    has one, so a v1/v2 rollout is unchanged."""
    shapes = {
        "rays": ((spec.ray_channels, spec.ray_width), torch.float32),
        "entities": ((spec.max_entities, spec.entity_feat_dim), torch.float32),
        "entity_type": ((spec.max_entities,), torch.long),
        "entity_mask": ((spec.max_entities,), torch.bool),
        "self": ((spec.self_dim,), torch.float32),
        "prev_action": ((spec.n_heads,), torch.long),
        "modality": ((spec.modality_dim,), torch.float32),
    }
    egomap = getattr(spec, "egomap", None)
    if egomap is not None:
        shapes[EGOMAP_KEY] = (tuple(egomap), torch.uint8)
    return shapes


@dataclass
class Rollout:
    obs: TensorDict
    telemetry_dropped: torch.Tensor   # (T, N) bool
    episode_start: torch.Tensor       # (T, N) float: reset hidden before obs[t]
    actions: torch.Tensor             # (T, N, K) long
    logp: torch.Tensor                # (T, N)
    values: torch.Tensor              # (T, N) in reward units
    rewards: torch.Tensor             # (T, N)
    terminated: torch.Tensor          # (T, N) bool
    truncated: torch.Tensor           # (T, N) bool
    bootstrap_value: torch.Tensor     # (T, N) V(true final obs) where truncated & fresh
    valid: torch.Tensor               # (T, N) bool, False = stale time-out transition
    keys: torch.Tensor                # (T, N, n_keys)
    keys_valid: torch.Tensor          # (T, N) bool
    damage: torch.Tensor              # (T, N) binary target
    damage_valid: torch.Tensor        # (T, N) bool
    monsters: torch.Tensor            # (T, N)
    h0: torch.Tensor                  # (N, H)
    last_value: torch.Tensor = field(default_factory=lambda: torch.zeros(0))  # (N,)
    # RND stream (train/rnd.py); empty tensors when RND is off.
    rnd_feat: torch.Tensor = field(default_factory=lambda: torch.zeros(0))       # (T, N, D) raw input of o_{t+1}
    rnd_ok: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))  # (T, N)
    values_int: torch.Tensor = field(default_factory=lambda: torch.zeros(0))     # (T, N) normalised units
    last_value_int: torch.Tensor = field(default_factory=lambda: torch.zeros(0))  # (N,)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.episode_start.shape)  # type: ignore[return-value]


def allocate(spec: ObsSpec, t_len: int, n: int, hidden: int, n_keys: int,
             device: torch.device | str = "cpu", rnd_dim: int = 0) -> Rollout:
    """`rnd_dim` > 0 also allocates the RND stream (features of width `rnd_dim`)."""
    obs = {k: torch.zeros((t_len, n, *shp), dtype=dt, device=device) for k, (shp, dt) in _obs_shapes(spec).items()}

    def z(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    rnd = {}
    if rnd_dim > 0:
        rnd = dict(rnd_feat=z(t_len, n, rnd_dim), rnd_ok=z(t_len, n, dtype=torch.bool),
                   values_int=z(t_len, n), last_value_int=z(n))
    return Rollout(
        **rnd,
        obs=obs,
        telemetry_dropped=z(t_len, n, dtype=torch.bool),
        episode_start=z(t_len, n),
        actions=z(t_len, n, spec.n_heads, dtype=torch.long),
        logp=z(t_len, n), values=z(t_len, n), rewards=z(t_len, n),
        terminated=z(t_len, n, dtype=torch.bool), truncated=z(t_len, n, dtype=torch.bool),
        bootstrap_value=z(t_len, n), valid=z(t_len, n, dtype=torch.bool),
        keys=z(t_len, n, n_keys), keys_valid=z(t_len, n, dtype=torch.bool),
        damage=z(t_len, n), damage_valid=z(t_len, n, dtype=torch.bool),
        monsters=z(t_len, n), h0=z(n, hidden), last_value=z(n),
    )


def rollout_to(ro: Rollout, device: torch.device | str) -> Rollout:
    """The same rollout with every tensor on `device` (a copy; `ro` is unchanged unless
    it is already there)."""
    kw = {}
    for f in fields(ro):
        v = getattr(ro, f.name)
        kw[f.name] = {k: t.to(device) for k, t in v.items()} if isinstance(v, dict) else v.to(device)
    return Rollout(**kw)


def select_envs(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index the env dim (dim 1) of a (T, N, ...) tensor."""
    return x.index_select(1, idx)
