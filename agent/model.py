"""Stage A0 agent network (ADR 0003): encoders -> fusion MLP -> GRU core -> heads.

Hidden-state convention (used identically in collection and in the update):
`episode_start[t]` is 1 when obs[t] is the first observation of an episode; the
hidden state is zeroed BEFORE obs[t] is processed wherever it is 1.

Auxiliary heads are training-only loss terms; nothing reads them to act.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from configs.goal_modes import N_GOAL_MODES
from configs.model_config import ModelConfig

from .init_schemes import apply_orthogonal_init, init_gru_update_bias
from .encoders import EgoMapEncoder, EntitySetEncoder, PrevActionEncoder, RayEncoder, SelfEncoder
from .obs_spec import EGOMAP_KEY, ObsSpec

TensorDict = dict[str, torch.Tensor]


@dataclass
class AgentOutput:
    logits: list[torch.Tensor]       # one (..., n_i) per head
    value_ext: torch.Tensor          # (...)
    value_int: torch.Tensor          # (...) reserved for intrinsic return (ADR 0005)
    aux_keys: torch.Tensor           # (..., n_key_bits) logits
    aux_damage: torch.Tensor         # (...) logit
    aux_monsters: torch.Tensor       # (...) regression
    # (..., N_GOAL_MODES) sub-goal logits when cfg.goal_head, else None. Auxiliary
    # prediction only (training signal and display); never read to choose actions.
    goal_logits: torch.Tensor | None = None


GOAL_HEAD_NAME = "goal_head"   # attribute name, hence the state-dict key prefix "goal_head."


class A0Agent(nn.Module):
    def __init__(self, spec: ObsSpec, cfg: ModelConfig) -> None:
        super().__init__()
        self.spec, self.cfg = spec, cfg
        self.rays = RayEncoder(spec.ray_channels, spec.ray_width, cfg)
        self.entities = EntitySetEncoder(spec.entity_feat_dim, spec.n_entity_types, cfg)
        self.self_enc = SelfEncoder(spec.self_dim, cfg)
        self.prev_action = PrevActionEncoder(spec.head_sizes, cfg)
        fused_in = (self.rays.out_dim + self.entities.out_dim + self.self_enc.out_dim
                    + self.prev_action.out_dim + spec.modality_dim)
        # Observation v3 only: built ONLY when the spec has an egomap, so a v1/v2 spec
        # builds exactly the earlier module list (and draws the same init RNG).
        self.egomap: EgoMapEncoder | None = None
        if spec.egomap is not None:
            self.egomap = EgoMapEncoder(spec.egomap, cfg)
            fused_in += self.egomap.out_dim
        self.fusion = nn.Sequential(nn.Linear(fused_in, cfg.fusion_dim), nn.ReLU())
        self.core = nn.GRUCell(cfg.fusion_dim, cfg.core_hidden)
        h = cfg.core_hidden
        self.policy_heads = nn.ModuleList(nn.Linear(h, n) for n in spec.head_sizes)
        self.value_ext = nn.Linear(h, 1)
        self.value_int = nn.Linear(h, 1)
        self.aux_keys = nn.Linear(h, cfg.n_key_bits)
        self.aux_damage = nn.Linear(h, 1)
        self.aux_monsters = nn.Linear(h, 1)
        for head in self.policy_heads:
            nn.init.orthogonal_(head.weight, gain=cfg.policy_head_init_gain)
            nn.init.zeros_(head.bias)
        if cfg.init_scheme == "orthogonal":
            apply_orthogonal_init(self, cfg, self.policy_heads,
                                  [self.value_ext, self.value_int, self.aux_keys, self.aux_damage,
                                   self.aux_monsters], self.core)
        elif cfg.init_scheme != "default":
            raise ValueError(f"unknown init_scheme {cfg.init_scheme!r}")
        if cfg.gru_update_bias_init:  # after any scheme: overrides "biases 0" (ADR 0008)
            init_gru_update_bias(self.core, cfg)
        # Sub-goal head: built LAST, after every other init, so the RNG draws (hence the
        # values) of every other parameter are the same with the head on or off.
        self.goal_head: nn.Linear | None = None
        if cfg.goal_head:
            self.goal_head = nn.Linear(h, N_GOAL_MODES)
            nn.init.orthogonal_(self.goal_head.weight, gain=cfg.orthogonal_unit_gain)
            nn.init.zeros_(self.goal_head.bias)

    # ------------------------------------------------------------ pieces
    def initial_state(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch, self.cfg.core_hidden, device=device)

    def encode(self, obs: TensorDict) -> torch.Tensor:
        """obs tensors with one leading batch dim -> (B, fusion_dim)."""
        b = obs["self"].shape[0]
        parts = [
            self.rays(obs["rays"]),
            self.entities(obs["entities"], obs["entity_type"], obs["entity_mask"]),
            self.self_enc(obs["self"]),
            self.prev_action(obs["prev_action"]),
            obs["modality"].float(),
        ]
        if self.egomap is not None:   # appended last: the v2 part of the fusion input is unchanged
            assert EGOMAP_KEY in obs, f"this model's spec has {EGOMAP_KEY!r} but the observation lacks it"
            parts.append(self.egomap(obs[EGOMAP_KEY]))
        z = torch.cat(parts, dim=-1)
        x = self.fusion(z)
        assert x.shape == (b, self.cfg.fusion_dim), f"fused expected (B, {self.cfg.fusion_dim})"
        return x

    def core_step(self, x: torch.Tensor, h: torch.Tensor, episode_start: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        assert h.shape == (b, self.cfg.core_hidden), \
            f"hidden expected (B, {self.cfg.core_hidden}), got {tuple(h.shape)}"
        assert episode_start.shape == (b,), f"episode_start expected (B,)={b}, got {tuple(episode_start.shape)}"
        keep = (1.0 - episode_start.float()).unsqueeze(-1)
        h = h * keep
        if not self.cfg.recurrent:
            h = torch.zeros_like(h)
        return self.core(x, h)

    def heads(self, h: torch.Tensor) -> AgentOutput:
        return AgentOutput(
            logits=[head(h) for head in self.policy_heads],
            value_ext=self.value_ext(h).squeeze(-1),
            value_int=self.value_int(h).squeeze(-1),
            aux_keys=self.aux_keys(h),
            aux_damage=self.aux_damage(h).squeeze(-1),
            aux_monsters=self.aux_monsters(h).squeeze(-1),
            goal_logits=None if self.goal_head is None else self.goal_head(h),
        )

    # ------------------------------------------------------------ public API
    def step(self, obs: TensorDict, h: torch.Tensor, episode_start: torch.Tensor) -> tuple[AgentOutput, torch.Tensor]:
        """One decision for B environments."""
        h_new = self.core_step(self.encode(obs), h, episode_start)
        return self.heads(h_new), h_new

    def forward_sequence(self, obs: TensorDict, h0: torch.Tensor, episode_start: torch.Tensor) -> tuple[AgentOutput, torch.Tensor]:
        """obs tensors (T, B, ...), h0 (B, H), episode_start (T, B) -> outputs (T, B, ...)."""
        t, b = episode_start.shape
        flat = {k: v.reshape(t * b, *v.shape[2:]) for k, v in obs.items()}
        x = self.encode(flat).reshape(t, b, -1)
        h = h0
        hs = []
        for i in range(t):
            h = self.core_step(x[i], h, episode_start[i])
            hs.append(h)
        hseq = torch.stack(hs, dim=0)
        assert hseq.shape == (t, b, self.cfg.core_hidden)
        return self.heads(hseq), h


def build_agent(spec: ObsSpec, cfg: ModelConfig, seed: int) -> A0Agent:
    """Construct with parameter init drawn from a seeded, forked RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return A0Agent(spec, cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
