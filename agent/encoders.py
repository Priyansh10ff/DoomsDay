"""Stage A0 input encoders (ADR 0003): rays, entity set, self vector, previous action."""
from __future__ import annotations

import torch
from torch import nn

from configs.model_config import ModelConfig


class RayEncoder(nn.Module):
    """1D convolutions over the depth rays (bands as channels) to a vector."""

    def __init__(self, channels: int, width: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.channels, self.width = channels, width
        layers: list[nn.Module] = []
        c_in = channels
        for c_out in cfg.ray_conv_channels:
            layers += [nn.Conv1d(c_in, c_out, cfg.ray_conv_kernel, cfg.ray_conv_stride,
                                 padding=cfg.ray_conv_kernel // 2), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, channels, width)).numel()
        self.out = nn.Sequential(nn.Linear(flat, cfg.ray_dim), nn.ReLU())
        self.out_dim = cfg.ray_dim

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        assert rays.dim() == 3 and rays.shape[1:] == (self.channels, self.width), \
            f"rays expected (B, {self.channels}, {self.width}), got {tuple(rays.shape)}"
        z = self.out(self.conv(rays).flatten(1))
        assert z.shape == (rays.shape[0], self.out_dim), f"ray code expected (B, {self.out_dim})"
        return z


class EntitySetEncoder(nn.Module):
    """Permutation-invariant encoder for 0..N visible entities.

    Per entity: type embedding ++ features -> shared MLP -> token. A learned
    "no entity" token is always present, so the empty set is well defined. One
    masked multi-head self-attention block with a residual, then a learned query
    pools the (masked) tokens. No positional encoding, so row order is irrelevant.
    Padded rows are replaced by zeros before use, so garbage there cannot leak.
    """

    def __init__(self, feat_dim: int, n_types: int, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.entity_token_dim
        self.feat_dim, self.n_types, self.out_dim = feat_dim, n_types, d
        self.type_embed = nn.Embedding(n_types, cfg.entity_type_embed_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(cfg.entity_type_embed_dim + feat_dim, cfg.entity_mlp_hidden), nn.ReLU(),
            nn.Linear(cfg.entity_mlp_hidden, d),
        )
        self.empty_token = nn.Parameter(torch.zeros(1, 1, d))
        self.attn = nn.MultiheadAttention(d, cfg.entity_attn_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d))
        self.pool = nn.MultiheadAttention(d, cfg.entity_pool_heads, batch_first=True)
        nn.init.normal_(self.empty_token, std=cfg.learned_token_init_std)
        nn.init.normal_(self.pool_query, std=cfg.learned_token_init_std)

    def forward(self, feats: torch.Tensor, types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n = mask.shape
        assert feats.shape == (b, n, self.feat_dim), \
            f"entities expected (B, N, {self.feat_dim}), got {tuple(feats.shape)}"
        assert types.shape == (b, n), f"entity_type expected (B, N)={b, n}, got {tuple(types.shape)}"
        mask = mask.bool()
        feats = torch.where(mask.unsqueeze(-1), feats, torch.zeros((), dtype=feats.dtype, device=feats.device))
        types = torch.where(mask, types.long(), torch.zeros((), dtype=torch.long, device=types.device))
        tokens = self.token_mlp(torch.cat([self.type_embed(types), feats], dim=-1))
        tokens = torch.cat([self.empty_token.expand(b, 1, -1), tokens], dim=1)
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        pad = ~valid
        att, _ = self.attn(tokens, tokens, tokens, key_padding_mask=pad, need_weights=False)
        tokens = self.norm(tokens + att)
        pooled, _ = self.pool(self.pool_query.expand(b, 1, -1), tokens, tokens,
                              key_padding_mask=pad, need_weights=False)
        z = pooled.squeeze(1)
        assert z.shape == (b, self.out_dim), f"entity code expected (B, {self.out_dim})"
        return z


class SelfEncoder(nn.Module):
    def __init__(self, self_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.in_dim, self.out_dim = self_dim, cfg.self_dim
        self.net = nn.Sequential(nn.Linear(self_dim, cfg.self_hidden), nn.ReLU(),
                                 nn.Linear(cfg.self_hidden, cfg.self_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.shape[1] == self.in_dim, \
            f"self expected (B, {self.in_dim}), got {tuple(x.shape)}"
        return self.net(x)


class EgoMapEncoder(nn.Module):
    """Small 2D CNN over the ego-centric map (C, H, W) uint8 0/1 -> egomap_dim.
    The map is agent-centred and heading-aligned (observation v3 contract), so no
    absolute coordinate reaches the model through it."""

    def __init__(self, shape: tuple[int, int, int], cfg: ModelConfig) -> None:
        super().__init__()
        self.shape = tuple(shape)
        layers: list[nn.Module] = []
        c_in = shape[0]
        for c_out in cfg.egomap_conv_channels:
            layers += [nn.Conv2d(c_in, c_out, cfg.egomap_conv_kernel, cfg.egomap_conv_stride,
                                 padding=cfg.egomap_conv_padding), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, *shape)).numel()
        self.out = nn.Sequential(nn.Linear(flat, cfg.egomap_dim), nn.ReLU())
        self.out_dim = cfg.egomap_dim

    def forward(self, egomap: torch.Tensor) -> torch.Tensor:
        assert egomap.dim() == 4 and tuple(egomap.shape[1:]) == self.shape, \
            f"egomap expected (B, {', '.join(map(str, self.shape))}), got {tuple(egomap.shape)}"
        z = self.out(self.conv(egomap.float()).flatten(1))
        assert z.shape == (egomap.shape[0], self.out_dim), f"egomap code expected (B, {self.out_dim})"
        return z


class PrevActionEncoder(nn.Module):
    """One embedding table per action head, summed into one vector."""

    def __init__(self, head_sizes: tuple[int, ...], cfg: ModelConfig) -> None:
        super().__init__()
        self.head_sizes, self.out_dim = head_sizes, cfg.prev_action_dim
        self.tables = nn.ModuleList(nn.Embedding(n, cfg.prev_action_dim) for n in head_sizes)

    def forward(self, prev: torch.Tensor) -> torch.Tensor:
        assert prev.dim() == 2 and prev.shape[1] == len(self.head_sizes), \
            f"prev_action expected (B, {len(self.head_sizes)}), got {tuple(prev.shape)}"
        prev = prev.long()
        z = sum(t(prev[:, i]) for i, t in enumerate(self.tables))
        assert isinstance(z, torch.Tensor)
        return z
