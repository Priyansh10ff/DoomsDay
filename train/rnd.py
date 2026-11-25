"""Random Network Distillation exploration bonus (idea: Burda et al. 2018, "Exploration
by Random Network Distillation"; written here from the paper's description).

TRAINING ONLY, trainer side (imports torch; lives in train/, not intrinsic/, so reward code stays
torch-free; env workers never import it). Nothing here reaches an observation or the
inference path: the bonus is a second reward stream inside the PPO update.

How it works:
  * A TARGET network with fixed random weights maps an RND input vector to a feature
    vector; a PREDICTOR network is trained to reproduce it. Inputs seen often are
    predicted well (small error), new ones badly (large error).
  * Intrinsic reward of transition t = mean squared prediction error on the NEXT
    observation o_{t+1}, computed with the predictor as it was BEFORE this rollout's
    training, divided by a running std of the discounted intrinsic return
    (`ReturnFilter`, per env, non-episodic, carried across rollouts).
  * RND input (`rnd_features`): rays (all bands, flattened) + self vector without the
    configured indices (ammo, damage pulse, selected-weapon one-hot) + any configured
    extra keys present (e.g. an ego-centric map), flattened. Never `prev_action`, so
    pressing buttons cannot by itself look novel.
  * The input is whitened by a running per-feature mean/std (std floored) and clipped
    to [-clip, clip], for the RND networks only (the policy sees raw observations).
  * The predictor trains on a random `predictor_fraction` of each rollout's usable rows.

A transition whose next observation is unavailable gets reward 0 and is left out of
every statistic: after death or level exit the env repeats the last observation
(`final_obs_fresh` False), and a stale time-out has no fresh observation either.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from configs.train_config import RNDConfig

RAYS_KEY = "rays"
SELF_KEY = "self"


# ---------------------------------------------------------------- input layout
MAP_NDIM = 3   # (C, H, W) extra keys are average-pooled spatially


@dataclass(frozen=True)
class FeatureLayout:
    ray_size: int                       # rays flattened (bands x columns)
    self_keep: tuple[int, ...]          # self-vector indices kept
    extra: tuple[tuple[str, int], ...]  # (key, flattened size after pooling) of extra keys present
    pool: int = 1                       # spatial pooling factor of (C, H, W) extra keys

    @property
    def dim(self) -> int:
        return self.ray_size + len(self.self_keep) + sum(n for _, n in self.extra)


def _pool_ok(shape: tuple[int, ...], pool: int) -> bool:
    return len(shape) == MAP_NDIM and pool > 1 and shape[1] % pool == 0 and shape[2] % pool == 0


def feature_layout(shapes: Mapping[str, tuple[int, ...]], cfg: RNDConfig) -> FeatureLayout:
    """`shapes`: per-key single-observation shape (e.g. from the observation space)."""
    if RAYS_KEY not in shapes or SELF_KEY not in shapes:
        raise KeyError(f"RND needs {RAYS_KEY!r} and {SELF_KEY!r} in the observation, got {sorted(shapes)}")
    self_dim = int(np.prod(shapes[SELF_KEY]))
    keep = tuple(i for i in range(self_dim) if i not in set(cfg.self_exclude))
    extra = []
    for k in cfg.extra_keys:
        if k in shapes:
            shp = tuple(shapes[k])
            n = int(np.prod(shp)) // (cfg.extra_pool ** 2 if _pool_ok(shp, cfg.extra_pool) else 1)
            if n > cfg.max_extra_dim:
                raise ValueError(f"RND extra key {k!r} has {n} values, above max_extra_dim {cfg.max_extra_dim}")
            extra.append((k, n))
    return FeatureLayout(int(np.prod(shapes[RAYS_KEY])), keep, tuple(extra), cfg.extra_pool)


def _flat_extra(x: np.ndarray, b: int, pool: int) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if _pool_ok(x.shape[1:], pool):
        _, c, h, w = x.shape
        x = x.reshape(b, c, h // pool, pool, w // pool, pool).mean(axis=(3, 5))
    return x.reshape(b, -1)


def rnd_features(obs: Mapping[str, np.ndarray], layout: FeatureLayout) -> np.ndarray:
    """Batched observation dict (B, ...) per key -> (B, layout.dim) float32."""
    b = obs[SELF_KEY].shape[0]
    parts = [np.asarray(obs[RAYS_KEY], np.float32).reshape(b, -1),
             np.asarray(obs[SELF_KEY], np.float32).reshape(b, -1)[:, list(layout.self_keep)]]
    parts += [_flat_extra(obs[k], b, layout.pool) for k, _ in layout.extra]
    out = np.concatenate(parts, axis=1)
    assert out.shape == (b, layout.dim), f"RND features expected ({b}, {layout.dim}), got {out.shape}"
    return out


# ---------------------------------------------------------------- running statistics
class RunningMeanStd:
    """Per-feature running mean / variance (parallel-batch update), float64 on `device`."""

    def __init__(self, shape: tuple[int, ...], eps: float, device: torch.device | str = "cpu") -> None:
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = float(eps)

    def update(self, x: torch.Tensor) -> None:
        """x: (B, *shape); an empty batch is ignored."""
        x = x.detach().to(self.mean.device, torch.float64)
        assert x.shape[1:] == self.mean.shape, f"RunningMeanStd expected (B, {tuple(self.mean.shape)}), got {tuple(x.shape)}"
        n = x.shape[0]
        if n == 0:
            return
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        tot = self.count + n
        delta = b_mean - self.mean
        m2 = self.var * self.count + b_var * n + delta.pow(2) * self.count * n / tot
        self.mean = self.mean + delta * n / tot
        self.var = m2 / tot
        self.count = tot

    def std(self, floor: float) -> torch.Tensor:
        return self.var.sqrt().clamp_min(floor)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        dev = self.mean.device
        self.mean, self.var, self.count = d["mean"].to(dev, torch.float64), d["var"].to(dev, torch.float64), float(d["count"])


class ReturnFilter:
    """Per-env discounted running sum of intrinsic rewards (non-episodic: never cut at an
    episode end), whose values feed a scalar running variance. The intrinsic reward is
    divided by that std (Burda et al. 2018, reward normalisation)."""

    def __init__(self, n_envs: int, gamma: float, eps: float, device: torch.device | str = "cpu") -> None:
        self.gamma = gamma
        self.ret = torch.zeros(n_envs, dtype=torch.float64, device=device)
        self.rms = RunningMeanStd((), eps, device)

    def update(self, rewards: torch.Tensor, ok: torch.Tensor) -> None:
        """rewards, ok: (T, N). Rows with ok False add nothing (reward 0, no statistic)."""
        t_len, n = rewards.shape
        assert ok.shape == (t_len, n), f"ok expected {(t_len, n)}, got {tuple(ok.shape)}"
        if self.ret.shape[0] != n:   # env count changed (e.g. a resume with another layout)
            self.ret = torch.zeros(n, dtype=torch.float64, device=self.ret.device)
        r = rewards.detach().to(self.ret.device, torch.float64)
        okd = ok.to(self.ret.device)
        vals = []
        for t in range(t_len):
            self.ret = self.ret * self.gamma + r[t]
            vals.append(self.ret[okd[t]])
        self.rms.update(torch.cat(vals))

    def std(self, floor: float) -> float:
        return float(self.rms.std(floor))

    def state_dict(self) -> dict:
        return {"ret": self.ret.cpu(), "rms": self.rms.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.ret = d["ret"].to(self.ret.device, torch.float64)
        self.rms.load_state_dict(d["rms"])


# ---------------------------------------------------------------- networks
def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


class RNDNetworks(nn.Module):
    """Fixed random target and trained predictor (one more hidden layer than the target,
    so the target function is inside the predictor's model class)."""

    def __init__(self, in_dim: int, cfg: RNDConfig) -> None:
        super().__init__()
        h, o = cfg.hidden_dim, cfg.out_dim
        self.in_dim, self.out_dim = in_dim, o
        self.target = _mlp([in_dim, h, h, o])
        self.predictor = _mlp([in_dim, h, h, h, o])
        for p in self.target.parameters():
            p.requires_grad_(False)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) whitened input -> (B,) mean squared prediction error."""
        assert x.dim() == 2 and x.shape[1] == self.in_dim, f"RND input expected (B, {self.in_dim}), got {tuple(x.shape)}"
        with torch.no_grad():
            tgt = self.target(x)
        return (self.predictor(x) - tgt).pow(2).mean(-1)


# ---------------------------------------------------------------- the bonus
class RND:
    """Owns the networks, normalisers, predictor optimizer and subsample generator.
    `net_seed` seeds the network init (forked RNG: global torch RNG untouched);
    `sample_seed` seeds the predictor's subsample."""

    def __init__(self, layout: FeatureLayout, cfg: RNDConfig, n_envs: int, net_seed: int, sample_seed: int,
                 device: torch.device | str = "cpu") -> None:
        self.layout, self.cfg, self.device = layout, cfg, torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(net_seed)
            self.nets = RNDNetworks(layout.dim, cfg)
        self.nets.to(self.device)
        self.optimizer = torch.optim.Adam(self.nets.predictor.parameters(), lr=cfg.predictor_lr)
        self.gen = torch.Generator(device="cpu").manual_seed(sample_seed)
        self.obs_rms = RunningMeanStd((layout.dim,), cfg.norm_eps, self.device)
        self.ret_filter = ReturnFilter(n_envs, cfg.gamma_int, cfg.norm_eps, self.device)

    def features(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return rnd_features(obs, self.layout)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.obs_rms.mean.float(), self.obs_rms.std(self.cfg.obs_std_floor).float()
        return ((x - mean) / std).clamp(-self.cfg.obs_clip, self.cfg.obs_clip)

    def _rows(self, feat: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        t_len, n, d = feat.shape
        assert d == self.layout.dim and ok.shape == (t_len, n), \
            f"RND rollout expected (T, N, {self.layout.dim}) / (T, N), got {tuple(feat.shape)} / {tuple(ok.shape)}"
        return feat.to(self.device)[ok.to(self.device)]

    @torch.no_grad()
    def intrinsic_rewards(self, feat: torch.Tensor, ok: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """feat (T, N, D) raw RND input of o_{t+1}; ok (T, N). Updates the observation
        normaliser with this rollout's usable rows, computes the raw error with the
        current predictor, updates the return filter, and returns the normalised reward
        (T, N) (0 where not ok) plus statistics."""
        okd = ok.to(self.device)
        self.obs_rms.update(self._rows(feat, ok))
        t_len, n, d = feat.shape
        raw = self.nets.error(self.normalise(feat.to(self.device).reshape(t_len * n, d))).reshape(t_len, n)
        raw = torch.where(okd, raw, torch.zeros_like(raw))
        self.ret_filter.update(raw, okd)
        std = self.ret_filter.std(self.cfg.return_std_floor)
        r = raw / std
        sel = r[okd]
        stats = {"rnd_int_reward_mean": float(sel.mean()) if sel.numel() else 0.0,
                 "rnd_int_reward_std": float(sel.std()) if sel.numel() > 1 else 0.0,
                 "rnd_int_reward_raw_mean": float(raw[okd].mean()) if sel.numel() else 0.0,
                 "rnd_int_return_std": std, "rnd_ok_frac": float(okd.float().mean())}
        return r.float(), stats

    def train_predictor(self, feat: torch.Tensor, ok: torch.Tensor) -> dict[str, float]:
        """Train the predictor on a random `predictor_fraction` of the usable rows."""
        x = self.normalise(self._rows(feat, ok).float())
        m = x.shape[0]
        k = int(round(self.cfg.predictor_fraction * m))
        if k < 1:
            return {"rnd_predictor_loss": float("nan"), "rnd_predictor_rows": 0.0}
        sel = torch.randperm(m, generator=self.gen)[:k]
        x = x[sel.to(self.device)]
        mbs = min(self.cfg.predictor_minibatches, k)
        total, steps = 0.0, 0
        for _ in range(self.cfg.predictor_epochs):
            perm = torch.randperm(k, generator=self.gen).to(self.device)
            for chunk in perm.chunk(mbs):
                loss = self.nets.error(x[chunk]).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.nets.predictor.parameters(), self.cfg.predictor_max_grad_norm)
                self.optimizer.step()
                total += float(loss.detach())
                steps += 1
        return {"rnd_predictor_loss": total / steps, "rnd_predictor_rows": float(k)}

    def state_dict(self) -> dict:
        return {"nets": {k: v.cpu() for k, v in self.nets.state_dict().items()},
                "optimizer": self.optimizer.state_dict(), "gen": self.gen.get_state(),
                "obs_rms": self.obs_rms.state_dict(), "ret_filter": self.ret_filter.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        self.nets.load_state_dict(d["nets"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.gen.set_state(d["gen"])
        self.obs_rms.load_state_dict(d["obs_rms"])
        self.ret_filter.load_state_dict(d["ret_filter"])
