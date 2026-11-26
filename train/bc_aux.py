"""Imitation (behaviour-cloning) auxiliary loss inside our own PPO
(configs.train_config.BCAuxConfig; 2026-11-25 evening, gap-analysis gap 3).

Data: the teacher datasets of train/bc.py (ep_*.npz + index.jsonl), loaded once into
RAM through train.bc_data.load_episodes, which validates every file against the run's
observation spec. A dataset recorded at another observation version (v1: rays (3, 80),
self (28,); v2: rays (4, 80), self (34,)) fails that check and is refused.

Sampling (`BCSampler.sample`): K start steps drawn uniformly over ALL dataset steps
(so later parts of long episodes, e.g. door passages, are trained too), seeded by its
own generator (never the PPO minibatch or action generators). For each start s:
  * burn-in window: the up-to-L steps before s, run under no_grad from a ZERO hidden
    state at the first real burn-in step (episode_start=1 there). Shorter burn-ins are
    front-padded with that first step; the reset makes the padding irrelevant.
  * graded window: steps s .. s+L-1 from the detached burned-in state; steps past the
    episode end repeat its last step and are masked out (valid=False). When s is the
    episode's first step the graded window itself starts with episode_start=1.
Loss: train.bc.bc_loss (sum over heads of weight x mean cross-entropy on valid steps),
every head weight 1. Telemetry dropout is not applied (A0 trains with probability 0).
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from agent.model import A0Agent, AgentOutput
from agent.obs_spec import ObsSpec
from agent.tensorize import torch_dtype
from configs.train_config import BCAuxConfig

from .bc import bc_loss, head_names, index_sha256
from .bc_data import _TORCH_DTYPES, Episode, load_episodes, step_bytes  # noqa: F401 (re-exported for tests)
from .bc_spec import DatasetMismatch
from .goal_head import GoalLabelError, goal_loss, require_goal_labels

BC_HEAD_WEIGHT = 1.0


class BCDataRefused(DatasetMismatch):
    """A BC dataset does not fit this run (observation layout/version, size, files)."""


def normalise_dirs(text_or_dirs: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated text (or a list) -> absolute, case-normalised dirs, order kept, no repeats."""
    vals = [text_or_dirs] if isinstance(text_or_dirs, str) else list(text_or_dirs)
    out: list[str] = []
    for v in vals:
        for d in v.split(","):
            d = d.strip()
            if d:
                p = os.path.normcase(os.path.abspath(d))
                if p not in out:
                    out.append(p)
    return tuple(out)


def dataset_hashes(dirs: tuple[str, ...]) -> tuple[str, ...]:
    """sha256 of each dir's index.jsonl (cheap; read before any big load)."""
    try:
        return tuple(index_sha256(d) for d in dirs)
    except FileNotFoundError as e:
        raise BCDataRefused(f"BC dataset missing: {e}") from e


def load_bc_data(dirs: tuple[str, ...], spec: ObsSpec, max_bytes: int, obs_version: int | None = None,
                 goal_required: bool = False) -> list[Episode]:
    """Load every episode of every dir, refusing (BCDataRefused) a dataset that does
    not match `spec` or whose arrays would exceed `max_bytes` in total, and, when
    `goal_required`, a dir in which no episode carries a `goal_mode` array."""
    total_steps = 0
    for d in dirs:
        from .bc_data import read_index
        try:
            total_steps += sum(int(r["length"]) for r in read_index(d))
        except (FileNotFoundError, DatasetMismatch) as e:
            raise BCDataRefused(f"BC dataset {d!r}: {e}") from e
    need = total_steps * step_bytes(spec)
    if need > max_bytes:
        raise BCDataRefused(f"BC datasets {list(dirs)} need about {need / 2**30:.2f} GiB in RAM, above the "
                            f"--bc-max-gb cap of {max_bytes / 2**30:.2f} GiB")
    eps: list[Episode] = []
    for d in dirs:
        try:
            part = load_episodes(d, spec, max_bytes)
            if goal_required:
                require_goal_labels(part, f"BC dataset {d!r}")
            eps.extend(part)
        except GoalLabelError as e:
            raise BCDataRefused(str(e)) from e
        except DatasetMismatch as e:
            v = f"v{obs_version}" if obs_version is not None else "this run's"
            raise BCDataRefused(f"BC dataset {d!r} does not match the run's observation layout ({v}; a "
                                f"dataset recorded at another observation version is refused): {e}") from e
    if not eps:
        raise BCDataRefused(f"BC datasets {list(dirs)}: no episodes")
    return eps


def coef_at(cfg: BCAuxConfig, policy_update: int) -> float:
    """Coefficient for the policy update with index `policy_update` (0 = first update
    after the value warm-up)."""
    if cfg.decay_mode == "exp":
        return float(max(cfg.coef * cfg.decay_rate ** max(policy_update, 0), cfg.coef_final))
    if cfg.decay_updates == 0:
        return float(cfg.coef)
    frac = min(max(policy_update, 0) / cfg.decay_updates, 1.0)
    return float(cfg.coef + frac * (cfg.coef_final - cfg.coef))


@dataclass
class BCBatch:
    burn_obs: dict[str, torch.Tensor]    # (L, K, ...)
    burn_start: torch.Tensor             # (L, K) float
    obs: dict[str, torch.Tensor]         # (L, K, ...)
    start: torch.Tensor                  # (L, K) float
    labels: torch.Tensor                 # (L, K, n_heads) long
    valid: torch.Tensor                  # (L, K) bool
    goal: torch.Tensor | None = None     # (L, K) long sub-goal labels (negative = none), if any episode has them


class BCSampler:
    def __init__(self, episodes: list[Episode], seq_len: int, batch_seqs: int, seed: int,
                 device: torch.device | str = "cpu") -> None:
        if not episodes:
            raise ValueError("BCSampler needs at least one episode")
        self.eps, self.l, self.k = episodes, seq_len, batch_seqs
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        lengths = np.array([e.length for e in episodes], np.int64)
        self.keys = tuple(episodes[0].obs)        # the model's keys (egomap included when loaded)
        if any(tuple(e.obs) != self.keys for e in episodes):
            raise ValueError(f"episodes carry different observation keys (first: {self.keys})")
        self.ends = np.cumsum(lengths)            # exclusive end of each episode in the flat step index
        self.total = int(self.ends[-1])
        self.has_goal = any(e.goal is not None for e in episodes)

    def draw_starts(self) -> list[tuple[int, int]]:
        """K (episode, start step) pairs, uniform over all dataset steps."""
        flat = torch.randint(0, self.total, (self.k,), generator=self.gen).numpy()
        ep = np.searchsorted(self.ends, flat, side="right")
        off = flat - (self.ends[ep] - np.array([self.eps[e].length for e in ep]))
        return [(int(e), int(o)) for e, o in zip(ep, off)]

    def _stack(self, idx: np.ndarray, eps: list[Episode]) -> dict[str, torch.Tensor]:
        """idx (K, L) step rows per sequence -> obs tensors (L, K, ...)."""
        out = {}
        for key in self.keys:
            arr = np.stack([eps[s].rows(key, idx[s]) for s in range(len(eps))], axis=1)   # unpacks egomap
            out[key] = torch.as_tensor(arr).to(device=self.device, dtype=torch_dtype(key, arr))
        return out

    def build(self, starts: list[tuple[int, int]]) -> BCBatch:
        l, k = self.l, len(starts)
        burn_idx = np.zeros((k, l), np.int64)
        burn_start = np.zeros((k, l), np.float32)
        main_idx = np.zeros((k, l), np.int64)
        main_start = np.zeros((k, l), np.float32)
        valid = np.zeros((k, l), bool)
        eps = [self.eps[e] for e, _ in starts]
        for s, (_, off) in enumerate(starts):
            n = eps[s].length
            s0 = max(0, off - l)
            b = off - s0                                   # real burn-in steps
            burn_idx[s, :l - b] = s0                       # front padding (reset below)
            burn_idx[s, l - b:] = np.arange(s0, off)
            if b > 0:
                burn_start[s, l - b] = 1.0                 # zero hidden state at the first real step
            real = min(l, n - off)
            main_idx[s, :real] = np.arange(off, off + real)
            main_idx[s, real:] = n - 1                     # tail padding, masked
            valid[s, :real] = True
            main_start[s, 0] = 1.0 if off == 0 else 0.0
        labels = np.stack([eps[s].labels[main_idx[s]] for s in range(k)], axis=1)
        dev = self.device
        goal = (torch.as_tensor(np.stack([eps[s].goal_rows(main_idx[s]) for s in range(k)], axis=1), device=dev).long()
                if self.has_goal else None)
        batch = BCBatch(self._stack(burn_idx, eps), torch.as_tensor(burn_start.T.copy(), device=dev),
                        self._stack(main_idx, eps), torch.as_tensor(main_start.T.copy(), device=dev),
                        torch.as_tensor(labels, device=dev).long(), torch.as_tensor(valid.T.copy(), device=dev), goal)
        assert batch.labels.shape[:2] == (l, k), f"BC labels expected (L={l}, K={k}, heads), got {tuple(batch.labels.shape)}"
        assert batch.valid.shape == (l, k) and batch.start.shape == (l, k) and batch.burn_start.shape == (l, k)
        return batch

    def sample(self) -> BCBatch:
        return self.build(self.draw_starts())


def batch_outputs(model: A0Agent, batch: BCBatch) -> AgentOutput:
    """Burn-in under no_grad from a zero state, then the graded window with gradient."""
    k = batch.start.shape[1]
    h = model.initial_state(k, batch.start.device)
    with torch.no_grad():
        _, h = model.forward_sequence(batch.burn_obs, h, batch.burn_start)
    out, _ = model.forward_sequence(batch.obs, h.detach(), batch.start)
    return out


def batch_logits(model: A0Agent, batch: BCBatch) -> list[torch.Tensor]:
    return batch_outputs(model, batch).logits


def bc_term(model: A0Agent, batch: BCBatch, goal_coef: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    """(loss tensor, stats): the summed per-head mean cross-entropy and per-head accuracy;
    with `goal_coef` > 0 plus goal_coef x the goal-head cross-entropy (train/goal_head.py)."""
    out = batch_outputs(model, batch)
    logits = out.logits
    c = bc_loss(logits, batch.labels, batch.valid, (BC_HEAD_WEIGHT,) * len(logits))
    n = max(c.n_valid, 1)
    stats = {"bc_loss": float(c.total.detach()), "bc_valid_steps": float(c.n_valid)}
    for h, name in enumerate(head_names(len(logits))):
        stats[f"bc_acc_{name}"] = c.head_correct[h] / n
    total = c.total
    if goal_coef > 0.0:
        if out.goal_logits is None or batch.goal is None:
            raise ValueError("BC goal_coef > 0 needs the model's goal head and goal_mode labels in the BC data")
        g = goal_loss(out.goal_logits, batch.goal, batch.valid)
        total = total + goal_coef * g.loss
        stats["bc_goal_loss"] = float(g.loss.detach())
        stats["bc_goal_acc"] = g.correct / max(g.n_labelled, 1)
        stats["bc_goal_labelled_steps"] = float(g.n_labelled)
    return total, stats


def make_term(sampler: BCSampler, goal_coef: float = 0.0
              ) -> Callable[[A0Agent], tuple[torch.Tensor, dict[str, float]]]:
    """A closure for ppo_update: draws a fresh batch per call."""
    return lambda model: bc_term(model, sampler.sample(), goal_coef)
