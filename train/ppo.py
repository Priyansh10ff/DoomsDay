"""Recurrent PPO loss and update (ADR 0004). Minibatches are whole env sequences;
each is recomputed from its stored h0 with the stored episode_start resets."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from agent.distributions import FactorizedCategorical
from agent.model import A0Agent
from configs.train_config import AuxConfig, PPOConfig

from .inverse_dynamics import InvDynTerm, inverse_dynamics_loss
from .rollout import Rollout, select_envs
from .telemetry_mask import mask_telemetry


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


@dataclass
class MinibatchResult:
    loss: torch.Tensor
    stats: dict[str, float]
    ratio: torch.Tensor
    logp: torch.Tensor


def evaluate_minibatch(model: A0Agent, ro: Rollout, idx: torch.Tensor, adv: torch.Tensor,
                       returns_scaled: torch.Tensor, ppo: PPOConfig, aux: AuxConfig,
                       policy_loss_scale: float = 1.0, int_returns: torch.Tensor | None = None,
                       inv_dyn: InvDynTerm | None = None) -> MinibatchResult:
    """`policy_loss_scale` multiplies the policy-gradient and entropy terms (0 during
    the value warm-up, PPOConfig.value_warmup_updates).

    `int_returns` (T, N) (RND on; None = off): adds value_coef x the `value_int` loss
    against these targets (already in normalised units). `inv_dyn` (None = off): adds
    coef x the inverse-dynamics loss (train/inverse_dynamics.py). With both None the
    loss is exactly the one without them."""
    obs = {k: select_envs(v, idx) for k, v in ro.obs.items()}
    dropped = select_envs(ro.telemetry_dropped, idx)
    obs_in = mask_telemetry(obs, dropped, model.cfg)
    out, _ = model.forward_sequence(obs_in, ro.h0.index_select(0, idx), select_envs(ro.episode_start, idx))
    dist = FactorizedCategorical(out.logits)
    actions = select_envs(ro.actions, idx)
    logp = dist.log_prob(actions)
    old_logp = select_envs(ro.logp, idx)
    valid = select_envs(ro.valid, idx)
    a = select_envs(adv, idx)
    vm = valid.float()
    a_mean = masked_mean(a, valid)
    a_std = torch.sqrt(masked_mean((a - a_mean) ** 2, valid))
    a = (a - a_mean) / (a_std + ppo.advantage_eps)

    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    pg = -torch.min(ratio * a, ratio.clamp(1.0 - ppo.clip, 1.0 + ppo.clip) * a)
    pg_loss = masked_mean(pg, valid)
    v_target = select_envs(returns_scaled, idx)
    v_loss = masked_mean(0.5 * (out.value_ext - v_target) ** 2, valid)
    ent_heads = dist.entropy_per_head()
    entropy = masked_mean(ent_heads.sum(-1), valid)

    k_ok = select_envs(ro.keys_valid, idx) & valid
    k_bce = F.binary_cross_entropy_with_logits(out.aux_keys, select_envs(ro.keys, idx), reduction="none").mean(-1)
    keys_loss = masked_mean(k_bce, k_ok)
    d_ok = select_envs(ro.damage_valid, idx) & valid
    d_bce = F.binary_cross_entropy_with_logits(out.aux_damage, select_envs(ro.damage, idx), reduction="none")
    damage_loss = masked_mean(d_bce, d_ok)
    monsters_loss = masked_mean((out.aux_monsters - select_envs(ro.monsters, idx)) ** 2, valid)

    loss = (policy_loss_scale * (pg_loss - ppo.entropy_coef * entropy) + ppo.value_coef * v_loss
            + aux.keys_weight * keys_loss + aux.damage_weight * damage_loss
            + aux.monsters_weight * monsters_loss)
    extra_stats: dict[str, float] = {}
    if int_returns is not None:
        v_int_loss = masked_mean(0.5 * (out.value_int - select_envs(int_returns, idx)) ** 2, valid)
        loss = loss + ppo.value_coef * v_int_loss
        extra_stats["v_int_loss"] = float(v_int_loss.detach())
    if inv_dyn is not None:
        inv_loss, inv_stats = inverse_dynamics_loss(model, inv_dyn.head, obs_in, actions,
                                                    select_envs(ro.episode_start, idx), valid)
        loss = loss + inv_dyn.coef * inv_loss
        extra_stats.update(inv_stats)
    with torch.no_grad():
        approx_kl = masked_mean((ratio - 1.0) - log_ratio, valid)
        clipfrac = masked_mean(((ratio - 1.0).abs() > ppo.clip).float(), valid)
    stats = {
        "loss": float(loss.detach()), "pg_loss": float(pg_loss.detach()), "v_loss": float(v_loss.detach()),
        "entropy": float(entropy.detach()), "approx_kl": float(approx_kl), "clipfrac": float(clipfrac),
        "aux_keys": float(keys_loss.detach()), "aux_damage": float(damage_loss.detach()),
        "aux_monsters": float(monsters_loss.detach()),
        "valid_frac": float(vm.mean()),
        # ADR 0011 F2: the minibatch's advantage spread BEFORE normalisation (logging only)
        "adv_std_prenorm": float(a_std), "adv_mean_prenorm": float(a_mean),
    }
    for i in range(ent_heads.shape[-1]):
        stats[f"entropy_head{i}"] = float(masked_mean(ent_heads[..., i].detach(), valid))
    stats.update(extra_stats)
    return MinibatchResult(loss, stats, ratio.detach(), logp.detach())


BCTerm = Callable[[A0Agent], tuple[torch.Tensor, dict[str, float]]]


def ppo_update(model: A0Agent, optimizer: torch.optim.Optimizer, ro: Rollout, adv: torch.Tensor,
               returns_scaled: torch.Tensor, ppo: PPOConfig, aux: AuxConfig,
               generator: torch.Generator, policy_loss_scale: float = 1.0,
               bc_term: BCTerm | None = None, bc_coef: float = 0.0,
               int_returns: torch.Tensor | None = None, inv_dyn: InvDynTerm | None = None) -> dict[str, float]:
    """`int_returns` / `inv_dyn`: see `evaluate_minibatch` (None = off, unchanged path).
    The inverse-dynamics head has its own optimizer and gradient clip, so the model's
    optimizer state and `clip_grad_norm_(model.parameters())` are the same as without it.
    `bc_term` (train/bc_aux.py; None = off, the code path is then unchanged) is
    called once per minibatch step and its loss added x `bc_coef`. `loss` in the stats
    is the PPO loss without it. `ppo.kl_stop` > 0: a minibatch whose approx KL exceeds
    it is not stepped, and the rest of the update is skipped (`kl_stopped`,
    `kl_stop_minibatch` = global minibatch index, -1 when not stopped)."""
    _, n = ro.shape
    assert n % ppo.num_minibatches == 0, f"num_envs {n} not divisible by num_minibatches {ppo.num_minibatches}"
    per = n // ppo.num_minibatches
    agg: dict[str, float] = {}
    count = 0
    first_ratio_dev = float("nan")
    stopped_at, stop_kl = -1, float("nan")
    for epoch in range(ppo.epochs):
        perm = torch.randperm(n, generator=generator)
        for mb in range(ppo.num_minibatches):
            idx = perm[mb * per:(mb + 1) * per].to(ro.h0.device)
            res = evaluate_minibatch(model, ro, idx, adv, returns_scaled, ppo, aux, policy_loss_scale,
                                     int_returns=int_returns, inv_dyn=inv_dyn)
            if epoch == 0 and mb == 0:
                v = select_envs(ro.valid, idx)
                first_ratio_dev = float((res.ratio - 1.0).abs()[v].max()) if bool(v.any()) else 0.0
            if ppo.kl_stop > 0 and count > 0 and res.stats["approx_kl"] > ppo.kl_stop:  # always step once
                stopped_at = epoch * ppo.num_minibatches + mb
                stop_kl = res.stats["approx_kl"]
                break
            loss = res.loss
            if bc_term is not None:
                bc_loss_t, bc_stats = bc_term(model)
                loss = loss + bc_coef * bc_loss_t
                res.stats.update(bc_stats)
            optimizer.zero_grad(set_to_none=True)
            if inv_dyn is not None:
                inv_dyn.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
            optimizer.step()
            if inv_dyn is not None:
                torch.nn.utils.clip_grad_norm_(inv_dyn.head.parameters(), inv_dyn.max_grad_norm)
                inv_dyn.optimizer.step()
            res.stats["grad_norm"] = float(gnorm)
            for k, val in res.stats.items():
                agg[k] = agg.get(k, 0.0) + val
            count += 1
        if stopped_at >= 0:
            break
    out = {k: v / count for k, v in agg.items()}
    out["first_minibatch_ratio_max_dev"] = first_ratio_dev
    if ppo.kl_stop > 0:
        out["kl_stopped"] = 1.0 if stopped_at >= 0 else 0.0
        out["kl_stop_minibatch"] = float(stopped_at)
        out["minibatches_stepped"] = float(count)
        out["kl_stop_value"] = stop_kl        # the KL that triggered the stop (NaN if none)
    return out
