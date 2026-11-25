"""Synchronous recurrent PPO trainer (ADR 0004, amended by ADR 0007), written
against `train.vec_env.VecEnv`.

Episode ends (ADR 0007 ruling 5; freshness is per step and per env):
  - terminated (death / exit) -> next value 0; the final obs is never fed to the net.
  - time-out, fresh final obs -> bootstrap_value[t] = V(final_obs), computed with the
                        hidden state that processed obs[t] (one more core step, no
                        reset) and the episode's telemetry mask. This is the normal
                        case for env.DoomEnv.
  - time-out, stale final obs -> fallback: the transition is dropped (valid[t]=False)
                        and step t-1 bootstraps from values[t]. Never V(stale obs).
                        Counted per rollout as `stale_timeouts`; non-zero is an env bug.
  - rollout ends mid-episode  -> V(next obs) with the carried hidden state.
"""
from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from agent.distributions import FactorizedCategorical
from agent.model import A0Agent, build_agent
from agent.obs_spec import ObsSpec, spec_from_spaces
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig
from configs.train_config import RunConfig

from . import aux_labels as AL
from .gae import compute_gae
from .ppo import ppo_update
from .device import device_roles
from .rollout import Rollout, allocate, rollout_to
from .running_stats import ReturnScale
from .telemetry_mask import EpisodeDropout, mask_telemetry
from .vec_env import VecEnv, stack_obs


# Parameter-name prefixes that still train during the value warm-up (agent/model.py).
WARMUP_TRAINABLE_PREFIXES: tuple[str, ...] = ("value_ext.", "value_int.", "aux_keys.", "aux_damage.",
                                              "aux_monsters.")


# Seed offsets from the run seed (+1 act, +2 minibatch, +3 telemetry dropout, +4 BC).
RND_NET_SEED_OFFSET = 5
RND_SAMPLE_SEED_OFFSET = 6
INV_DYN_SEED_OFFSET = 7


class RolloutAbandoned(RuntimeError):
    """A stop was requested during collection; no weights were changed."""


def session_env_seed(run_seed: int, session: int) -> int:
    """Session 0 uses the run seed itself (first session unchanged)."""
    if session == 0:
        return run_seed
    return int(np.random.SeedSequence([run_seed, session]).generate_state(1)[0])


@dataclass
class EpisodeLog:
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    # Unshaped engine return per episode: `engine_return` when shaping is on, else
    # `return` (they are the same quantity then). ADR 0011 F7.
    engine_returns: list[float] = field(default_factory=list)
    # Per-episode records not yet written to disk; train/run_loop.py drains them
    # into episodes.jsonl (ADR 0009 Decision 5 item 2).
    pending: list[dict] = field(default_factory=list)


# How an episode ended, read from the env's end-of-episode info (env.DoomEnv keys).
END_KEYS = ("level_exit", "dead", "timeout")
# Added by rewards.wrapper.ShapedRewardEnv on an ending step (ADR 0010). Read by key
# only, so this module never imports rewards/.
SHAPING_EPISODE_KEY = "shaping_episode"
ENGINE_RETURN_KEY = "engine_return_unscaled"
# Per-step shaping values (info["shaping"]) averaged per update into metrics.jsonl
# (ADR 0010 change 3). For an env that ended, the step's info is final_infos[i].
SHAPING_STEP_KEY = "shaping"
SHAPING_STEP_FIELDS = ("extrinsic", "intrinsic", "intrinsic_raw", "intrinsic_coef", "total")
CELLS_VISITED_KEY = "cells_visited"
# Added by train.curriculum_vec_env.CurriculumVecEnv on an ending step: the stage and
# map the episode was played on. Logging only (episodes.jsonl); never an input.
CURRICULUM_KEY = "curriculum"
CURRICULUM_FIELDS = ("stage", "map_id", "replay")


def action_shares(actions: torch.Tensor, head_sizes: tuple[int, ...]) -> list[list[float]]:
    """Share of each option chosen per head over a rollout's (T, N, K) actions
    (ADR 0011 F1). Every sampled action counts, including stale time-out steps."""
    assert actions.dim() == 3 and actions.shape[-1] == len(head_sizes),         f"actions expected (T, N, {len(head_sizes)}), got {tuple(actions.shape)}"
    flat = actions.reshape(-1, len(head_sizes)).cpu()
    total = max(flat.shape[0], 1)
    return [(torch.bincount(flat[:, h], minlength=k)[:k].double() / total).tolist()
            for h, k in enumerate(head_sizes)]


def episode_record(update_idx: int, decision: int, env_index: int, ret: float, length: int,
                   terminated: bool, truncated: bool, final_info: dict | None) -> dict:
    """One episodes.jsonl line. `success` = ended by level exit.

    With reward shaping on, `return` is the shaped return; the record then also gets
    `engine_return` (the unshaped engine return, comparable with unshaped runs) and
    `shaping_episode` (the wrapper's per-term episode totals). Without shaping the
    record is unchanged. On a curriculum run it also gets `stage`, `map_id`, `replay`."""
    info = final_info or {}
    rec = {"update_idx": int(update_idx), "decision": int(decision), "env_index": int(env_index),
           "return": float(ret), "length": int(length),
           "terminated": bool(terminated), "truncated": bool(truncated)}
    for k in END_KEYS:
        rec[k] = bool(info.get(k, False))
    rec["success"] = rec["level_exit"]
    shaping = info.get(SHAPING_EPISODE_KEY)
    if shaping is not None:
        rec["engine_return"] = float(shaping[ENGINE_RETURN_KEY])
        rec[SHAPING_EPISODE_KEY] = dict(shaping)
    cur = info.get(CURRICULUM_KEY)
    if cur is not None:  # curriculum runs only; otherwise the record is unchanged
        for k in CURRICULUM_FIELDS:
            rec[k] = cur[k]
    return rec


class Trainer:
    def __init__(self, vec_env: VecEnv, model_cfg: ModelConfig, run: RunConfig) -> None:
        self.env, self.run, self.ppo = vec_env, run, run.ppo
        torch.set_num_threads(run.torch_threads)  # user's load cap (ADR 0004)
        # `device` runs rollout inference (and holds hidden states and rollouts);
        # `learn_device` runs the PPO update. They differ only for run.device "hybrid".
        act_dev, learn_dev = device_roles(run.device)
        self.device, self.learn_device = torch.device(act_dev), torch.device(learn_dev)
        self.n = vec_env.num_envs
        if self.n % self.ppo.num_minibatches != 0:
            raise ValueError(f"num_envs {self.n} not divisible by num_minibatches {self.ppo.num_minibatches}")
        self.spec: ObsSpec = spec_from_spaces(vec_env.single_observation_space, vec_env.single_action_space)
        self.model: A0Agent = build_agent(self.spec, model_cfg, seed=run.seed).to(self.device)
        # The learner owns the optimizer. Same object as `model` unless hybrid, where it is
        # a copy on the learn device and `model` (the actor) is refreshed from it by
        # `sync_actor()` after every update and every load.
        self.learner: A0Agent = (self.model if self.learn_device == self.device
                                 else copy.deepcopy(self.model).to(self.learn_device))
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.ppo.lr, eps=self.ppo.adam_eps)
        self.act_gen = torch.Generator(device="cpu").manual_seed(run.seed + 1)
        self.mb_gen = torch.Generator(device="cpu").manual_seed(run.seed + 2)
        self.dropout = EpisodeDropout(self.n, run.telemetry_mask.episode_prob, seed=run.seed + 3)
        self.scale = ReturnScale(self.ppo.return_scale_eps, self.ppo.return_scale_min)
        self.update_idx = 0
        self.decisions_done = 0       # LR decay position (ADR 0008 point 1), saved in checkpoints
        self.session = 0              # resume counter; seeds env resets (ADR 0008 Decision 4)
        self.last_stale_timeouts = 0
        self.stale_timeouts_total = 0
        self.last_action_shares: list[list[float]] = []
        self.last_shaping_means: dict[str, float] = {}
        self.log = EpisodeLog()
        # Imitation auxiliary loss (train/bc_aux.py): data attached by `attach_bc`.
        self.bc_sampler = None
        # RND bonus and inverse-dynamics head: built only when switched on (off = nothing
        # built, no RNG drawn, the update is bit-identical). Both live on the learn device.
        self.rnd = None
        if run.rnd.enabled:
            from .rnd import RND, feature_layout
            shapes = {k: tuple(sp.shape) for k, sp in vec_env.single_observation_space.spaces.items()}
            self.rnd = RND(feature_layout(shapes, run.rnd), run.rnd, self.n, net_seed=run.seed + RND_NET_SEED_OFFSET,
                           sample_seed=run.seed + RND_SAMPLE_SEED_OFFSET, device=self.learn_device)
        self.inv_dyn = None
        if run.inv_dyn.enabled:
            from .inverse_dynamics import build_head, make_term
            head = build_head(model_cfg.fusion_dim, self.spec.head_sizes, run.inv_dyn,
                              run.seed + INV_DYN_SEED_OFFSET, self.learn_device)
            self.inv_dyn = make_term(head, run.inv_dyn)
        self.restart_episodes(run.seed)

    def attach_bc(self, episodes: list) -> None:
        """Give the trainer its BC episodes (loaded and validated by the caller against
        this trainer's obs spec). The sampler has its own seeded generator (seed + 4)."""
        from .bc_aux import BCSampler
        cfg = self.run.bc_aux
        if not cfg.enabled:
            raise ValueError("attach_bc: the run config has the BC auxiliary loss off")
        self.bc_sampler = BCSampler(episodes, cfg.seq_len, cfg.batch_seqs, self.run.seed + 4, self.learn_device)

    def restart_episodes(self, env_seed: int) -> None:
        """Reset every env with `env_seed`; zero hidden state; fresh telemetry-mask draws."""
        self.obs, self.infos = self.env.reset(seed=env_seed)
        self.episode_start = np.ones(self.n, np.float32)
        self.dropped = self.dropout.on_episode_start(np.ones(self.n, bool))
        self.h = self.model.initial_state(self.n, self.device)
        self._ep_ret = np.zeros(self.n, np.float64)
        self._ep_len = np.zeros(self.n, np.int64)

    # ------------------------------------------------------------ helpers
    def _t(self, x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, device=self.device).to(dtype)

    def _policy_input(self, obs_np: dict[str, np.ndarray], dropped: np.ndarray) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        true = to_tensors(obs_np, self.device)
        return true, mask_telemetry(true, self._t(dropped, torch.bool), self.model.cfg)

    def _rnd_next_features(self, st: Any, ended: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """RND input of each env's NEXT observation o_{t+1}: `st.obs` where the episode
        goes on, the fresh final observation where it ended; usable flag False (reward
        0, no statistic) where the final observation is stale (death, exit, stale
        time-out)."""
        assert self.rnd is not None
        feat = self.rnd.features(st.obs)
        ok = np.ones(self.n, bool)
        for i in np.flatnonzero(ended):
            fo = st.final_obs[i]
            if st.final_obs_fresh[i] and fo is not None:
                feat[i] = self.rnd.features({k: v[None] for k, v in fo.items()})[0]
            else:
                ok[i] = False
                feat[i] = 0.0
        return feat, ok

    # ------------------------------------------------------------ collection
    @torch.no_grad()
    def collect(self, should_stop: Callable[[], bool] = lambda: False) -> Rollout:
        """Raises RolloutAbandoned if `should_stop()` turns true mid-rollout (weights unchanged)."""
        t_len, aux = self.ppo.rollout_len, self.run.aux
        rnd = self.rnd
        ro = allocate(self.spec, t_len, self.n, self.model.cfg.core_hidden, self.model.cfg.n_key_bits, self.device,
                      rnd_dim=rnd.layout.dim if rnd is not None else 0)
        ro.h0.copy_(self.h)
        events = np.zeros((t_len, self.n), np.float32)
        events_ok = np.zeros((t_len, self.n), bool)
        ended_all = np.zeros((t_len, self.n), bool)
        stale_count = 0
        scale = self.scale.scale
        sh_sums = dict.fromkeys(SHAPING_STEP_FIELDS, 0.0)
        sh_steps = 0
        cells: list[float] = []
        for t in range(t_len):
            if should_stop():
                raise RolloutAbandoned(f"stop requested at rollout step {t}")
            true, inp = self._policy_input(self.obs, self.dropped)
            starts = self._t(self.episode_start)
            out, h_new = self.model.step(inp, self.h, starts)
            dist = FactorizedCategorical(out.logits)
            action = dist.sample(self.act_gen)
            for k, v in true.items():
                ro.obs[k][t] = v
            ro.telemetry_dropped[t] = self._t(self.dropped, torch.bool)
            ro.episode_start[t] = starts
            ro.actions[t] = action
            ro.logp[t] = dist.log_prob(action)
            ro.values[t] = out.value_ext * scale
            if rnd is not None:
                ro.values_int[t] = out.value_int
            keys, keys_ok = AL.keys_from_infos(self.infos)
            ro.keys[t], ro.keys_valid[t] = self._t(keys), self._t(keys_ok, torch.bool)
            ro.monsters[t] = self._t(AL.visible_monsters(self.obs["entity_type"], self.obs["entity_mask"],
                                                         aux.monster_type_id, aux.monster_count_cap))

            st = self.env.step(action.cpu().numpy())
            ended = st.terminated | st.truncated
            ro.rewards[t] = self._t(st.reward)
            ro.terminated[t] = self._t(st.terminated, torch.bool)
            ro.truncated[t] = self._t(st.truncated, torch.bool)
            stale_timeout = st.truncated & ~st.final_obs_fresh
            stale_count += int(stale_timeout.sum())
            ro.valid[t] = self._t(~stale_timeout, torch.bool)
            boot = np.flatnonzero(st.truncated & st.final_obs_fresh)
            if boot.size:
                fo = stack_obs([st.final_obs[i] for i in boot])  # type: ignore[misc]
                ft, fin = self._policy_input(fo, self.dropped[boot])
                bi = torch.as_tensor(boot, device=self.device)
                fout, _ = self.model.step(fin, h_new.index_select(0, bi), torch.zeros(boot.size, device=self.device))
                ro.bootstrap_value[t, bi] = fout.value_ext * scale
            if rnd is not None:
                feat, ok = self._rnd_next_features(st, ended)
                ro.rnd_feat[t], ro.rnd_ok[t] = self._t(feat), self._t(ok, torch.bool)
            final_self = [None if fo is None else fo["self"] for fo in st.final_obs]
            events[t], events_ok[t] = AL.damage_events(st.obs["self"], ended, final_self, st.final_obs_fresh,
                                                      st.final_infos, aux.self_damage_index)
            ended_all[t] = ended
            for i in range(self.n):  # read by key only: this module never imports rewards/
                info = st.final_infos[i] if ended[i] else st.infos[i]
                sh = info.get(SHAPING_STEP_KEY) if info else None
                if sh is not None:
                    sh_steps += 1
                    for k in SHAPING_STEP_FIELDS:
                        sh_sums[k] += float(sh[k])

            self._ep_ret += st.reward
            self._ep_len += 1
            for i in np.flatnonzero(ended):
                rec = episode_record(self.update_idx, self.decisions_done + (t + 1) * self.n, i,
                                     self._ep_ret[i], self._ep_len[i], st.terminated[i], st.truncated[i],
                                     st.final_infos[i])
                self.log.returns.append(rec["return"])
                self.log.lengths.append(rec["length"])
                self.log.successes.append(rec["success"])
                self.log.engine_returns.append(rec.get("engine_return", rec["return"]))
                self.log.pending.append(rec)
                cv = rec.get(SHAPING_EPISODE_KEY, {}).get(CELLS_VISITED_KEY)
                if cv is not None:
                    cells.append(float(cv))
                self._ep_ret[i], self._ep_len[i] = 0.0, 0
            self.h = h_new
            self.obs, self.infos = st.obs, st.infos
            self.episode_start = ended.astype(np.float32)
            self.dropped = self.dropout.on_episode_start(ended)

        tgt, ok = AL.damage_window_targets(events, events_ok, ended_all, aux.damage_horizon_k)
        ro.damage.copy_(self._t(tgt))
        ro.damage_valid.copy_(self._t(ok, torch.bool))
        _, inp = self._policy_input(self.obs, self.dropped)
        last, _ = self.model.step(inp, self.h, self._t(self.episode_start))
        ro.last_value.copy_(last.value_ext * scale)
        if rnd is not None:
            ro.last_value_int.copy_(last.value_int)
        self.last_stale_timeouts = stale_count
        self.stale_timeouts_total += stale_count
        self.last_action_shares = action_shares(ro.actions, self.spec.head_sizes)
        self.last_shaping_means = {}
        if sh_steps:
            self.last_shaping_means = {f"shaping_mean_{k}": v / sh_steps for k, v in sh_sums.items()}
            # wiring check: equals rollout_len x num_envs when every step carried shaping info
            self.last_shaping_means["shaping_steps_logged"] = float(sh_steps)
        if cells:
            self.last_shaping_means["shaping_cells_visited_mean"] = sum(cells) / len(cells)
        return ro

    # ------------------------------------------------------------ update
    def advantages(self, ro: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_gae(ro.rewards, ro.values, ro.last_value, ro.terminated, ro.truncated,
                           ro.bootstrap_value, ro.valid, self.ppo.gamma, self.ppo.gae_lambda)

    def _rnd_advantages(self, ro: Rollout, adv_ext: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Intrinsic rewards (normalised) -> NON-episodic GAE with `value_int` (no
        termination or truncation cut; a dropped stale step still gets advantage 0) ->
        combined advantage ext_coef x A_ext + int_coef x A_int. Returns (advantage,
        intrinsic value targets, statistics)."""
        assert self.rnd is not None
        cfg = self.run.rnd
        r_int, stats = self.rnd.intrinsic_rewards(ro.rnd_feat, ro.rnd_ok)
        r_int = r_int.to(ro.values_int.device)
        never = torch.zeros_like(ro.valid)
        adv_int, ret_int = compute_gae(r_int, ro.values_int, ro.last_value_int, never, never,
                                       torch.zeros_like(ro.values_int), ro.valid, cfg.gamma_int, cfg.gae_lambda_int)
        return cfg.ext_coef * adv_ext + cfg.int_coef * adv_int, ret_int, stats

    def _set_lr(self) -> None:
        """Constant unless a final fixed-length decay is configured, defined over decisions
        (ADR 0008 point 1); never below lr_decay_floor_fraction (enforced in PPOConfig)."""
        final, span = self.ppo.lr_final_fraction, self.ppo.lr_decay_decisions
        frac = min(self.decisions_done / span, 1.0) if (final < 1.0 and span) else 0.0
        lr = self.ppo.lr * (1.0 - frac * (1.0 - final)) * self.lr_warmup_factor()
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    @property
    def policy_update_idx(self) -> int:
        """Index of the next update counted from the end of the value warm-up (negative
        during it). Clock of the LR warm-up and of the BC coefficient schedule."""
        return self.update_idx - self.ppo.value_warmup_updates

    def lr_warmup_factor(self) -> float:
        """(k+1)/N for policy update k < N (PPOConfig.lr_warmup_updates); 1 otherwise,
        including during the value warm-up."""
        n, k = self.ppo.lr_warmup_updates, self.policy_update_idx
        return min((k + 1) / n, 1.0) if (n and k >= 0) else 1.0

    def bc_coef_now(self) -> float:
        """Effective BC coefficient of the next update (0 during the value warm-up)."""
        from .bc_aux import coef_at
        if not self.run.bc_aux.enabled or self.in_value_warmup:
            return 0.0
        return coef_at(self.run.bc_aux, self.policy_update_idx)

    @property
    def split_devices(self) -> bool:
        return self.learner is not self.model

    @torch.no_grad()
    def sync_actor(self) -> None:
        """Copy the learner's parameters and buffers into the actor (hybrid only)."""
        if self.split_devices:
            self.model.load_state_dict(self.learner.state_dict())

    @property
    def in_value_warmup(self) -> bool:
        """True while the next update is a value warm-up update (PPOConfig.value_warmup_updates)."""
        return self.update_idx < self.ppo.value_warmup_updates

    def _freeze_all_but_value_and_aux(self, freeze: bool) -> None:
        """Warm-up: every learner parameter except the value and auxiliary heads gets
        requires_grad False, so it receives no gradient and Adam skips it."""
        for name, p in self.learner.named_parameters():
            if not name.startswith(WARMUP_TRAINABLE_PREFIXES):
                p.requires_grad_(not freeze)

    def update(self, ro: Rollout) -> dict[str, float]:
        if self.split_devices:
            ro = rollout_to(ro, self.learn_device)
        adv, ret = self.advantages(ro)
        self.scale.update(ret[ro.valid])
        ret_scaled = ret / self.scale.scale
        adv_used, ret_int, rnd_stats = adv, None, {}
        if self.rnd is not None:
            adv_used, ret_int, rnd_stats = self._rnd_advantages(ro, adv)
        self._set_lr()
        warm = self.in_value_warmup
        bc_coef, bc_fn = 0.0, None
        if self.run.bc_aux.enabled:
            if self.bc_sampler is None:
                raise RuntimeError("BC auxiliary loss configured but no BC data attached (Trainer.attach_bc)")
            bc_coef = self.bc_coef_now()
            if bc_coef > 0.0:  # coefficient 0: nothing drawn, nothing computed
                from .bc_aux import make_term
                bc_fn = make_term(self.bc_sampler)
        if warm:
            self._freeze_all_but_value_and_aux(True)
        try:
            stats = ppo_update(self.learner, self.optimizer, ro, adv_used, ret_scaled, self.ppo, self.run.aux,
                               self.mb_gen, policy_loss_scale=0.0 if warm else 1.0,
                               bc_term=bc_fn, bc_coef=bc_coef, int_returns=ret_int, inv_dyn=self.inv_dyn)
        finally:
            if warm:
                self._freeze_all_but_value_and_aux(False)
        if self.rnd is not None:
            stats.update(rnd_stats)
            stats.update(self.rnd.train_predictor(ro.rnd_feat, ro.rnd_ok))
            with torch.no_grad():
                vi, ri = ro.values_int[ro.valid], ret_int[ro.valid]
                var_ri = float(ri.var()) if ri.numel() > 1 else 0.0
                stats["value_int_explained_variance"] = (1.0 - float((ri - vi).var()) / var_ri
                                                         if var_ri > 0 else float("nan"))
        if self.run.bc_aux.enabled:
            stats["bc_coef"] = bc_coef
        stats["value_warmup_active"] = 1.0 if warm else 0.0
        # 1.0 on the last warm-up update only; train/run_loop.py logs a line for it
        stats["value_warmup_ended"] = 1.0 if (warm and self.update_idx + 1 == self.ppo.value_warmup_updates) else 0.0
        self.sync_actor()
        with torch.no_grad():
            v, r = ro.values[ro.valid], ret[ro.valid]
            var_r = float(r.var()) if r.numel() > 1 else 0.0
            stats["explained_variance"] = 1.0 - float((r - v).var()) / var_r if var_r > 0 else float("nan")
        stats["return_scale"] = self.scale.scale
        stats["lr"] = self.optimizer.param_groups[0]["lr"]
        self.update_idx += 1
        self.decisions_done += ro.shape[0] * ro.shape[1]
        return stats

    def train_iteration(self, should_stop: Callable[[], bool] = lambda: False) -> dict[str, float]:
        """One rollout and one update. A stop request abandons the rollout; once the
        update has started it always completes, so weights stay on an update boundary."""
        t0 = time.perf_counter()
        ro = self.collect(should_stop)
        t1 = time.perf_counter()
        stats = self.update(ro)
        t2 = time.perf_counter()
        steps = ro.shape[0] * ro.shape[1]
        stats["sample_dps"] = steps / max(t1 - t0, 1e-9)
        stats["update_dps"] = steps / max(t2 - t1, 1e-9)
        stats["stale_timeouts"] = float(self.last_stale_timeouts)
        stats["stale_timeouts_total"] = float(self.stale_timeouts_total)
        stats["update_idx"] = float(self.update_idx)
        for h, shares in enumerate(self.last_action_shares):
            for k, v in enumerate(shares):
                stats[f"action_share_head{h}_opt{k}"] = v
        stats.update(self.last_shaping_means)
        return stats

    # ------------------------------------------------------------ resume
    def state_dict(self) -> dict:
        """Everything needed to resume learning, in types `torch.load(weights_only=True)`
        accepts (the NumPy RNG state is stored as a JSON string). Env episodes are not
        resumed: see `resume_session`."""
        d = {
            "model": self.learner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "return_scale": self.scale.state_dict(),
            "update_idx": self.update_idx,
            "decisions_done": self.decisions_done,
            "session": self.session,
            "stale_timeouts_total": self.stale_timeouts_total,
            "act_gen": self.act_gen.get_state(),
            "mb_gen": self.mb_gen.get_state(),
            "dropout_rng_json": json.dumps(self.dropout.rng.bit_generator.state),
        }
        if self.bc_sampler is not None:
            d["bc_gen"] = self.bc_sampler.gen.get_state()
        if self.rnd is not None:
            d["rnd"] = self.rnd.state_dict()
        if self.inv_dyn is not None:
            d["inv_dyn"] = {"head": {k: v.cpu() for k, v in self.inv_dyn.head.state_dict().items()},
                            "optimizer": self.inv_dyn.optimizer.state_dict()}
        return d

    def load_state_dict(self, d: dict) -> None:
        self.learner.load_state_dict(d["model"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.sync_actor()
        self.scale.load_state_dict(d["return_scale"])
        self.update_idx = int(d["update_idx"])
        self.decisions_done = int(d["decisions_done"])
        self.session = int(d["session"])
        self.stale_timeouts_total = int(d["stale_timeouts_total"])
        self.act_gen.set_state(d["act_gen"])
        self.mb_gen.set_state(d["mb_gen"])
        self.dropout.rng.bit_generator.state = json.loads(d["dropout_rng_json"])
        if self.bc_sampler is not None and "bc_gen" in d:
            self.bc_sampler.gen.set_state(d["bc_gen"])
        # The switches are compared by train.checkpoint.resume before this is called.
        if self.rnd is not None:
            self.rnd.load_state_dict(d["rnd"])
        if self.inv_dyn is not None:
            self.inv_dyn.head.load_state_dict(d["inv_dyn"]["head"])
            self.inv_dyn.optimizer.load_state_dict(d["inv_dyn"]["optimizer"])

    def resume_session(self) -> int:
        """After load_state_dict: start a new session whose env seeds are derived from
        the run seed and the session counter, so resumed runs do not replay the same
        episodes (ADR 0008 Decision 4). Returns the env seed used."""
        self.session += 1
        seed = session_env_seed(self.run.seed, self.session)
        self.restart_episodes(seed)
        return seed
