"""Recurrent PPO settings (ADR 0004, amended by ADR 0007) and run layout / safety limits.

No torch import (env worker processes may import configs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024 ** 3


@dataclass(frozen=True)
class PPOConfig:
    rollout_len: int = 64
    num_minibatches: int = 4          # minibatches are whole env sequences
    epochs: int = 3
    clip: float = 0.2
    lr: float = 3e-4
    # ADR 0008 point 1: constant learning rate. A decay is allowed only for a final
    # run of fixed length; it is defined over DECISIONS (`lr_decay_decisions`, None =
    # constant), its position is saved in the checkpoint, and it never goes below
    # `lr_decay_floor_fraction` of the base rate.
    lr_final_fraction: float = 1.0
    lr_decay_decisions: int | None = None
    lr_decay_floor_fraction: float = 0.1
    max_grad_norm: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01        # on the summed entropy of all heads
    gamma: float = 0.997
    gae_lambda: float = 0.95
    adam_eps: float = 1e-5            # accepted by ADR 0007
    # Value targets divided by the running std of GAE returns, floored at
    # `return_scale_min` (ADR 0008 point 3: 0.1, i.e. at most 10x amplification).
    return_scale_eps: float = 1e-8
    return_scale_min: float = 0.1
    advantage_eps: float = 1e-8
    # Value warm-up (coordinator ruling 2, 2026-11-25; meant for --init-from): for the
    # first N updates the policy-gradient and entropy terms are multiplied by 0 and
    # every parameter except the value and auxiliary heads is frozen, so only those
    # heads move. 0 = off.
    value_warmup_updates: int = 0
    # Linear LR warm-up (2026-11-25 evening, after the BC hand-off's first policy update
    # measured approx KL 0.155): the first N POLICY updates (counted after the value
    # warm-up) use lr x (k+1)/N, k = 0..N-1; the value warm-up updates use the full rate.
    # Multiplies the decisions-based decay above. 0 = off.
    lr_warmup_updates: int = 0
    # KL early stop (2026-11-25 strategy, C2): within one update, once a minibatch's
    # approx KL (measured before its optimizer step) exceeds this, that step and every
    # remaining minibatch/epoch of the update are skipped. 0 = off.
    kl_stop: float = 0.0

    def __post_init__(self) -> None:
        if self.kl_stop < 0:
            raise ValueError(f"kl_stop must be >= 0, got {self.kl_stop}")
        if self.lr_warmup_updates < 0:
            raise ValueError(f"lr_warmup_updates must be >= 0, got {self.lr_warmup_updates}")
        if not (self.lr_decay_floor_fraction <= self.lr_final_fraction <= 1.0):
            raise ValueError(f"lr_final_fraction {self.lr_final_fraction} must be in "
                             f"[{self.lr_decay_floor_fraction}, 1.0] (ADR 0008: never decay to zero)")
        if self.lr_final_fraction < 1.0 and not self.lr_decay_decisions:
            raise ValueError("a decaying learning rate needs lr_decay_decisions (ADR 0008 point 1)")
        if self.value_warmup_updates < 0:
            raise ValueError(f"value_warmup_updates must be >= 0, got {self.value_warmup_updates}")


@dataclass(frozen=True)
class AuxConfig:
    # Starting weights accepted by ADR 0007 (keys weight marked for an experiment).
    keys_weight: float = 0.1
    damage_weight: float = 0.1
    monsters_weight: float = 0.1
    # Damage head: "any damage within the next k decisions" (binary), ADR 0007.
    damage_horizon_k: int = 8
    # Monster-count head target: min(count, cap) / cap, MSE (ADR 0007 ruling 2).
    monster_count_cap: int = 8
    # Coupled to env tables; tests fail if the env tables move (ADR 0007 ruling 2).
    # Index of "monster" in env.observation.ENTITY_TYPES.
    monster_type_id: int = 1
    # Index of "damage taken since last decision" in `self` (env.observation.self_vector).
    self_damage_index: int = 3


@dataclass(frozen=True)
class TelemetryMaskConfig:
    """ADR 0006: trainer-side modality dropout. A0 trains with it off (0.0)."""
    episode_prob: float = 0.0


@dataclass(frozen=True)
class HealthMonitorConfig:
    """ADR 0011 Decision 2: the training-health monitor (F4). Training-side logging
    only; it never reaches an observation or an action. Thresholds are ADR 0011's [J]
    values. Success fractions are compared exactly, as counts over the window.

    Coordinator rulings (2026-11-25): full windows only (no success value before
    `window_episodes` episodes, no mean before its update count exists); head maxima
    are exact ln(option count)."""
    enabled: bool = True
    window_episodes: int = 500            # trailing finished episodes behind "success"
    watch_mean_updates: int = 100         # summed-entropy mean for the watch level
    watch_entropy_fraction: float = 0.5   # watch: summed-entropy mean below this x max
    watch_success_below: float = 0.5      # ... while success is below this
    warn_mean_updates: int = 50           # per-head entropy mean for the warning level
    warn_head_fraction: float = 0.1       # warning: a head's mean below this x ln(k)
    event_drop_from_best: float = 0.15    # event: success at least this far below run best
    event_margin_over_p0: float = 0.10    # ... or success below p0 + this
    stop_after_updates: int = 600         # stop: an event still active this long after it began
    stop_enabled: bool = True             # False: log the stop level but never request a stop
    # Random-policy baseline p0 of the env configuration (ADR 0011 Decision 5): A0 TRAIN
    # on my_way_home, 28 of 200 (reports/runs/2026-11-25-random-baseline.json).
    p0_successes: int = 28
    p0_episodes: int = 200
    # Names for the env's action heads in log lines (env/actions.py HEAD_NAMES; a test
    # checks they agree). Heads beyond this tuple are named head<i>.
    head_names: tuple[str, ...] = ("move", "strafe", "turn", "attack", "use", "speed")


BC_DECAY_MODES: tuple[str, ...] = ("linear", "exp")


@dataclass(frozen=True)
class BCAuxConfig:
    """Imitation (behaviour-cloning) auxiliary loss inside PPO (train/bc_aux.py;
    2026-11-25 evening, gap-analysis gap 3: PPO from a cloned model erased it).

    Every PPO minibatch step adds coef x sum over heads of the cross-entropy of the
    policy logits against the teacher label, on `batch_seqs` sequences of `seq_len`
    decisions sampled from the teacher datasets. Coefficient schedule over POLICY
    updates k (counted after the value warm-up, during which the term is off):
      decay_mode "linear": decay_updates == 0 -> `coef` always; else linear from `coef`
        to `coef_final` over `decay_updates` updates, then `coef_final`.
      decay_mode "exp": max(coef x decay_rate^k, coef_final) (`coef_final` is the floor).
    Training-only: nothing here reaches an
    observation or the inference path."""
    data_dirs: tuple[str, ...] = ()          # absolute dataset dirs (ep_*.npz + index.jsonl)
    data_sha256: tuple[str, ...] = ()        # sha256 of each dir's index.jsonl (recorded, compared on resume)
    coef: float = 0.0
    coef_final: float = 0.0
    decay_updates: int = 0
    decay_mode: str = "linear"               # "linear" | "exp"
    decay_rate: float = 0.999                # per policy update, decay_mode "exp" only
    batch_seqs: int = 16
    seq_len: int = 64
    max_bytes: int = 2 * GIB                 # refuse datasets whose arrays exceed this in RAM

    def __post_init__(self) -> None:
        if self.coef < 0 or self.coef_final < 0 or self.decay_updates < 0:
            raise ValueError("BC coef, coef_final and decay_updates must be >= 0")
        if self.batch_seqs < 1 or self.seq_len < 1:
            raise ValueError("BC batch_seqs and seq_len must be >= 1")
        if self.decay_mode not in BC_DECAY_MODES:
            raise ValueError(f"BC decay_mode must be one of {BC_DECAY_MODES}, got {self.decay_mode!r}")
        if not (0.0 < self.decay_rate <= 1.0):
            raise ValueError(f"BC decay_rate must be in (0, 1], got {self.decay_rate}")
        if self.enabled_by_coef and not self.data_dirs:
            raise ValueError("a BC coefficient > 0 needs at least one BC dataset dir")

    @property
    def enabled_by_coef(self) -> bool:
        if self.decay_mode == "exp":
            return self.coef > 0.0 or self.coef_final > 0.0
        return self.coef > 0.0 or (self.decay_updates > 0 and self.coef_final > 0.0)

    @property
    def enabled(self) -> bool:
        return bool(self.data_dirs) and self.enabled_by_coef

    def resume_identity(self) -> dict:
        """The settings a resume must not change (everything except the RAM cap)."""
        return {"data_dirs": list(self.data_dirs), "data_sha256": list(self.data_sha256),
                "coef": self.coef, "coef_final": self.coef_final, "decay_updates": self.decay_updates,
                "decay_mode": self.decay_mode, "decay_rate": self.decay_rate, "batch_seqs": self.batch_seqs, "seq_len": self.seq_len}


@dataclass(frozen=True)
class RNDConfig:
    """Random Network Distillation exploration bonus (Burda et al. 2018; train/rnd.py).
    TRAINING ONLY: the bonus is a second reward stream inside the PPO update; nothing
    here reaches an observation or the inference path.

    `int_coef` 0 = off (no RND module is built and the update is bit-identical to a run
    without it). On: advantages are `ext_coef * A_ext + int_coef * A_int`, where A_int
    is a NON-episodic GAE over the intrinsic reward with the model's `value_int` head.

    RND input (`train.rnd.rnd_features`): the NEXT observation's rays (all bands),
    the self vector without `self_exclude` (ammo, damage pulse, selected-weapon one-hot:
    pressing attack / being hit must not farm novelty), any of `extra_keys` present in
    the observation (flattened), never `prev_action`."""
    int_coef: float = 0.0
    ext_coef: float = 1.0
    gamma_int: float = 0.99               # paper: 0.99 (0.999 hurt)
    gae_lambda_int: float = 0.95
    hidden_dim: int = 256
    out_dim: int = 128
    predictor_lr: float = 1e-4            # paper: 1e-4
    predictor_fraction: float = 0.25      # share of each rollout the predictor trains on (paper)
    predictor_epochs: int = 3
    predictor_minibatches: int = 4
    predictor_max_grad_norm: float = 1.0
    obs_clip: float = 5.0                 # whitened RND input clipped to [-5, 5] (paper)
    obs_std_floor: float = 1e-2           # per-feature std floor (constant features stay finite)
    return_std_floor: float = 1e-2        # floor of the intrinsic-return std the reward is divided by
    norm_eps: float = 1e-4                # initial count of the running mean/std
    self_exclude: tuple[int, ...] = tuple(range(2, 14))  # ammo 2, damage 3, weapon one-hot 4-13
    extra_keys: tuple[str, ...] = ("egomap",)
    extra_pool: int = 2                   # (C, H, W) extra keys: average-pool H and W by this factor
    max_extra_dim: int = 4096             # refuse an extra key larger than this after pooling ("small")

    def __post_init__(self) -> None:
        if self.int_coef < 0 or self.ext_coef < 0:
            raise ValueError("RND int_coef and ext_coef must be >= 0")
        if not (0.0 < self.predictor_fraction <= 1.0):
            raise ValueError(f"predictor_fraction must be in (0, 1], got {self.predictor_fraction}")
        if not (0.0 <= self.gamma_int < 1.0):
            raise ValueError(f"gamma_int must be in [0, 1), got {self.gamma_int}")
        if self.predictor_epochs < 1 or self.predictor_minibatches < 1:
            raise ValueError("predictor_epochs and predictor_minibatches must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.int_coef > 0.0


@dataclass(frozen=True)
class InvDynConfig:
    """Inverse-dynamics auxiliary loss (idea from Pathak et al. 2017, ICM; used as a
    representation loss only, NOT as a curiosity reward; train/inverse_dynamics.py).
    A trainer-owned head predicts the six-head action a_t from the model's encoder
    features of (o_t, o_{t+1}); `coef` 0 = off (nothing built, bit-identical).
    Training only: the head is never used to act."""
    coef: float = 0.0
    hidden_dim: int = 256
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.coef < 0:
            raise ValueError(f"inverse-dynamics coef must be >= 0, got {self.coef}")

    @property
    def enabled(self) -> bool:
        return self.coef > 0.0


@dataclass(frozen=True)
class CurriculumLogConfig:
    """Logging of the curriculum (train/curriculum_vec_env.py). Logging only: it never
    changes which map is chosen (curriculum/config.json owns that)."""
    # Finished episodes per stage behind `curriculum_recent_success_<stage>`.
    recent_window: int = 100


@dataclass(frozen=True)
class SafetyConfig:
    """ADR 0004 / 0007 ruling 7: what keeps the laptop safe and lets the user stop."""
    below_normal_priority: bool = True
    checkpoint_interval_s: float = 600.0          # save every 10 minutes (on an update boundary)
    # Also save (main + numbered copy) whenever update_idx is a multiple of this. 0 = off.
    checkpoint_every_updates: int = 0
    memory_cap_bytes: int = 8 * GIB               # whole process tree; above it: save and stop
    # ADR 0008 Decision 5: memory watched on a clock, not per update.
    memory_check_interval_s: float = 5.0
    min_available_bytes: int = int(1.5 * GIB)     # system available below this: save and stop
    emergency_available_bytes: int = int(0.75 * GIB)  # below this: close engines at once
    cooldown_s: float = 0.0                       # pause between updates
    # ADR 0011 F3: every checkpoint the run loop writes is also kept under a numbered
    # name; when the numbered set exceeds the budget, every second one is removed
    # (oldest first; the newest is always kept).
    keep_numbered_checkpoints: bool = True
    numbered_checkpoint_budget_bytes: int = 10 ** 9   # "1 GB" (ADR 0011 F3)
    # Engine process names; a run refuses to start while any is alive.
    engine_process_names: tuple[str, ...] = ("vizdoom.exe", "vizdoom")
    worker_join_timeout_s: float = 10.0
    worker_start_method: str = "spawn"
    # ADR 0008 point 6: target design is non-daemon workers that poll for a dead
    # parent; daemon workers stay selectable until trainer-ops' hard-kill test decides.
    worker_daemon: bool = False
    parent_poll_interval_s: float = 1.0
    # Layout-refusal memory estimate (ADR 0008 Decision 5). Sources: ADR 0006 single
    # instance (Python 65 MB, vizdoom.exe 72 MB) and the 1 x 1 smoke run's tree peak.
    est_engine_bytes: int = 72 * 2**20
    est_worker_bytes: int = 65 * 2**20
    est_trainer_bytes: int = 330 * 2**20


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    # ADR 0008 Decision 5: one engine instance by default. 4 x 8 (ADR 0006) is the
    # layout to ASK the user for; more than one instance needs --allow-multi-instance N.
    num_workers: int = 1
    envs_per_worker: int = 1
    torch_threads: int = 2
    # Resolved torch device ("cpu", "cuda" or "hybrid"); train.device.resolve_device
    # turns the --device choice into this. cpu / cuda: one model on that device does
    # rollout inference and the PPO update. hybrid: inference on CPU, update on CUDA
    # (train.device.DEVICE_ROLES). Samplers' generators stay on CPU.
    device: str = "cpu"
    # Reward-shaping preset name (train.run_doom.SHAPING_PRESETS) or None. Recorded in
    # checkpoints so a resume with a different preset is refused (ADR 0011 F6).
    shaping: str | None = None
    # Map source preset (train.run_doom.MAPS_PRESETS): "my_way_home" (bundled scenario,
    # default) or "generated" (maps/ TRAIN bundle through the curriculum). Recorded in
    # checkpoints so a resume with a different map source is refused.
    maps: str = "my_way_home"
    # --init-from: the weights file a fresh run started from and its sha256 (None when
    # the run started from random weights). Recorded only; never read to act.
    init_from: str | None = None
    init_from_sha256: str | None = None
    # --init-from growth (train/egomap_growth.py): canonical JSON of what was grown when
    # the file's obs spec had fewer egomap channels (None = strict copy). Recorded only.
    init_from_growth: str | None = None
    # --start-stage: the curriculum stage a fresh generated-maps run starts in (None =
    # the first stage).
    start_stage: str | None = None
    curriculum_log: CurriculumLogConfig = field(default_factory=CurriculumLogConfig)
    # --bc-*: imitation auxiliary loss (off by default; train/bc_aux.py)
    bc_aux: BCAuxConfig = field(default_factory=BCAuxConfig)
    # --rnd-*: RND exploration bonus; --inv-dyn-coef: inverse-dynamics auxiliary loss.
    # Both off by default. A resume refuses a change of either switch.
    rnd: RNDConfig = field(default_factory=RNDConfig)
    inv_dyn: InvDynConfig = field(default_factory=InvDynConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)
    telemetry_mask: TelemetryMaskConfig = field(default_factory=TelemetryMaskConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    health: HealthMonitorConfig = field(default_factory=HealthMonitorConfig)

    @property
    def num_envs(self) -> int:
        return self.num_workers * self.envs_per_worker
