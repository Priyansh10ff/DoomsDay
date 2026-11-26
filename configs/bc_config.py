"""Behaviour-cloning settings (CLAUDE.md s7, 2026-11-25 evening plan change: clone our
own scripted teacher, then fine-tune with our own PPO). No torch import.

Head order is env/actions.py's: move, strafe, turn, attack, use, speed.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 1024 ** 3

# Default per-head loss weights: every head cloned, attack included. Attack was 0.0
# until 2026-11-25 evening: measured then, weight 0 leaves the attack head at 50/50,
# which fires at random and breaks closed-loop play (coordinator ruling). A zero
# weight is still possible via --head-weights.
DEFAULT_HEAD_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
# Head order index of USE (env/actions.py) and its "press" label.
USE_HEAD_INDEX = 4
USE_PRESS_LABEL = 1


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 10
    seq_len: int = 64                 # truncated-BPTT chunk length (decisions)
    batch_seqs: int = 32              # parallel episode streams ("slots") per batch
    lr: float = 3e-4
    adam_eps: float = 1e-5            # same as PPOConfig.adam_eps
    max_grad_norm: float = 1.0        # same as PPOConfig.max_grad_norm
    seed: int = 0
    val_frac: float = 0.1             # fraction of MAP IDS held out of cloning
    head_weights: tuple[float, ...] = DEFAULT_HEAD_WEIGHTS
    max_seconds: float = float("inf")
    torch_threads: int = 2            # same default as RunConfig.torch_threads
    # Refuse to load a dataset whose arrays would exceed this many bytes in RAM
    # (the laptop is shared; 15.4 GB total).
    max_dataset_bytes: int = 3 * GIB
    below_normal_priority: bool = True
    # USE head (index USE_HEAD_INDEX): label-1 (press) steps weigh this much in its
    # cross-entropy (2026-11-25 strategy C1: presses are rare). 1.0 = unweighted, exactly
    # the earlier loss.
    use_pos_weight: float = 1.0
    # Previous-action dropout (2026-11-26, fix for the wall-wedge regression in
    # reports/eval/2026-11-26-bc-v3-regression.md): in BC TRAINING steps only, each
    # (step, sequence) independently has its `prev_action` input replaced, with this
    # probability, by the env's reset value (no previous action). 0 = off, exactly the
    # earlier training. Draws come from their own generator, seeded seed + offset, so
    # the episode shuffle is the same at every probability.
    prev_action_dropout: float = 0.0
    prev_action_dropout_seed_offset: int = 1
    # Sub-goal head (train/goal_head.py): loss += goal_head_coef x mean cross-entropy
    # of the goal logits against the dataset's `goal_mode` (unlabelled steps masked).
    # 0 = off, exactly the earlier loss; > 0 needs a model built with the head.
    goal_head_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.seq_len < 1 or self.batch_seqs < 1 or self.epochs < 0:
            raise ValueError("seq_len and batch_seqs must be >= 1, epochs >= 0")
        if not (0.0 <= self.val_frac < 1.0):
            raise ValueError(f"val_frac must be in [0, 1), got {self.val_frac}")
        if any(w < 0 for w in self.head_weights):
            raise ValueError(f"head weights must be non-negative, got {self.head_weights}")
        if not self.use_pos_weight > 0:
            raise ValueError(f"use_pos_weight must be > 0, got {self.use_pos_weight}")
        if not (0.0 <= self.prev_action_dropout <= 1.0):
            raise ValueError(f"prev_action_dropout must be in [0, 1], got {self.prev_action_dropout}")
        if not self.goal_head_coef >= 0.0:
            raise ValueError(f"goal_head_coef must be >= 0, got {self.goal_head_coef}")
