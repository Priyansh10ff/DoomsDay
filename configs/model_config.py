"""Stage A0 model sizes (ADR 0003). Observation and action sizes are NOT here:
they are read from the observation / action space at build time (ADR 0006)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # --- ray encoder: Conv1d stack over rays (C bands x W columns) -> ray_dim
    ray_conv_channels: tuple[int, ...] = (16, 32)
    ray_conv_kernel: int = 5
    ray_conv_stride: int = 2
    ray_dim: int = 64

    # --- entity set encoder
    entity_type_embed_dim: int = 16
    entity_token_dim: int = 64
    entity_mlp_hidden: int = 64
    entity_attn_heads: int = 4
    entity_pool_heads: int = 4

    # --- self / prev_action encoders
    self_hidden: int = 64
    self_dim: int = 32
    # ADR 0003 "six small embeddings to 16": one embedding table per head, each of
    # width prev_action_dim, summed (accepted by ADR 0007).
    prev_action_dim: int = 16

    # --- egomap encoder (observation v3 only; built only when the spec has `egomap`):
    # Conv2d stack over the (C, H, W) uint8 map cast to float -> egomap_dim. With the
    # 32x32 map, stride 2 and padding 1 the grid goes 32 -> 16 -> 8 -> 4.
    egomap_conv_channels: tuple[int, ...] = (16, 32, 32)
    egomap_conv_kernel: int = 3
    egomap_conv_stride: int = 2
    egomap_conv_padding: int = 1
    egomap_dim: int = 128

    # --- fusion and core
    fusion_dim: int = 192
    core_hidden: int = 256
    # Ablation switch for the toy memory test (advisor): when False the SAME
    # network is used but its hidden state is zeroed before every step, in
    # collection and in the update. Not used by the A0 agent.
    recurrent: bool = True

    # --- initialization (accepted by ADR 0007)
    policy_head_init_gain: float = 0.01     # near-uniform initial action distribution
    learned_token_init_std: float = 0.02    # "no entity" token and pooling query
    # ADR 0008 point 5: "default" = as built; "orthogonal" = 0007-recipe arm I
    # (hidden layers gain sqrt 2, GRU and value/aux heads gain 1, policy heads
    # policy_head_init_gain, biases 0).
    init_scheme: str = "default"
    orthogonal_hidden_gain: float = 2.0 ** 0.5
    orthogonal_unit_gain: float = 1.0
    # ADR 0008 Decision 3, arm B1: GRU update-gate bias initialised so units start with
    # retention times spread log-uniformly over [min, max] decisions. Off in the base.
    gru_update_bias_init: bool = False
    gru_retention_min: float = 2.0
    gru_retention_max: float = 256.0

    # --- auxiliary heads (training-only outputs, never read to choose actions)
    n_key_bits: int = 3
    # Sub-goal prediction head (2026-11-26, user request; no ADR): GRU state ->
    # configs.goal_modes.N_GOAL_MODES logits, trained against the teacher's per-step
    # `goal_mode` label. Auxiliary only: never feeds the policy heads. Off = the model
    # is built exactly as before (same modules, same init RNG draws, same outputs).
    goal_head: bool = False

    # --- observation layout facts from ADR 0001 that are not visible in the space
    # Number of trailing dims of `self` that are ego-motion (masked with telemetry).
    self_ego_dims: int = 4
    # Indices in `modality`: telemetry valid, ego-motion valid.
    modality_telemetry_index: int = 0
    modality_ego_index: int = 2
