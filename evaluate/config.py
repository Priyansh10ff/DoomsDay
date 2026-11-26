"""Evaluation run settings. Every number used by `evaluate` is a named field here."""
from __future__ import annotations

from dataclasses import dataclass

from agent.inference import DEFAULT_TEMPERATURE, check_temperature

SPLIT_CHOICES: tuple[str, ...] = ("val", "heldout", "my_way_home")
POLICY_SAMPLE = "sample"
POLICY_ARGMAX = "argmax"
POLICY_RANDOM = "random"
POLICY_CHOICES: tuple[str, ...] = (POLICY_SAMPLE, POLICY_ARGMAX, POLICY_RANDOM)

# Stage label used for the bundled scenario (it has no generator stage).
MY_WAY_HOME_STAGE = "my_way_home"
MY_WAY_HOME_ID = "my_way_home"
# loader mode used to resolve each split (maps.loader.MODES). Held-out maps are
# reachable only through "eval", and only from this package.
LOADER_MODE = {"val": "val", "heldout": "eval"}


@dataclass(frozen=True)
class EvalRunConfig:
    split: str
    episodes_per_map: int = 5
    max_maps: int | None = None          # first N maps of the split (None = all)
    policy: str = POLICY_SAMPLE           # sample | argmax | random
    base_seed: int = 20261125
    # episode seed = base_seed + map_index * seed_stride_map + episode_index
    seed_stride_map: int = 10_000
    device: str = "cpu"
    torch_threads: int = 1                # one engine, light CPU load (laptop limits)
    # Wilson score interval, 95 % (z for a two-sided 95 % normal quantile).
    ci_z: float = 1.959963984540054
    # Success needs the final-step engine reward to reach this fraction of the
    # config's map_exit_reward (ADR 0007-recipe Decision 5.1: two agreeing signals).
    exit_reward_fraction: float = 1.0
    # Optional override of every map's time limit (tests only; None = map default).
    timeout_tics_override: int | None = None
    # Play-time decoding temperature for sampled heads: softmax(logits / T), the same T
    # for every head (agent.inference). 1.0 = plain sampling; != 1.0 only with "sample".
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        if self.split not in SPLIT_CHOICES:
            raise ValueError(f"split {self.split!r} not in {SPLIT_CHOICES}")
        if self.policy not in POLICY_CHOICES:
            raise ValueError(f"policy {self.policy!r} not in {POLICY_CHOICES}")
        check_temperature(self.temperature)
        if self.temperature != DEFAULT_TEMPERATURE and self.policy != POLICY_SAMPLE:
            raise ValueError(f"temperature {self.temperature} needs policy {POLICY_SAMPLE!r}, "
                             f"got {self.policy!r} (argmax and random take no temperature)")
        if self.episodes_per_map < 1:
            raise ValueError("episodes_per_map must be >= 1")
        if self.max_maps is not None and self.max_maps < 1:
            raise ValueError("max_maps must be >= 1 or None")

    def episode_seed(self, map_index: int, episode_index: int) -> int:
        assert 0 <= episode_index < self.seed_stride_map, "episode index exceeds the seed stride"
        return int(self.base_seed + map_index * self.seed_stride_map + episode_index)

    def argmax_heads(self, n_heads: int) -> tuple[bool, ...]:
        """Per-head sample/argmax choice, fixed by configuration (ADR 0002)."""
        if self.policy == POLICY_RANDOM:
            raise ValueError("the random baseline has no network heads")
        return (self.policy == POLICY_ARGMAX,) * n_heads
