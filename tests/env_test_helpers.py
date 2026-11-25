"""Shared helpers for the env engine tests (env-engineer's tests only).

FD2_ENGINE_TEST_CONFIG is a TRAIN-type config on freedoom2 MAP01. It exists only
so engine tests can use the geometry probed there (door sector 27, warp
coordinates). It is never a training config: Freedoom Phase 2 is never trained on
(ADR 0005); the real TRAIN_CONFIG is the bundled my_way_home scenario."""
from env.config import TRAIN_CONFIG, VANILLA_ENGINE_REWARDS

FD2_ENGINE_TEST_CONFIG = TRAIN_CONFIG.replace(
    name="fd2-engine-test", scenario_wad=None, maps=("MAP01",), skill=3,
    episode_timeout_tics=4200, engine_rewards=VANILLA_ENGINE_REWARDS,
    phase2_engine_test=True,  # explicit opt-in; training configs are refused (ADR 0008)
)
