"""trainer-ops launcher: start `train.run_doom` from a JSON run config in configs/ops/.

    python -u -m configs.ops_launch configs/ops/<name>.json reports/runs/<run>

`train.run_doom` exposes no command-line flag for the episode time limit or for the
worker design (daemon / non-daemon). This launcher reads a JSON file with:
    argv              extra run_doom arguments (layout, threads, limits)
    env_overrides     fields replaced on the A0 TRAIN env config (configs.a0_env)
    safety_overrides  fields replaced on SafetyConfig (configs.train_config)
    health_overrides  fields replaced on RunConfig.health (HealthMonitorConfig), e.g.
                      {"stop_enabled": false} for a flag-only health monitor
substitutes the names `train.run_doom.main` reads (`a0_train_env_config`,
`generated_maps_env_config`, `SafetyConfig`, and `RunConfig` when health_overrides is
non-empty) for this process only, prints and asserts the effective values,
copies the JSON into the run directory, and calls `main()`. No file outside configs/
is changed and the defaults in configs/*.py are untouched.

Top level is torch-free and the entry point is guarded (Windows spawn re-imports
the main module in every worker).
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys


def launch(config_path: str, run_dir: str, extra: list[str] | None = None) -> int:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    env_over = cfg.get("env_overrides", {})
    safety_over = cfg.get("safety_overrides", {})
    health_over = cfg.get("health_overrides", {})
    argv = list(cfg.get("argv", [])) + ["--run-dir", run_dir] + list(extra or [])

    from configs import a0_env
    from configs.train_config import RunConfig, SafetyConfig

    import train.run_doom as R

    base_env_fn = a0_env.a0_train_env_config

    def env_fn():
        return base_env_fn().replace(**env_over) if env_over else base_env_fn()

    def safety_cls(**kw):
        return dataclasses.replace(SafetyConfig(**kw), **safety_over)

    base_gen_fn = a0_env.generated_maps_env_config

    def gen_fn(*a, **k):  # --maps generated: the overrides apply to the final config too
        cfg = base_gen_fn(*a, **k)
        return cfg.replace(**env_over) if env_over else cfg

    R.a0_train_env_config = env_fn
    R.generated_maps_env_config = gen_fn
    R.SafetyConfig = safety_cls

    def run_cls(**kw):  # run_doom builds its RunConfig only through this name
        run = RunConfig(**kw)
        return dataclasses.replace(run, health=dataclasses.replace(run.health, **health_over))

    if health_over:
        R.RunConfig = run_cls

    eff_env, eff_safety = env_fn(), safety_cls()
    for k, v in env_over.items():
        assert getattr(eff_env, k) == v, f"env override {k} not applied"
    for k, v in safety_over.items():
        assert getattr(eff_safety, k) == v, f"safety override {k} not applied"
    eff_health = R.build_run_config(R.parse_args(argv)).health  # the config main() will build
    for k, v in health_over.items():
        assert getattr(eff_health, k) == v, f"health override {k} not applied"
    os.makedirs(run_dir, exist_ok=True)
    if not extra:  # a resume keeps the copy made at the run's start
        shutil.copyfile(config_path, os.path.join(run_dir, "config.json"))
    print(f"ops_launch: config {config_path} -> {run_dir}")
    print(f"ops_launch: effective episode_timeout_tics={eff_env.episode_timeout_tics} (my_way_home base; "
          f"--maps generated uses configs.a0_env.GENERATED_MAPS unless overridden here) "
          f"frame_skip={eff_env.frame_skip} worker_daemon={eff_safety.worker_daemon} "
          f"memory_cap_gib={eff_safety.memory_cap_bytes / 2**30:.1f} "
          f"below_normal={eff_safety.below_normal_priority} "
          f"health_enabled={eff_health.enabled} health_stop_enabled={eff_health.stop_enabled}")
    print(f"ops_launch: run_doom argv {argv}", flush=True)
    return R.main(argv)


if __name__ == "__main__":
    # Only one extra argument is accepted: `--resume <checkpoint>` (a pointer to the
    # state to continue from, not a setting). Every setting lives in the JSON.
    extra = sys.argv[3:]
    if len(sys.argv) < 3 or extra not in ([],) and not (len(extra) == 2 and extra[0] == "--resume"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(launch(sys.argv[1], sys.argv[2], extra))
