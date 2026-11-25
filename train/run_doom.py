"""DOOM training entry point for Stage A0 (ADR 0007 ruling 7, ADR 0008 Decision 5).

    python -m train.run_doom                      # 1 worker x 1 env, 2 torch threads
    python -m train.run_doom --workers 2 --envs-per-worker 4 --allow-multi-instance 8
    python -m train.run_doom --shaping my_way_home  # ADR 0010 reward + coverage bonus
    python -m train.run_doom --shaping passthrough  # engine reward only; logs cells_visited
    python -m train.run_doom --entropy-coef 0.03 --device cpu
    python -m train.run_doom --maps generated --shaping full_level   # maps/ TRAIN bundle + curriculum
    python -m train.run_doom --maps generated --shaping full_level_doors  # + door-progress term

--maps my_way_home (default, unchanged) | generated. `generated` trains on the TRAIN
split of maps/ only: the bundle comes from maps.loader.training_bundle (train mode:
held-out and val maps refused, hashes verified once, here in the main process), and
every episode's map is chosen by curriculum.scheduler (train/curriculum_vec_env.py).
Shaping presets allowed per map source: SHAPING_FOR_MAPS. A resume refuses a
different --maps; on `generated` it restores the curriculum state from the checkpoint.

--shaping is off by default (engine reward only, the env factory is exactly
DoomEnvFactory). Shaping is training-only: evaluation builds env.DoomEnv directly.
`passthrough` wraps the env like a shaped run but pays no bonus (coverage on with a
zero coefficient), so `cells_visited` is logged for the with/without-bonus comparison.

--init-from <file> starts a FRESH run (new optimizer, return scale, curriculum, run
dir; update 0) with model weights from a behaviour-cloning file (train/bc.py) or any
trainer checkpoint: strict load, refused on a different sensor signature, model
config or observation spec. The path and sha256 go into the run config (so into
every checkpoint) and run_start.jsonl. Not allowed together with --resume.
--value-warmup-updates N (default 0; meant for --init-from): for the first N updates
only the value and auxiliary heads train (policy-gradient and entropy terms x 0,
everything else frozen); the end of the warm-up is logged.
--start-stage S0|S1|S2|S3 (--maps generated, fresh runs only): the curriculum starts
in that stage instead of the first (earlier stages are still replayed).

PPO knobs (defaults unchanged): --lr, --clip, --ppo-epochs, --lr-warmup-updates N
(linear LR warm-up over the first N policy updates after the value warm-up),
--kl-stop K (skip the rest of an update once a minibatch's approx KL exceeds K).
--checkpoint-every-updates N: also checkpoint (+ numbered copy) every N updates.

Imitation auxiliary loss (train/bc_aux.py; off unless a coefficient > 0):
  --bc-data DIR[,DIR...] --bc-coef C [--bc-coef-final C2] [--bc-decay-mode linear|exp]
  [--bc-decay-updates N] [--bc-decay-rate R] [--bc-batch-seqs K] [--bc-seq-len L] [--bc-max-gb G]
Each PPO minibatch step adds coef x (sum over heads of CE against the teacher label)
on K sequences of L decisions from the datasets (burn-in from a zero state). The
schedule counts policy updates after the value warm-up (the term is off during it).
The data is loaded and validated (observation version must match) BEFORE any engine
starts. A resume refuses changed BC settings.

--resume reads the checkpoint before the vec env is built and refuses a --shaping
different from the one the checkpoint was trained with (ADR 0011 F6); the anneal
offset uses the env count recorded in the checkpoint.

--device cpu (default, today's behaviour) | cuda (refused when CUDA is unavailable)
| auto | hybrid (rollout inference on CPU, PPO update on CUDA, weights copied to the
CPU actor after every update; refused when CUDA is unavailable).

With no flags exactly one engine instance starts. More than one engine instance is
refused unless --allow-multi-instance equals the total instance count; only
trainer-ops passes it, for a layout the user approved. Top-level imports here must
stay torch-free: worker processes (spawn) re-import this module.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Any

from configs.a0_env import a0_train_env_config, generated_maps_env_config
from configs.train_config import (BC_DECAY_MODES, GIB, BCAuxConfig, InvDynConfig, PPOConfig, RNDConfig, RunConfig,
                                  SafetyConfig)

from . import process_control as PC
from .doom_factory import DoomEnvFactory
from .layout_guard import LayoutRefused, check_layout, plan_text
from .subproc_vec_env import SubprocVecEnv

DEFAULT_RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "a0")
# --shaping choices: name -> (RewardConfig, CoverageConfig) attribute names in
# configs.reward_config plus field overrides on the CoverageConfig, resolved lazily so
# the default path imports no reward code.
SHAPING_PRESETS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "my_way_home": ("MY_WAY_HOME_REWARD", "MY_WAY_HOME_COVERAGE", {}),
    # The no-bonus arm (ADR 0010 validation 3, ADR 0011 E1): engine reward only, the
    # coverage tracker runs so `cells_visited` is logged, and its coefficient is 0.
    "passthrough": ("PASSTHROUGH_REWARD", "MY_WAY_HOME_COVERAGE", {"coef_start": 0.0, "coef_end": 0.0}),
    # Generated maps (real exit line/switch, keys, monsters; vanilla engine knobs:
    # map_exit_reward 1.0 only): our terms pay, the engine reward is ignored except as
    # the second exit signal (configs/reward_config.py FULL_LEVEL_*).
    "full_level": ("FULL_LEVEL_REWARD", "FULL_LEVEL_COVERAGE", {}),
    # full_level + a once-per-door progress term (rewards/doors.py); door geometry is
    # read from the generated bundle in the main process, for reward only.
    "full_level_doors": ("FULL_LEVEL_DOORS_REWARD", "FULL_LEVEL_COVERAGE", {}),
    # + USE-miss and door-reversal penalties, door pay for on-path doors only (rewards/use_doors.py).
    "full_level_doors_v2": ("FULL_LEVEL_DOORS_V2_REWARD", "FULL_LEVEL_COVERAGE", {}),
}
MAPS_PRESETS: tuple[str, ...] = ("my_way_home", "generated")
# Which --shaping presets fit which --maps (None = engine reward only, always allowed).
#  * my_way_home on generated maps: it declares the exit paid by the scenario's ACS and
#    zeroes keys/kills/damage/death; the generated maps have no ACS, and have keys and
#    monsters. It passes rewards.validate but is the wrong description of these maps.
#  * passthrough on generated maps: rewards.validate refuses it ("the scenario pays the
#    exit: declare scenario_exit_value"); a generated-maps passthrough preset
#    (scenario_exit_value=1.0) would have to be added by reward-designer.
#  * full_level on my_way_home: the bundled scenario pays its goal through ACS with no
#    map_exit_reward, so the two-signal exit term could never pay.
SHAPING_FOR_MAPS: dict[str, tuple[str, ...]] = {
    "my_way_home": ("my_way_home", "passthrough"),
    "generated": ("full_level", "full_level_doors", "full_level_doors_v2"),
}
EXIT_DEVICE_UNAVAILABLE = 6
EXIT_RESUME_REFUSED = 5
EXIT_INIT_REFUSED = 7
EXIT_BC_REFUSED = 8


def build_env_factory(env_cfg: Any, shaping: str | None, decisions_done: int, num_envs: int) -> Any:
    """The picklable per-env factory. `shaping=None` -> exactly DoomEnvFactory(env_cfg)
    (today's behaviour). Otherwise the ADR 0010 wrapper around it, with the anneal
    clock offset by the per-env decisions already done (resume)."""
    inner = DoomEnvFactory(env_cfg)
    if shaping is None:
        return inner
    from rewards.factory import ShapedEnvFactory, per_env_decisions
    reward, coverage = shaping_configs(shaping)
    doors = diag = None
    if reward.door != 0.0:  # privileged geometry for the reward only, never an input
        from maps.doors import door_table
        from train.episode_diag import diag_table  # C4: per-episode USE/door logging only
        doors, diag = door_table(env_cfg.map_wad, env_cfg.maps), diag_table(env_cfg.map_wad, env_cfg.maps)
    return ShapedEnvFactory(inner, reward, coverage,
                            start_decisions=per_env_decisions(decisions_done, num_envs), doors=doors, diag=diag)


def build_generated_maps(base_env_cfg: Any) -> tuple[Any, Any]:
    """(CurriculumScheduler, env config) for --maps generated. The scheduler's pool and
    the bundle path come from maps.loader.training_bundle (training mode: held-out and
    val maps refused, every hash verified, once, in this process)."""
    from curriculum.scheduler import CurriculumScheduler
    scheduler = CurriculumScheduler.from_bundle()
    lumps = tuple(sorted(lump for s in scheduler.stages for lump, _ in scheduler.pool[s]))
    return scheduler, generated_maps_env_config(base_env_cfg, scheduler.bundle_path, lumps)


def shaping_configs(shaping: str) -> tuple[Any, Any]:
    """(RewardConfig, CoverageConfig) of a --shaping preset, imported lazily."""
    if shaping not in SHAPING_PRESETS:
        raise ValueError(f"unknown --shaping {shaping!r}; choices: {sorted(SHAPING_PRESETS)}")
    import configs.reward_config as RC
    reward_name, coverage_name, coverage_over = SHAPING_PRESETS[shaping]
    return getattr(RC, reward_name), dataclasses.replace(getattr(RC, coverage_name), **coverage_over)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    d, s, p = RunConfig(), SafetyConfig(), PPOConfig()
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--workers", type=int, default=d.num_workers)
    a.add_argument("--envs-per-worker", type=int, default=d.envs_per_worker)
    a.add_argument("--torch-threads", type=int, default=d.torch_threads)
    a.add_argument("--seed", type=int, default=d.seed)
    a.add_argument("--rollout-len", type=int, default=p.rollout_len)
    a.add_argument("--minibatches", type=int, default=None,
                   help=f"default: largest divisor of the env count not above {p.num_minibatches}")
    a.add_argument("--allow-multi-instance", type=int, default=None, metavar="N",
                   help="required for more than one engine instance; N must equal workers x envs")
    a.add_argument("--max-updates", type=int, default=10**9)
    a.add_argument("--max-seconds", type=float, default=float("inf"))
    a.add_argument("--checkpoint-interval", type=float, default=s.checkpoint_interval_s)
    a.add_argument("--memory-cap-gb", type=float, default=s.memory_cap_bytes / GIB)
    a.add_argument("--cooldown", type=float, default=s.cooldown_s)
    a.add_argument("--run-dir", default=None)
    a.add_argument("--resume", default=None, help="checkpoint path to resume from")
    a.add_argument("--shaping", default=None, choices=sorted(SHAPING_PRESETS),
                   help="training-only reward shaping + coverage bonus (ADR 0010); default: off")
    a.add_argument("--entropy-coef", type=float, default=p.entropy_coef,
                   help=f"entropy bonus coefficient on the summed head entropy (ADR 0011 F5); default {p.entropy_coef}")
    a.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto", "hybrid"),
                   help="torch device for the model (rollout inference and update); hybrid = rollout "
                        "inference on CPU, PPO update on CUDA; default cpu")
    a.add_argument("--maps", default=MAPS_PRESETS[0], choices=MAPS_PRESETS,
                   help="map source: my_way_home (default) or generated (maps/ TRAIN bundle, curriculum)")
    a.add_argument("--init-from", default=None, metavar="PATH",
                   help="start a fresh run with model weights from this file (BC output or checkpoint)")
    a.add_argument("--value-warmup-updates", type=int, default=p.value_warmup_updates, metavar="N",
                   help="first N updates train only the value and auxiliary heads; default 0")
    a.add_argument("--start-stage", default=None, metavar="STAGE",
                   help="curriculum stage a fresh --maps generated run starts in (e.g. S2)")
    a.add_argument("--lr", type=float, default=p.lr)
    a.add_argument("--clip", type=float, default=p.clip)
    a.add_argument("--ppo-epochs", type=int, default=p.epochs)
    a.add_argument("--lr-warmup-updates", type=int, default=p.lr_warmup_updates, metavar="N")
    a.add_argument("--kl-stop", type=float, default=p.kl_stop, metavar="K",
                   help="skip the rest of an update once a minibatch's approx KL exceeds K; 0 = off")
    a.add_argument("--checkpoint-every-updates", type=int, default=s.checkpoint_every_updates, metavar="N")
    b = BCAuxConfig()
    a.add_argument("--bc-data", default=None, metavar="DIR[,DIR...]",
                   help="teacher datasets (ep_*.npz + index.jsonl) for the imitation auxiliary loss")
    a.add_argument("--bc-coef", type=float, default=b.coef)
    a.add_argument("--bc-coef-final", type=float, default=None,
                   help="linear: value after --bc-decay-updates; exp: floor. default: --bc-coef")
    a.add_argument("--bc-decay-mode", default=b.decay_mode, choices=BC_DECAY_MODES)
    a.add_argument("--bc-decay-updates", type=int, default=b.decay_updates)
    a.add_argument("--bc-decay-rate", type=float, default=b.decay_rate, help="exp mode: factor per policy update")
    a.add_argument("--bc-batch-seqs", type=int, default=b.batch_seqs)
    a.add_argument("--bc-seq-len", type=int, default=b.seq_len)
    a.add_argument("--bc-max-gb", type=float, default=b.max_bytes / GIB)
    r, iv = RNDConfig(), InvDynConfig()
    a.add_argument("--rnd-int-coef", type=float, default=r.int_coef,
                   help="RND exploration bonus: weight of the intrinsic advantage; 0 = off (default)")
    a.add_argument("--rnd-ext-coef", type=float, default=r.ext_coef,
                   help="weight of the extrinsic advantage when RND is on; default 1")
    a.add_argument("--inv-dyn-coef", type=float, default=iv.coef,
                   help="inverse-dynamics auxiliary loss weight; 0 = off (default)")
    ns = a.parse_args(argv)
    if not (ns.rnd_int_coef >= 0 and ns.rnd_ext_coef >= 0 and ns.inv_dyn_coef >= 0):
        a.error("--rnd-int-coef, --rnd-ext-coef and --inv-dyn-coef must be >= 0")
    if ns.rnd_ext_coef != r.ext_coef and ns.rnd_int_coef == 0:
        a.error("--rnd-ext-coef only applies with RND on (--rnd-int-coef > 0)")
    if ns.bc_coef_final is None:
        ns.bc_coef_final = ns.bc_coef
    if ns.bc_data and not (ns.bc_coef > 0 or ns.bc_coef_final > 0):
        a.error("--bc-data needs --bc-coef > 0 (or a positive --bc-coef-final)")
    if (ns.bc_coef > 0 or ns.bc_coef_final > 0) and not ns.bc_data:
        a.error("--bc-coef > 0 needs --bc-data")
    if min(ns.bc_coef, ns.bc_coef_final, ns.bc_decay_updates, ns.lr_warmup_updates, ns.kl_stop,
           ns.checkpoint_every_updates) < 0 or ns.lr <= 0 or ns.clip <= 0 or ns.ppo_epochs < 1:
        a.error("BC coefficients, decay updates, --lr-warmup-updates, --kl-stop and --checkpoint-every-updates "
                "must be >= 0; --lr and --clip > 0; --ppo-epochs >= 1")
    if (ns.bc_decay_mode == "linear" and ns.bc_decay_updates == 0 and ns.bc_coef_final != ns.bc_coef):
        a.error(f"--bc-coef-final {ns.bc_coef_final} differs from --bc-coef {ns.bc_coef} but no decay is set: "
                f"give --bc-decay-updates N (linear) or --bc-decay-mode exp")
    if not (0.0 < ns.bc_decay_rate <= 1.0):
        a.error(f"--bc-decay-rate must be in (0, 1], got {ns.bc_decay_rate}")
    if ns.init_from is not None and ns.resume is not None:
        a.error("--init-from starts a fresh run; it cannot be combined with --resume")
    if ns.value_warmup_updates < 0:
        a.error(f"--value-warmup-updates must be >= 0, got {ns.value_warmup_updates}")
    if ns.start_stage is not None and ns.maps != "generated":
        a.error("--start-stage needs --maps generated")
    if ns.start_stage is not None and ns.resume is not None:
        a.error("--start-stage applies to a fresh run; a resume continues the saved curriculum")
    if ns.shaping is not None and ns.shaping not in SHAPING_FOR_MAPS[ns.maps]:
        a.error(f"--shaping {ns.shaping} does not fit --maps {ns.maps}; allowed: "
                f"{list(SHAPING_FOR_MAPS[ns.maps])} or none (see SHAPING_FOR_MAPS in train/run_doom.py)")
    if not (ns.entropy_coef >= 0.0):
        a.error(f"--entropy-coef must be a non-negative number, got {ns.entropy_coef}")
    return ns


def default_minibatches(num_envs: int, preferred: int) -> int:
    """Largest divisor of num_envs not above `preferred` (1 x 1 -> 1): minibatches are
    whole env sequences, so they must divide the env count (train/ppo.py)."""
    return max(d for d in range(1, max(1, min(preferred, num_envs)) + 1) if num_envs % d == 0)


def build_run_config(ns: argparse.Namespace) -> RunConfig:
    safety = dataclasses.replace(SafetyConfig(), checkpoint_interval_s=ns.checkpoint_interval,
                                 memory_cap_bytes=int(ns.memory_cap_gb * GIB), cooldown_s=ns.cooldown,
                                 checkpoint_every_updates=getattr(ns, "checkpoint_every_updates", 0))
    n_envs = ns.workers * ns.envs_per_worker
    mb = ns.minibatches if ns.minibatches is not None else default_minibatches(n_envs, PPOConfig().num_minibatches)
    ppo = dataclasses.replace(PPOConfig(), rollout_len=ns.rollout_len, num_minibatches=mb,
                              entropy_coef=ns.entropy_coef,
                              value_warmup_updates=getattr(ns, "value_warmup_updates", 0),
                              lr=getattr(ns, "lr", PPOConfig.lr), clip=getattr(ns, "clip", PPOConfig.clip),
                              epochs=getattr(ns, "ppo_epochs", PPOConfig.epochs),
                              lr_warmup_updates=getattr(ns, "lr_warmup_updates", 0),
                              kl_stop=getattr(ns, "kl_stop", 0.0))
    return dataclasses.replace(RunConfig(), seed=ns.seed, num_workers=ns.workers,
                               envs_per_worker=ns.envs_per_worker, torch_threads=ns.torch_threads,
                               ppo=ppo, safety=safety, shaping=ns.shaping, maps=ns.maps,
                               start_stage=getattr(ns, "start_stage", None), bc_aux=bc_aux_from_args(ns),
                               rnd=dataclasses.replace(RNDConfig(), int_coef=getattr(ns, "rnd_int_coef", 0.0),
                                                       ext_coef=getattr(ns, "rnd_ext_coef", RNDConfig.ext_coef)),
                               inv_dyn=dataclasses.replace(InvDynConfig(), coef=getattr(ns, "inv_dyn_coef", 0.0)))


def bc_aux_from_args(ns: argparse.Namespace) -> BCAuxConfig:
    """BC auxiliary settings from the flags (dirs absolute; index hashes filled in by
    main once the dirs are known to exist). Off when no --bc-data."""
    if not getattr(ns, "bc_data", None):
        return BCAuxConfig()
    dirs = tuple(os.path.normcase(os.path.abspath(d.strip())) for d in ns.bc_data.split(",") if d.strip())
    dirs = tuple(dict.fromkeys(dirs))
    return BCAuxConfig(data_dirs=dirs, coef=ns.bc_coef, coef_final=ns.bc_coef_final,
                       decay_updates=ns.bc_decay_updates, decay_mode=ns.bc_decay_mode,
                       decay_rate=ns.bc_decay_rate, batch_seqs=ns.bc_batch_seqs, seq_len=ns.bc_seq_len,
                       max_bytes=int(ns.bc_max_gb * GIB))


def apply_start_stage(scheduler: Any, stage: str | None) -> None:
    """Put a fresh scheduler into `stage` before any episode is chosen."""
    if stage is None:
        return
    if stage not in scheduler.stages:
        raise ValueError(f"--start-stage {stage!r} is not a curriculum stage; stages: {scheduler.stages}")
    if scheduler.issued or scheduler.episodes:
        raise ValueError("--start-stage must be applied before the curriculum issues any episode")
    scheduler.stage_idx = scheduler.stages.index(stage)


def run_settings_line(run: RunConfig, device_choice: str) -> str:
    return (f"run settings: entropy_coef={run.ppo.entropy_coef} shaping={run.shaping} maps={run.maps} "
            f"value_warmup_updates={run.ppo.value_warmup_updates} "
            f"start_stage={run.start_stage} lr={run.ppo.lr} clip={run.ppo.clip} epochs={run.ppo.epochs} "
            f"lr_warmup_updates={run.ppo.lr_warmup_updates} kl_stop={run.ppo.kl_stop} "
            f"bc_coef={run.bc_aux.coef if run.bc_aux.enabled else 0} "
            f"rnd_int_coef={run.rnd.int_coef} rnd_ext_coef={run.rnd.ext_coef} inv_dyn_coef={run.inv_dyn.coef} "
            f"device={run.device} (--device {device_choice}) seed={run.seed} "
            f"layout={run.num_workers}x{run.envs_per_worker}")


RUN_START_NAME = "run_start.jsonl"


def write_run_start(run_dir: str, run: RunConfig, ns: argparse.Namespace, argv: list[str] | None) -> None:
    """One line per session: the settings this session ran with (entropy coefficient,
    shaping, device; ADR 0011 F5). The checkpoint also stores the full run config."""
    os.makedirs(run_dir, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": list(sys.argv[1:] if argv is None else argv),
           "entropy_coef": run.ppo.entropy_coef, "shaping": run.shaping, "maps": run.maps, "device": run.device,
           "device_choice": ns.device, "resume": ns.resume, "seed": run.seed,
           "init_from": run.init_from, "init_from_sha256": run.init_from_sha256,
           "init_from_growth": run.init_from_growth,
           "value_warmup_updates": run.ppo.value_warmup_updates, "start_stage": run.start_stage,
           "num_workers": run.num_workers, "envs_per_worker": run.envs_per_worker,
           "lr": run.ppo.lr, "clip": run.ppo.clip, "ppo_epochs": run.ppo.epochs,
           "lr_warmup_updates": run.ppo.lr_warmup_updates, "kl_stop": run.ppo.kl_stop,
           "checkpoint_every_updates": run.safety.checkpoint_every_updates,
           "bc_aux": dataclasses.asdict(run.bc_aux),
           "rnd": dataclasses.asdict(run.rnd), "inv_dyn": dataclasses.asdict(run.inv_dyn)}
    with open(os.path.join(run_dir, RUN_START_NAME), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    run = build_run_config(ns)
    safety = run.safety
    try:  # before any process starts
        check_layout(run.num_envs, ns.allow_multi_instance, run.num_workers, run.torch_threads, safety)
    except LayoutRefused as e:
        print(str(e), file=sys.stderr)
        return 4
    print(plan_text(run.num_workers, run.num_envs, run.torch_threads, safety))
    leftovers = PC.engine_processes(safety.engine_process_names)
    if leftovers:
        print(f"refusing to start: engine processes still alive from an earlier run: "
              f"{[(p.pid, p.name()) for p in leftovers]}", file=sys.stderr)
        return 2
    if safety.below_normal_priority:
        PC.set_below_normal()  # before spawning, so workers and engines inherit it
    # device resolved before any engine starts (torch imported only here, not at module level)
    from .device import DeviceUnavailable, resolve_device
    try:
        run = dataclasses.replace(run, device=resolve_device(ns.device))
    except DeviceUnavailable as e:
        print(str(e), file=sys.stderr)
        return EXIT_DEVICE_UNAVAILABLE
    print(run_settings_line(run, ns.device))
    env_cfg = a0_train_env_config()
    scheduler = None
    if run.maps == "generated":
        scheduler, env_cfg = build_generated_maps(env_cfg)
        try:
            apply_start_stage(scheduler, run.start_stage)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        print(f"maps: generated bundle {env_cfg.map_wad} ({len(env_cfg.maps)} train maps), stages "
              f"{scheduler.stages}, skill {env_cfg.skill}, episode limit {env_cfg.episode_timeout_tics} tics")
    assert env_cfg.screen_enabled is False, "Stage A0 trains with screen off (ADR 0006/0007)"
    signature = env_cfg.sensor_signature()
    run_dir = ns.run_dir or os.path.join(DEFAULT_RUNS_DIR, time.strftime("%Y%m%d-%H%M%S"))
    init_src = None
    if ns.init_from:
        # read and hashed BEFORE any engine starts; sensor signature checked here
        from .checkpoint import CheckpointMismatch
        from .init_from import read_init_source
        try:
            init_src = read_init_source(ns.init_from, signature)
        except (CheckpointMismatch, FileNotFoundError) as e:
            print(f"refusing --init-from: {e}", file=sys.stderr)
            return EXIT_INIT_REFUSED
        run = dataclasses.replace(run, init_from=init_src.loaded_from, init_from_sha256=init_src.sha256)
        print(f"init-from: {init_src.loaded_from} ({init_src.kind}, sha256 {init_src.sha256})")
    if run.bc_aux.enabled:  # cheap: index hashes now, so a resume can compare them
        from .bc_aux import BCDataRefused, dataset_hashes
        try:
            run = dataclasses.replace(run, bc_aux=dataclasses.replace(
                run.bc_aux, data_sha256=dataset_hashes(run.bc_aux.data_dirs)))
        except BCDataRefused as e:
            print(f"refusing --bc-data: {e}", file=sys.stderr)
            return EXIT_BC_REFUSED
    decisions_done, clock_envs = 0, run.num_envs
    if ns.resume:
        # read BEFORE the vec env is built: refuse a shaping change (ADR 0011 F6) and
        # continue the bonus anneal on the checkpoint's own per-env clock (ADR 0010)
        from .checkpoint import CheckpointMismatch, check_shaping_matches, checkpoint_resume_info
        from .checkpoint import check_maps_matches
        info = checkpoint_resume_info(ns.resume)
        try:
            check_shaping_matches(info, ns.shaping, info["loaded_from"])
            check_maps_matches(info["maps"], run.maps, info["loaded_from"])
            from .checkpoint import check_bc_matches, check_switches_match, switches_requested
            check_bc_matches(info["bc_aux"], run.bc_aux, info["loaded_from"])
            check_switches_match(info["switches"], switches_requested(run), info["loaded_from"])
            if scheduler is not None and info["curriculum_json"] is None:
                raise CheckpointMismatch(f"checkpoint {info['loaded_from']!r} is a generated-maps run "
                                         f"without curriculum state; refusing to restart the curriculum")
        except CheckpointMismatch as e:
            print(str(e), file=sys.stderr)
            return EXIT_RESUME_REFUSED
        decisions_done, clock_envs = info["decisions_done"], info["num_envs"]
        if info["legacy"]:
            print(f"resume: {info['loaded_from']} has no shaping record; read as unshaped "
                  f"with {clock_envs} envs from its run config")
    bc_episodes, bc_spec = None, None
    if run.bc_aux.enabled:  # loaded and validated BEFORE any engine starts
        from configs.a0_env import obs_version

        from .bc_aux import BCDataRefused, load_bc_data
        from .bc_spec import a0_training_spec_and_signature
        bc_spec, bc_sig = a0_training_spec_and_signature()
        try:
            if bc_sig != signature:
                raise BCDataRefused("the engine-free A0 sensor signature differs from this run's")
            bc_episodes = load_bc_data(run.bc_aux.data_dirs, bc_spec, run.bc_aux.max_bytes, obs_version())
        except BCDataRefused as e:
            print(f"refusing --bc-data: {e}", file=sys.stderr)
            return EXIT_BC_REFUSED
        print(f"bc aux: {len(bc_episodes)} episodes, {sum(e.length for e in bc_episodes)} steps from "
              f"{list(run.bc_aux.data_dirs)}; coef {run.bc_aux.coef} -> {run.bc_aux.coef_final} "
              f"({run.bc_aux.decay_mode}), K={run.bc_aux.batch_seqs} L={run.bc_aux.seq_len}")
    factory = build_env_factory(env_cfg, ns.shaping, decisions_done, clock_envs)
    if ns.shaping is not None:
        print(f"reward shaping: {ns.shaping}, anneal start at {factory.start_decisions} per-env decisions "
              f"(trainer decisions {decisions_done} / {clock_envs} envs)")
    venv = None
    left: list = []
    try:
        venv = SubprocVecEnv([[factory] * run.envs_per_worker for _ in range(run.num_workers)], safety,
                             allow_multi_instance=ns.allow_multi_instance, torch_threads=run.torch_threads)
        curriculum = None
        if scheduler is not None:
            from .curriculum_vec_env import CurriculumVecEnv
            from maps.loader import sha256_file
            venv = curriculum = CurriculumVecEnv(venv, scheduler, run.curriculum_log.recent_window,
                                                 bundle_sha256=sha256_file(env_cfg.map_wad))
            if ns.resume:  # before the trainer's first reset, so the curriculum continues
                try:
                    curriculum.load_state_json(info["curriculum_json"])
                except ValueError as e:
                    print(f"refusing to resume: {e}", file=sys.stderr)
                    return EXIT_RESUME_REFUSED
                print(f"curriculum resumed: stage {scheduler.stage}, {scheduler.episodes} outcomes recorded, "
                      f"{scheduler.issued} choices issued, promotions {scheduler.history}, "
                      f"dangling restored choices {curriculum.dangling_restored}")
        # torch-using modules imported only here, never at module level (spawn re-import)
        from configs.model_config import ModelConfig

        from .checkpoint import resume
        from .run_loop import make_health_monitor, run_training
        from .trainer import Trainer

        trainer = Trainer(venv, ModelConfig(), run)
        if bc_episodes is not None:
            if trainer.spec != bc_spec:
                print(f"refusing --bc-data: run obs spec {trainer.spec} differs from the spec the BC data "
                      f"was validated against {bc_spec}", file=sys.stderr)
                return EXIT_BC_REFUSED
            trainer.attach_bc(bc_episodes)
        blob = None
        if init_src is not None:
            from .checkpoint import CheckpointMismatch
            from .init_from import apply_init
            try:
                growth = apply_init(trainer, init_src)
            except (CheckpointMismatch, RuntimeError, KeyError, AssertionError) as e:
                print(f"refusing --init-from: {e}", file=sys.stderr)
                return EXIT_INIT_REFUSED
            if growth is not None:
                from .checkpoint import canonical_json
                from .init_from import growth_record
                grown_json = canonical_json(growth_record(init_src, growth))
                run = dataclasses.replace(run, init_from_growth=grown_json)
                trainer.run = dataclasses.replace(trainer.run, init_from_growth=grown_json)
                print(f"init-from GROWTH: {growth.describe()}; obs_version "
                      f"{init_src.obs_version_change or 'unchanged'}")
            print(f"fresh run initialised from {init_src.loaded_from}; value warm-up "
                  f"{run.ppo.value_warmup_updates} updates")
            if run.rnd.enabled or run.inv_dyn.enabled:
                print(f"init-from: model weights only; RND networks and normalisers "
                      f"({'on' if run.rnd.enabled else 'off'}) and the inverse-dynamics head "
                      f"({'on' if run.inv_dyn.enabled else 'off'}) start fresh, nothing copied from the file")
        if ns.resume:
            blob = resume(trainer, ns.resume, signature)
            print(f"resumed from {blob['loaded_from']} at update {trainer.update_idx}, session {trainer.session}, "
                  f"env seed {blob['session_env_seed']}")
        monitor = make_health_monitor(trainer, blob)
        write_run_start(run_dir, run, ns, argv)
        summary = run_training(trainer, run_dir, signature, safety, ns.max_updates, ns.max_seconds,
                               extra_pids=venv.process_pids, emergency=venv.kill_engines, monitor=monitor,
                               curriculum=curriculum)
        print("summary: " + json.dumps(dataclasses.asdict(summary)))
    finally:
        if venv is not None:
            venv.close()
        left = PC.engine_processes(safety.engine_process_names)
        print(f"engine processes alive after shutdown: {len(left)}")
    return 3 if left else 0


if __name__ == "__main__":
    sys.exit(main())
