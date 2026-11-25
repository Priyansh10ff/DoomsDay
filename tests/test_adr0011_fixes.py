"""ADR 0011 fix requests F1-F7, ADR 0010 change 3 (shaping means), the passthrough
preset, and the --device option. No engine is started: toy and fake envs only.

The monitor test replays the first DOOM run's logs and must reproduce the E0
forensics table `adr0011_levels_primary.csv` exactly (coordinator rulings: full
windows only, exact ln maxima)."""
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from a0_helpers import ROOT
from configs.a0_env import a0_train_env_config
from configs.reward_config import MY_WAY_HOME_COVERAGE, PASSTHROUGH_REWARD
from configs.train_config import HealthMonitorConfig, PPOConfig, RunConfig, SafetyConfig
from rewards.factory import ShapedEnvFactory
from test_shaping_integration import SIG, SMALL, _PosFactory
from test_train_ppo import SMALL as PPO_SMALL
from test_train_ppo import make_trainer
from train import checkpoint as CK
from train import run_doom
from train.device import DeviceUnavailable, resolve_device
from train.health_monitor import HealthMonitor
from train.run_loop import CHECKPOINT_NAME, METRICS_NAME, run_training
from train.trainer import Trainer, action_shares
from train.vec_env import SyncVecEnv

RUN_DIR = os.path.join(ROOT, "reports", "runs", "20261125-nogit-7a675759")
REF_CSV = os.path.join(ROOT, "reports", "eval", "2026-11-25-first-run-log-forensics-data",
                       "adr0011_levels_primary.csv")
DOOM_HEADS = (3, 3, 7, 2, 2, 2)
QUIET = dict(log=lambda m: None, install_sigint=False)


def _jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _replay(mon: HealthMonitor, metrics: list[dict], eps: list[dict], start: int = 0, seen: int = 0):
    """Feed metrics lines [start:] in update order; returns the readings and episodes seen."""
    out = []
    for m in metrics[start:]:
        n = int(m["episodes_done"])
        out.append(mon.observe(int(m["update_idx"]), [m[f"entropy_head{h}"] for h in range(6)],
                               [e["success"] for e in eps[seen:n]]))
        seen = n
    return out, seen


def _opt(v: Any) -> str:
    return "" if v is None else str(v)


# ---------------------------------------------------------------- F4: the monitor
@pytest.mark.skipif(not (os.path.exists(REF_CSV) and os.path.exists(os.path.join(RUN_DIR, METRICS_NAME))),
                    reason="first-run logs or E0 forensics table not present")
def test_monitor_reproduces_e0_forensics_table_exactly() -> None:
    metrics = _jsonl(os.path.join(RUN_DIR, METRICS_NAME))
    eps = _jsonl(os.path.join(RUN_DIR, "episodes.jsonl"))
    with open(REF_CSV, encoding="utf-8") as f:
        ref = list(csv.DictReader(f))
    assert len(ref) == len(metrics) == 1814
    mon = HealthMonitor(HealthMonitorConfig(), DOOM_HEADS)
    readings, _ = _replay(mon, metrics, eps)
    first_diff = None
    for r, rd in zip(ref, readings):
        exact = {"update": str(rd.update), "level": rd.level, "watch": str(rd.watch), "warning": str(rd.warning),
                 "warn_heads": "+".join(rd.warn_heads), "drop_cond": str(rd.drop), "low_cond": str(rd.low),
                 "event": str(rd.event), "succ_count": _opt(rd.success_count)}
        bad = [k for k, v in exact.items() if r[k] != v]
        if bad and first_diff is None:
            first_diff = (r["update"], bad)
        for col, val in (("run_best", None if rd.run_best_count is None else rd.run_best_count / 500),
                         ("ent_sum", rd.entropy_sum), ("ent_sum_mean100", rd.entropy_sum_mean)):
            assert (r[col] == "") == (val is None), (r["update"], col)
            if val is not None:
                assert float(r[col]) == pytest.approx(val, rel=0, abs=1e-12), (r["update"], col)
    assert first_diff is None, f"first differing update {first_diff}"
    events = [rd.update for rd in readings if rd.event]
    assert events == list(range(814, 1106)), "one collapse event, updates 814-1105"
    assert not any(rd.stop for rd in readings) and "stop" not in mon.levels_reached
    assert not mon.event_active
    assert all(rd.success_count is None and rd.level in ("none", "warning") for rd in readings[:117])


@pytest.mark.skipif(not os.path.exists(os.path.join(RUN_DIR, METRICS_NAME)), reason="first-run logs not present")
def test_monitor_state_survives_a_resume_mid_event() -> None:
    metrics = _jsonl(os.path.join(RUN_DIR, METRICS_NAME))[:1200]
    eps = _jsonl(os.path.join(RUN_DIR, "episodes.jsonl"))
    whole, _ = _replay(HealthMonitor(HealthMonitorConfig(), DOOM_HEADS), metrics, eps)
    a = HealthMonitor(HealthMonitorConfig(), DOOM_HEADS)
    first, seen = _replay(a, metrics[:900], eps)
    b = HealthMonitor(HealthMonitorConfig(), DOOM_HEADS)
    b.load_json(a.to_json())               # what the checkpoint carries
    assert b.event_start == 814
    second, _ = _replay(b, metrics, eps, start=900, seen=seen)
    assert first + second == whole         # deterministic, and identical across the resume


def test_monitor_full_windows_only_and_exact_lines() -> None:
    cfg = HealthMonitorConfig(window_episodes=4, watch_mean_updates=2, warn_mean_updates=2,
                              p0_successes=1, p0_episodes=5)   # p0 + 0.10 = 0.3 -> count < 1.2
    mon = HealthMonitor(cfg, (2, 3))
    low = [0.0, 0.0]                                           # far below every entropy line
    r1 = mon.observe(1, low, [True, True, True])               # 3 episodes: window not full
    assert r1.success_count is None and r1.head_means == (None, None) and r1.level == "none"
    r2 = mon.observe(2, low, [True])                           # full window 4/4, warning on both heads
    assert (r2.success_count, r2.run_best_count, r2.level) == (4, 4, "warning")
    r3 = mon.observe(3, low, [False])                          # 3/4: drop 0.25 >= 0.15 -> event
    assert r3.drop and r3.event and r3.event_start == 3 and r3.level == "event"
    r4 = mon.observe(4, low, [True])                           # 3/4 again: the same event goes on
    assert r4.event and r4.event_start == 3
    r5 = mon.observe(5, [math.log(2), math.log(3)], [])        # 2-update means at half max -> event over
    assert not r5.warning and not r5.event and r5.event_start is None and r5.level == "none"
    r6 = mon.observe(6, low, [])                               # means 0.5 x max: still no warning
    assert r6.level == "none"
    r7 = mon.observe(7, low, [])                               # a new event starts at 7, not 3
    assert r7.event and r7.event_start == 7
    with pytest.raises(ValueError):
        HealthMonitor(cfg, (1, 3))


def test_head_names_match_the_env_action_heads() -> None:
    actions = pytest.importorskip("env.actions")
    assert HealthMonitorConfig().head_names == tuple(actions.HEAD_NAMES)
    assert actions.head_sizes(7) == DOOM_HEADS


def test_stop_level_saves_and_stops_the_run(tmp_path: "os.PathLike[str]") -> None:
    tr = make_trainer(n_envs=4, t_len=10)
    # every head always "warns", success is always "low": an event from the first full window
    tr.run = replace(tr.run, health=HealthMonitorConfig(window_episodes=2, watch_mean_updates=1, warn_mean_updates=1,
                                                         warn_head_fraction=1.01, p0_successes=1, p0_episodes=1,
                                                         stop_after_updates=3))
    d = str(tmp_path)
    s = run_training(tr, d, SIG, SafetyConfig(), max_updates=50, max_seconds=120, **QUIET)
    assert s.stop_reason.startswith("health:") and s.health_stop_update == 4 and tr.update_idx == 4
    assert "stop" in s.health_levels_reached and s.health_event_active_at_end
    lines = _jsonl(os.path.join(d, METRICS_NAME))
    assert [m["health_level"] for m in lines] == ["event", "event", "event", "stop"]
    assert [m["health_event_start"] for m in lines] == [1, 1, 1, 1]
    blob = CK.load_checkpoint(os.path.join(d, CHECKPOINT_NAME))
    assert int(blob["trainer"]["update_idx"]) == 4 and CK.HEALTH_KEY in blob


def test_agent_imports_no_monitor_reward_or_intrinsic_code_and_monitor_is_torch_free() -> None:
    bad = re.compile(r"^\s*(?:from|import)\s+(train|rewards|intrinsic|configs\.reward_config)\b"
                     r"|health_monitor", re.M)
    offenders = []
    for base, _, files in os.walk(os.path.join(ROOT, "agent")):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(base, f), encoding="utf-8") as fh:
                    if bad.search(fh.read()):
                        offenders.append(f)
    assert not offenders, offenders
    code = "import sys, train.health_monitor; print(int('torch' in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["0"], out.stderr


# ---------------------------------------------------------------- F1, F2: histograms, advantage spread
def test_action_shares_sum_to_one_per_head_and_match_counts() -> None:
    g = torch.Generator().manual_seed(0)
    sizes = (3, 7, 2)
    acts = torch.stack([torch.randint(0, k, (5, 4), generator=g) for k in sizes], dim=-1)
    shares = action_shares(acts, sizes)
    assert [len(s) for s in shares] == list(sizes)
    for h, s in enumerate(shares):
        assert sum(s) == pytest.approx(1.0, abs=1e-12)
        assert s == [float((acts[..., h] == k).sum()) / 20 for k in range(sizes[h])]


def test_metrics_carry_histograms_advantage_spread_and_engine_return(tmp_path: "os.PathLike[str]") -> None:
    tr = make_trainer(n_envs=4, t_len=10)
    d = str(tmp_path)
    run_training(tr, d, SIG, SafetyConfig(), max_updates=2, max_seconds=120, **QUIET)
    m = _jsonl(os.path.join(d, METRICS_NAME))[-1]
    for h, k in enumerate(tr.spec.head_sizes):
        sh = [m[f"action_share_head{h}_opt{o}"] for o in range(k)]
        assert sum(sh) == pytest.approx(1.0, abs=1e-9) and min(sh) >= 0.0
        assert f"action_share_head{h}_opt{k}" not in m
    assert m["adv_std_prenorm"] > 0 and math.isfinite(m["adv_mean_prenorm"])
    assert m["recent_engine_return"] == m["recent_return"]   # unshaped: the same quantity
    assert not any(k.startswith("shaping_") for k in m)
    assert m["health_level"] in ("none", "watch", "warning", "event", "stop")


def test_advantage_spread_is_the_pre_normalisation_std() -> None:
    from train.ppo import evaluate_minibatch
    tr = make_trainer(n_envs=2, t_len=6)
    ro = tr.collect()
    adv, ret = tr.advantages(ro)
    idx = torch.arange(2)
    with torch.no_grad():
        res = evaluate_minibatch(tr.model, ro, idx, adv, ret, tr.ppo, tr.run.aux)
    a = adv[ro.valid]
    assert res.stats["adv_std_prenorm"] == pytest.approx(float(a.std(unbiased=False)), rel=1e-5)


# ---------------------------------------------------------------- F3: numbered checkpoints
def test_numbered_checkpoints_kept_loadable_and_thinned(tmp_path: "os.PathLike[str]") -> None:
    tr = make_trainer(n_envs=2, t_len=4)
    d = str(tmp_path)
    safety = replace(SafetyConfig(), checkpoint_interval_s=0.0)    # periodic save after every update
    s = run_training(tr, d, SIG, safety, max_updates=4, max_seconds=120, **QUIET)
    files = CK.numbered_checkpoints(d)
    assert [os.path.basename(f) for f in files] == [f"checkpoint-u{u:07d}.pt" for u in (1, 2, 3, 4)]
    assert os.path.exists(os.path.join(d, CHECKPOINT_NAME)) and os.path.exists(os.path.join(d, CHECKPOINT_NAME + ".prev"))
    for u, f in zip((1, 2, 3, 4), files):
        raw = torch.load(f, map_location="cpu", weights_only=True)
        assert int(raw["trainer"]["update_idx"]) == u
    assert s.numbered_checkpoint_bytes == os.path.getsize(files[-1]) > 0
    sizes = [os.path.getsize(f) for f in files]
    assert CK.thin_numbered(d, budget_bytes=sum(sizes)) == []          # at the budget: nothing removed
    removed = CK.thin_numbered(d, budget_bytes=int(3.5 * min(sizes)))  # one thinning pass
    assert removed == [files[1]] and CK.numbered_checkpoints(d) == [files[0], files[2], files[3]]
    removed = CK.thin_numbered(d, budget_bytes=int(1.5 * min(sizes)))  # never below two, newest kept
    assert removed == [files[2]] and CK.numbered_checkpoints(d) == [files[0], files[3]]


# ---------------------------------------------------------------- F5: entropy coefficient flag
def test_entropy_coef_flag_default_value_and_refusal() -> None:
    assert run_doom.parse_args([]).entropy_coef == PPOConfig().entropy_coef
    run = run_doom.build_run_config(run_doom.parse_args(["--entropy-coef", "0.03"]))
    assert run.ppo.entropy_coef == 0.03
    assert "entropy_coef=0.03" in run_doom.run_settings_line(run, "cpu")
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--entropy-coef", "-0.1"])


class _Built(Exception):
    pass


def _main_until_vec_env(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict:
    seen: dict = {}

    def fake_vec_env(layout: list, *a: Any, **k: Any) -> None:
        seen["factory"] = layout[0][0]
        raise _Built

    monkeypatch.setattr(run_doom, "SubprocVecEnv", fake_vec_env)
    monkeypatch.setattr(run_doom.PC, "engine_processes", lambda names: [])
    monkeypatch.setattr(run_doom.PC, "set_below_normal", lambda *a: None)
    try:
        seen["rc"] = run_doom.main(argv)
    except _Built:
        seen["rc"] = None
    return seen


# ---------------------------------------------------------------- F6: shaping recorded, mismatch refused
@pytest.mark.parametrize("stored,requested", [(None, "my_way_home"), ("my_way_home", None),
                                              ("passthrough", "my_way_home"), ("my_way_home", "passthrough")])
def test_resume_with_a_different_shaping_is_refused(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture, stored: str | None,
                                                    requested: str | None) -> None:
    tr = make_trainer(n_envs=2, t_len=4)
    tr.run = replace(tr.run, shaping=stored)
    ckpt = os.path.join(str(tmp_path), "c.pt")
    CK.save_checkpoint(ckpt, tr, SIG)
    rec = CK.shaping_record(CK.load_checkpoint(ckpt))
    assert rec == {"preset": stored, "num_envs": 2, "legacy": False}
    argv = ["--resume", ckpt, "--run-dir", str(tmp_path)] + (["--shaping", requested] if requested else [])
    seen = _main_until_vec_env(monkeypatch, argv)
    assert seen["rc"] == run_doom.EXIT_RESUME_REFUSED and "factory" not in seen
    assert "refusing to resume" in capsys.readouterr().err
    other = make_trainer(n_envs=2, t_len=4)
    other.run = replace(other.run, shaping=requested)
    with pytest.raises(CK.CheckpointMismatch, match="shaping"):
        CK.resume(other, ckpt, SIG)


def test_legacy_checkpoint_reads_as_unshaped_with_its_run_config_env_count(tmp_path: "os.PathLike[str]") -> None:
    tr = make_trainer(n_envs=4, t_len=4)
    ckpt = os.path.join(str(tmp_path), "c.pt")
    CK.save_checkpoint(ckpt, tr, SIG)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    del blob[CK.SHAPING_KEY]
    run = json.loads(blob["run_config_json"])
    del run["shaping"]
    run["num_workers"], run["envs_per_worker"] = 4, 8         # as the first run's checkpoint
    blob["run_config_json"] = json.dumps(run)
    torch.save(blob, ckpt)
    assert CK.shaping_record(CK.load_checkpoint(ckpt)) == {"preset": None, "num_envs": 32, "legacy": True}
    with pytest.raises(CK.CheckpointMismatch, match="legacy"):
        CK.check_shaping_matches(CK.shaping_record(CK.load_checkpoint(ckpt)), "my_way_home")


# ---------------------------------------------------------------- B: shaping means, passthrough
def _shaped_trainer(preset: str) -> Trainer:
    reward, coverage = run_doom.shaping_configs(preset)
    venv = SyncVecEnv([ShapedEnvFactory(_PosFactory(i == 1), reward, coverage) for i in range(2)])
    run = replace(RunConfig(), num_workers=1, envs_per_worker=2, torch_threads=2, shaping=preset,
                  ppo=PPOConfig(rollout_len=10, num_minibatches=1, epochs=1))
    return Trainer(venv, SMALL, run)


def test_passthrough_preset_pays_exactly_the_engine_reward_and_logs_cells(tmp_path: "os.PathLike[str]") -> None:
    reward, coverage = run_doom.shaping_configs("passthrough")
    assert reward == PASSTHROUGH_REWARD and coverage.enabled
    assert (coverage.coef_start, coverage.coef_end) == (0.0, 0.0)
    assert replace(coverage, coef_start=MY_WAY_HOME_COVERAGE.coef_start,
                   coef_end=MY_WAY_HOME_COVERAGE.coef_end) == MY_WAY_HOME_COVERAGE
    f = run_doom.build_env_factory(a0_train_env_config(), "passthrough", 0, 1)
    assert isinstance(f, ShapedEnvFactory) and f.coverage == coverage
    tr = _shaped_trainer("passthrough")
    d = str(tmp_path)
    run_training(tr, d, SIG, SafetyConfig(), max_updates=1, max_seconds=60, **QUIET)
    recs = _jsonl(os.path.join(d, "episodes.jsonl"))
    assert recs
    for r in recs:
        assert r["return"] == r["engine_return"] == r["length"] * 1.0
        assert isinstance(r["shaping_episode"]["cells_visited"], int) and r["shaping_episode"]["cells_visited"] > 1
        assert r["shaping_episode"]["intrinsic"] == 0.0 and r["shaping_episode"]["intrinsic_raw"] > 0
    m = _jsonl(os.path.join(d, METRICS_NAME))[-1]
    assert m["shaping_mean_intrinsic"] == 0.0 and m["shaping_mean_intrinsic_coef"] == 0.0
    assert m["shaping_mean_intrinsic_raw"] > 0 and m["shaping_mean_extrinsic"] == pytest.approx(1.0)
    assert m["recent_engine_return"] == m["recent_return"]
    assert m["shaping_cells_visited_mean"] == pytest.approx(
        np.mean([r["shaping_episode"]["cells_visited"] for r in recs]))


def test_shaped_metrics_log_term_means_and_the_engine_return_separately(tmp_path: "os.PathLike[str]") -> None:
    tr = _shaped_trainer("my_way_home")
    d = str(tmp_path)
    run_training(tr, d, SIG, SafetyConfig(), max_updates=1, max_seconds=60, **QUIET)
    recs = _jsonl(os.path.join(d, "episodes.jsonl"))
    m = _jsonl(os.path.join(d, METRICS_NAME))[-1]
    # 20 env-steps, every one carries info["shaping"] (ending steps through final_infos)
    assert m["shaping_mean_intrinsic_coef"] == pytest.approx(MY_WAY_HOME_COVERAGE.coef_start, rel=1e-3)
    assert m["shaping_mean_intrinsic"] > 0
    assert m["shaping_mean_total"] == pytest.approx(m["shaping_mean_extrinsic"] + m["shaping_mean_intrinsic"])
    k = min(len(recs), 20)
    assert m["recent_engine_return"] == pytest.approx(np.mean([r["engine_return"] for r in recs[-k:]]))
    assert m["recent_engine_return"] != m["recent_return"]


def test_shaping_step_means_read_final_infos_for_ended_envs() -> None:
    tr = _shaped_trainer("my_way_home")
    tr.collect()
    # env 0 times out every 5 decisions, env 1 exits every 3: the ending steps' values are
    # only in final_infos (infos then holds the reset info), so all 2 x 10 steps count only
    # if final_infos are read
    assert tr.last_shaping_means["shaping_steps_logged"] == 20.0
    assert tr.last_shaping_means["shaping_mean_intrinsic_raw"] == pytest.approx(1.0 / MY_WAY_HOME_COVERAGE.budget_cells)


# ---------------------------------------------------------------- C: device
def test_resolve_device_never_falls_back_silently() -> None:
    assert resolve_device("cpu", lambda: True) == "cpu"
    assert resolve_device("auto", lambda: False) == "cpu"
    assert resolve_device("auto", lambda: True) == "cuda"
    assert resolve_device("cuda", lambda: True) == "cuda"
    with pytest.raises(DeviceUnavailable, match="no automatic fallback"):
        resolve_device("cuda", lambda: False)
    with pytest.raises(ValueError):
        resolve_device("tpu")
    assert run_doom.parse_args([]).device == "cpu" and RunConfig().device == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA refusal")
def test_device_cuda_without_cuda_is_refused_before_any_engine(tmp_path: "os.PathLike[str]",
                                                               monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture) -> None:
    seen = _main_until_vec_env(monkeypatch, ["--device", "cuda", "--run-dir", str(tmp_path)])
    assert seen["rc"] == run_doom.EXIT_DEVICE_UNAVAILABLE and "factory" not in seen
    assert "torch.cuda.is_available() is False" in capsys.readouterr().err


def test_default_device_run_is_unchanged_and_deterministic() -> None:
    def params(device: str) -> list[torch.Tensor]:
        tr = make_trainer(seed=3)
        assert tr.run.device == device and tr.device == torch.device("cpu")
        tr.train_iteration()
        return [p.detach().clone() for p in tr.model.parameters()]
    a, b = params("cpu"), params("cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available (CPU-only torch build)")
def test_cuda_training_and_checkpoints_load_across_devices(tmp_path: "os.PathLike[str]") -> None:
    from a0_helpers import FakeDoomEnv
    venv = SyncVecEnv([lambda: FakeDoomEnv(7) for _ in range(4)])
    run = replace(RunConfig(), device="cuda", num_workers=1, envs_per_worker=4,
                  ppo=PPOConfig(rollout_len=10, num_minibatches=2, epochs=2))
    g = Trainer(venv, PPO_SMALL, run)             # same model config as make_trainer's
    stats = g.train_iteration()
    assert np.isfinite(stats["loss"]) and next(g.model.parameters()).is_cuda
    path = os.path.join(str(tmp_path), "g.pt")
    CK.save_checkpoint(path, g, SIG)
    c = make_trainer(n_envs=4)
    CK.resume(c, path, SIG)
    assert not next(c.model.parameters()).is_cuda
    for (k, x), y in zip(g.model.state_dict().items(), c.model.state_dict().values()):
        assert torch.equal(x.cpu(), y), k
    c.train_iteration()                       # optimizer state moved to CPU with the params
    path2 = os.path.join(str(tmp_path), "c.pt")
    CK.save_checkpoint(path2, c, SIG)
    CK.resume(g, path2, SIG)
    g.train_iteration()
