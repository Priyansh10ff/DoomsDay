"""run_doom flags for the BC hand-off (--init-from, --value-warmup-updates,
--start-stage) and the trainer's value warm-up. No engine is started."""
from __future__ import annotations

import os

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from a0_helpers import FakeDoomEnv
from configs.model_config import ModelConfig
from configs.train_config import PPOConfig, RunConfig
from train import run_doom
from train.trainer import WARMUP_TRAINABLE_PREFIXES, Trainer
from train.vec_env import SyncVecEnv

SMALL = ModelConfig(core_hidden=32, fusion_dim=32, ray_dim=16, entity_token_dim=16, entity_mlp_hidden=16,
                    self_dim=16, self_hidden=16)


def make_trainer(warmup: int) -> Trainer:
    venv = SyncVecEnv([lambda: FakeDoomEnv(5) for _ in range(2)])
    run = replace(RunConfig(), seed=0, num_workers=1, envs_per_worker=2,
                  ppo=PPOConfig(rollout_len=8, num_minibatches=1, epochs=2, value_warmup_updates=warmup))
    return Trainer(venv, SMALL, run)


def snapshot(tr: Trainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in tr.learner.named_parameters()}


def changed(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> set[str]:
    return {k for k in a if not torch.equal(a[k], b[k])}


def test_value_warmup_moves_only_value_and_aux_heads_then_unfreezes() -> None:
    tr = make_trainer(warmup=1)
    p0 = snapshot(tr)
    s1 = tr.train_iteration()
    p1 = snapshot(tr)
    moved = changed(p0, p1)
    assert moved, "value head must train during warm-up"
    assert all(k.startswith(WARMUP_TRAINABLE_PREFIXES) for k in moved), sorted(moved)
    assert any(k.startswith("value_ext.") for k in moved)
    assert s1["value_warmup_active"] == 1.0 and s1["value_warmup_ended"] == 1.0
    assert all(p.requires_grad for p in tr.learner.parameters()), "must unfreeze after the warm-up update"
    s2 = tr.train_iteration()
    moved2 = changed(p1, snapshot(tr))
    assert any(k.startswith("policy_heads.") for k in moved2) and any(k.startswith("core.") for k in moved2)
    assert s2["value_warmup_active"] == 0.0 and s2["value_warmup_ended"] == 0.0


def test_no_warmup_by_default_moves_policy_on_first_update() -> None:
    tr = make_trainer(warmup=0)
    p0 = snapshot(tr)
    s = tr.train_iteration()
    assert any(k.startswith("policy_heads.") for k in changed(p0, snapshot(tr)))
    assert s["value_warmup_active"] == 0.0


def test_parse_args_init_from_resume_and_start_stage_rules() -> None:
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--init-from", "a.pt", "--resume", "b.pt"])
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--start-stage", "S2"])                    # needs --maps generated
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--maps", "generated", "--start-stage", "S2", "--resume", "b.pt"])
    with pytest.raises(SystemExit):
        run_doom.parse_args(["--value-warmup-updates", "-1"])
    ns = run_doom.parse_args(["--maps", "generated", "--shaping", "full_level", "--init-from", "a.pt",
                              "--value-warmup-updates", "5", "--start-stage", "S2"])
    run = run_doom.build_run_config(ns)
    assert run.ppo.value_warmup_updates == 5 and run.start_stage == "S2" and ns.init_from == "a.pt"
    assert run_doom.build_run_config(run_doom.parse_args([])).ppo.value_warmup_updates == 0


def test_apply_start_stage() -> None:
    sch = SimpleNamespace(stages=["S0", "S1", "S2", "S3"], stage_idx=0, issued=0, episodes=0)
    run_doom.apply_start_stage(sch, None)
    assert sch.stage_idx == 0
    run_doom.apply_start_stage(sch, "S2")
    assert sch.stage_idx == 2
    with pytest.raises(ValueError, match="not a curriculum stage"):
        run_doom.apply_start_stage(sch, "S9")
    sch.issued = 3
    with pytest.raises(ValueError, match="before"):
        run_doom.apply_start_stage(sch, "S1")


def test_run_config_records_init_source_fields() -> None:
    run = replace(RunConfig(), init_from="x/bc_model.pt", init_from_sha256="0" * 64)
    from train.checkpoint import canonical_json
    import dataclasses
    text = canonical_json(dataclasses.asdict(run))
    assert '"init_from": "x/bc_model.pt"' in text and '"init_from_sha256"' in text


def test_run_loop_logs_the_end_of_the_value_warmup(tmp_path: "os.PathLike[str]") -> None:
    from configs.train_config import SafetyConfig
    from train.run_loop import run_training
    tr = make_trainer(warmup=1)
    lines: list[str] = []
    safety = replace(SafetyConfig(), keep_numbered_checkpoints=False)
    run_training(tr, str(tmp_path), (("fake", True),), safety, max_updates=2, max_seconds=120,
                 log=lines.append, install_sigint=False, monitor=None)
    ended = [m for m in lines if "value warm-up ended" in m]
    assert len(ended) == 1 and "update 1" in ended[0]


def test_start_stage_on_a_real_scheduler_issues_that_stage() -> None:
    import copy
    from curriculum.scheduler import CurriculumScheduler, load_config
    pool = {s: [(f"DD{i:04d}{s}", f"{s}-train-{i}") for i in range(3)] for s in ("S0", "S1", "S2", "S3")}
    c = copy.deepcopy(load_config())
    c["replay"]["fraction"] = 0.0
    sch = CurriculumScheduler(c, pool, seed=1)
    run_doom.apply_start_stage(sch, "S2")
    assert sch.stage == "S2"
    assert {sch.next_episode().stage for _ in range(5)} == {"S2"}
