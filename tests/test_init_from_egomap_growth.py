"""--init-from / --init-from-bc egomap growth (train/egomap_growth.py, train/init_from.py,
train/bc.py read_init_bc): a v3 model (egomap 5 channels) starting a v4 run (6
channels) computes exactly what it computed before, gradients reach the new slices,
any other difference is still refused, and equal specs load strictly as before.
No engine is started."""
from __future__ import annotations

import dataclasses
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent.model import A0Agent, build_agent
from agent.obs_spec import EGOMAP_KEY, EgoMapObsSpec, ObsSpec
from configs.model_config import ModelConfig
from configs.train_config import RunConfig
from train import bc as BC
from train import checkpoint as CK
from train.egomap_growth import (FIRST_EGOMAP_CONV_KEY, EgomapGrowth, egomap_growth, grow_state,
                                 obs_version_change, state_shapes)
from train.init_from import InitSource, apply_init, growth_record, read_init_source

CFG = ModelConfig()
V2 = dict(ray_channels=4, ray_width=80, max_entities=24, entity_feat_dim=8, n_entity_types=14,
          self_dim=34, modality_dim=3, head_sizes=(3, 3, 7, 2, 2, 2))
HW = (32, 32)
V3 = EgoMapObsSpec(**V2, egomap_shape=(5, *HW))
V4 = EgoMapObsSpec(**V2, egomap_shape=(6, *HW))
BASE_SIG = (("frame_skip", 4), ("ray_bands", 4))
SIG3 = BASE_SIG + (("obs_version", 3),)
SIG4 = BASE_SIG + (("obs_version", 4),)
B, T = 5, 7


def spec_json(spec: ObsSpec) -> str:
    return CK.canonical_json(dataclasses.asdict(spec))


def random_obs(spec: ObsSpec, lead: tuple[int, ...], seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ego = spec.egomap
    assert ego is not None
    return {
        "rays": torch.rand(*lead, spec.ray_channels, spec.ray_width, generator=g),
        "entities": torch.randn(*lead, spec.max_entities, spec.entity_feat_dim, generator=g),
        "entity_type": torch.randint(0, spec.n_entity_types, (*lead, spec.max_entities), generator=g),
        "entity_mask": torch.rand(*lead, spec.max_entities, generator=g) < 0.5,
        "self": torch.randn(*lead, spec.self_dim, generator=g),
        "prev_action": torch.stack([torch.randint(0, n, lead, generator=g) for n in spec.head_sizes], -1),
        "modality": torch.rand(*lead, spec.modality_dim, generator=g),
        EGOMAP_KEY: (torch.rand(*lead, *ego, generator=g) < 0.3).to(torch.uint8),
    }


def extend_egomap(obs: dict[str, torch.Tensor], extra: int, seed: int) -> dict[str, torch.Tensor]:
    """The same observation with `extra` arbitrary egomap channels appended at the end."""
    e = obs[EGOMAP_KEY]
    g = torch.Generator().manual_seed(seed)
    more = torch.randint(0, 2, (*e.shape[:-3], extra, *e.shape[-2:]), generator=g).to(torch.uint8)
    return {**obs, EGOMAP_KEY: torch.cat([e, more], dim=-3)}


def grown_v4_from(v3: A0Agent, seed: int = 11) -> A0Agent:
    v4 = build_agent(V4, CFG, seed=seed)           # different seed: every weight must come from v3
    growth = egomap_growth(spec_json(V3), V4)
    assert growth is not None
    v4.load_state_dict(grow_state(v3.state_dict(), growth, state_shapes(v4)), strict=True)
    return v4


def outputs(model: A0Agent, obs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    starts = torch.zeros(T, B)
    starts[0] = 1
    starts[3, 1] = 1
    with torch.no_grad():
        out, h = model.forward_sequence(obs, model.initial_state(B), starts)
    return [*out.logits, out.value_ext, out.value_int, out.aux_keys, out.aux_damage, out.aux_monsters, h]


# ---------------------------------------------------------------- spec comparison
def test_growth_detected_only_for_appended_egomap_channels() -> None:
    assert egomap_growth(spec_json(V3), V4) == EgomapGrowth(5, 6)
    assert egomap_growth(spec_json(V3), V3) is None                      # equal: nothing to grow
    assert egomap_growth(spec_json(V4), V3) is None                      # shrink
    assert egomap_growth(spec_json(V3), dataclasses.replace(V4, ray_channels=5)) is None
    assert egomap_growth(spec_json(V3), EgoMapObsSpec(**V2, egomap_shape=(6, 16, 32))) is None
    assert egomap_growth(spec_json(ObsSpec(**V2)), V4) is None           # no egomap in the source
    assert egomap_growth(spec_json(V3), ObsSpec(**V2)) is None


def test_obs_version_change_only_when_it_is_the_sole_upward_difference() -> None:
    j = CK.canonical_json
    assert obs_version_change(j(SIG3), j(SIG4)) == (3, 4)
    assert obs_version_change(j(SIG4), j(SIG3)) is None
    assert obs_version_change(j(SIG3), j(SIG3)) is None
    assert obs_version_change(j(SIG3), j((("frame_skip", 2), ("ray_bands", 4), ("obs_version", 4)))) is None
    assert obs_version_change(j(SIG3), j(SIG4 + (("extra", 1),))) is None


# ---------------------------------------------------------------- grown model
def test_grown_model_outputs_equal_the_v3_model_exactly() -> None:
    v3 = build_agent(V3, CFG, seed=3)
    v4 = grown_v4_from(v3)
    w = v4.state_dict()[FIRST_EGOMAP_CONV_KEY]
    assert w.shape[1] == 6 and torch.count_nonzero(w[:, 5:]) == 0
    assert torch.equal(w[:, :5], v3.state_dict()[FIRST_EGOMAP_CONV_KEY])
    for k, v in v3.state_dict().items():
        if k != FIRST_EGOMAP_CONV_KEY:
            assert torch.equal(v, v4.state_dict()[k]), k
    for seed in (0, 1):
        obs3 = random_obs(V3, (T, B), seed)
        a, b = outputs(v3, obs3), outputs(v4, extend_egomap(obs3, 1, seed + 100))
        for x, y in zip(a, b):
            diff = (x - y).abs().max().item()
            assert torch.equal(x, y), f"grown output differs by up to {diff}"


def test_gradients_reach_the_new_slices() -> None:
    v4 = grown_v4_from(build_agent(V3, CFG, seed=3))
    obs = extend_egomap(random_obs(V3, (T, B), 4), 1, 5)
    obs[EGOMAP_KEY][..., 5, :, :] = 1                # the new channel carries signal
    starts = torch.zeros(T, B)
    starts[0] = 1
    out, _ = v4.forward_sequence(obs, v4.initial_state(B), starts)
    (sum(l.sum() for l in out.logits) + out.value_ext.sum()).backward()
    grad = v4.egomap.conv[0].weight.grad
    assert grad is not None and torch.count_nonzero(grad[:, 5:]) > 0


def test_grow_state_is_deterministic_and_does_not_touch_the_source() -> None:
    v3 = build_agent(V3, CFG, seed=3)
    before = {k: v.clone() for k, v in v3.state_dict().items()}
    shapes = state_shapes(build_agent(V4, CFG, seed=0))
    s1 = grow_state(v3.state_dict(), EgomapGrowth(5, 6), shapes)
    s2 = grow_state(v3.state_dict(), EgomapGrowth(5, 6), shapes)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert all(torch.equal(before[k], v) for k, v in v3.state_dict().items())


# ---------------------------------------------------------------- --init-from
def stub_trainer(spec: ObsSpec, seed: int = 21) -> SimpleNamespace:
    model = build_agent(spec, CFG, seed=seed)
    return SimpleNamespace(model=model, learner=model, spec=spec, sync_actor=lambda: None)


def source(model: A0Agent, spec: ObsSpec, sig: tuple, change: tuple[int, int] | None) -> InitSource:
    blob = {"model_config": CFG, "obs_spec_json": spec_json(spec), "sensor_signature_json": CK.canonical_json(sig),
            "trainer": {"model": model.state_dict()}}
    return InitSource("x.pt", "x.pt", "0" * 64, "weights_only", blob, change)


def save_file(path: str, model: A0Agent, spec: ObsSpec, sig: tuple) -> str:
    CK.save_weights_checkpoint(path, model, CFG, spec, RunConfig(), sig, {"kind": "behaviour_cloning",
                                                                          "val_map_ids": ["DD0001"]})
    return path


def test_apply_init_grows_and_records(tmp_path: "os.PathLike[str]") -> None:
    v3 = build_agent(V3, CFG, seed=3)
    path = save_file(os.path.join(str(tmp_path), "v3.pt"), v3, V3, SIG3)
    src = read_init_source(path, SIG4)
    assert src.obs_version_change == (3, 4)
    tr = stub_trainer(V4)
    growth = apply_init(tr, src)
    assert growth == EgomapGrowth(5, 6)
    rec = growth_record(src, growth)
    assert rec["old_channels"] == 5 and rec["new_channels"] == 6 and rec["obs_version_to"] == 4
    json.dumps(rec)
    obs3 = random_obs(V3, (T, B), 9)
    for x, y in zip(outputs(v3, obs3), outputs(tr.learner, extend_egomap(obs3, 1, 10))):
        assert torch.equal(x, y)
    assert "init_from_growth" in {f.name for f in dataclasses.fields(RunConfig)}


def test_apply_init_strict_path_unchanged_for_equal_specs() -> None:
    v3 = build_agent(V3, CFG, seed=3)
    tr = stub_trainer(V3)
    assert apply_init(tr, source(v3, V3, SIG3, None)) is None
    for k, v in v3.state_dict().items():
        assert torch.equal(v, tr.learner.state_dict()[k])


def test_apply_init_refuses_every_other_difference(tmp_path: "os.PathLike[str]") -> None:
    v3 = build_agent(V3, CFG, seed=3)
    other = EgoMapObsSpec(**{**V2, "ray_channels": 5}, egomap_shape=(6, *HW))
    with pytest.raises(CK.CheckpointMismatch, match="observation spec differs"):
        apply_init(stub_trainer(other), source(v3, V3, SIG3, (3, 4)))
    v4 = build_agent(V4, CFG, seed=3)
    with pytest.raises(CK.CheckpointMismatch, match="observation spec differs"):   # shrink
        apply_init(stub_trainer(V3), source(v4, V4, SIG4, None))
    with pytest.raises(CK.CheckpointMismatch, match="obs_version differs"):        # version without growth
        apply_init(stub_trainer(V3), source(v3, V3, SIG3, (3, 4)))
    path = save_file(os.path.join(str(tmp_path), "v3.pt"), v3, V3, SIG3)
    with pytest.raises(CK.CheckpointMismatch, match="sensor"):
        read_init_source(path, (("frame_skip", 2), ("ray_bands", 4), ("obs_version", 4)))
    with pytest.raises(CK.CheckpointMismatch, match="sensor"):                     # downgrade
        read_init_source(save_file(os.path.join(str(tmp_path), "v4.pt"), v4, V4, SIG4), SIG3)


def test_growth_record_reaches_the_checkpoint_run_config(tmp_path: "os.PathLike[str]") -> None:
    """The way train.run_doom records it: replace trainer.run, then every checkpoint
    the run writes carries init_from_growth in run_config_json."""
    from a0_helpers import FakeDoomEnv
    from configs.train_config import PPOConfig
    from train.trainer import Trainer
    from train.vec_env import SyncVecEnv
    venv = SyncVecEnv([lambda: FakeDoomEnv(7) for _ in range(2)])
    run = dataclasses.replace(RunConfig(), num_workers=1, envs_per_worker=2,
                              ppo=PPOConfig(rollout_len=4, num_minibatches=1, epochs=1))
    tr = Trainer(venv, CFG, run)
    grown_json = CK.canonical_json(growth_record(source(build_agent(V3, CFG, seed=3), V3, SIG3, (3, 4)),
                                                 EgomapGrowth(5, 6)))
    tr.run = dataclasses.replace(tr.run, init_from_growth=grown_json)
    path = os.path.join(str(tmp_path), "checkpoint.pt")
    CK.save_checkpoint(path, tr, SIG4)
    rc = json.loads(CK.load_checkpoint(path)["run_config_json"])
    assert json.loads(rc["init_from_growth"]) == {"kind": "egomap_channel_growth", "key": FIRST_EGOMAP_CONV_KEY,
                                                  "old_channels": 5, "new_channels": 6, "new_slices_init": "zeros",
                                                  "obs_version_from": 3, "obs_version_to": 4}
    venv.close()
    from train import run_doom
    ns = run_doom.parse_args(["--init-from", "x.pt"])
    run_doom.write_run_start(str(tmp_path), dataclasses.replace(run, init_from_growth=grown_json), ns, [])
    line = open(os.path.join(str(tmp_path), run_doom.RUN_START_NAME), encoding="utf-8").read().splitlines()[-1]
    assert json.loads(line)["init_from_growth"] == grown_json


# ---------------------------------------------------------------- --init-from-bc
def test_read_init_bc_grows_and_refuses_other_differences(tmp_path: "os.PathLike[str]") -> None:
    v3 = build_agent(V3, CFG, seed=3)
    path = save_file(os.path.join(str(tmp_path), "bc_model.pt"), v3, V3, SIG3)
    state, rec = BC.read_init_bc(path, V4, CFG, SIG4)
    assert rec["val_map_ids"] == ["DD0001"]
    g = rec[BC.INIT_GROWTH_KEY]
    assert (g["old_channels"], g["new_channels"], g["obs_version_from"], g["obs_version_to"]) == (5, 6, 3, 4)
    model = build_agent(V4, CFG, seed=0)
    model.load_state_dict(state, strict=True)
    obs3 = random_obs(V3, (T, B), 12)
    for x, y in zip(outputs(v3, obs3), outputs(model, extend_egomap(obs3, 1, 13))):
        assert torch.equal(x, y)
    same_state, same_rec = BC.read_init_bc(path, V3, CFG, SIG3)                    # strict path
    assert BC.INIT_GROWTH_KEY not in same_rec and all(torch.equal(same_state[k], v)
                                                      for k, v in v3.state_dict().items())
    with pytest.raises(CK.CheckpointMismatch, match="observation spec"):
        BC.read_init_bc(path, dataclasses.replace(V4, ray_channels=5), CFG, SIG4)
    with pytest.raises(CK.CheckpointMismatch, match="sensor signature"):
        BC.read_init_bc(path, V4, CFG, (("frame_skip", 2), ("ray_bands", 4), ("obs_version", 4)))
    with pytest.raises(CK.CheckpointMismatch, match="obs_version differs"):
        BC.read_init_bc(path, V3, CFG, SIG4)
