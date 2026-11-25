"""Observation v3 `egomap` model input (agent/encoders.EgoMapEncoder, agent/obs_spec,
agent/tensorize, train/bc_data + bc_aux): shapes, determinism, hidden-state reset, and
that a v1/v2 spec builds EXACTLY the earlier model (golden captured from the code
before the egomap change: tests/fixtures/a0_v2_golden.torch). No engine is started."""
from __future__ import annotations

import dataclasses
import json
import os
import pickle

import numpy as np
import pytest
import torch

from a0_helpers import ROOT, doom_like_spaces, random_obs
from agent.model import build_agent, count_parameters
from agent.obs_spec import EGOMAP_KEY, EgoMapObsSpec, ObsSpec, model_keys, spec_from_spaces
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig
from gymnasium import spaces

CFG = ModelConfig()
EGO = (5, 32, 32)   # contract 2026-11-26: 4 occupancy channels + key seen
V2 = dict(ray_channels=4, ray_width=80, max_entities=24, entity_feat_dim=8, n_entity_types=14,
          self_dim=34, modality_dim=3, head_sizes=(3, 3, 7, 2, 2, 2))
# Literal obs_spec_json of a real v2 BC checkpoint (reports/eval/2026-11-26-val/bc-v2-usew8-full/summary.json).
V2_JSON = ('{"entity_feat_dim": 8, "head_sizes": [3, 3, 7, 2, 2, 2], "max_entities": 24, "modality_dim": 3, '
           '"n_entity_types": 14, "ray_channels": 4, "ray_width": 80, "self_dim": 34}')
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "a0_v2_golden.torch")
REAL_V2_CKPT = os.path.join(ROOT, "reports", "runs", "20261125-bc-v2-usew8", "bc_model.pt")


def canonical(spec: ObsSpec) -> str:
    return json.dumps(dataclasses.asdict(spec), sort_keys=True)   # == train.checkpoint.canonical_json


def v3_spec() -> EgoMapObsSpec:
    return EgoMapObsSpec(**V2, egomap_shape=EGO)


def v3_spaces() -> tuple[spaces.Dict, spaces.MultiDiscrete]:
    obs_space, act = doom_like_spaces()
    return spaces.Dict({**obs_space.spaces, EGOMAP_KEY: spaces.Box(0, 1, EGO, np.uint8)}), act


def v3_obs(space: spaces.Dict, b: int, rng: np.random.Generator, lead: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    o = random_obs(space, b, rng, lead=lead)
    o[EGOMAP_KEY] = (rng.random((*lead, b, *EGO)) < 0.3).astype(np.uint8)
    return o


# ---------------------------------------------------------------- v2 unchanged
def test_v2_spec_json_is_byte_identical_to_old_checkpoints() -> None:
    spec = ObsSpec(**V2)
    assert type(spec) is ObsSpec and spec.egomap is None
    assert canonical(spec) == V2_JSON
    assert model_keys(spec) == ("rays", "entities", "entity_type", "entity_mask", "self", "prev_action", "modality")


def test_v2_model_matches_pre_egomap_golden() -> None:
    g = torch.load(GOLDEN, weights_only=False)
    model = build_agent(ObsSpec(**V2), CFG, seed=g["seed"])
    assert model.egomap is None
    assert [(k, tuple(v.shape)) for k, v in model.state_dict().items()] == [tuple(x) for x in g["keys_shapes"]]
    assert count_parameters(model) == g["n_params"]
    with torch.no_grad():
        out, h = model.forward_sequence(g["obs"], model.initial_state(g["h"].shape[0]), g["starts"])
    for a, b in zip(out.logits, g["logits"]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
    torch.testing.assert_close(out.value_ext, g["value_ext"], atol=1e-6, rtol=0)
    torch.testing.assert_close(h, g["h"], atol=1e-6, rtol=0)


@pytest.mark.skipif(not os.path.exists(REAL_V2_CKPT), reason="real v2 BC checkpoint not present")
def test_real_v2_checkpoint_still_loads_strictly() -> None:
    from evaluate.checkpoint_io import load_model
    loaded = load_model(REAL_V2_CKPT)
    assert type(loaded.spec) is ObsSpec and loaded.model.egomap is None
    assert canonical(loaded.spec) == loaded.obs_spec_json


# ---------------------------------------------------------------- v3 spec plumbing
def test_v3_spec_round_trips_through_checkpoint_json_loaders() -> None:
    spec = v3_spec()
    d = json.loads(canonical(spec))
    assert d["egomap_shape"] == list(EGO)
    d["head_sizes"] = tuple(d["head_sizes"])
    back = ObsSpec(**d)                       # what evaluate/checkpoint_io and teachers/collect do
    assert type(back) is EgoMapObsSpec and back == spec and back.egomap == EGO
    from evaluate.checkpoint_io import spec_from_json
    assert spec_from_json(canonical(spec)) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert dataclasses.replace(spec, self_dim=28).egomap == EGO
    assert spec != ObsSpec(**V2) and ObsSpec(**V2) != spec
    assert model_keys(spec)[-1] == EGOMAP_KEY
    with pytest.raises(ValueError):
        EgoMapObsSpec(**V2, egomap_shape=(32, 32))


def test_spec_from_spaces_detects_egomap() -> None:
    assert type(spec_from_spaces(*doom_like_spaces())) is ObsSpec
    spec = spec_from_spaces(*v3_spaces())
    assert isinstance(spec, EgoMapObsSpec) and spec.egomap == EGO


def test_to_tensors_keeps_v2_keys_and_egomap_uint8() -> None:
    space, _ = doom_like_spaces()
    t2 = to_tensors(random_obs(space, 2, np.random.default_rng(0)))
    assert EGOMAP_KEY not in t2
    t3 = to_tensors(v3_obs(v3_spaces()[0], 2, np.random.default_rng(0)))
    assert t3[EGOMAP_KEY].dtype == torch.uint8 and t3[EGOMAP_KEY].shape == (2, *EGO)


# ---------------------------------------------------------------- v3 model
def test_v3_model_shapes_and_parameter_count() -> None:
    space, act = v3_spaces()
    spec = spec_from_spaces(space, act)
    model = build_agent(spec, CFG, seed=0)
    v2 = build_agent(spec_from_spaces(*doom_like_spaces()), CFG, seed=0)
    enc = model.egomap
    assert enc is not None and enc.out_dim == CFG.egomap_dim
    n_enc, n_v3, n_v2 = count_parameters(enc), count_parameters(model), count_parameters(v2)
    # conv C->16, 16->32, 32->32 (k3), linear 32*4*4 -> 128, plus 128 extra fusion inputs
    expect_enc = (EGO[0] * 16 * 9 + 16) + (16 * 32 * 9 + 32) + (32 * 32 * 9 + 32) + (512 * 128 + 128)
    assert n_enc == expect_enc
    assert n_v3 - n_v2 == expect_enc + CFG.egomap_dim * CFG.fusion_dim
    print(f"egomap encoder {n_enc} params; v3 model {n_v3} (v2-shaped {n_v2}, +{n_v3 - n_v2})")
    rng = np.random.default_rng(1)
    b, t = 5, 6
    obs = to_tensors(v3_obs(space, b, rng))
    assert enc(obs[EGOMAP_KEY]).shape == (b, CFG.egomap_dim)
    out, h = model.step(obs, model.initial_state(b), torch.zeros(b))
    assert [lg.shape for lg in out.logits] == [(b, n) for n in spec.head_sizes] and h.shape == (b, CFG.core_hidden)
    seq = to_tensors(v3_obs(space, b, rng, lead=(t,)))
    out_s, h_s = model.forward_sequence(seq, model.initial_state(b), torch.zeros(t, b))
    assert [lg.shape for lg in out_s.logits] == [(t, b, n) for n in spec.head_sizes]
    assert out_s.value_ext.shape == (t, b) and h_s.shape == (b, CFG.core_hidden)


def test_v3_model_needs_egomap_and_uses_it() -> None:
    space, act = v3_spaces()
    model = build_agent(spec_from_spaces(space, act), CFG, seed=0)
    obs = to_tensors(v3_obs(space, 3, np.random.default_rng(2)))
    with pytest.raises(AssertionError, match="egomap"):
        model.step({k: v for k, v in obs.items() if k != EGOMAP_KEY}, model.initial_state(3), torch.zeros(3))
    with pytest.raises(AssertionError, match="egomap expected"):
        model.egomap(obs[EGOMAP_KEY][:, :2])
    a, _ = model.step(obs, model.initial_state(3), torch.zeros(3))
    flipped = dict(obs, egomap=1 - obs[EGOMAP_KEY])
    b, _ = model.step(flipped, model.initial_state(3), torch.zeros(3))
    assert not torch.allclose(a.logits[0], b.logits[0]), "the egomap must reach the policy"


def test_v3_build_deterministic_per_seed_and_orthogonal_init() -> None:
    space, act = v3_spaces()
    spec = spec_from_spaces(space, act)
    m1, m2, m3 = build_agent(spec, CFG, seed=4), build_agent(spec, CFG, seed=4), build_agent(spec, CFG, seed=5)
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p, q)
    assert not all(torch.equal(p, q) for p, q in zip(m1.parameters(), m3.parameters()))
    orth = build_agent(spec, dataclasses.replace(CFG, init_scheme="orthogonal"), seed=4)
    conv = orth.egomap.conv[0]
    assert torch.count_nonzero(conv.bias) == 0          # orthogonal scheme zeroes hidden biases, Conv2d included
    obs = to_tensors(v3_obs(space, 2, np.random.default_rng(6)))
    o1, _ = m1.step(obs, m1.initial_state(2), torch.zeros(2))
    o2, _ = m2.step(obs, m2.initial_state(2), torch.zeros(2))
    torch.testing.assert_close(o1.logits[2], o2.logits[2], rtol=0, atol=0)


def test_v3_hidden_reset_only_for_ended_envs() -> None:
    space, act = v3_spaces()
    model = build_agent(spec_from_spaces(space, act), CFG, seed=0)
    b = 4
    obs = to_tensors(v3_obs(space, b, np.random.default_rng(3)))
    h_prev = torch.randn(b, CFG.core_hidden, generator=torch.Generator().manual_seed(1))
    starts = torch.tensor([0.0, 1.0, 0.0, 1.0])
    _, h_mixed = model.step(obs, h_prev, starts)
    _, h_keep = model.step(obs, h_prev, torch.zeros(b))
    _, h_fresh = model.step(obs, model.initial_state(b), torch.zeros(b))
    for i in (0, 2):
        torch.testing.assert_close(h_mixed[i], h_keep[i])
        assert not torch.allclose(h_mixed[i], h_fresh[i])
    for i in (1, 3):
        torch.testing.assert_close(h_mixed[i], h_fresh[i])


# ---------------------------------------------------------------- BC data with egomap
def _v3_bc_spec() -> EgoMapObsSpec:
    return EgoMapObsSpec(**dataclasses.asdict(spec_from_spaces(*doom_like_spaces())), egomap_shape=EGO)


def _write_v3_dataset(root: str, n_eps: int, length: int, with_egomap: bool = True) -> str:
    from test_bc import make_episode_arrays
    rng = np.random.default_rng(0)
    os.makedirs(root, exist_ok=True)
    rows = []
    for k in range(n_eps):
        a = make_episode_arrays(length, rng)
        if with_egomap:
            a[EGOMAP_KEY] = (rng.random((length, *EGO)) < 0.3).astype(np.uint8)
        name = f"ep_{k:06d}.npz"
        np.savez_compressed(os.path.join(root, name), **a)
        rows.append({"file": name, "map_id": f"DD{k:04d}", "stage": "S0", "length": length,
                     "success": True, "seed": k, "noise": 0.0})
    with open(os.path.join(root, "index.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return root


def test_bc_data_loads_egomap_and_counts_its_memory(tmp_path: "os.PathLike[str]") -> None:
    from train.bc_data import ChunkStream, load_episodes, step_bytes
    from train.bc_spec import DatasetMismatch
    spec, spec_v2 = _v3_bc_spec(), spec_from_spaces(*doom_like_spaces())
    assert step_bytes(spec) - step_bytes(spec_v2) == int(np.prod(EGO)) // 8 == 640   # bit-packed in RAM
    assert step_bytes(spec, pack=False) - step_bytes(spec_v2) == int(np.prod(EGO))     # 5120 unpacked
    root = _write_v3_dataset(os.path.join(str(tmp_path), "v3"), 3, 11)
    eps = load_episodes(root, spec, 1 << 30)
    assert all(e.obs[EGOMAP_KEY].shape == (11, 640) and e.packed == {EGOMAP_KEY: EGO} for e in eps)
    assert all(e.rows(EGOMAP_KEY, np.arange(11)).shape == (11, *EGO) for e in eps)
    assert EGOMAP_KEY not in load_episodes(root, spec_v2, 1 << 30)[0].obs          # v2 spec ignores it
    with pytest.raises(MemoryError):
        load_episodes(root, spec, step_bytes(spec) * 33 - 1)
    chunk = next(iter(ChunkStream(eps, 4, 2, torch.Generator().manual_seed(0))))
    assert chunk.obs[EGOMAP_KEY].dtype == torch.uint8 and chunk.obs[EGOMAP_KEY].shape == (4, 2, *EGO)
    model = build_agent(spec, ModelConfig(), seed=0)
    out, _ = model.forward_sequence(chunk.obs, model.initial_state(2), chunk.episode_start)
    assert out.logits[0].shape == (4, 2, 3)
    bad = _write_v3_dataset(os.path.join(str(tmp_path), "v2only"), 1, 5, with_egomap=False)
    with pytest.raises(DatasetMismatch, match="egomap"):
        load_episodes(bad, spec, 1 << 30)


def test_bc_aux_sampler_carries_egomap(tmp_path: "os.PathLike[str]") -> None:
    from train.bc_aux import BCSampler, bc_term, load_bc_data
    spec = _v3_bc_spec()
    root = _write_v3_dataset(os.path.join(str(tmp_path), "v3"), 2, 9)
    eps = load_bc_data((root,), spec, 1 << 30)
    batch = BCSampler(eps, 4, 3, seed=0).sample()
    assert batch.obs[EGOMAP_KEY].shape == (4, 3, *EGO) and batch.burn_obs[EGOMAP_KEY].dtype == torch.uint8
    loss, stats = bc_term(build_agent(spec, ModelConfig(), seed=0), batch)
    assert torch.isfinite(loss) and stats["bc_valid_steps"] > 0


# ---------------------------------------------------------------- bit-packed egomap in RAM
def test_pack_unpack_round_trip_is_exact() -> None:
    from train.bc_data import pack_steps, packed_step_bytes, unpack_steps
    from train.bc_spec import DatasetMismatch
    rng = np.random.default_rng(0)
    for shape in (EGO, (3, 5, 7)):                       # 5120 bits, and a count not divisible by 8
        a = (rng.random((13, *shape)) < 0.4).astype(np.uint8)
        p = pack_steps(a)
        assert p.shape == (13, packed_step_bytes(shape)) and p.dtype == np.uint8
        back = unpack_steps(p, shape)
        assert back.dtype == np.uint8 and np.array_equal(back, a)
        idx = np.array([4, 0, 12, 4])
        assert np.array_equal(unpack_steps(p[idx], shape), a[idx])       # row indexing commutes with packing
    assert packed_step_bytes(EGO) == 640
    with pytest.raises(DatasetMismatch, match="0/1"):
        pack_steps(np.full((2, *EGO), 2, np.uint8))


def test_bc_epoch_same_loss_packed_and_unpacked(tmp_path: "os.PathLike[str]") -> None:
    from train import bc as BC
    from train.bc_aux import BCSampler, bc_term
    from train.bc_data import ChunkStream, load_episodes
    spec = _v3_bc_spec()
    root = _write_v3_dataset(os.path.join(str(tmp_path), "v3"), 4, 13)
    cfg = ModelConfig(core_hidden=32, fusion_dim=32, ray_dim=16, entity_token_dim=16, entity_mlp_hidden=16,
                      self_dim=16, self_hidden=16, egomap_dim=16)

    def epoch(pack: bool) -> tuple[float, dict]:
        eps = load_episodes(root, spec, 1 << 30, pack=pack)
        assert bool(eps[0].packed) == pack
        model = build_agent(spec, cfg, seed=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        acc, _ = BC.run_epoch(model, ChunkStream(eps, 5, 2, torch.Generator().manual_seed(0)), (1.0,) * 6,
                              opt, 1.0, prev_action_dropout=0.3, dropout_generator=torch.Generator().manual_seed(1))
        aux, _ = bc_term(model, BCSampler(eps, 4, 3, seed=0).sample())
        return acc.weighted_loss_sum + float(aux.detach()), model.state_dict()
    (lp, sp), (lu, su) = epoch(True), epoch(False)
    assert lp == lu
    for k in sp:
        assert torch.equal(sp[k], su[k]), k
