"""Tests for evaluate/ (inference entry point). One engine instance at a time; the
engine tests use my_way_home only (never a held-out map)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from gymnasium import spaces

from a0_helpers import doom_like_spaces, random_obs  # noqa: F401  (also caps torch threads)
from agent.model import build_agent
from agent.obs_spec import A0_KEYS, ObsSpec, spec_from_spaces
from configs.a0_env import GENERATED_MAPS, a0_eval_env_config, a0_train_env_config, generated_maps_env_config
from configs.model_config import ModelConfig
from env import observation as O
from env.actions import head_sizes
from evaluate import map_jobs
from evaluate.checkpoint_io import LoadedCheckpoint, load_model
from evaluate.config import LOADER_MODE, EvalRunConfig
from evaluate.episode import run_episode
from evaluate.policies import NetworkPolicy, RandomPolicy
from evaluate.run import evaluate, verify_configs
from evaluate.sensors import (SensorSignatureError, canonical_json, check_sensors,
                              trained_config_from_signature)
from maps import loader as L

EVAL_SIG = canonical_json(a0_eval_env_config().sensor_signature())


# ---------------------------------------------------------------- sensors / vanilla
def test_matching_signature_passes() -> None:
    trained = trained_config_from_signature(EVAL_SIG, a0_eval_env_config())
    check_sensors(trained, a0_eval_env_config())


def test_missing_signature_refused() -> None:
    with pytest.raises(SensorSignatureError):
        trained_config_from_signature(None, a0_eval_env_config())


def test_screen_on_checkpoint_refused_against_a0_eval() -> None:
    sig = canonical_json(a0_eval_env_config().replace(screen_enabled=True).sensor_signature())
    trained = trained_config_from_signature(sig, a0_eval_env_config())
    with pytest.raises(AssertionError, match="screen_enabled"):
        check_sensors(trained, a0_eval_env_config())


def test_tampered_or_unknown_signature_refused() -> None:
    pairs = json.loads(EVAL_SIG)
    bad_bands = [[k, [[0, 1], [1, 2], [2, 3]]] if k == "ray_bands" else [k, v] for k, v in pairs]
    with pytest.raises(SensorSignatureError):
        trained_config_from_signature(json.dumps(bad_bands, sort_keys=True), a0_eval_env_config())
    with pytest.raises(SensorSignatureError):
        trained_config_from_signature(json.dumps(pairs + [["fov", 90]]), a0_eval_env_config())
    with pytest.raises(SensorSignatureError):
        trained_config_from_signature(json.dumps(pairs[1:]), a0_eval_env_config())


@pytest.mark.parametrize("change", [
    {"scenario_wad": a0_train_env_config().scenario_wad},
    {"game_args": ("+sv_cheats 1",)},
    {"vanilla_eval": False},
    {"objects_info": True},
    {"engine_rewards": (("living_reward", -0.0001),)},
])
def test_non_vanilla_eval_config_refused(change: dict) -> None:
    trained = trained_config_from_signature(EVAL_SIG, a0_eval_env_config())
    with pytest.raises(AssertionError):
        check_sensors(trained, a0_eval_env_config().replace(**change))


def test_checkpoint_without_signature_refused(tmp_path) -> None:
    p = tmp_path / "nosig.pt"
    torch.save({"trainer": {"model": {}}, "model_config_json": "{}", "obs_spec_json": "{}"}, str(p))
    with pytest.raises(Exception, match="sensor signature"):
        load_model(str(p))


# ---------------------------------------------------------------- map plans / held-out
def _heldout_entry() -> dict:
    return next(e for _, e in L.load_splits().entries("heldout"))


def test_heldout_refused_on_every_training_path() -> None:
    e = _heldout_entry()
    for mode in ("train", "val"):
        with pytest.raises(L.HeldOutAccessError):
            L.resolve(e["id"], mode)
    path = os.path.join(L.REPO_ROOT, e["file"])
    train_cfg = a0_train_env_config().replace(scenario_wad=None, map_wad=path, maps=(e["map"],))
    with pytest.raises(ValueError, match="heldout_maps"):
        train_cfg.validate()
    with pytest.raises(ValueError, match="heldout_maps"):
        generated_maps_env_config(a0_train_env_config(), path, (e["map"],))
    assert LOADER_MODE["heldout"] == "eval" and LOADER_MODE["val"] == "val"


def test_val_plan_never_contains_heldout_and_is_staged_outside_repo() -> None:
    splits = L.load_splits()
    plan = map_jobs.plan_maps(EvalRunConfig(split="val", max_maps=2), splits)
    try:
        heldout_ids = {e["id"] for _, e in splits.entries("heldout")}
        heldout_hashes = splits.heldout_hashes()
        assert len(plan.jobs) == 2
        for j in plan.jobs:
            assert j.map_id not in heldout_ids and j.wad_sha256 not in heldout_hashes
            assert j.staged and not os.path.normcase(os.path.realpath(j.env_config.map_wad)).startswith(
                os.path.normcase(os.path.realpath(L.REPO_ROOT)))
            assert L.sha256_file(j.env_config.map_wad) == j.wad_sha256
            j.env_config.assert_vanilla()
            assert j.env_config.skill == GENERATED_MAPS.skill
            assert j.env_config.episode_timeout_tics == GENERATED_MAPS.episode_timeout_tics
    finally:
        plan.cleanup()
    assert not os.path.isdir(plan.staging_dir)


def test_staging_refuses_train_hash(tmp_path) -> None:
    splits = L.load_splits()
    e = next(e for _, e in splits.entries("train"))
    with pytest.raises(L.HeldOutAccessError):
        map_jobs._stage_copy(os.path.join(L.REPO_ROOT, e["file"]), e["sha256"], str(tmp_path),
                             map_jobs._train_hashes(splits))


# ---------------------------------------------------------------- policy inputs
class _FakeEnv:
    """Gym-like env whose info carries privileged fields (map id, pose)."""

    def __init__(self, length: int) -> None:
        self.observation_space, self.action_space = doom_like_spaces()
        self.length, self.t = length, 0
        self.rng = np.random.default_rng(0)

    def _obs(self) -> dict:
        return {k: v[0] for k, v in random_obs(self.observation_space, 1, self.rng).items()}

    def _info(self) -> dict:
        return {"map": "S0-val-SECRET", "pos_x": 123.0, "episode_seed": 7, "health": 100.0}

    def reset(self, seed: int):
        self.t = 0
        return self._obs(), self._info()

    def step(self, action):
        self.t += 1
        done = self.t >= self.length
        info = self._info() | ({"timeout": True} if done else {})
        return self._obs(), 0.0, False, done, info


def _fresh_model(seed: int = 0):
    obs_space, act_space = doom_like_spaces()
    return build_agent(spec_from_spaces(obs_space, act_space), ModelConfig(), seed)


def test_map_ids_never_reach_model_inputs() -> None:
    model = _fresh_model()
    seen: list[dict] = []
    orig = model.step

    def spy(obs, h, start):
        seen.append(obs)
        return orig(obs, h, start)

    model.step = spy  # type: ignore[method-assign]
    env = _FakeEnv(length=5)
    rec = run_episode(env, NetworkPolicy(model, (False,) * 6), seed=3, exit_reward_threshold=1.0)
    assert len(seen) == 5 and rec["length"] == 5
    for obs in seen:
        assert set(obs) == set(A0_KEYS)
        for k, v in obs.items():
            assert isinstance(v, torch.Tensor) and v.shape[0] == 1, f"{k} must be a batched tensor"
            assert v.shape[1:] == env.observation_space[k].shape
    assert rec["map_lump"] == "S0-val-SECRET"          # logged ...
    assert not rec["success"]                           # ... and timeouts are not successes


def test_network_policy_deterministic_and_fresh_per_episode() -> None:
    model = _fresh_model()
    env = _FakeEnv(length=4)
    obs, _ = env.reset(0)
    p = NetworkPolicy(model, (False,) * 6)
    p.begin_episode(11)
    a1 = [p.act(obs, i == 0) for i in range(4)]
    p.begin_episode(11)
    a2 = [p.act(obs, i == 0) for i in range(4)]
    assert all(np.array_equal(x, y) for x, y in zip(a1, a2))
    assert a1[0].shape == (6,)


def test_random_policy_seeded_and_in_range() -> None:
    space = spaces.MultiDiscrete(head_sizes(7))
    p, q = RandomPolicy(space), RandomPolicy(space)
    p.begin_episode(5)
    q.begin_episode(5)
    a = [p.act({}, False) for _ in range(20)]
    b = [q.act({}, False) for _ in range(20)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert all(((x >= 0) & (x < space.nvec)).all() for x in a)


def test_seed_rule_is_arithmetic() -> None:
    r = EvalRunConfig(split="val", base_seed=100, seed_stride_map=1000)
    assert r.episode_seed(2, 3) == 2103


# ---------------------------------------------------------------- engine, end to end
def _fresh_loaded() -> LoadedCheckpoint:
    c = a0_eval_env_config()
    from env import egomap as E   # observation v3: the model consumes `egomap` (agent.obs_spec.EgoMapObsSpec)
    extra = {"egomap_shape": tuple(E.EGOMAP_SHAPE)} if c.egomap else {}
    spec = ObsSpec(**extra, ray_channels=O.ray_channels(len(c.ray_bands), c.obs_version), ray_width=O.N_RAY_COLS,
                   max_entities=c.max_entities, entity_feat_dim=8, n_entity_types=len(O.ENTITY_TYPES),
                   self_dim=O.self_dim(c.obs_version),
                   modality_dim=3, head_sizes=head_sizes(len(c.turn_bins)))
    model = build_agent(spec, ModelConfig(), seed=0).eval()
    return LoadedCheckpoint(model=model, spec=spec, signature_json=EVAL_SIG, loaded_from="fresh",
                            sha256="none", obs_spec_json="{}", model_config={}, run_config_json=None)


def test_engine_episode_end_to_end_fresh_model(tmp_path) -> None:
    run = EvalRunConfig(split="my_way_home", episodes_per_map=2, timeout_tics_override=80)
    loaded = _fresh_loaded()
    s1 = evaluate(run, str(tmp_path / "a"), loaded)
    rows1 = (tmp_path / "a" / "results.jsonl").read_text().splitlines()
    assert len(rows1) == 2 and s1["summary"]["overall"]["episodes"] == 2
    r = json.loads(rows1[0])
    for k in ("map_id", "stage", "success", "return", "length", "death", "timeout", "kills",
              "keys_picked", "final_health"):
        assert k in r
    assert r["length"] >= 1
    meta = json.loads((tmp_path / "a" / "summary.json").read_text())["meta"]
    assert meta["versions"]["torch"] == torch.__version__ and meta["engine_instances"] == 1


def test_verify_configs_refuses_screen_checkpoint() -> None:
    loaded = _fresh_loaded()
    loaded.signature_json = canonical_json(a0_eval_env_config().replace(screen_enabled=True).sensor_signature())
    plan = map_jobs.plan_maps(EvalRunConfig(split="my_way_home"))
    with pytest.raises(AssertionError):
        verify_configs(plan, loaded)
