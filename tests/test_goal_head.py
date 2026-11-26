"""Sub-goal prediction head (agent/model.py goal_head, train/goal_head.py): off = the
model of before (pinned by a golden captured from the pre-change code), on = an
auxiliary output that never changes the policy; BC loss with masking and refusals;
head addition by --init-from-bc and --init-from. No engine is started."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from a0_helpers import FakeDoomEnv
from agent.model import GOAL_HEAD_NAME, build_agent
from agent.obs_spec import EgoMapObsSpec, ObsSpec
from configs.bc_config import BCConfig
from configs.goal_modes import N_GOAL_MODES
from configs.model_config import ModelConfig
from configs.train_config import BCAuxConfig, PPOConfig, RunConfig
from train import bc as BC
from train import checkpoint as CK
from train.bc_aux import BCDataRefused, BCSampler, bc_term, load_bc_data
from train.bc_data import ChunkStream, load_episodes
from train.bc_spec import a0_training_spec_and_signature
from train.goal_head import (GOAL_KEY_PREFIX, GOAL_LABEL_KEY, GOAL_UNLABELLED, GoalLabelError, goal_head_added,
                             goal_loss)
from train.init_from import InitSource, apply_init, sha256_file
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

from test_bc import SMALL, write_dataset

# Golden of the model BEFORE the goal head existed (captured 2026-11-26 from the
# unmodified agent/model.py, CPU torch 2.9.1): GOLDEN_SPEC, ModelConfig(), seed 0.
GOLDEN_SPEC = EgoMapObsSpec(ray_channels=4, ray_width=80, max_entities=24, entity_feat_dim=8, n_entity_types=14,
                            self_dim=34, modality_dim=3, head_sizes=(3, 3, 7, 2, 2, 2), egomap_shape=(5, 32, 32))
GOLDEN_N_KEYS, GOLDEN_N_PARAMS = 69, 579802
GOLDEN_KEYS_SHA = "8d47c60b59c4864943dabe05be6372f07790d7b3a8b60736cf98450d16125d48"
GOLDEN_PARAM_SHA = "0e27699c588ea6ab764948cbafda258f31c3e3d9439ca387866ccf54a8444a86"
GOLDEN_OUT_SHA = "c2187c66500ffd710ecc28398a0c0a3e7cffd3a19f78e8e6513cf248162e9458"
GOLDEN_OBS_SEED, GOLDEN_BATCH, GOLDEN_ROUND = 123, 4, 5
GOAL_DIMS = slice(8, 16)   # test_bc's REACT_DIMS: the per-step sign the goal is made from


@pytest.fixture(autouse=True, scope="module")
def _obs_v1():
    """test_bc's synthetic datasets are observation v1; pin the BC pipeline to v1."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DOOMSDAY_OBS_VERSION", "1")
        yield


def golden_obs(spec: ObsSpec, b: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand((b, spec.max_entities), generator=g) < 0.5
    o = {"rays": torch.rand((b, spec.ray_channels, spec.ray_width), generator=g),
         "entities": torch.rand((b, spec.max_entities, spec.entity_feat_dim), generator=g) * mask.unsqueeze(-1),
         "entity_type": torch.randint(1, spec.n_entity_types, (b, spec.max_entities), generator=g) * mask,
         "entity_mask": mask,
         "self": torch.rand((b, spec.self_dim), generator=g),
         "prev_action": torch.stack([torch.randint(0, n, (b,), generator=g) for n in spec.head_sizes], -1),
         "modality": torch.ones((b, spec.modality_dim))}
    if spec.egomap is not None:
        o["egomap"] = (torch.rand((b, *spec.egomap), generator=g) < 0.3).to(torch.uint8)
    return o


# ---------------------------------------------------------------- model
def test_head_off_is_the_model_of_before() -> None:
    m = build_agent(GOLDEN_SPEC, ModelConfig(), seed=0)
    assert m.goal_head is None and not ModelConfig().goal_head
    sd = m.state_dict()
    h = hashlib.sha256()
    for k in sd:
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    assert len(sd) == GOLDEN_N_KEYS and sum(p.numel() for p in m.parameters()) == GOLDEN_N_PARAMS
    assert hashlib.sha256("|".join(sd).encode()).hexdigest() == GOLDEN_KEYS_SHA
    assert h.hexdigest() == GOLDEN_PARAM_SHA
    with torch.no_grad():
        out, hn = m.step(golden_obs(GOLDEN_SPEC, GOLDEN_BATCH, GOLDEN_OBS_SEED), m.initial_state(GOLDEN_BATCH),
                         torch.ones(GOLDEN_BATCH))
    ho = hashlib.sha256()
    for t in [*out.logits, out.value_ext, out.value_int, hn]:
        ho.update(t.numpy().round(GOLDEN_ROUND).tobytes())
    assert ho.hexdigest() == GOLDEN_OUT_SHA
    assert out.goal_logits is None


def test_head_on_adds_only_the_head_and_leaves_policy_bit_identical() -> None:
    off = build_agent(GOLDEN_SPEC, ModelConfig(), seed=3)
    on = build_agent(GOLDEN_SPEC, ModelConfig(goal_head=True), seed=3)
    so, sn = off.state_dict(), on.state_dict()
    assert set(sn) - set(so) == {f"{GOAL_HEAD_NAME}.weight", f"{GOAL_HEAD_NAME}.bias"} and set(so) <= set(sn)
    for k in so:
        assert torch.equal(so[k], sn[k]), k
    b, t = 3, 5
    obs = golden_obs(GOLDEN_SPEC, b, 7)
    with torch.no_grad():
        o1, h1 = off.step(obs, off.initial_state(b), torch.ones(b))
        o2, h2 = on.step(obs, on.initial_state(b), torch.ones(b))
    for a, c in zip([*o1.logits, o1.value_ext, o1.value_int, h1], [*o2.logits, o2.value_ext, o2.value_int, h2]):
        assert torch.equal(a, c)
    assert o2.goal_logits is not None and o2.goal_logits.shape == (b, N_GOAL_MODES)
    seq = {k: torch.stack([v] * t) for k, v in obs.items()}
    starts = torch.zeros(t, b)
    starts[0] = 1.0
    with torch.no_grad():
        so_, _ = on.forward_sequence(seq, on.initial_state(b), starts)
    assert so_.goal_logits.shape == (t, b, N_GOAL_MODES)
    # determinism: same seed, same head
    again = build_agent(GOLDEN_SPEC, ModelConfig(goal_head=True), seed=3)
    assert torch.equal(again.goal_head.weight, on.goal_head.weight)


def test_hidden_state_reset_reaches_the_goal_head() -> None:
    """episode_start zeroes the state before the step, so the goal logits of a restarted
    env equal a fresh env's, while a continuing env's differ."""
    m = build_agent(GOLDEN_SPEC, ModelConfig(goal_head=True), seed=1)
    obs = golden_obs(GOLDEN_SPEC, 2, 9)
    with torch.no_grad():
        _, h = m.step(obs, m.initial_state(2), torch.ones(2))
        cont, _ = m.step(obs, h, torch.tensor([0.0, 1.0]))
        fresh, _ = m.step(obs, m.initial_state(2), torch.ones(2))
    assert torch.equal(cont.goal_logits[1], fresh.goal_logits[1])
    assert not torch.equal(cont.goal_logits[0], fresh.goal_logits[0])


def test_old_checkpoint_config_without_the_field_reads_as_head_off(tmp_path) -> None:
    spec, sig = a0_training_spec_and_signature()
    m = build_agent(spec, ModelConfig(), seed=0)
    path = os.path.join(str(tmp_path), "old.pt")
    CK.save_weights_checkpoint(path, m, ModelConfig(), spec, RunConfig(), sig, {})
    blob = torch.load(path, weights_only=True)
    cfg = json.loads(blob["model_config_json"])
    cfg.pop("goal_head")
    blob["model_config_json"] = CK.canonical_json(cfg)
    torch.save(blob, path)
    loaded = CK.load_checkpoint(path)
    assert loaded["model_config"] == ModelConfig()
    build_agent(spec, loaded["model_config"], seed=0).load_state_dict(loaded["trainer"]["model"], strict=True)


# ---------------------------------------------------------------- loss
def test_goal_loss_masks_padding_and_unlabelled_steps() -> None:
    t, b = 4, 3
    g = torch.Generator().manual_seed(0)
    goal = torch.randint(0, N_GOAL_MODES, (t, b), generator=g)
    goal[1, 0] = GOAL_UNLABELLED
    valid = torch.ones(t, b, dtype=torch.bool)
    valid[3, 2] = False
    lg = torch.randn(t, b, N_GOAL_MODES, generator=g, requires_grad=True)
    r = goal_loss(lg, goal, valid)
    assert r.n_labelled == t * b - 2
    r.loss.backward()
    assert torch.all(lg.grad[1, 0] == 0) and torch.all(lg.grad[3, 2] == 0)
    lg2 = lg.detach().clone()
    lg2[1, 0] += 100.0
    lg2[3, 2] -= 100.0
    assert torch.allclose(goal_loss(lg2, goal, valid).loss, r.loss.detach())
    none = goal_loss(lg, torch.full((t, b), GOAL_UNLABELLED), valid)
    assert none.n_labelled == 0 and float(none.loss.detach()) == 0.0


# ---------------------------------------------------------------- data
def add_goal_labels(root: str, unlabelled_every: int = 5) -> None:
    """goal_mode = 3 where the reactive sign is +, 0 where -, and GOAL_UNLABELLED on every
    `unlabelled_every`-th step: inferable from the current observation."""
    for name in sorted(os.listdir(root)):
        if not name.endswith(".npz"):
            continue
        p = os.path.join(root, name)
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        goal = np.where(arrays["self"][:, GOAL_DIMS.start] > 0, 3, 0).astype(np.int64)
        goal[::unlabelled_every] = GOAL_UNLABELLED
        np.savez_compressed(p, **arrays, **{GOAL_LABEL_KEY: goal})


def test_dataset_goal_labels_load_stream_and_refuse(tmp_path) -> None:
    spec, _ = a0_training_spec_and_signature()
    t = str(tmp_path)
    plain = write_dataset(os.path.join(t, "plain"), n_maps=2, eps_per_map=2, length=6)
    assert all(e.goal is None for e in load_episodes(plain, spec, 10**9))
    lab = write_dataset(os.path.join(t, "lab"), n_maps=2, eps_per_map=2, length=6)
    add_goal_labels(lab)
    eps = load_episodes(lab, spec, 10**9)
    assert all(e.goal is not None and e.goal.dtype == np.int64 and e.goal.shape == (6,) for e in eps)
    chunk = next(iter(ChunkStream(eps, 4, 3, torch.Generator().manual_seed(0))))
    assert chunk.goal is not None and chunk.goal.shape == (4, 3)
    assert next(iter(ChunkStream(load_episodes(plain, spec, 10**9), 4, 3,
                                 torch.Generator().manual_seed(0)))).goal is None
    # out-of-range id refused at load
    bad = write_dataset(os.path.join(t, "bad"), n_maps=1, eps_per_map=1, length=6)
    p = os.path.join(bad, "ep_000000.npz")
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    np.savez_compressed(p, **arrays, **{GOAL_LABEL_KEY: np.full(6, N_GOAL_MODES, np.int64)})
    with pytest.raises(GoalLabelError, match="N_GOAL_MODES"):
        load_episodes(bad, spec, 10**9)
    # coefficient > 0 refuses a dir without labels; 0 accepts it
    with pytest.raises(GoalLabelError, match=GOAL_LABEL_KEY):
        BC.prepare(plain, spec, BCConfig(goal_head_coef=1.0), log=lambda s: None)
    BC.prepare(plain, spec, BCConfig(), log=lambda s: None)
    with pytest.raises(GoalLabelError):
        BC.prepare([lab, plain], spec, BCConfig(goal_head_coef=1.0), log=lambda s: None)
    with pytest.raises(BCDataRefused, match=GOAL_LABEL_KEY):
        load_bc_data((lab, plain), spec, 10**9, goal_required=True)
    assert len(load_bc_data((lab, plain), spec, 10**9)) == 8


# ---------------------------------------------------------------- BC training
def test_bc_goal_loss_falls_and_accuracy_is_logged(tmp_path) -> None:
    spec, sig = a0_training_spec_and_signature()
    data = write_dataset(os.path.join(str(tmp_path), "d"), n_maps=4, eps_per_map=6, length=12, seed=0)
    add_goal_labels(data)
    cfg = BCConfig(epochs=25, seq_len=4, batch_seqs=8, lr=3e-3, seed=0, val_frac=0.25, torch_threads=2,
                   goal_head_coef=1.0)
    logs: list[str] = []
    tr, va, rec = BC.prepare(data, spec, cfg, log=logs.append)
    mcfg = replace(SMALL, goal_head=True)
    out = os.path.join(str(tmp_path), "run")
    s = BC.train_bc(tr, va, spec, mcfg, cfg, "cpu", out, sig, rec, log=logs.append)
    rows = [json.loads(x) for x in open(os.path.join(out, BC.METRICS_NAME), encoding="utf-8")]
    first, last = rows[0], rows[-1]
    assert last["train_goal_loss"] < 0.5 * first["train_goal_loss"], (first["train_goal_loss"], last["train_goal_loss"])
    assert last["train_goal_acc"] > 0.9 and last["val_goal_acc"] > 0.85
    # unlabelled steps (every 5th) are excluded from the count
    assert last["train_goal_labelled_steps"] < last["train_steps"]
    assert any("goal acc" in line for line in logs) and s["goal_head"] and s["goal_head_coef"] == 1.0
    # coef > 0 without the head is refused
    with pytest.raises(ValueError, match="goal head"):
        BC.train_bc(tr, va, spec, SMALL, cfg, "cpu", os.path.join(str(tmp_path), "x"), sig, rec, log=logs.append)


def test_bc_coef_zero_logs_no_goal_keys(tmp_path) -> None:
    spec, sig = a0_training_spec_and_signature()
    data = write_dataset(os.path.join(str(tmp_path), "d"), n_maps=2, eps_per_map=2, length=6)
    add_goal_labels(data)
    cfg = BCConfig(epochs=1, seq_len=4, batch_seqs=2, seed=0, val_frac=0.5, torch_threads=2)
    tr, va, rec = BC.prepare(data, spec, cfg, log=lambda s: None)
    out = os.path.join(str(tmp_path), "run")
    BC.train_bc(tr, va, spec, SMALL, cfg, "cpu", out, sig, rec, log=lambda s: None)
    row = json.loads(open(os.path.join(out, BC.METRICS_NAME), encoding="utf-8").readline())
    assert not any("goal" in k for k in row)


def test_cli_goal_head_flags(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from train import process_control as PC
    monkeypatch.setattr(PC, "set_below_normal", lambda pid=None: None)
    t = str(tmp_path)
    data = write_dataset(os.path.join(t, "d"), n_maps=3, eps_per_map=2, length=6)
    common = ["--seq-len", "4", "--batch-seqs", "2", "--torch-threads", "2", "--val-frac", "0.34"]
    with pytest.raises(SystemExit):
        BC.main(["--data", data, "--out", os.path.join(t, "o0"), "--goal-head-coef", "1", *common])
    o1 = os.path.join(t, "o1")
    assert BC.main(["--data", data, "--out", o1, "--epochs", "1", *common]) == 0    # head off
    plain = os.path.join(t, "plain_out")
    with pytest.raises(GoalLabelError):    # labels missing, coef > 0
        BC.main(["--data", data, "--out", plain, "--epochs", "1", "--goal-head", "--goal-head-coef", "1", *common])
    add_goal_labels(data)
    o2 = os.path.join(t, "o2")
    assert BC.main(["--data", data, "--out", o2, "--epochs", "0", "--goal-head", "--goal-head-coef", "1",
                    "--init-from-bc", os.path.join(o1, BC.MODEL_NAME), *common]) == 0
    s2 = json.load(open(os.path.join(o2, BC.SUMMARY_NAME), encoding="utf-8"))
    assert s2["init_from_bc_goal_head"]["kind"] == "goal_head_added"
    b2 = CK.load_checkpoint(os.path.join(o2, BC.MODEL_NAME))
    assert b2["model_config"].goal_head


# ---------------------------------------------------------------- head addition
def _src_file(tmp_path, cfg: ModelConfig, spec: ObsSpec, seed: int = 11) -> tuple[str, dict]:
    _, sig = a0_training_spec_and_signature()
    m = build_agent(spec, cfg, seed=seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.5)          # distinguishable from any fresh build
    path = os.path.join(str(tmp_path), f"src_{cfg.goal_head}.pt")
    CK.save_weights_checkpoint(path, m, cfg, spec, RunConfig(), sig, {"val_map_ids": ["DD0000"]})
    return path, {k: v.clone() for k, v in m.state_dict().items()}


def test_init_from_bc_adds_a_fresh_head_and_keeps_every_other_weight(tmp_path) -> None:
    spec, sig = a0_training_spec_and_signature()
    path, src = _src_file(tmp_path, ModelConfig(), spec)
    on = ModelConfig(goal_head=True)
    state, rec = BC.read_init_bc(path, spec, on, sig, seed=5)
    fresh = build_agent(spec, on, seed=5).state_dict()
    assert set(state) == set(fresh) and BC.INIT_GOAL_HEAD_KEY in rec
    for k, v in state.items():
        ref = fresh[k] if k.startswith(GOAL_KEY_PREFIX) else src[k]
        assert torch.equal(v, ref), k
    build_agent(spec, on, seed=0).load_state_dict(state, strict=True)
    # no head on either side: unchanged path, no record
    _, rec0 = BC.read_init_bc(path, spec, ModelConfig(), sig)
    assert BC.INIT_GOAL_HEAD_KEY not in rec0
    # removing the head, or any other config change, is refused
    path_on, _ = _src_file(tmp_path, on, spec)
    with pytest.raises(CK.CheckpointMismatch, match="model config"):
        BC.read_init_bc(path_on, spec, ModelConfig(), sig)
    with pytest.raises(CK.CheckpointMismatch, match="model config"):
        BC.read_init_bc(path, spec, replace(on, core_hidden=64), sig)


def _fake_trainer(cfg: ModelConfig, seed: int = 0, bc: BCAuxConfig | None = None) -> Trainer:
    venv = SyncVecEnv([lambda: FakeDoomEnv(7) for _ in range(2)])
    run = replace(RunConfig(), seed=seed, num_workers=1, envs_per_worker=2,
                  ppo=PPOConfig(rollout_len=6, num_minibatches=1, epochs=1),
                  **({"bc_aux": bc} if bc is not None else {}))
    return Trainer(venv, cfg, run)


def _init_source(path: str) -> InitSource:
    return InitSource(path, path, sha256_file(path), "weights_only", CK.load_checkpoint(path))


def test_init_from_adds_a_fresh_head_and_keeps_every_other_weight(tmp_path) -> None:
    probe = _fake_trainer(ModelConfig())
    spec = probe.spec
    path, src = _src_file(tmp_path, ModelConfig(), spec)
    on = ModelConfig(goal_head=True)
    tr = _fake_trainer(on, seed=4)
    fresh_head = {k: v.clone() for k, v in tr.learner.state_dict().items() if k.startswith(GOAL_KEY_PREFIX)}
    s = _init_source(path)
    assert apply_init(tr, s) is None and s.goal_head_added
    for k, v in tr.learner.state_dict().items():
        ref = fresh_head[k] if k.startswith(GOAL_KEY_PREFIX) else src[k]
        assert torch.equal(v.cpu(), ref), k
    same = _init_source(path)
    apply_init(_fake_trainer(ModelConfig()), same)
    assert not same.goal_head_added
    path_on, _ = _src_file(tmp_path, on, spec)
    with pytest.raises(CK.CheckpointMismatch, match="model config"):
        apply_init(_fake_trainer(ModelConfig()), _init_source(path_on))


def test_goal_head_added_rules() -> None:
    off, on = ModelConfig(), ModelConfig(goal_head=True)
    assert goal_head_added(off, on) and not goal_head_added(on, on) and not goal_head_added(off, off)
    for a, b in ((on, off), (off, replace(on, fusion_dim=8))):
        with pytest.raises(ValueError):
            goal_head_added(a, b)


# ---------------------------------------------------------------- PPO BC term
def test_ppo_bc_term_adds_the_goal_loss(tmp_path) -> None:
    spec, _ = a0_training_spec_and_signature()
    data = write_dataset(os.path.join(str(tmp_path), "d"), n_maps=1, eps_per_map=3, length=8)
    add_goal_labels(data)
    eps = load_bc_data((data,), spec, 10**9, goal_required=True)
    model = build_agent(spec, replace(SMALL, goal_head=True), seed=0)

    def term(coef: float) -> tuple[torch.Tensor, dict]:
        return bc_term(model, BCSampler(eps, 4, 3, seed=1).sample(), coef)
    base, st0 = term(0.0)
    with_goal, st1 = term(2.0)
    assert "bc_goal_acc" not in st0 and 0.0 <= st1["bc_goal_acc"] <= 1.0
    assert torch.allclose(with_goal - base, torch.tensor(2.0 * st1["bc_goal_loss"]), atol=1e-5)
    with pytest.raises(ValueError, match="goal head"):
        bc_term(build_agent(spec, SMALL, seed=0), BCSampler(eps, 4, 3, seed=1).sample(), 1.0)
    cfg = BCAuxConfig(data_dirs=(data,), coef=1.0, goal_coef=0.5)
    assert cfg.resume_identity()["goal_coef"] == 0.5
    # a checkpoint written before goal_coef existed resumes with the same BC settings
    old = dataclasses.asdict(BCAuxConfig(data_dirs=(data,), coef=1.0))
    old.pop("goal_coef")
    blob = {"run_config_json": json.dumps({"bc_aux": old})}
    CK.check_bc_matches(CK.bc_aux_of(blob), BCAuxConfig(data_dirs=(data,), coef=1.0))
    with pytest.raises(CK.CheckpointMismatch):
        CK.check_bc_matches(CK.bc_aux_of(blob), cfg)


def test_run_doom_goal_flags_parse() -> None:
    from train.run_doom import bc_aux_from_args, parse_args
    ns = parse_args(["--bc-data", "x", "--bc-coef", "1", "--goal-head", "--goal-head-coef", "0.3"])
    assert ns.goal_head and bc_aux_from_args(ns).goal_coef == 0.3
    for bad in (["--goal-head-coef", "1"], ["--bc-data", "x", "--bc-coef", "1", "--goal-head-coef", "1"]):
        with pytest.raises(SystemExit):
            parse_args(bad)
    assert bc_aux_from_args(parse_args([])).goal_coef == 0.0
