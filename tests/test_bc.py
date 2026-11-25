"""Behaviour cloning (train/bc.py, bc_data.py, bc_spec.py): spec, map-id split, chunk
streaming with carried state, padded-step masking, hidden-state reset between
episodes, learning a reactive and a memory label, and the saved file loading through
evaluate's loader, --init-from and the --resume refusal. No engine is started."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from a0_helpers import FakeDoomEnv, doom_like_spaces, random_obs
from agent.model import build_agent
from agent.obs_spec import A0_KEYS, spec_from_spaces
from configs.bc_config import BCConfig
from configs.model_config import ModelConfig
from configs.train_config import PPOConfig, RunConfig
from train import bc as BC
from train import checkpoint as CK
from train.bc_data import ChunkStream, Episode, load_episodes, split_map_ids
from train.bc_spec import DatasetMismatch, a0_training_spec_and_signature, validate_arrays
from train.trainer import Trainer
from train.vec_env import SyncVecEnv

SMALL = ModelConfig(core_hidden=32, fusion_dim=32, ray_dim=16, entity_token_dim=16, entity_mlp_hidden=16,
                    self_dim=16, self_hidden=16)
SPEC = spec_from_spaces(*doom_like_spaces())


@pytest.fixture(autouse=True, scope="module")
def _obs_v1():
    """These synthetic datasets use the observation v1 layout (doom_like_spaces), so the
    BC pipeline is pinned to v1 here (module scope: module fixtures build files too);
    v2 sizes are checked in tests/test_obs_v2_unit.py."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DOOMSDAY_OBS_VERSION", "1")
        yield
# `self` dims carrying the memory cue and the reactive signal. The signal is copied over
# 8 dims with the rest of `self` zero: measured 2026-11-25, the A0 network sits on a
# long plateau on a single noisy dim (and the SMALL test config did not leave it in
# 600 steps), while this version is learned in about 50 epochs by ModelConfig().
CUE_DIMS, REACT_DIMS = slice(0, 8), slice(8, 16)
MOVE, STRAFE, ATTACK = 0, 1, 3


# ---------------------------------------------------------------- synthetic data
def make_episode_arrays(length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Labels: move = reactive (sign of self[REACT_DIMS] at this step); strafe = memory:
    the sign of a cue shown ONLY at step 0 (self[CUE_DIMS] = +-1, 0 afterwards), held
    for the whole episode; attack constant 0; turn constant 3; use/speed constant 0.
    prev_action is random so it cannot leak the cue (in real data it is the executed
    action). Rays and entities are random distractors."""
    space, _ = doom_like_spaces()
    obs = random_obs(space, length, rng)
    react = np.where(rng.random(length) < 0.5, 1.0, -1.0).astype(np.float32)
    cue = 1.0 if rng.random() < 0.5 else -1.0
    obs["self"] = np.zeros((length, SPEC.self_dim), np.float32)
    obs["self"][:, REACT_DIMS] = react[:, None]
    obs["self"][0, CUE_DIMS] = cue
    obs["prev_action"] = rng.integers(0, 2, (length, 6)).astype(np.int64)
    action = np.zeros((length, 6), np.int64)
    action[:, MOVE] = (react > 0).astype(np.int64)
    action[:, STRAFE] = 2 if cue > 0 else 0
    action[:, 2] = 3
    return {**{k: obs[k] for k in A0_KEYS}, "action": action, "executed_action": action.copy(),
            "engine_reward": np.zeros(length, np.float32), "done": np.eye(1, length, length - 1, dtype=bool)[0]}


def write_dataset(root: str, n_maps: int, eps_per_map: int, length: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    rows, k = [], 0
    for m in range(n_maps):
        for _ in range(eps_per_map):
            name = f"ep_{k:06d}.npz"
            np.savez_compressed(os.path.join(root, name), **make_episode_arrays(length, rng))
            rows.append({"file": name, "map_id": f"DD{m:04d}", "stage": "S0", "length": length,
                         "success": True, "seed": k, "noise": 0.0})
            k += 1
    with open(os.path.join(root, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return root


def episode(length: int, rng: np.random.Generator, map_id: str = "DD0000") -> Episode:
    a = make_episode_arrays(length, rng)
    return Episode({k: a[k] for k in A0_KEYS}, a["action"], map_id, "S0", "x")


# ---------------------------------------------------------------- spec and validation
def test_synthetic_labels_follow_the_rule() -> None:
    a = make_episode_arrays(9, np.random.default_rng(3))
    assert (a["action"][:, MOVE] == (a["self"][:, 8] > 0)).all()
    assert (a["self"][1:, CUE_DIMS] == 0).all() and len(set(a["action"][:, STRAFE])) == 1


def test_engine_free_spec_matches_the_env_space_layout() -> None:
    spec, sig = a0_training_spec_and_signature()
    assert spec == SPEC, "train/bc_spec.py drifted from the DoomEnv space layout used by the tests"
    assert sig and isinstance(sig, tuple)


def test_validate_rejects_wrong_shape_dtype_and_label_range() -> None:
    good = make_episode_arrays(5, np.random.default_rng(0))
    assert validate_arrays(good, SPEC, "g") == 5
    bad = dict(good, rays=good["rays"][:, :2])
    with pytest.raises(DatasetMismatch, match="rays shape"):
        validate_arrays(bad, SPEC, "b")
    with pytest.raises(DatasetMismatch, match="dtype"):
        validate_arrays(dict(good, action=good["action"].astype(np.int32)), SPEC, "b")
    lab = good["action"].copy()
    lab[0, 2] = 7
    with pytest.raises(DatasetMismatch, match="label head 2"):
        validate_arrays(dict(good, action=lab), SPEC, "b")
    with pytest.raises(DatasetMismatch, match="missing"):
        validate_arrays({k: v for k, v in good.items() if k != "modality"}, SPEC, "b")


# ---------------------------------------------------------------- split
def test_map_id_split_is_disjoint_seeded_and_nonempty() -> None:
    ids = [f"DD{i:04d}" for i in range(10)] * 3
    tr, va = split_map_ids(ids, 0.2, seed=1)
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(ids) and len(va) == 2
    assert split_map_ids(ids, 0.2, seed=1) == (tr, va)
    tr2, va2 = split_map_ids(["A", "B"], 0.1, seed=0)          # rounds to 0 -> still one held out
    assert len(va2) == 1 and len(tr2) == 1
    assert split_map_ids(["A", "B"], 0.0, seed=0) == (["A", "B"], [])
    assert split_map_ids(["A"], 0.5, seed=0) == (["A"], [])


# ---------------------------------------------------------------- streaming
def test_chunk_stream_covers_each_step_once_with_starts_and_padding() -> None:
    rng = np.random.default_rng(0)
    eps = [episode(n, rng) for n in (5, 8, 3, 9)]
    for i, e in enumerate(eps):
        e.obs["self"][:, 0] = np.arange(e.length) + 100 * i      # step tag
    stream = ChunkStream(eps, seq_len=4, batch_seqs=2, generator=torch.Generator().manual_seed(0))
    seen, starts = [], 0
    for c in stream:
        assert c.obs["rays"].shape == (4, 2, 3, 80) and c.labels.shape == (4, 2, 6)
        tags = c.obs["self"][..., 0][c.valid]
        seen += tags.long().tolist()
        starts += int(c.episode_start.sum())
        # padded steps never carry an episode start
        assert bool((c.episode_start[c.valid.logical_not()] == 0).all())
    want = sorted(int(t) for i, e in enumerate(eps) for t in np.arange(e.length) + 100 * i)
    assert sorted(seen) == want and starts == len(eps)


def test_chunk_stream_is_deterministic_per_seed() -> None:
    rng = np.random.default_rng(0)
    eps = [episode(6, rng) for _ in range(5)]

    def tags(seed: int) -> list[list[float]]:
        s = ChunkStream(eps, 3, 2, torch.Generator().manual_seed(seed))
        return [c.obs["self"][..., 3].flatten().tolist() for c in s]
    assert tags(4) == tags(4)
    assert tags(4) != tags(5)


def test_hidden_state_resets_between_episodes_in_one_slot() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    a, b = episode(6, rng), episode(5, rng)
    model = build_agent(SPEC, SMALL, seed=0).eval()

    def logits_of(eps: list[Episode]) -> list[torch.Tensor]:
        out, h = [], model.initial_state(1)
        with torch.no_grad():
            for c in ChunkStream(eps, 4, 1, torch.Generator(), shuffle=False):
                o, h = model.forward_sequence(c.obs, h, c.episode_start)
                out.append(o.logits[1][c.valid])
        return out
    both = torch.cat(logits_of([a, b]))[a.length:]
    alone = torch.cat(logits_of([b]))
    torch.testing.assert_close(both, alone)


# ---------------------------------------------------------------- loss masking
def test_bc_loss_ignores_padded_steps_in_value_and_gradient() -> None:
    g = torch.Generator().manual_seed(0)
    t, b = 5, 3
    valid = torch.ones(t, b, dtype=torch.bool)
    valid[3:, 1] = False
    valid[1:, 2] = False
    labels = torch.stack([torch.randint(0, n, (t, b), generator=g) for n in SPEC.head_sizes], -1)
    base = [torch.randn(t, b, n, generator=g) for n in SPEC.head_sizes]
    w = (1.0, 1.0, 1.0, 0.0, 1.0, 1.0)

    def run(lg_vals: list[torch.Tensor], lab: torch.Tensor) -> tuple[float, list[torch.Tensor]]:
        lg = [x.clone().requires_grad_(True) for x in lg_vals]
        c = BC.bc_loss(lg, lab, valid, w)
        c.total.backward()
        return float(c.total.detach()), [torch.zeros_like(x) if x.grad is None else x.grad for x in lg]
    l1, g1 = run(base, labels)
    pad = valid.logical_not()
    junk = [x.masked_fill(pad.unsqueeze(-1), 50.0) for x in base]
    lab2 = labels.clone()
    lab2[pad] = 0
    l2, g2 = run(junk, lab2)
    assert l1 == pytest.approx(l2)
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y)
        assert bool((x[pad] == 0).all())
    assert bool((g1[ATTACK] == 0).all()), "zero-weight head must get no gradient"
    # value: mean CE over the 9 valid steps per weighted head
    n = int(valid.sum())
    ref = sum(w[h] * torch.nn.functional.cross_entropy(base[h][valid], labels[..., h][valid], reduction="sum") / n
              for h in range(6))
    assert l1 == pytest.approx(float(ref), rel=1e-5)


# ---------------------------------------------------------------- learning + outputs
EPOCHS = 80


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One BC run shared by the output tests (default ModelConfig, as PPO uses; CPU; about 20 s)."""
    root = tmp_path_factory.mktemp("bc")
    data = write_dataset(os.path.join(root, "data"), n_maps=6, eps_per_map=10, length=12, seed=0)
    spec, sig = a0_training_spec_and_signature()
    # attack weight 0 set explicitly (the default is 1.0): these tests check that a
    # zero-weight head is logged and keeps its entropy
    cfg = BCConfig(epochs=EPOCHS, seq_len=4, batch_seqs=8, lr=1e-3, seed=0, val_frac=0.34, torch_threads=4,
                   head_weights=(1.0, 1.0, 1.0, 0.0, 1.0, 1.0))
    logs: list[str] = []
    tr, va, rec = BC.prepare(data, spec, cfg, log=logs.append)
    out = os.path.join(root, "run")
    summary = BC.train_bc(tr, va, spec, ModelConfig(), cfg, "cpu", out, sig, rec, log=logs.append)
    return {"summary": summary, "out": out, "sig": sig, "spec": spec, "logs": logs, "data": data,
            "train": tr, "val": va}


def test_bc_reaches_high_accuracy_including_the_memory_label(trained: dict) -> None:
    f = trained["summary"]["final"]
    assert set(e.map_id for e in trained["train"]).isdisjoint(e.map_id for e in trained["val"])
    assert trained["val"], "val split must not be empty"
    assert f["val_acc_move"] > 0.9
    # The strafe label depends on a cue seen only at step 0 of a 12-step episode cut into
    # 4-step chunks: without the state carried across chunks, steps 4..11 (2/3 of all
    # steps) are at chance, capping accuracy near 0.67. Above 0.9 needs the carry.
    assert f["val_acc_strafe"] > 0.9, f
    assert f["train_entropy_attack"] > 0.5, "zero-weight attack head must keep its entropy (ln2 = 0.69)"
    logs = "\n".join(trained["logs"])
    assert "attack: loss weight 0" in logs and "teacher label constant" in logs and "value head untouched" in logs


def test_bc_outputs_and_record(trained: dict) -> None:
    out = trained["out"]
    for name in (BC.MODEL_NAME, BC.METRICS_NAME, BC.SUMMARY_NAME):
        assert os.path.exists(os.path.join(out, name))
    lines = open(os.path.join(out, BC.METRICS_NAME), encoding="utf-8").read().splitlines()
    assert len(lines) == EPOCHS and "val_acc_turn" in json.loads(lines[-1])
    blob = CK.load_checkpoint(os.path.join(out, BC.MODEL_NAME))
    rec = json.loads(blob[CK.WEIGHTS_ONLY_KEY])
    assert rec["sensor_signature_assumed"] is True and "assumed" in rec["sensor_signature_source"]
    assert rec["val_map_ids"] and set(rec["val_map_ids"]).isdisjoint(rec["train_map_ids"])


def test_bc_file_loads_through_the_evaluate_loader_and_sensor_check(trained: dict) -> None:
    from configs.a0_env import a0_eval_env_config
    from evaluate.checkpoint_io import load_model
    from evaluate.sensors import check_sensors, trained_config_from_signature
    path = os.path.join(trained["out"], BC.MODEL_NAME)
    loaded = load_model(path)
    assert loaded.spec == trained["spec"]
    blob = CK.load_checkpoint(path)
    for k, v in loaded.model.state_dict().items():
        torch.testing.assert_close(v, blob["trainer"]["model"][k])
    ev = a0_eval_env_config()
    check_sensors(trained_config_from_signature(loaded.signature_json, ev), ev)


def test_resume_refuses_a_bc_file(trained: dict) -> None:
    path = os.path.join(trained["out"], BC.MODEL_NAME)
    with pytest.raises(CK.CheckpointMismatch, match="--init-from"):
        CK.checkpoint_resume_info(path)


def _fake_trainer(model_cfg: ModelConfig = SMALL, warmup: int = 0, seed: int = 0) -> Trainer:
    venv = SyncVecEnv([lambda: FakeDoomEnv(7) for _ in range(2)])
    run = replace(RunConfig(), seed=seed, num_workers=1, envs_per_worker=2,
                  ppo=PPOConfig(rollout_len=6, num_minibatches=1, epochs=1, value_warmup_updates=warmup))
    return Trainer(venv, model_cfg, run)


def test_init_from_loads_weights_strictly_and_refuses_mismatches(trained: dict) -> None:
    from train.init_from import apply_init, read_init_source
    path = os.path.join(trained["out"], BC.MODEL_NAME)
    src = read_init_source(path, trained["sig"])
    assert src.kind == "weights_only" and len(src.sha256) == 64
    tr = _fake_trainer(model_cfg=ModelConfig(), seed=7)
    apply_init(tr, src)
    for k, v in tr.learner.state_dict().items():
        torch.testing.assert_close(v, src.blob["trainer"]["model"][k])
    assert tr.update_idx == 0
    with pytest.raises(CK.CheckpointMismatch, match="sensor"):
        read_init_source(path, (("frame_skip", 2),))
    other = _fake_trainer(model_cfg=SMALL)
    with pytest.raises(CK.CheckpointMismatch, match="model config"):
        apply_init(other, src)


# ---------------------------------------------------------------- CLI
def test_cli_end_to_end_small(tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch) -> None:
    from train import process_control as PC
    monkeypatch.setattr(PC, "set_below_normal", lambda pid=None: None)
    data = write_dataset(os.path.join(str(tmp_path), "d"), n_maps=3, eps_per_map=2, length=6)
    out = os.path.join(str(tmp_path), "o")
    assert BC.main(["--data", data, "--out", out, "--epochs", "1", "--seq-len", "4", "--batch-seqs", "2",
                    "--val-frac", "0.34", "--torch-threads", "2"]) == 0
    s = json.load(open(os.path.join(out, BC.SUMMARY_NAME), encoding="utf-8"))
    assert s["epochs_done"] == 1 and s["head_weights"][3] == 1.0 and len(s["val_map_ids"]) == 1
    assert len(load_episodes(data, SPEC, 10**9)) == 6


def test_cli_several_data_dirs_and_init_from_bc(tmp_path: "os.PathLike[str]",
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """DAgger rounds: --data repeated and comma-separated trains on the union;
    --init-from-bc starts from the earlier weights and keeps its BC-val map ids."""
    from train import process_control as PC
    monkeypatch.setattr(PC, "set_below_normal", lambda pid=None: None)
    t = str(tmp_path)
    d1 = write_dataset(os.path.join(t, "d1"), n_maps=3, eps_per_map=2, length=6, seed=1)
    d2 = write_dataset(os.path.join(t, "d2"), n_maps=3, eps_per_map=1, length=5, seed=2)
    d3 = write_dataset(os.path.join(t, "d3"), n_maps=3, eps_per_map=1, length=7, seed=3)
    o1, o2 = os.path.join(t, "o1"), os.path.join(t, "o2")
    common = ["--seq-len", "4", "--batch-seqs", "2", "--torch-threads", "2"]
    assert BC.main(["--data", d1, "--out", o1, "--epochs", "1", "--val-frac", "0.34", *common]) == 0
    s1 = json.load(open(os.path.join(o1, BC.SUMMARY_NAME), encoding="utf-8"))
    # a different seed would split differently: the val ids must come from the init model
    assert BC.main(["--data", d1, "--data", f"{d2},{d3}", "--out", o2, "--epochs", "0",
                    "--init-from-bc", os.path.join(o1, BC.MODEL_NAME), "--seed", "99", *common]) == 0
    s2 = json.load(open(os.path.join(o2, BC.SUMMARY_NAME), encoding="utf-8"))
    assert s2["val_map_ids"] == s1["val_map_ids"]
    assert len(s2["data_dirs"]) == 3
    assert s2["n_train_steps"] + s2["n_val_steps"] == 6 * 6 + 3 * 5 + 3 * 7
    w1 = CK.load_checkpoint(os.path.join(o1, BC.MODEL_NAME))["trainer"]["model"]
    w2 = CK.load_checkpoint(os.path.join(o2, BC.MODEL_NAME))["trainer"]["model"]
    for k in w1:                                     # 0 epochs: weights are exactly the init's
        torch.testing.assert_close(w1[k], w2[k])
    assert BC.data_dirs_from_args([d1, f"{d1},{d2}"]) == [d1, d2]


def test_init_from_bc_refuses_a_different_model_config(trained: dict) -> None:
    path = os.path.join(trained["out"], BC.MODEL_NAME)
    state, rec = BC.read_init_bc(path, trained["spec"], ModelConfig(), trained["sig"])
    assert rec["val_map_ids"] and set(state) == set(build_agent(trained["spec"], ModelConfig(), seed=0).state_dict())
    with pytest.raises(CK.CheckpointMismatch, match="model config"):
        BC.read_init_bc(path, trained["spec"], SMALL, trained["sig"])


def test_dataset_size_guard(tmp_path: "os.PathLike[str]") -> None:
    data = write_dataset(os.path.join(str(tmp_path), "d"), n_maps=1, eps_per_map=1, length=6)
    with pytest.raises(MemoryError):
        load_episodes(data, SPEC, max_bytes=100)


def test_carried_state_over_chunks_equals_one_whole_episode_pass() -> None:
    """Truncated BPTT carry: three 4-step chunks with the state carried give the same
    logits as one 12-step pass."""
    ep = episode(12, np.random.default_rng(5))
    model = build_agent(SPEC, SMALL, seed=0).eval()
    with torch.no_grad():
        whole = next(iter(ChunkStream([ep], 12, 1, torch.Generator(), shuffle=False)))
        ref, _ = model.forward_sequence(whole.obs, model.initial_state(1), whole.episode_start)
        parts, h = [], model.initial_state(1)
        for c in ChunkStream([ep], 4, 1, torch.Generator(), shuffle=False):
            o, h = model.forward_sequence(c.obs, h, c.episode_start)
            parts.append(torch.stack(o.logits[1:2], 0))
    torch.testing.assert_close(torch.cat(parts, dim=1), torch.stack(ref.logits[1:2], 0))


def test_bc_training_is_deterministic_per_seed(tmp_path: "os.PathLike[str]") -> None:
    rng = np.random.default_rng(2)
    tr = [episode(7, rng, f"M{i % 2}") for i in range(6)]
    spec, sig = a0_training_spec_and_signature()
    cfg = BCConfig(epochs=1, seq_len=3, batch_seqs=2, lr=1e-3, seed=3)

    def weights(sub: str) -> dict[str, torch.Tensor]:
        out = os.path.join(str(tmp_path), sub)
        BC.train_bc(tr, [], spec, SMALL, cfg, "cpu", out, sig, {"kind": "test"}, log=lambda m: None)
        return CK.load_checkpoint(os.path.join(out, BC.MODEL_NAME))["trainer"]["model"]
    a, b = weights("a"), weights("b")
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    c = build_agent(spec, SMALL, seed=3).state_dict()
    assert any(not torch.equal(a[k], c[k]) for k in a), "one epoch must change the weights"


def test_default_head_weights_clone_attack() -> None:
    """Attack weight default is 1.0 (2026-11-25: weight 0 left the head at 50/50)."""
    from configs.bc_config import DEFAULT_HEAD_WEIGHTS
    assert BCConfig().head_weights == DEFAULT_HEAD_WEIGHTS == (1.0,) * 6
