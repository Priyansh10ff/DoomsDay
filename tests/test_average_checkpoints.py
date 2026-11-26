"""tools/average_checkpoints.py: uniform weight averaging of checkpoints of one run.
No engine is started."""
from __future__ import annotations

import dataclasses
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from a0_helpers import doom_like_spaces, random_obs  # noqa: F401  (also caps torch threads)
from agent.model import A0Agent, build_agent
from agent.obs_spec import ObsSpec, spec_from_spaces
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig
from configs.train_config import RunConfig
from evaluate.checkpoint_io import load_model
from tools import average_checkpoints as AV
from train import checkpoint as CK
from train.init_from import apply_init, read_init_source

CFG = ModelConfig()
SIG = (("frame_skip", 4), ("obs_version", 2))
BATCH = 3


def spec_of(ray_bands: int = 3) -> ObsSpec:
    return spec_from_spaces(*doom_like_spaces(ray_bands=ray_bands))


def save(path: str, model: A0Agent, spec: ObsSpec, sig: tuple = SIG, cfg: ModelConfig = CFG) -> str:
    CK.save_weights_checkpoint(path, model, cfg, spec, RunConfig(), sig, {"kind": "test"})
    return path


def outputs(model: A0Agent, seed: int) -> list[torch.Tensor]:
    obs_space, _ = doom_like_spaces()
    obs = to_tensors(random_obs(obs_space, BATCH, np.random.default_rng(seed)))
    with torch.no_grad():
        out, h = model.step(obs, model.initial_state(BATCH), torch.ones(BATCH))
    return [*out.logits, out.value_ext, h]


def test_two_copies_of_one_model_give_the_same_outputs(tmp_path: "os.PathLike[str]") -> None:
    spec = spec_of()
    m = build_agent(spec, CFG, seed=1).eval()
    a = save(os.path.join(str(tmp_path), "a.pt"), m, spec)
    b = save(os.path.join(str(tmp_path), "b.pt"), m, spec)
    out = os.path.join(str(tmp_path), "avg.pt")
    rec = AV.average_checkpoints([a, b], out)
    loaded = load_model(out)
    for x, y in zip(outputs(m, 5), outputs(loaded.model, 5)):
        assert torch.equal(x, y)
    assert rec["kind"] == AV.AVERAGE_KIND and rec["n_sources"] == 2
    assert [s["sha256"] for s in rec["sources"]] == [AV.sha256_file(p) for p in (a, b)]


def test_two_models_give_the_parameter_mean(tmp_path: "os.PathLike[str]") -> None:
    spec = spec_of()
    m1, m2 = build_agent(spec, CFG, seed=1), build_agent(spec, CFG, seed=2)
    p1 = save(os.path.join(str(tmp_path), "1.pt"), m1, spec)
    p2 = save(os.path.join(str(tmp_path), "2.pt"), m2, spec)
    out = os.path.join(str(tmp_path), "avg.pt")
    AV.average_checkpoints([p1, p2], out)
    got = load_model(out).model.state_dict()
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert set(got) == set(s1)
    differing = 0
    for k, v in got.items():
        if torch.is_floating_point(v):
            want = ((s1[k].double() + s2[k].double()) / 2).to(v.dtype)
            assert torch.equal(v, want), k
            differing += int(not torch.equal(s1[k], s2[k]))
        else:
            assert torch.equal(v, s2[k]), k
    assert differing > 0, "the two source models must actually differ"


def test_non_float_tensors_come_from_the_last_source() -> None:
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor([1], dtype=torch.int64),
         "h": torch.tensor([1.0], dtype=torch.float16)}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor([7], dtype=torch.int64),
         "h": torch.tensor([2.0], dtype=torch.float16)}
    out = AV.average_states([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0])) and out["n"].item() == 7
    assert out["h"].dtype == torch.float16 and out["h"].item() == 1.5


def test_output_is_a_weights_only_trainer_format_blob(tmp_path: "os.PathLike[str]") -> None:
    spec = spec_of()
    paths = [save(os.path.join(str(tmp_path), f"{i}.pt"), build_agent(spec, CFG, seed=i), spec) for i in range(3)]
    out = os.path.join(str(tmp_path), "avg.pt")
    AV.average_checkpoints(paths, out)
    blob = torch.load(out, map_location="cpu", weights_only=True)      # what every loader does
    last = torch.load(paths[-1], map_location="cpu", weights_only=True)
    for k in ("model_config_json", "obs_spec_json", "sensor_signature_json", "run_config_json", CK.SHAPING_KEY):
        assert blob[k] == last[k], k
    rec = json.loads(blob[CK.WEIGHTS_ONLY_KEY])
    assert [os.path.normcase(s["loaded_from"]) for s in rec["sources"]] == \
        [os.path.normcase(os.path.abspath(p)) for p in paths]
    assert all(len(s["sha256"]) == 64 for s in rec["sources"])
    # --init-from accepts it (weights-only); --resume refuses it
    src = read_init_source(out, SIG)
    assert src.kind == "weights_only"
    fresh = build_agent(spec, CFG, seed=99)
    stub = SimpleNamespace(model=fresh, learner=fresh, spec=spec, sync_actor=lambda: None)
    assert apply_init(stub, src) is None                          # model config, spec, strict load
    assert all(torch.equal(v, blob["trainer"]["model"][k]) for k, v in fresh.state_dict().items())
    with pytest.raises(CK.CheckpointMismatch, match="weights-only"):
        CK.checkpoint_resume_info(out)


@pytest.mark.parametrize("what", ["obs_spec", "signature", "model_config"])
def test_mismatched_inputs_refused(tmp_path: "os.PathLike[str]", what: str) -> None:
    spec = spec_of()
    a = save(os.path.join(str(tmp_path), "a.pt"), build_agent(spec, CFG, seed=1), spec)
    b_path = os.path.join(str(tmp_path), "b.pt")
    if what == "obs_spec":
        s4 = spec_of(ray_bands=4)
        save(b_path, build_agent(s4, CFG, seed=1), s4)
    elif what == "signature":
        save(b_path, build_agent(spec, CFG, seed=1), spec, sig=(("frame_skip", 4), ("obs_version", 3)))
    else:
        cfg2 = dataclasses.replace(CFG, ray_dim=CFG.ray_dim // 2)
        save(b_path, build_agent(spec, cfg2, seed=1), spec, cfg=cfg2)
    out = os.path.join(str(tmp_path), "avg.pt")
    with pytest.raises(CK.CheckpointMismatch):
        AV.average_checkpoints([a, b_path], out)
    assert not os.path.exists(out)
    assert AV.main(["--out", out, a, b_path]) == 2 and not os.path.exists(out)


def test_too_few_duplicate_or_overwriting_inputs_refused(tmp_path: "os.PathLike[str]") -> None:
    spec = spec_of()
    a = save(os.path.join(str(tmp_path), "a.pt"), build_agent(spec, CFG, seed=1), spec)
    b = save(os.path.join(str(tmp_path), "b.pt"), build_agent(spec, CFG, seed=2), spec)
    out = os.path.join(str(tmp_path), "avg.pt")
    with pytest.raises(ValueError, match="at least"):
        AV.average_checkpoints([a], out)
    with pytest.raises(ValueError, match="same file"):
        AV.average_checkpoints([a, a], out)
    with pytest.raises(ValueError, match="overwrite"):
        AV.average_checkpoints([a, b], b)
    with pytest.raises(SystemExit):
        AV.main(["--out", out, a])


def test_cli_writes_a_loadable_checkpoint(tmp_path: "os.PathLike[str]", capsys: pytest.CaptureFixture) -> None:
    spec = spec_of()
    a = save(os.path.join(str(tmp_path), "a.pt"), build_agent(spec, CFG, seed=1), spec)
    b = save(os.path.join(str(tmp_path), "b.pt"), build_agent(spec, CFG, seed=2), spec)
    out = os.path.join(str(tmp_path), "sub", "avg.pt")
    assert AV.main(["--out", out, a, b]) == 0
    assert "averaged 2 checkpoints" in capsys.readouterr().out
    loaded = load_model(out)
    assert loaded.spec == spec and loaded.signature_json == CK.canonical_json(SIG)
