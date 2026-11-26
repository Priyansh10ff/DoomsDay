"""Play-time decoding temperature (agent/inference.py, evaluate/, ablation script).
No engine is started."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from a0_helpers import doom_like_spaces, random_obs  # noqa: F401  (also caps torch threads)
from agent.distributions import FactorizedCategorical
from agent.inference import PolicyRunner, check_temperature, tempered_logits
from agent.model import A0Agent, build_agent
from agent.obs_spec import spec_from_spaces
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig
from evaluate import record as REC
from evaluate import run as RUN
from evaluate.config import EvalRunConfig
from evaluate.policies import NetworkPolicy

N_ENVS = 3
STEPS = 6
HEADS = (3, 3, 7, 2, 2, 2)
ABLATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "reports", "eval", "scripts", "ablation_table.py")


def model_and_obs(seed: int = 0) -> tuple[A0Agent, list[dict[str, np.ndarray]]]:
    obs_space, act_space = doom_like_spaces()
    model = build_agent(spec_from_spaces(obs_space, act_space), ModelConfig(), seed).eval()
    rng = np.random.default_rng(seed)
    return model, [random_obs(obs_space, N_ENVS, rng) for _ in range(STEPS)]


def starts(t: int) -> np.ndarray:
    return np.full(N_ENVS, 1.0 if t == 0 else 0.0, np.float32)


def test_t1_is_bit_identical_to_plain_sampling() -> None:
    """The pre-temperature sampling path, written out: same seed, same actions."""
    model, seq = model_and_obs(1)
    runner = PolicyRunner(model, N_ENVS, (False,) * 6, seed=9, temperature=1.0)
    default = PolicyRunner(model, N_ENVS, (False,) * 6, seed=9)
    gen = torch.Generator(device="cpu").manual_seed(9)
    h = model.initial_state(N_ENVS, "cpu")
    for t, obs in enumerate(seq):
        with torch.no_grad():
            out, h = model.step(to_tensors(obs), h, torch.as_tensor(starts(t)))
        want = FactorizedCategorical(out.logits).sample(gen).numpy().astype(np.int64)
        a, b = runner.act(obs, starts(t)), default.act(obs, starts(t))
        assert np.array_equal(a, want) and np.array_equal(b, want) and a.shape == (N_ENVS, 6)


def test_tempered_logits_identity_at_one_and_scaled_otherwise() -> None:
    lg = [torch.randn(2, n, generator=torch.Generator().manual_seed(n)) for n in HEADS]
    assert tempered_logits(lg, 1.0) is lg
    for x, y in zip(lg, tempered_logits(lg, 0.5)):
        assert torch.equal(y, x / 0.5)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            check_temperature(bad)


class _FixedLogitsModel:
    """Stub with well-separated fixed logits per head (not the A0 model)."""

    def __init__(self, margin: float) -> None:
        self.spec = SimpleNamespace(n_heads=len(HEADS))
        g = torch.Generator().manual_seed(3)
        self.base = [torch.randn(n, generator=g) * margin for n in HEADS]

    def initial_state(self, n: int, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(n, 1)

    def step(self, obs: dict[str, torch.Tensor], h: torch.Tensor,
             start: torch.Tensor) -> tuple[SimpleNamespace, torch.Tensor]:
        n = h.shape[0]
        return SimpleNamespace(logits=[b.expand(n, -1).clone() for b in self.base]), h


def test_small_temperature_approaches_argmax() -> None:
    model = _FixedLogitsModel(margin=1.0)
    want = torch.stack([b.argmax() for b in model.base]).numpy()
    obs = random_obs(doom_like_spaces()[0], N_ENVS, np.random.default_rng(0))
    cold = PolicyRunner(model, N_ENVS, (False,) * 6, seed=4, temperature=1e-3)   # type: ignore[arg-type]
    warm = PolicyRunner(model, N_ENVS, (False,) * 6, seed=4, temperature=1.0)    # type: ignore[arg-type]
    cold_acts = np.concatenate([cold.act(obs, starts(t)) for t in range(50)])
    warm_acts = np.concatenate([warm.act(obs, starts(t)) for t in range(50)])
    assert (cold_acts == want).all()
    assert not (warm_acts == want).all()          # plain sampling is not argmax here
    dist = FactorizedCategorical(tempered_logits([b[None] for b in model.base], 1e-3))
    assert torch.allclose(dist.log_probs[2].exp().max(), torch.tensor(1.0))


def test_network_policy_passes_temperature() -> None:
    model, _ = model_and_obs(0)
    p = NetworkPolicy(model, (False,) * 6, temperature=0.7)
    p.begin_episode(3)
    assert p.runner is not None and p.runner.temperature == 0.7


# ---------------------------------------------------------------- recorded / CLI
def test_temperature_recorded_in_run_config_and_rows() -> None:
    run = EvalRunConfig(split="val", temperature=0.5)
    assert dataclasses.asdict(run)["temperature"] == 0.5                      # -> meta.run_config
    assert RUN.decode_fields(run) == {"policy": "sample", "temperature": 0.5}  # -> every results row
    assert RUN.decode_fields(EvalRunConfig(split="val", policy="argmax")) == {"policy": "argmax", "temperature": 1.0}
    assert RUN.decode_fields(EvalRunConfig(split="val", policy="random"))["temperature"] is None
    assert dataclasses.asdict(EvalRunConfig(split="val"))["temperature"] == 1.0
    with pytest.raises(ValueError):
        EvalRunConfig(split="val", policy="argmax", temperature=0.5)
    with pytest.raises(ValueError):
        EvalRunConfig(split="val", temperature=0.0)


def test_cli_temperature_default_and_exclusions() -> None:
    a = RUN.parse_args(["--split", "val", "--checkpoint", "c.pt", "--temperature", "0.5"])
    assert a.temperature == 0.5 and a.policy == "sample"
    assert RUN.parse_args(["--split", "val", "--checkpoint", "c.pt"]).temperature == 1.0
    assert RUN.parse_args(["--split", "val", "--sample", "--temperature", "2"]).temperature == 2.0
    for extra in (["--argmax"], ["--random-policy"]):
        with pytest.raises(SystemExit):
            RUN.parse_args(["--split", "val", *extra, "--temperature", "0.5"])
    with pytest.raises(SystemExit):
        RUN.parse_args(["--split", "val", "--temperature", "0"])
    r = REC.parse_args(["--checkpoint", "c.pt", "--out", "o", "--split", "val", "--map-id", "x",
                        "--temperature", "0.25"])
    assert r.temperature == 0.25
    assert REC.parse_args(["--checkpoint", "c.pt", "--out", "o", "--split", "val", "--map-id", "x"]).temperature == 1.0
    with pytest.raises(SystemExit):
        REC.parse_args(["--checkpoint", "c.pt", "--out", "o", "--split", "val", "--map-id", "x",
                        "--argmax", "--temperature", "0.5"])


class _FakeEnv:
    """Stands in for DoomEnv inside evaluate(): no engine; times out after `LENGTH` steps."""
    LENGTH = 3

    def __init__(self, env_config: object, render_mode: str | None = None) -> None:
        self.observation_space, self.action_space = doom_like_spaces()
        self.t, self.rng = 0, np.random.default_rng(0)

    def _obs(self) -> dict[str, np.ndarray]:
        return {k: v[0] for k, v in random_obs(self.observation_space, 1, self.rng).items()}

    def reset(self, seed: int) -> tuple[dict[str, np.ndarray], dict]:
        self.t = 0
        return self._obs(), {"episode_seed": seed}

    def step(self, action: np.ndarray) -> tuple:
        self.t += 1
        done = self.t >= self.LENGTH
        return self._obs(), 0.0, False, done, ({"timeout": True} if done else {})

    def close(self) -> None:
        pass


def test_evaluate_writes_temperature_into_rows_and_meta(tmp_path: "os.PathLike[str]",
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from configs.a0_env import a0_eval_env_config
    from evaluate.checkpoint_io import LoadedCheckpoint
    from evaluate.sensors import canonical_json
    monkeypatch.setattr(RUN, "DoomEnv", _FakeEnv)
    model, _ = model_and_obs(0)
    loaded = LoadedCheckpoint(model=model, spec=model.spec,
                              signature_json=canonical_json(a0_eval_env_config().sensor_signature()),
                              loaded_from="fresh", sha256="none", obs_spec_json="{}", model_config={},
                              run_config_json=None)
    run = EvalRunConfig(split="my_way_home", episodes_per_map=2, temperature=0.5)
    out = os.path.join(str(tmp_path), "ev")
    RUN.evaluate(run, out, loaded)
    with open(os.path.join(out, "results.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2 and all(r["temperature"] == 0.5 and r["policy"] == "sample" for r in rows)
    with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
        assert json.load(f)["meta"]["run_config"]["temperature"] == 0.5


# ---------------------------------------------------------------- ablation script
def ablation() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ablation_table_under_test", ABLATION)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_summary(d: str, run_config: dict) -> None:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"run_config": run_config, "checkpoint": {"sha256": "s"}, "maps": []}}, f)


def test_ablation_manifest_temperature_passthrough_and_protocol(tmp_path: "os.PathLike[str]") -> None:
    A = ablation()
    assert "--temperature" not in A.eval_cmd("c.pt", "sample", 3, 1, "o")
    cmd = A.eval_cmd("c.pt", "sample", 3, 1, "o", 0.5)
    assert cmd[cmd.index("--temperature") + 1] == "0.5" and float(cmd[cmd.index("--temperature") + 1]) == 0.5
    assert A.row_temperature({"decode": "sample"}) == (1.0, None)
    assert A.row_temperature({"decode": "sample", "temperature": 0.5}) == (0.5, None)
    assert A.row_temperature({"decode": "argmax", "temperature": 0.5})[1] is not None
    assert A.row_temperature({"checkpoint": "random", "decode": "sample", "temperature": 0.5})[1] is not None
    assert A.row_temperature({"decode": "sample", "temperature": -1})[1] is not None
    assert A.decode_label("sample", 0.5) == "sample T=0.5" and A.decode_label("sample", 1.0) == "sample"
    base = {"split": "val", "episodes_per_map": 3, "policy": "sample", "base_seed": 1, "seed_stride_map": 10_000,
            "max_maps": None, "exit_reward_fraction": 1.0, "timeout_tics_override": None}
    kw = dict(policy="sample", episodes=3, seed=1, ckpt_sha="s", maps=[])
    old = os.path.join(str(tmp_path), "old")
    write_summary(old, base)                                   # written before the field existed
    assert A.protocol_mismatch(old, **kw) is None
    assert "temperature" in A.protocol_mismatch(old, **kw, temperature=0.5)
    new = os.path.join(str(tmp_path), "new")
    write_summary(new, {**base, "temperature": 0.5})
    assert A.protocol_mismatch(new, **kw, temperature=0.5) is None
    assert "temperature" in A.protocol_mismatch(new, **kw)
