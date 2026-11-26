"""Demo dashboard (tools/dashboard*.py): composer on synthetic arrays, the read-only
model probe (same actions as the plain evaluation path, attention weights recomputed
exactly), the pass-through env proxy, and the streaming writer. No engine is started."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import a0_helpers  # noqa: F401  (caps torch threads, puts the repo on sys.path)
from agent.inference import PolicyRunner
from agent.model import build_agent
from agent.obs_spec import EgoMapObsSpec
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig
from env.egomap import CH_DOOR, CH_FREE, CH_KEY, CH_LOCKED, CH_VISITED, CH_WALL
from evaluate.episode import run_episode
from evaluate.policies import NetworkPolicy
from tools.dashboard import DashboardEnv, door_columns, frame_data, make_policy
from tools.dashboard_compose import FrameData, compose, egomap_rgb
from tools.dashboard_config import DashboardConfig
from tools.dashboard_probe import ATTN_MATCH_TOL, ModelProbe
from tools.dashboard_writer import EpisodeWriter

CFG = DashboardConfig()
HEADS = (3, 3, 7, 2, 2, 2)
TURN_BINS = (-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0)
N_ENT, RAY_BANDS = 24, 3


def spec(channels: int = 5) -> EgoMapObsSpec:
    return EgoMapObsSpec(ray_channels=RAY_BANDS + 1, ray_width=80, max_entities=N_ENT, entity_feat_dim=8,
                         n_entity_types=14, self_dim=34, modality_dim=3, head_sizes=HEADS,
                         egomap_shape=(channels, 32, 32))


def synth_obs(rng: np.random.Generator, channels: int = 5, n_visible: int = 3) -> dict[str, np.ndarray]:
    mask = np.zeros(N_ENT, bool)
    mask[:n_visible] = True
    ents = np.zeros((N_ENT, 8), np.float32)
    ents[:n_visible] = rng.uniform(-0.6, 0.6, (n_visible, 8)).astype(np.float32)
    ents[:n_visible, 2:4] = 0.1
    return {
        "rays": rng.random((RAY_BANDS + 1, 80), dtype=np.float32),
        "entities": ents,
        "entity_type": np.where(mask, rng.integers(1, 14, N_ENT), 0).astype(np.int64),
        "entity_mask": mask,
        "self": rng.uniform(0, 1, 34).astype(np.float32),
        "prev_action": rng.integers(0, 2, 6).astype(np.int64),
        "modality": np.ones(3, np.float32),
        "egomap": (rng.random((channels, 32, 32)) < 0.2).astype(np.uint8),
    }


def synth_frame(rng: np.random.Generator, channels: int = 5, goal: bool = False, banner: str | None = None,
                screen: bool = True) -> FrameData:
    o = synth_obs(rng, channels)
    attn = np.zeros(N_ENT)
    attn[:3] = (0.6, 0.25, 0.05)
    return FrameData(
        screen=rng.integers(0, 256, (120, 160, 3)).astype(np.uint8) if screen else None,
        door_cols=(rng.random(80) < 0.3).astype(np.float32), egomap=o["egomap"], entities=o["entities"],
        entity_type=o["entity_type"], entity_mask=o["entity_mask"], attn=attn, attn_empty=0.1,
        values=list(rng.normal(size=50)), action=np.array([1, 2, 5, 1, 0, 1]),
        action_probs=np.array([0.9, 0.5, 0.3, 0.7, 0.99, 0.6]), turn_bins=TURN_BINS, title="test",
        step=7, health=87.0, keys_held={"blue": True}, goal=("getting key", 0.8) if goal else None, banner=banner)


# ---------------------------------------------------------------- composer
@pytest.mark.parametrize("channels,goal,banner,screen", [(5, False, None, True), (6, True, None, True),
                                                         (5, True, "DIED", False)])
def test_compose_shape_and_dtype(channels: int, goal: bool, banner: str | None, screen: bool) -> None:
    img = compose(synth_frame(np.random.default_rng(0), channels, goal, banner, screen), CFG)
    assert img.shape == (720, 1280, 3) and img.dtype == np.uint8


def test_compose_without_egomap_door_or_values() -> None:
    d = synth_frame(np.random.default_rng(1))
    d.egomap, d.door_cols, d.values, d.attn, d.attn_empty, d.action_probs = None, None, [], None, None, None
    assert compose(d, CFG).shape == (720, 1280, 3)


def test_compose_deterministic() -> None:
    a = compose(synth_frame(np.random.default_rng(3), goal=True), CFG)
    b = compose(synth_frame(np.random.default_rng(3), goal=True), CFG)
    assert np.array_equal(a, b)


def test_goal_panel_only_when_goal_present() -> None:
    d = synth_frame(np.random.default_rng(4))
    without = compose(d, CFG)
    d.goal = ("to exit", 0.7)
    assert not np.array_equal(without, compose(d, CFG))


def test_attention_intensity_follows_weight() -> None:
    """Two equal boxes on a black screen: the heavier-weighted one is brighter."""
    d = synth_frame(np.random.default_rng(5))
    d.screen = np.zeros((120, 160, 3), np.uint8)
    d.entity_mask = np.zeros(N_ENT, bool)
    d.entity_mask[:2] = True
    d.entities = np.zeros((N_ENT, 8), np.float32)
    d.entities[0, :4] = (-0.5, 0.2, 0.15, 0.2)
    d.entities[1, :4] = (0.5, 0.2, 0.15, 0.2)
    d.attn = np.zeros(N_ENT)
    d.attn[:2] = (0.9, 0.02)
    img = compose(d, CFG).astype(np.int64)
    s, x0, y0 = CFG.screen_scale, CFG.margin, CFG.header_h + 8

    def box_sum(u: float) -> int:
        cx, cy = int(x0 + (u + 1) / 2 * 160 * s), int(y0 + 0.6 * 120 * s)
        return int(img[cy - 20:cy + 20, cx - 20:cx + 20].sum())
    assert box_sum(-0.5) > 2 * box_sum(0.5) > 0


def test_egomap_rgb_colours_and_agent_arrow() -> None:
    em = np.zeros((6, 32, 32), np.uint8)
    em[CH_FREE, 2, 2] = 1
    em[CH_WALL, 3, 3] = 1
    em[CH_VISITED, 4, 4] = 1
    em[CH_DOOR, 5, 5] = 1
    em[CH_KEY, 6, 6] = 1
    em[CH_LOCKED, 7, 7] = 1
    rgb = egomap_rgb(em, CFG)
    k, c = CFG.egomap_cell_px, CFG.egomap
    assert rgb.shape == (32 * k, 32 * k, 3)
    at = lambda r, col: tuple(int(v) for v in rgb[r * k + k // 2, col * k + k // 2])  # noqa: E731
    assert at(0, 0) == c.unknown and at(2, 2) == c.free and at(3, 3) == c.wall and at(4, 4) == c.visited
    assert at(5, 5) == c.door and at(6, 6) == c.key and at(7, 7) == c.locked
    assert at(16, 16) == c.agent


def test_door_columns_only_with_v2_row() -> None:
    rays = np.zeros((4, 80), np.float32)
    rays[3, 10] = 1.0
    assert door_columns(rays, 3)[10] == 1.0
    assert door_columns(rays[:3], 3) is None


# ---------------------------------------------------------------- probe
def model(channels: int = 5, seed: int = 0) -> Any:
    m = build_agent(spec(channels), ModelConfig(), seed=seed)
    # non-zero pooling query / tokens so the attention is not uniform
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed)
        m.entities.pool_query.copy_(torch.randn(m.entities.pool_query.shape, generator=g))
    return m.eval()


def test_probe_actions_and_hidden_state_identical_to_plain_runner() -> None:
    """Same seed, same observations, including an episode reset mid-sequence: the probe
    changes neither the actions nor the hidden state (and resets it where the plain
    runner does)."""
    m = model()
    rng = np.random.default_rng(7)
    obs_seq = [synth_obs(rng) for _ in range(8)]
    starts = [1, 0, 0, 0, 1, 0, 0, 0]
    plain = PolicyRunner(m, 1, (False,) * 6, seed=11)
    probe = ModelProbe(m)
    probed = PolicyRunner(probe, 1, (False,) * 6, seed=11)  # type: ignore[arg-type]
    for o, s in zip(obs_seq, starts):
        b = {k: v[None] for k, v in o.items()}
        es = np.array([float(s)], np.float32)
        assert np.array_equal(plain.act(b, es), probed.act(b, es))
        assert torch.equal(plain.h, probed.h)
        if s:  # hidden state was zeroed before this step: equals a fresh runner's first step
            fresh = PolicyRunner(m, 1, (False,) * 6, seed=0)
            fresh.act(b, es)
            assert torch.allclose(fresh.h, probed.h)
    probe.close()


def test_probe_attention_weights_match_model() -> None:
    m = model(seed=3)
    probe = ModelProbe(m)
    o = synth_obs(np.random.default_rng(9), n_visible=4)
    with torch.no_grad():
        probe.step(to_tensors({k: v[None] for k, v in o.items()}), m.initial_state(1), torch.ones(1))
    cap = probe.last
    assert cap is not None and cap.attn is not None and cap.attn.shape == (N_ENT,)
    assert cap.attn_max_abs_diff is not None and cap.attn_max_abs_diff <= ATTN_MATCH_TOL
    assert np.all(cap.attn[4:] == 0.0) and np.all(cap.attn[:4] > 0.0)
    assert abs(cap.attn.sum() + cap.attn_empty - 1.0) < 1e-5
    assert cap.obs["egomap"].shape == (5, 32, 32) and cap.goal_logits is None
    probe.close()
    assert len(m.entities.pool._forward_hooks) == 0


def test_probe_unverified_attention_is_not_shown(monkeypatch: Any) -> None:
    import tools.dashboard_probe as P
    monkeypatch.setattr(P, "ATTN_MATCH_TOL", -1.0)          # every check fails
    m = model(seed=2)
    o = to_tensors({k: v[None] for k, v in synth_obs(np.random.default_rng(0)).items()})
    probe = ModelProbe(m)
    with torch.no_grad():
        probe.step(o, m.initial_state(1), torch.ones(1))
    assert probe.last.attn is None and probe.attn_mismatches == 1
    probe.close()
    strict = ModelProbe(m, strict=True)
    with pytest.raises(AssertionError), torch.no_grad():
        strict.step(o, m.initial_state(1), torch.ones(1))
    strict.close()


def test_probe_is_deterministic() -> None:
    caps = []
    for _ in range(2):
        m = model(seed=4)
        probe = ModelProbe(m)
        o = synth_obs(np.random.default_rng(1))
        with torch.no_grad():
            probe.step(to_tensors({k: v[None] for k, v in o.items()}), m.initial_state(1), torch.ones(1))
        caps.append(probe.last)
        probe.close()
    assert caps[0].value == caps[1].value and np.array_equal(caps[0].attn, caps[1].attn)


def test_probe_temperature_display_only_output_untouched() -> None:
    """The probe returns the model's own logits; T only changes the displayed probabilities."""
    m = model(seed=5)
    o = {k: v[None] for k, v in synth_obs(np.random.default_rng(2)).items()}
    with torch.no_grad():
        out, _ = m.step(to_tensors(o), m.initial_state(1), torch.ones(1))
        probe = ModelProbe(m, display_temperature=2.0)
        pout, _ = probe.step(to_tensors(o), m.initial_state(1), torch.ones(1))
    for lg, plg in zip(out.logits, pout.logits):
        assert torch.equal(plg, lg)
    assert np.allclose(probe.last.probs[2], torch.softmax(out.logits[2][0] / 2.0, -1).numpy(), atol=1e-6)
    with pytest.raises(ValueError):
        ModelProbe(m, display_temperature=0.0)
    probe.close()


def test_tempered_actions_identical_to_plain_network_policy() -> None:
    m = model(seed=10)
    rng = np.random.default_rng(12)
    obs_seq = [synth_obs(rng) for _ in range(6)]
    plain = NetworkPolicy(m, (False,) * 6, temperature=1.7)
    dash, probe = make_policy(m, (False,) * 6, "cpu", 1.7)
    plain.begin_episode(5)
    dash.begin_episode(5)
    for i, o in enumerate(obs_seq):
        assert np.array_equal(plain.act(o, i == 0), dash.act(o, i == 0))
    probe.close()


def test_goal_names_from_goal_modes_in_id_order() -> None:
    from configs.goal_modes import GOAL_MODES
    from tools.dashboard import goal_names
    names = goal_names()
    assert len(names) == len(GOAL_MODES) and names[0] == "exploring" and names[1] == "getting key"


def test_probe_reads_goal_logits_when_model_has_goal_head() -> None:
    import dataclasses
    m = build_agent(spec(), dataclasses.replace(ModelConfig(), goal_head=True), seed=0).eval()
    probe = ModelProbe(m)
    o = synth_obs(np.random.default_rng(4))
    with torch.no_grad():
        probe.step(to_tensors({k: v[None] for k, v in o.items()}), m.initial_state(1), torch.ones(1))
    assert probe.last is not None and probe.last.goal_logits is not None
    assert probe.last.goal_logits.shape == (m.goal_head.out_features,)
    probe.close()


# ---------------------------------------------------------------- env proxy
class FakeRenderEnv:
    """Doom-shaped fake with render(); terminates after `length` steps (no frame then)."""

    def __init__(self, length: int = 6) -> None:
        self.render_mode = "rgb_array"
        self.cfg = SimpleNamespace(turn_bins=TURN_BINS, ray_bands=((0, 1),) * RAY_BANDS)
        self.length, self.t, self.rng, self.terminal = length, 0, np.random.default_rng(0), False

    def reset(self, *, seed: int | None = None) -> tuple[dict, dict]:
        self.rng, self.t, self.terminal = np.random.default_rng(seed), 0, False
        return synth_obs(self.rng), {"health": 100.0, "keys_held": {}}

    def step(self, action: np.ndarray) -> tuple:
        self.t += 1
        self.terminal = self.t >= self.length
        info: dict[str, Any] = {"health": 100.0 - self.t, "keys_held": {"red": True}}
        if self.terminal:
            info.update(dead=True, level_exit=False, timeout=False)
        return synth_obs(self.rng), 0.0, self.terminal, False, info

    def render(self) -> np.ndarray | None:
        return None if self.terminal else self.rng.integers(0, 256, (120, 160, 3)).astype(np.uint8)


def test_dashboard_env_is_pass_through_and_draws_every_decision() -> None:
    m = model(seed=6)
    plain_policy = NetworkPolicy(m, (False,) * 6)
    plain_actions: list[np.ndarray] = []

    class Spy(FakeRenderEnv):
        def step(self, action: np.ndarray) -> tuple:
            plain_actions.append(np.array(action))
            return super().step(action)
    run_episode(Spy(), plain_policy, 42, None)

    policy, probe = make_policy(m, (False,) * 6, "cpu", 1.0)
    drawn: list[FrameData] = []
    env = FakeRenderEnv()
    dash = DashboardEnv(env, probe, lambda img, d: drawn.append(d), CFG, "t", ("a",))  # type: ignore[arg-type]
    rec = run_episode(dash, policy, 42, None)
    probe.close()
    assert rec["length"] == len(drawn) == dash.decisions == len(plain_actions)
    assert all(np.array_equal(d.action, a) for d, a in zip(drawn, plain_actions))
    assert all(d.screen is not None and d.screen.shape == (120, 160, 3) for d in drawn)
    assert drawn[0].health == 100.0 and drawn[1].keys_held == {"red": True}
    assert len(drawn[-1].values) == len(drawn) and dash.attn_steps == len(drawn)


def test_frame_data_goal_head_names() -> None:
    m = model(seed=8)
    probe = ModelProbe(m)
    o = synth_obs(np.random.default_rng(3))
    with torch.no_grad():
        probe.step(to_tensors({k: v[None] for k, v in o.items()}), m.initial_state(1), torch.ones(1))
    cap = probe.last
    d = frame_data(cap, None, {}, np.zeros(6, np.int64), [0.0], TURN_BINS, RAY_BANDS, "t", 0, ("x", "y"))
    assert d.goal is None
    cap.goal_logits = np.array([0.0, 3.0])
    d = frame_data(cap, None, {}, np.zeros(6, np.int64), [0.0], TURN_BINS, RAY_BANDS, "t", 0, ("x", "y"))
    assert d.goal is not None and d.goal[0] == "y" and d.goal[1] > 0.9
    probe.close()


# ---------------------------------------------------------------- writer
def test_episode_writer_streams_mp4_and_gif(tmp_path: Any) -> None:
    stem = os.path.join(str(tmp_path), "ep")
    w = EpisodeWriter(stem, 8.75, CFG)
    rng = np.random.default_rng(0)
    for _ in range(5):
        w.add(compose(synth_frame(rng), CFG))
    out = w.close()
    assert out["frames"] == "5" and os.path.getsize(out["gif"]) > 0
    if "mp4" in out:
        assert os.path.getsize(out["mp4"]) > 0
