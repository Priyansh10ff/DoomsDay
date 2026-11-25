"""Aux label construction and trainer-side telemetry masking (ADR 0006)."""
import numpy as np
import pytest
import torch

from a0_helpers import doom_like_spaces, random_obs
from agent.tensorize import to_tensors
from configs.model_config import ModelConfig
from configs.train_config import AuxConfig
from train import aux_labels as AL
from train.telemetry_mask import EpisodeDropout, mask_telemetry


def test_keys_from_infos_valid_only_when_present() -> None:
    k, ok = AL.keys_from_infos([{"keys_held": {"blue": True, "red": True}}, {}, {"keys_held": {}}])
    assert ok.tolist() == [True, False, True]
    assert k[0].tolist() == [1.0, 0.0, 1.0] and k[2].tolist() == [0.0, 0.0, 0.0]


def test_visible_monsters_counts_only_masked_monsters_capped() -> None:
    types = np.array([[1, 1, 2, 1], [0, 0, 0, 0]])
    mask = np.array([[True, True, True, False], [False] * 4])
    out = AL.visible_monsters(types, mask, monster_type_id=1, cap=8)
    assert out.tolist() == pytest.approx([2 / 8, 0.0])
    many = AL.visible_monsters(np.ones((1, 24), np.int64), np.ones((1, 24), bool), 1, cap=8)
    assert many.tolist() == pytest.approx([1.0])


def test_coupled_indices_match_env_tables() -> None:
    obs_mod = pytest.importorskip("env.observation")  # imports vizdoom, does not start it
    aux = AuxConfig()
    assert obs_mod.ENTITY_TYPES.index("monster") == aux.monster_type_id
    zeros = np.zeros(obs_mod.N_WEAPONS)
    base = obs_mod.self_vector(50, 0, 10, 0.0, 2, zeros, np.zeros(4, np.float32))
    hit = obs_mod.self_vector(50, 0, 10, 40.0, 2, zeros, np.zeros(4, np.float32))
    changed = np.flatnonzero(base != hit)
    assert changed.tolist() == [aux.self_damage_index]
    assert ModelConfig().self_ego_dims == 4 and obs_mod.SELF_DIM - 4 == 24  # ego is the tail of `self`


def test_damage_events_ended_use_final_info() -> None:
    nxt = np.zeros((5, 28), np.float32)
    nxt[0, 3] = 0.5
    ended = np.array([False, True, True, True, True])
    fs = [None, np.full(28, 0.2, np.float32), np.full(28, 0.3, np.float32), None, None]
    fresh = np.array([False, False, True, False, False])
    infos = [None, {}, {}, {"dead": True, "last_step_damage_taken": 0.0},
             {"dead": False, "last_step_damage_taken": 7.0}]
    ev, ok = AL.damage_events(nxt, ended, fs, fresh, infos, damage_index=3)
    # env 1: stale and no info -> invalid; env 2: fresh obs fallback; 3: death; 4: info damage
    assert ok.tolist() == [True, False, True, True, True]
    assert ev.tolist() == pytest.approx([0.5, 0.0, 0.3, 1.0, 7.0])


def test_damage_window_targets() -> None:
    # one env, T=6, k=3, damage on transition 2, episode ends at transition 3
    ev = np.array([[0], [0], [1], [0], [0], [0]], np.float32)
    ok = np.ones((6, 1), bool)
    ended = np.array([[0], [0], [0], [1], [0], [0]], bool)
    tgt, valid = AL.damage_window_targets(ev, ok, ended, k=3)
    assert tgt[:, 0].tolist() == [1, 1, 1, 0, 0, 0]
    # t=3: window stops at the episode end -> complete; t=4,5: window runs past rollout end
    assert valid[:, 0].tolist() == [True, True, True, True, False, False]
    ok[2, 0] = False
    _, valid2 = AL.damage_window_targets(ev, ok, ended, k=3)
    assert valid2[:, 0].tolist() == [False, False, False, True, False, False]


def test_mask_telemetry_only_dropped_rows() -> None:
    cfg = ModelConfig()
    space, _ = doom_like_spaces()
    obs = to_tensors(random_obs(space, 4, np.random.default_rng(0)))
    obs["entity_mask"][:] = True
    dropped = torch.tensor([False, True, False, True])
    out = mask_telemetry(obs, dropped, cfg)
    for i in (1, 3):
        assert torch.all(out["rays"][i] == 0) and torch.all(out["entities"][i] == 0)
        assert not bool(out["entity_mask"][i].any())
        assert torch.all(out["self"][i, -cfg.self_ego_dims:] == 0)
        torch.testing.assert_close(out["self"][i, :-cfg.self_ego_dims], obs["self"][i, :-cfg.self_ego_dims])
        assert out["modality"][i, cfg.modality_telemetry_index] == 0 and out["modality"][i, cfg.modality_ego_index] == 0
        assert out["modality"][i, 1] == obs["modality"][i, 1]
    for i in (0, 2):
        for k in obs:
            assert torch.equal(out[k][i], obs[k][i])
    # original untouched
    assert bool(obs["entity_mask"].all())


def test_mask_telemetry_sequence_shape() -> None:
    space, _ = doom_like_spaces()
    obs = to_tensors(random_obs(space, 3, np.random.default_rng(1), lead=(5,)))
    d = torch.zeros(5, 3, dtype=torch.bool)
    d[2, 1] = True
    out = mask_telemetry(obs, d, ModelConfig())
    assert torch.all(out["rays"][2, 1] == 0) and torch.equal(out["rays"][1, 1], obs["rays"][1, 1])


def test_episode_dropout_seeded_and_redrawn_only_at_start() -> None:
    a, b = EpisodeDropout(6, 0.5, seed=3), EpisodeDropout(6, 0.5, seed=3)
    s = np.ones(6, bool)
    assert np.array_equal(a.on_episode_start(s), b.on_episode_start(s))
    before = a.dropped.copy()
    after = a.on_episode_start(np.zeros(6, bool))
    assert np.array_equal(before, after)
