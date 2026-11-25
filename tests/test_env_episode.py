"""Reward accumulation over skipped tics and terminated/truncated semantics.
Each test opens and closes its own DoomGame; never two at once."""
import contextlib

import numpy as np
import pytest

from env import TRAIN_CONFIG, DoomEnv
from env_test_helpers import FD2_ENGINE_TEST_CONFIG as FD2


@contextlib.contextmanager
def make(cfg):
    env = DoomEnv(cfg)
    try:
        yield env
    finally:
        env.close()


def test_reward_accumulated_over_frame_skip_with_use_held():
    cfg = FD2.replace(maps=("MAP01",), engine_rewards=(("living_reward", -1.0),),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=3)
        rewards = []
        for i in range(60):
            a = [i % 3, (i // 3) % 3, i % 7, i % 2, (i // 2) % 2, 0]  # half the steps press USE
            _, r, te, tr, _ = env.step(a)
            rewards.append(r)
            assert not (te or tr)
        assert all(abs(r - (-4.0)) < 1e-6 for r in rewards), rewards[:8]
        assert abs(sum(rewards) - env.game.get_total_reward()) < 1e-6


def test_timeout_is_truncated_not_terminated():
    cfg = FD2.replace(maps=("MAP01",), episode_timeout_tics=41)
    with make(cfg) as env:
        obs, _ = env.reset(seed=1)
        for i in range(30):
            obs, r, te, tr, info = env.step([0, 0, 3, 0, 0, 0])
            if te or tr:
                break
        assert tr and not te
        assert info["timeout"] and not info["dead"]
        assert info["final_obs_fresh"] is True  # wrapper's time limit: observation is current
        assert info["last_step_damage_taken"] == 0.0
        assert env.observation_space.contains(obs)


def test_obs_fresh_flag_every_step_and_stale_after_death():
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        _, info = env.reset(seed=1)
        assert info["final_obs_fresh"] is True
        for _ in range(3):
            prev_obs, _, _, _, info = env.step([0, 0, 3, 0, 0, 0])
            assert info["final_obs_fresh"] is True and "last_step_damage_taken" not in info
        env.game.send_game_command("kill")
        for _ in range(10):
            obs, r, te, tr, info = env.step([0, 0, 3, 0, 0, 0])
            if te or tr:
                break
            prev_obs = obs
        assert te and info["dead"]
        assert info["final_obs_fresh"] is False
        for k in obs:  # the stale observation is the previous one, repeated
            np.testing.assert_array_equal(obs[k], prev_obs[k])
        # console `kill` is not counted in DAMAGE_TAKEN by the engine (probe
        # `damage_after_end`), so 0.0 is the correct value here
        assert info["last_step_damage_taken"] == 0.0
        assert "final_obs_fresh" not in obs and "last_step_damage_taken" not in obs


def test_last_step_damage_reported_on_lethal_real_damage():
    """Two barrels exploded in front of the player: the second one kills."""
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=1)
        for _ in range(3):
            env.step([0, 0, 3, 0, 0, 0])
        te = tr = False
        for barrel in range(4):
            env.game.send_game_command("summon ExplosiveBarrel 48")
            for _ in range(12):
                obs, r, te, tr, info = env.step([0, 0, 3, 1, 0, 0])
                if te or tr:
                    break
            if te or tr:
                break
        assert te and info["dead"], "player did not die from barrels"
        assert info["final_obs_fresh"] is False
        assert info["last_step_damage_taken"] > 0, info


def test_death_is_terminated():
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=1)
        env.step([0, 0, 3, 0, 0, 0])
        env.game.send_game_command("kill")  # console suicide, TRAIN-only cheat
        for _ in range(10):
            obs, r, te, tr, info = env.step([0, 0, 3, 0, 0, 0])
            if te or tr:
                break
        assert te and not tr
        assert info["dead"]


def test_key_pickup_reaches_info_only():
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=1)
        env.step([0, 0, 3, 0, 0, 0])
        env.step([0, 0, 3, 0, 0, 0])
        env.game.send_game_command("summon BlueCard 64")
        obs, *_ = env.step([0, 0, 3, 0, 0, 0])  # stand still: key visible 64 units ahead
        key_rows = obs["entity_mask"] & (obs["entity_type"] == 7)
        assert key_rows.any(), "summoned blue key not in entity set"
        d0 = float(obs["entities"][key_rows][0, 4])
        # 64 map units at ~8 units per depth level ~= 8/255 ~= 0.03; the wall behind is farther
        assert 0.01 < d0 < 0.06, d0
        # The summoned key spawns close and low in the view; one decision later it is
        # fully under the weapon sprite. Probe R4.2: a label with no visible pixel gives
        # no row (before, a d = NO_READING row was emitted here). Any key row that does
        # appear must be a real reading.
        depths = [d0]
        for _ in range(12):
            obs, r, te, tr, info = env.step([1, 0, 3, 0, 0, 0])
            rows = obs["entity_mask"] & (obs["entity_type"] == 7)
            if rows.any():
                depths.append(float(obs["entities"][rows][0, 4]))
            if info["keys_held"]["blue"]:
                break
        assert all(d > 0 for d in depths), depths
        assert info["keys_held"] == {"blue": True, "yellow": False, "red": False}
        assert "keys_held" not in obs


def test_time_limit_returns_live_final_state():
    with make(FD2.replace(maps=("MAP01",), episode_timeout_tics=41)) as env:
        env.reset(seed=1)
        for _ in range(30):
            obs, r, te, tr, info = env.step([1, 0, 4, 0, 0, 0])
            if te or tr:
                break
        assert tr and not te and info["timeout"]
        assert info["tic"] >= 41  # observation of the final decision, not the previous one


def test_key_picked_up_on_final_decision_at_time_limit():
    """Item 7 (final step): pickup during the decision that hits the time limit."""
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",),
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=1)
        for _ in range(3):
            env.step([0, 0, 3, 0, 0, 0])
        # Doom only checks touches when the player moves: place the key just ahead
        # and run onto it during the decision that hits the time limit.
        env.game.send_game_command("summon BlueCard 24")
        obs, r, te, tr, info = env.step([1, 0, 3, 0, 0, 1])  # accelerating from rest
        assert not info["keys_held"]["blue"]  # measured: pickup lands in the 2nd decision
        env._timeout_tics = env.game.get_episode_time() + 4  # next decision ends the episode
        obs, r, te, tr, info = env.step([1, 0, 3, 0, 0, 1])
        assert tr and not te
        assert info["keys_held"]["blue"] is True


def test_bad_timer_override_fails_before_engine_starts(monkeypatch):
    from env import doom_env, wintimer
    started = []
    monkeypatch.setattr(doom_env.DoomEnv, "_make_game", lambda self: started.append(1))
    monkeypatch.setattr(wintimer, "_winmm", lambda: (_ for _ in ()).throw(AssertionError("no")))
    monkeypatch.setenv("DOOMSDAY_WIN_TIMER_MS", "fast")
    with pytest.raises(ValueError):
        DoomEnv(TRAIN_CONFIG)
    assert started == []


def test_screen_disabled_removes_key_and_flags_modality():
    with make(FD2.replace(maps=("MAP01",), screen_enabled=False)) as env:
        assert "screen" not in env.observation_space.spaces
        obs, _ = env.reset(seed=0)
        for _ in range(3):
            obs, *_ = env.step(env.action_space.sample())
            assert "screen" not in obs
            assert obs["modality"].tolist() == [1.0, 0.0, 1.0]
            assert env.observation_space.contains(obs)


def test_held_use_opens_door_once_and_keeps_it_open():
    """ADR 0006 ruling 2 on the probed MAP01 door (sector 27, x 704-720)."""
    cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",), sectors_info=True,
                               episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=0)
        for _ in range(3):  # past the 7-tic turn dead time
            env.step([0, 0, 3, 0, 0, 0])
        env.game.send_game_command("warp 670 -160")  # facing east (spawn angle 0)
        env.step([0, 0, 3, 0, 0, 0])
        gaps = []
        for _ in range(10):  # USE held on 10 consecutive decisions
            env.step([0, 0, 3, 0, 1, 0])
            sec = env.game.get_state().sectors[27]
            gaps.append(round(sec.ceiling_height - sec.floor_height))
        assert max(gaps) >= 90 and gaps[-1] == max(gaps), gaps  # opened, not reversed


def test_locked_door_message_does_not_set_keys_held():
    """Item 7 regression with real engine text: freedoom2 MAP02 line 215 (yellow DR)."""
    cfg = FD2.replace(maps=("MAP02",), game_args=("+sv_cheats 1",),
                               sectors_info=True, episode_timeout_tics=100000)
    with make(cfg) as env:
        env.reset(seed=0)
        for _ in range(3):
            env.step([0, 0, 3, 0, 0, 0])
        env.game.send_game_command("warp 224 -320")
        env.step([0, 0, 3, 0, 0, 0])
        for turn in (0, 0, 2, 2):  # spawn heading 90 -> ~178 (bins -10,-10,-1,-1 deg/tic)
            env.step([0, 0, turn, 0, 0, 0])
        texts = []
        for k in range(4):
            obs, r, te, tr, info = env.step([0, 0, 3, 0, 1 if k < 2 else 0, 0])
            texts.append(env.game.get_state().notifications_buffer)
        assert any("key needed" in (t or "") for t in texts), texts  # message really occurred
        assert info["keys_held"] == {"blue": False, "yellow": False, "red": False}


def test_timer_pairs_once_on_real_env(monkeypatch):
    fake_calls = {"begin": 0, "end": 0}

    class Fake:
        def timeBeginPeriod(self, p):
            fake_calls["begin"] += 1
            return 0

        def timeEndPeriod(self, p):
            fake_calls["end"] += 1
            return 0

    from env import wintimer
    monkeypatch.setattr(wintimer, "_winmm", lambda: Fake())
    monkeypatch.delenv("DOOMSDAY_WIN_TIMER_MS", raising=False)
    env = DoomEnv(FD2.replace(maps=("MAP01",)))
    try:
        env.reset(seed=0)
        env.reset(seed=1)
        env.step([0, 0, 3, 0, 0, 0])
    finally:
        env.close()
    env.close()
    assert fake_calls == {"begin": 1, "end": 1}
    env2 = DoomEnv(FD2.replace(maps=("MAP01",), win_timer_ms=None))
    env2.close()
    assert fake_calls == {"begin": 1, "end": 1}


@pytest.mark.parametrize("timeout", [41, 42, 43, 44])
def test_episode_ending_with_use_held_is_truncated(timeout):
    """USE on every step; timeouts that end mid-decision still give `truncated`."""
    with make(FD2.replace(maps=("MAP01",), episode_timeout_tics=timeout)) as env:
        obs, _ = env.reset(seed=2)
        for _ in range(40):
            obs, r, te, tr, info = env.step([0, 0, 3, 0, 1, 0])
            if te or tr:
                break
        assert tr and not te
        assert env.observation_space.contains(obs)


def test_viz_nocheat_fallback():
    cfg = FD2.replace(maps=("MAP01",), game_args=("+viz_nocheat 1",))
    with make(cfg) as env:
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            obs, *_ = env.step([1, 0, 4, 0, 0, 0])
        assert obs["modality"].tolist() == [0.0, 1.0, 0.0]
        assert env.observation_space.contains(obs)
        assert obs["self"][0] > 0.0          # health survives
        assert (obs["self"][24:28] == 0).all()  # ego-motion unavailable
        assert not obs["entity_mask"].any() and (obs["rays"] == 0).all()
        assert obs["screen"].any()


def test_runtime_seed_is_applied():
    with make(FD2.replace(maps=("MAP01",))) as env:
        _, info = env.reset(seed=5)
        assert env.game.get_seed() == info["episode_seed"]


def test_prev_action_is_last_action():
    with make(FD2.replace(maps=("MAP01",))) as env:
        obs, _ = env.reset(seed=0)
        assert (obs["prev_action"] == 0).all()
        a = np.array([1, 2, 5, 1, 0, 1])
        obs, *_ = env.step(a)
        assert (obs["prev_action"] == a).all()


def test_my_way_home_goal_rewarded_once_and_ends_as_exit():
    """Bundled reward (probe `mwh_goal`): -0.0001 per tic, +1.0 on the armour pickup
    (ACS), then Exit_Normal. Through DoomEnv the goal must pay once, not twice."""
    cfg = TRAIN_CONFIG.replace(game_args=("+sv_cheats 1",), objects_info=True)
    with make(cfg) as env:
        env.reset(seed=0)
        armour = [o for o in env.game.get_state().objects if o.name == "GreenArmor"][0]
        env.game.send_game_command(f"warp {armour.position_x:.0f} {armour.position_y:.0f}")
        rewards, te, tr = [], False, False
        for _ in range(6):
            obs, r, te, tr, info = env.step([1, 0, 3, 0, 0, 0])  # moving: touch needs motion
            rewards.append(r)
            if te or tr:
                break
        assert te and not tr and info["level_exit"] and not info["dead"]
        assert info["final_obs_fresh"] is False
        assert max(rewards) < 1.0 + 1e-6 and max(rewards) > 0.99, rewards  # one +1, not two
        assert abs(sum(rewards) - env.game.get_total_reward()) < 1e-6
