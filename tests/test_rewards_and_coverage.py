"""ADR 0010: each reward term in isolation, the scaled sum, logging, the coverage
bonus, its per-episode reset and annealed coefficient, no double counting, and the
integration point. Synthetic `info` only; no engine is started."""
import dataclasses
import os
import pickle
import re
import subprocess
import sys
from functools import partial

import numpy as np
import pytest

from reward_test_helpers import CELL, COV, FULL, ROOT, ScriptEnv, end, episode, mk, paid, play

from configs.reward_config import (CAPPED_TERMS, FULL_LEVEL_COVERAGE, FULL_LEVEL_REWARD, KNOB_PAYS,
                                   MY_WAY_HOME_COVERAGE, MY_WAY_HOME_REWARD, NO_COVERAGE,
                                   PASSTHROUGH_REWARD, TERMS, CoverageConfig, RewardConfig)
from intrinsic.coverage import EpisodicCoverage
from intrinsic.learned_novelty import LearnedEpisodicNovelty
from intrinsic.schedule import LinearAnneal
from rewards.factory import ShapedEnvFactory, per_env_decisions
from rewards.terms import ExtrinsicReward, MissingInfoField
from rewards.validate import RewardConfigError, check_reward_setup
from rewards.wrapper import INFO_EPISODE, INFO_STEP, ShapedRewardEnv

NOCOV = NO_COVERAGE
ZERO = dataclasses.replace(PASSTHROUGH_REWARD, engine_reward_scale=0.0)


def only(term: str, **kw) -> RewardConfig:
    """Every scale 0 except `term` (at its default), so the term is tested alone."""
    return dataclasses.replace(ZERO, **{term: FULL.scale(term)}, **kw)


# ---------------------------------------------------------------- terms in isolation
def test_term_list_is_the_nine_of_the_brief():
    assert TERMS == ("level_exit", "keycard", "kill", "damage_taken", "health", "ammo", "armor",
                     "death", "time")
    assert all(hasattr(FULL_LEVEL_REWARD, t) for t in TERMS)


def test_level_exit_pays_once_on_exit_only():
    r, out = play([mk(), mk(), end(mk(), "exit")], only("level_exit"), NOCOV)
    assert r == [0.0, 1.0] and paid(out, "level_exit") == 1.0
    r, _ = play([mk(), end(mk(), "timeout")], only("level_exit"), NOCOV)
    assert r == [0.0]
    r, _ = play([mk(), end(mk(), "dead")], only("level_exit"), NOCOV)
    assert r == [0.0]


def test_keycard_pays_per_new_colour():
    blue = {"blue": True, "yellow": False, "red": False}
    both = {"blue": True, "yellow": False, "red": True}
    r, _ = play([mk(), mk(keys=blue), mk(keys=blue), mk(keys=both), end(mk(keys=both), "timeout")],
                only("keycard"), NOCOV)
    assert r == pytest.approx([0.1, 0.0, 0.1, 0.0])


def test_kill_pays_per_kill():
    r, _ = play([mk(), mk(kills=1), mk(kills=1), mk(kills=3), end(mk(kills=3), "timeout")],
                only("kill"), NOCOV)
    assert r == pytest.approx([0.02, 0.0, 0.04, 0.0])


def test_damage_taken_is_negative_per_point():
    r, _ = play([mk(), mk(damage=10), mk(damage=10), mk(damage=35), end(mk(damage=35), "timeout")],
                only("damage_taken"), NOCOV)
    assert r == pytest.approx([-0.03, 0.0, -0.075, 0.0])


def test_health_armor_ammo_pay_gains_only():
    r, _ = play([mk(health=50), mk(health=75), mk(health=40), end(mk(health=40), "timeout")],
                only("health"), NOCOV)
    assert r == pytest.approx([0.05, 0.0, 0.0])
    r, _ = play([mk(armor=0), mk(armor=100), mk(armor=60), end(mk(armor=60), "timeout")],
                only("armor"), NOCOV)
    assert r == pytest.approx([0.1, 0.0, 0.0])
    r, _ = play([mk(ammo=(50, 0)), mk(ammo=(60, 0)), mk(ammo=(55, 4)), end(mk(ammo=(55, 4)), "timeout")],
                only("ammo"), NOCOV)
    assert r == pytest.approx([0.005, 0.002, 0.0])  # +10 bullets = 1 unit; +4 shells = 0.4 unit


def test_death_and_time():
    r, _ = play([mk(), mk(), end(mk(), "dead")], only("death"), NOCOV)
    assert r == pytest.approx([0.0, -0.75])
    r, _ = play([mk(), mk(), end(mk(), "timeout")], only("time"), NOCOV)
    assert r == pytest.approx([-0.0002, -0.0002])


# ---------------------------------------------------------------- scaled sum and logging
def test_total_is_the_scaled_sum_and_every_component_is_logged():
    blue = {"blue": True, "yellow": False, "red": False}
    infos = [mk(health=50), mk(x=CELL * 1.5, health=60, kills=1, keys=blue, damage=5, armor=10, ammo=(70, 0)),
             end(mk(x=CELL * 1.5, health=60, kills=1, keys=blue, damage=5, armor=10, ammo=(70, 0)), "exit")]
    r, out = play(infos, FULL, COV)
    for rew, info in zip(r, out):
        s = info[INFO_STEP]
        assert set(s["raw"]) == set(s["paid"]) == set(TERMS)
        for t in TERMS:
            assert s["paid"][t] == pytest.approx(FULL.scale(t) * s["raw"][t])
        assert s["extrinsic"] == pytest.approx(sum(s["paid"].values()) + s["engine"])
        assert s["intrinsic"] == pytest.approx(s["intrinsic_coef"] * s["intrinsic_raw"])
        assert rew == pytest.approx(s["extrinsic"] + s["intrinsic"]) == pytest.approx(s["total"])
    s0 = out[0][INFO_STEP]
    expect = 0.1 + 0.02 - 0.003 * 5 + 0.002 * 10 + 0.005 * 2 + 0.001 * 10 - 0.0002
    assert s0["extrinsic"] == pytest.approx(expect)
    assert s0["intrinsic_raw"] == pytest.approx(1 / COV.budget_cells) and s0["intrinsic_coef"] == COV.coef_start
    ep = episode(out)
    assert ep["total"] == pytest.approx(sum(r)) and ep["length"] == 2
    assert ep["extrinsic"] + ep["intrinsic"] == pytest.approx(ep["total"])
    for t in TERMS:
        assert ep[f"paid/{t}"] == pytest.approx(paid(out, t))
    assert ep["paid/level_exit"] == 1.0 and ep["cells_visited"] == 2
    assert INFO_EPISODE not in out[0]


def test_intrinsic_can_be_kept_out_of_the_reward_for_the_two_stream_trainer():
    cfg = dataclasses.replace(ZERO, sum_intrinsic_into_reward=False)
    r, out = play([mk(), mk(x=CELL * 1.5), end(mk(x=CELL * 1.5), "timeout")], cfg, COV)
    assert r == [0.0, 0.0]
    assert out[0][INFO_STEP]["intrinsic"] == pytest.approx(COV.coef_start / COV.budget_cells)


def test_missing_info_field_raises_instead_of_paying_zero():
    info = mk()
    del info["health"]
    env = ShapedRewardEnv(ScriptEnv([info, info]), FULL, COV)
    with pytest.raises(MissingInfoField, match="health"):
        env.reset()
    no_pose = {"final_obs_fresh": True}
    with pytest.raises(KeyError, match="pos_x"):
        ShapedRewardEnv(ScriptEnv([no_pose, no_pose]), ZERO, COV).reset()
    ShapedRewardEnv(ScriptEnv([no_pose, no_pose]), ZERO, NOCOV).reset()  # everything off: fine


def test_deltas_start_from_the_reset_info_and_never_cross_episodes():
    # Episode 1 ends hurt and rich; episode 2 starts at 100 health with other counters.
    script = [mk(health=30, armor=0, kills=0, ammo=(5, 0), damage=0),
              end(mk(health=30, kills=4, damage=70), "timeout")]
    env = ShapedRewardEnv(ScriptEnv(script), FULL, COV)
    env.reset()
    env.step(np.zeros(2, np.int64))
    env.env.infos = [mk(health=100, armor=50, kills=0, ammo=(50, 8), damage=0, x=5 * CELL),
                     mk(health=100, armor=50, kills=0, ammo=(50, 8), damage=0, x=5 * CELL),
                     end(mk(health=100, armor=50, x=5 * CELL, ammo=(50, 8)), "timeout")]
    env.reset()
    _, r, *_, info = env.step(np.zeros(2, np.int64))
    assert r == pytest.approx(-FULL.time)
    assert all(info[INFO_STEP]["paid"][t] == 0.0 for t in ("health", "armor", "ammo", "kill"))


def test_damage_has_one_source_per_step():
    # Death: the cumulative counter is stale (copy of the previous info); the env's
    # last_step_damage_taken is the only source and is paid once.
    r, out = play([mk(), mk(damage=10), end(mk(damage=10), "dead", last_damage=40, stale=True)],
                  only("damage_taken"), NOCOV)
    assert r == pytest.approx([-0.03, -0.12])
    # Fresh time-out: both sources present and equal; still paid once.
    r, _ = play([mk(), mk(damage=10), end(mk(damage=25), "timeout", last_damage=15)],
                only("damage_taken"), NOCOV)
    assert r == pytest.approx([-0.03, -0.045])


def test_stale_terminal_info_pays_no_delta_and_no_coverage():
    last = mk(x=CELL * 1.5, kills=2, health=80)
    r, out = play([mk(health=80), last, end(last, "dead", stale=True)], FULL, COV)
    s = out[-1][INFO_STEP]
    assert s["intrinsic_raw"] == 0.0 and s["paid"]["kill"] == 0.0 and s["paid"]["health"] == 0.0
    assert r[-1] == pytest.approx(-0.75 - 0.0002)


def test_two_signal_exit_rule():
    cfg = only("level_exit", exit_confirm_engine_reward=0.5)
    r, out = play([mk(), end(mk(), "exit")], cfg, NOCOV, engine=[1.0])
    assert r == [1.0] and episode(out)["exit_signal_disagreements"] == 0
    r, out = play([mk(), end(mk(), "exit")], cfg, NOCOV, engine=[0.0])
    assert r == [0.0] and episode(out)["exit_signal_disagreements"] == 1
    with pytest.raises(RewardConfigError, match="map_exit_reward"):
        check_reward_setup(FULL_LEVEL_REWARD, COV, (), 2100)  # the second signal must exist


# ---------------------------------------------------------------- no double counting
def test_my_way_home_goal_is_paid_exactly_once():
    # Probe R3.4: the goal decision's engine reward is +1 minus the per-tic cost.
    engine = [-0.0004, -0.0004, 0.9996]
    infos = [mk(), mk(x=40), mk(x=80), end(mk(x=80, armor=100), "exit")]
    r, out = play(infos, MY_WAY_HOME_REWARD, NOCOV, engine=engine)
    assert r == pytest.approx(engine)
    assert sum(r) == pytest.approx(0.9988)
    assert all(v == 0.0 for i in out for v in i[INFO_STEP]["paid"].values())
    assert episode(out)["engine_return_unscaled"] == pytest.approx(sum(engine))


MWH_KNOBS = (("living_reward", -0.0001),)


@pytest.mark.parametrize("change,knobs", [
    ({"level_exit": 1.0}, MWH_KNOBS),                      # ACS pays the goal
    ({"time": 0.0002}, MWH_KNOBS),                         # living_reward pays time
    ({"level_exit": 1.0, "scenario_script_pays": ()}, (("map_exit_reward", 1.0),)),
    ({"kill": 0.02}, (("kill_reward", 1.0),)),
    ({"death": 0.75}, (("death_penalty", 1.0),)),
    ({"health": 0.0}, (("made_up_knob", 1.0),)),           # unknown knob: must be classified
])
def test_double_pay_setups_are_refused(change, knobs):
    cfg = dataclasses.replace(MY_WAY_HOME_REWARD, **change)
    with pytest.raises(RewardConfigError):
        check_reward_setup(cfg, MY_WAY_HOME_COVERAGE, knobs, 525)


def test_shipped_presets_validate_and_ignored_engine_reward_cannot_double_pay():
    check_reward_setup(MY_WAY_HOME_REWARD, MY_WAY_HOME_COVERAGE, MWH_KNOBS, 525)
    check_reward_setup(FULL_LEVEL_REWARD, FULL_LEVEL_COVERAGE, (("map_exit_reward", 1.0),), 2100)
    check_reward_setup(PASSTHROUGH_REWARD, NO_COVERAGE, MWH_KNOBS, 525)
    # Full level: engine reward scale 0, so the engine's +1 at the exit is not added.
    r, out = play([mk(), end(mk(), "exit")], FULL_LEVEL_REWARD, NOCOV, engine=[1.0])
    assert r == pytest.approx([1.0 - 0.0002])
    assert episode(out)["engine"] == 0.0 and episode(out)["engine_return_unscaled"] == 1.0


def test_every_engine_knob_is_classified():
    knobs = re.search(r"ENGINE_REWARD_KNOBS = \((.*?)\)", open(os.path.join(ROOT, "env", "config.py")).read(), re.S)
    names = re.findall(r'"(\w+)"', knobs.group(1))
    assert len(names) >= 13 and set(names) <= set(KNOB_PAYS)


def test_passthrough_is_the_identity_on_the_toy_env():
    from train.toy_memory import ToyConfig, ToyMemoryEnv
    from train.vec_env import SyncVecEnv
    toy = partial(ToyMemoryEnv, ToyConfig(answer_step=3))
    plain = SyncVecEnv([toy, toy])
    shaped = SyncVecEnv([ShapedEnvFactory(toy, PASSTHROUGH_REWARD, NO_COVERAGE)] * 2)
    plain.reset(seed=7), shaped.reset(seed=7)
    rng = np.random.default_rng(0)
    for _ in range(12):
        a = rng.integers(0, 2, (2, 2))
        p, s = plain.step(a), shaped.step(a)
        assert np.array_equal(p.reward, s.reward) and np.array_equal(p.terminated, s.terminated)
        assert all(np.array_equal(p.obs[k], s.obs[k]) for k in p.obs)
        assert np.array_equal(p.final_obs_fresh, s.final_obs_fresh)
        for fi in s.final_infos:
            assert fi is None or INFO_EPISODE in fi


# ---------------------------------------------------------------- coverage bonus
def test_coverage_pays_first_visits_and_is_bounded_by_one():
    cov = EpisodicCoverage(CoverageConfig(cell_size=100.0, budget_cells=4))
    cov.reset(mk(x=50, y=50))
    got = [cov.step(mk(x=x, y=50)) for x in (60, 150, 250, 150, 350, 450, 550, 650)]
    assert got == [0.0, 0.25, 0.25, 0.0, 0.25, 0.25, 0.0, 0.0]
    assert sum(got) == 1.0 and cov.cells_visited == 7
    assert cov.step(mk(x=-50, y=-50)) == 0.0  # negative coordinates floor to their own cell


def test_coverage_resets_every_episode():
    script = [mk(), mk(x=CELL * 1.5), end(mk(x=CELL * 2.5), "timeout")]
    env = ShapedRewardEnv(ScriptEnv(script), ZERO, COV)
    per_episode = []
    for _ in range(3):
        env.reset()
        a = env.step(np.zeros(2, np.int64))[4]
        b = env.step(np.zeros(2, np.int64))[4]
        per_episode.append((a[INFO_STEP]["intrinsic_raw"], b[INFO_STEP]["intrinsic_raw"],
                            b[INFO_EPISODE]["cells_visited"], b[INFO_EPISODE]["intrinsic_raw"]))
    assert per_episode == [(1 / 64, 1 / 64, 3, 2 / 64)] * 3


def test_anneal_schedule():
    s = LinearAnneal(0.25, 0.05, hold_decisions=100, anneal_decisions=1000)
    assert [s(0), s(100)] == [0.25, 0.25]
    assert s(600) == pytest.approx(0.15) and s(1100) == pytest.approx(0.05) and s(10**9) == pytest.approx(0.05)
    vals = [s(d) for d in range(0, 2000, 50)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    with pytest.raises(RewardConfigError):
        check_reward_setup(MY_WAY_HOME_REWARD, CoverageConfig(coef_start=0.1, coef_end=0.2), MWH_KNOBS, 525)


def test_wrapper_anneals_on_its_own_clock_and_resumes_from_an_offset():
    cov = dataclasses.replace(COV, anneal_decisions=10)
    script = [mk()] * 30
    _, out = play(script, ZERO, cov)
    coefs = [i[INFO_STEP]["intrinsic_coef"] for i in out]
    assert coefs[0] == 0.25 and coefs[5] == pytest.approx(0.15) and coefs[10] == pytest.approx(0.05)
    assert coefs[-1] == pytest.approx(0.05)
    _, resumed = play(script, ZERO, cov, start_decisions=5)
    assert [i[INFO_STEP]["intrinsic_coef"] for i in resumed][:20] == pytest.approx(coefs[5:25])
    assert per_env_decisions(6400, 32) == 200


def test_learned_novelty_is_a_marked_placeholder():
    with pytest.raises(NotImplementedError, match="ADR 0005"):
        LearnedEpisodicNovelty()


# ---------------------------------------------------------------- integration point
class _Inner:
    def __init__(self, engine: bool, config=None):
        self.is_engine_instance, self.config = engine, config

    def __call__(self):
        return ScriptEnv([mk(), end(mk(), "timeout")])


class _Cfg:
    engine_rewards = (("living_reward", -0.0001),)
    episode_timeout_tics, frame_skip, vanilla_eval = 2100, 4, False


def test_factory_keeps_the_layout_guard_and_pickles():
    from train.doom_factory import DoomEnvFactory
    from train.layout_guard import count_engine_instances
    doom = ShapedEnvFactory(DoomEnvFactory(_Cfg()), MY_WAY_HOME_REWARD, MY_WAY_HOME_COVERAGE)
    toy = ShapedEnvFactory(_Inner(False), PASSTHROUGH_REWARD, NO_COVERAGE)
    assert count_engine_instances([doom, toy, doom]) == 2
    f = ShapedEnvFactory(partial(ScriptEnv, [mk(), mk()]), ZERO, COV, start_decisions=9)
    g = pickle.loads(pickle.dumps(f))
    assert (g.reward, g.coverage, g.start_decisions) == (f.reward, f.coverage, 9) and g().decisions == 9


def test_factory_validates_from_the_env_config_before_any_env_is_built():
    with pytest.raises(RewardConfigError):   # my_way_home config + our own exit term
        ShapedEnvFactory(_Inner(True, _Cfg()), dataclasses.replace(MY_WAY_HOME_REWARD, level_exit=1.0), COV)

    class Vanilla(_Cfg):
        vanilla_eval = True
    with pytest.raises(ValueError, match="vanilla-eval"):
        ShapedEnvFactory(_Inner(True, Vanilla()), MY_WAY_HOME_REWARD, COV)

    class Phase2:
        phase2_engine_test = True
    from train.doom_factory import DoomEnvFactory
    with pytest.raises(ValueError, match="phase2"):  # inner refusals still run
        ShapedEnvFactory(DoomEnvFactory(Phase2()), MY_WAY_HOME_REWARD, COV)


def test_observation_is_untouched_and_info_keys_are_only_added():
    inner = ScriptEnv([mk(), mk(x=CELL * 2), end(mk(), "timeout")])
    env = ShapedRewardEnv(inner, FULL, COV)
    assert env.observation_space is inner.observation_space and env.action_space is inner.action_space
    obs, info = env.reset(seed=3)
    assert set(obs) == {"self"} and info == mk()
    seen = {}
    orig = inner.step
    inner.step = lambda a: seen.setdefault("out", orig(a))
    obs, _, _, _, info = env.step(np.zeros(2, np.int64))
    assert obs is seen["out"][0]
    assert set(info) - set(seen["out"][4]) == {INFO_STEP, "engine_reward"}
    assert all(info[k] == v for k, v in seen["out"][4].items())
    assert INFO_STEP not in seen["out"][4]  # the env's own dict was not mutated


def test_reward_code_is_training_only_and_torch_free():
    def sources(*dirs):
        for d in dirs:
            for base, _, files in os.walk(os.path.join(ROOT, d)):
                for f in files:
                    if f.endswith(".py"):
                        yield os.path.join(base, f), open(os.path.join(base, f), encoding="utf-8").read()
    imp = re.compile(r"^\s*(?:from|import)\s+(rewards|intrinsic|configs\.reward_config)\b", re.M)
    offenders = [p for p, src in sources("agent", "env") if imp.search(src)]
    assert not offenders, f"inference-path packages import reward code: {offenders}"
    bad = re.compile(r"^\s*(?:from|import)\s+(torch|vizdoom|agent|env)\b", re.M)
    own = list(sources("rewards", "intrinsic")) + [(p, s) for p, s in sources("configs") if p.endswith("reward_config.py")]
    assert len(own) >= 9 and not [p for p, src in own if bad.search(src)]
    code = ("import sys, configs.reward_config, rewards.factory, rewards.wrapper, intrinsic.coverage, "
            "intrinsic.schedule, intrinsic.learned_novelty; "
            "print(int('torch' in sys.modules), int('vizdoom' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["0", "0"], out.stderr


def test_step_signatures_take_no_action_or_observation():
    import inspect
    assert list(inspect.signature(ExtrinsicReward.step).parameters) == ["self", "info", "ended"]
    assert list(inspect.signature(EpisodicCoverage.step).parameters) == ["self", "info"]
    assert set(CAPPED_TERMS) == {"kill", "health", "ammo", "armor"}


def test_the_exact_integration_line_given_to_worker():
    """Builds the factory only; no engine starts until the factory is called."""
    from configs.a0_env import a0_train_env_config
    from train.doom_factory import DoomEnvFactory
    from train.layout_guard import count_engine_instances
    f = ShapedEnvFactory(DoomEnvFactory(a0_train_env_config()), MY_WAY_HOME_REWARD, MY_WAY_HOME_COVERAGE,
                         start_decisions=per_env_decisions(64_000, 32))
    assert count_engine_instances([f]) == 1 and f.start_decisions == 2000
    g = pickle.loads(pickle.dumps(f))
    assert g == f and g.is_engine_instance
    with pytest.raises(RewardConfigError):  # the full-level preset would pay time twice here
        ShapedEnvFactory(DoomEnvFactory(a0_train_env_config()), FULL_LEVEL_REWARD, FULL_LEVEL_COVERAGE)
