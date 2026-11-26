"""PLR map ordering in curriculum/scheduler.py (no engine), plus pins on shuffled_cycle."""
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter

import pytest

from curriculum.scheduler import CurriculumScheduler, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR = os.path.join(ROOT, "curriculum")
POOL5 = {s: [(f"DD{i:04d}{s}", f"{s}-train-{i}") for i in range(5)] for s in ("S0", "S1", "S2", "S3")}
POOL20 = {s: [(f"DD{i:04d}{s}", f"{s}-train-{i}") for i in range(20)] for s in ("S0", "S1", "S2", "S3")}


def base_cfg(window=8, fraction=0.25, threshold=None):
    c = copy.deepcopy(load_config(os.path.join(CUR, "config.json")))
    c["promotion"]["window"] = window
    c["promotion"]["threshold"] = threshold or {"S0": 0.5, "S1": 0.5, "S2": 0.5}
    c["replay"]["fraction"] = fraction
    return c


def plr_cfg(rho=0.3, beta=0.3, window=5, progress_weight=0.0, **kw):
    c = base_cfg(**kw)
    c["map_order"] = "plr"
    c["plr"] = {"rho": rho, "beta": beta, "window": window, "progress_weight": progress_weight}
    return c


def run(sch, outcome, n):
    log = []
    for _ in range(n):
        ch = sch.next_episode()
        sch.record(ch, outcome(ch))
        log.append((ch.stage, ch.map_id, ch.replay))
    return log


# ------------------------------------------------------------------ shuffled_cycle pins
def test_shuffled_cycle_sequence_and_state_pinned():
    """Hashes captured from the code BEFORE PLR was added (2026-11-26). The live run's
    resume relies on both the choice sequence and the state format being unchanged."""
    s = CurriculumScheduler(base_cfg(), POOL5, seed=42)
    hist = iter(random.Random(5).choices([True, False], k=400))
    log = [list(t) for t in run(s, lambda ch: next(hist), 400)]
    assert hashlib.sha256(json.dumps(log).encode()).hexdigest() == \
        "59fb27ff490b79b88ce746a08159fecd05b9254ce2c30c179b161bceeeb670cb"
    assert hashlib.sha256(json.dumps(s.state_dict(), sort_keys=True).encode()).hexdigest() == \
        "b57fdfff0a2697923ca25575aa66b97103156021b62110900e957f3a0f39258d"
    assert "plr" not in s.state_dict() and s.plr_metrics() == {}


def test_existing_configs_keep_shuffled_cycle():
    for name in ("config.json", "config_mixed.json"):
        assert load_config(os.path.join(CUR, name))["map_order"] == "shuffled_cycle"


def test_config_mixed_plr_differs_only_in_map_order():
    a = load_config(os.path.join(CUR, "config_mixed.json"))
    b = load_config(os.path.join(CUR, "config_mixed_plr.json"))
    assert b["map_order"] == "plr" and set(b["plr"]) >= {"rho", "beta", "window"}
    strip = lambda d: {k: v for k, v in d.items() if k not in ("map_order", "plr", "_comment")}
    assert strip(a) == strip(b)
    CurriculumScheduler(b, POOL20, seed=0)   # valid


def test_bad_plr_config_refused():
    with pytest.raises(ValueError):
        c = base_cfg(); c["map_order"] = "random"; CurriculumScheduler(c, POOL5)
    with pytest.raises(ValueError):
        c = base_cfg(); c["map_order"] = "plr"; CurriculumScheduler(c, POOL5)   # no plr block
    for bad in ({"rho": 1.5}, {"beta": 0.0}, {"window": 0}, {"progress_weight": -1.0}):
        c = plr_cfg()
        c["plr"].update(bad)
        with pytest.raises(ValueError):
            CurriculumScheduler(c, POOL5)


# ------------------------------------------------------------------ PLR behaviour
FAIL = {f"S0-train-{i}" for i in range(4)}   # 4 of 20 S0 maps always fail, the rest always succeed


def test_unseen_maps_first_then_prefers_failing_maps():
    s = CurriculumScheduler(plr_cfg(threshold={"S0": 2.0, "S1": 2.0, "S2": 2.0}), POOL20, seed=3)
    log = run(s, lambda ch: ch.map_id not in FAIL, 2000)
    assert sorted(m for _, m, _ in log[:20]) == sorted(m for _, m in POOL20["S0"])   # each once first
    counts = Counter(m for _, m, _ in log[20:])
    fail_share = sum(counts[m] for m in FAIL) / sum(counts.values())
    assert fail_share > 0.6          # uniform would be 4/20 = 0.2
    assert all(counts[m] > 0 for _, m in POOL20["S0"])   # staleness: nothing starves


def test_staleness_keeps_every_map_visited():
    """Sharp score distribution: one failing map, the rest succeed. With rho > 0 every map
    is revisited within a bounded gap."""
    s = CurriculumScheduler(plr_cfg(rho=0.3, beta=0.1, threshold={"S0": 2.0, "S1": 2.0, "S2": 2.0}),
                            POOL20, seed=8)
    log = run(s, lambda ch: ch.map_id != "S0-train-0", 3000)
    last = {}
    gaps = []
    for k, (_, m, _) in enumerate(log):
        if m in last:
            gaps.append(k - last[m])
        last[m] = k
    assert set(last) == {m for _, m in POOL20["S0"]}
    assert max(gaps) < 400
    # rho = 0 with a sharp beta: successful maps are (almost) never revisited
    s0 = CurriculumScheduler(plr_cfg(rho=0.0, beta=0.1, threshold={"S0": 2.0, "S1": 2.0, "S2": 2.0}),
                             POOL20, seed=8)
    c0 = Counter(m for _, m, _ in run(s0, lambda ch: ch.map_id != "S0-train-0", 3000)[20:])
    assert c0["S0-train-0"] / sum(c0.values()) > 0.9


def test_tied_scores_give_uniform_distribution():
    s = CurriculumScheduler(plr_cfg(rho=0.0), POOL20, seed=1)
    run(s, lambda ch: False, 20)         # every map visited once, all fail -> all tied
    p = s.plr_distribution("S0")
    assert max(p) == pytest.approx(min(p)) and sum(p) == pytest.approx(1.0)


def test_progress_term_raises_learning_maps():
    s = CurriculumScheduler(plr_cfg(window=2, progress_weight=1.0), POOL5, seed=0)
    s.plr_outcomes["S0"] = [[False, False, True, True], [True, True, True, True],
                            [True] * 4, [True] * 4, [True] * 4]
    sc = s._plr_scores("S0")
    assert sc[0] == pytest.approx(1.0) and sc[1] == pytest.approx(0.0)


def test_parallel_outstanding_choices_spread_over_maps():
    s = CurriculumScheduler(plr_cfg(), POOL20, seed=2)
    run(s, lambda ch: ch.map_id not in FAIL, 40)
    chs = [s.next_episode() for _ in range(12)]   # 12 envs reset, no outcomes yet
    assert len({c.map_id for c in chs}) >= 6      # staleness stamped at issue time
    for c in reversed(chs):
        s.record(c, c.map_id not in FAIL)


def test_replay_outcomes_update_their_own_stage_stats():
    s = CurriculumScheduler(plr_cfg(window=5, fraction=0.5, threshold={"S0": 0.5, "S1": 2.0, "S2": 2.0}),
                            POOL5, seed=4)
    run(s, lambda ch: True, 8)           # promotion window 8
    assert s.stage == "S1"
    before = sum(len(h) for h in s.plr_outcomes["S0"])
    log = run(s, lambda ch: False, 200)
    n_rep = sum(1 for st, _, r in log if r and st == "S0")
    assert n_rep > 0 and sum(len(h) for h in s.plr_outcomes["S0"]) > before
    assert all(len(h) <= 10 for v in s.plr_outcomes.values() for h in v)   # capped at 2*window
    m = s.plr_metrics()
    for st in s.stages:
        for k in ("plr_mean_score_", "plr_visited_frac_", "plr_entropy_", "plr_entropy_norm_", "plr_max_prob_"):
            assert k + st in m
    assert m["plr_visited_frac_S0"] == 1.0 and math.isnan(m["plr_mean_score_S3"])


def test_plr_deterministic_given_seed_and_history():
    hist = random.Random(5).choices([True, False], k=800)

    def go(seed):
        s = CurriculumScheduler(plr_cfg(), POOL20, seed=seed)
        it = iter(hist)
        return run(s, lambda ch: next(it), 800)
    assert go(42) == go(42)
    assert go(42) != go(43)


def test_plr_state_round_trip_with_outstanding_choices():
    hist = random.Random(9).choices([True, False], k=1000)
    it = iter(hist)
    a = CurriculumScheduler(plr_cfg(fraction=0.5), POOL20, seed=4)
    run(a, lambda ch: next(it), 300)
    out = [a.next_episode() for _ in range(6)]   # outstanding at checkpoint time
    sd = json.loads(json.dumps(a.state_dict()))
    b = CurriculumScheduler(plr_cfg(fraction=0.5), POOL20, seed=4)
    b.load_state_dict(sd)
    assert json.loads(json.dumps(b.state_dict())) == sd
    rest = hist[300:]
    for x in (a, b):
        r = iter(rest)
        for c in reversed(out):
            x.record(c, next(r))
        x._log = run(x, lambda ch: next(r), 300)
    assert a._log == b._log and a.state_dict() == b.state_dict()
    assert a.plr_metrics() == b.plr_metrics() or all(
        (math.isnan(v) and math.isnan(b.plr_metrics()[k])) or v == b.plr_metrics()[k]
        for k, v in a.plr_metrics().items())


def test_shuffled_state_loads_into_plr_with_fresh_stats():
    a = CurriculumScheduler(base_cfg(), POOL20, seed=4)
    run(a, lambda ch: True, 50)
    pend = [a.next_episode() for _ in range(3)]
    sd = json.loads(json.dumps(a.state_dict()))
    b = CurriculumScheduler(plr_cfg(), POOL20, seed=4)
    b.load_state_dict(sd)
    assert b.stage == a.stage and b.issued == a.issued
    assert all(st is None for v in b.plr_stamp.values() for st in v)
    for c in pend:
        b.record(c, True)            # pending choices from shuffled mode still record
    run(b, lambda ch: True, 50)


def test_plr_state_refused_on_different_pool():
    a = CurriculumScheduler(plr_cfg(), POOL20, seed=4)
    run(a, lambda ch: True, 10)
    sd = json.loads(json.dumps(a.state_dict()))
    other = {s: v[:19] for s, v in POOL20.items()}
    b = CurriculumScheduler(plr_cfg(), other, seed=4)
    with pytest.raises(ValueError):
        b.load_state_dict(sd)


def shipped_plr_cfg(**kw):
    """plr_cfg with the plr block of the shipped curriculum/config_mixed_plr.json."""
    p = load_config(os.path.join(CUR, "config_mixed_plr.json"))["plr"]
    return plr_cfg(rho=p["rho"], beta=p["beta"], window=p["window"],
                   progress_weight=p.get("progress_weight", 0.0), **kw)


NEVER = {"S0": 2.0, "S1": 2.0, "S2": 2.0}


def test_shipped_params_prefer_failing_maps_and_visit_all():
    s = CurriculumScheduler(shipped_plr_cfg(threshold=NEVER), POOL20, seed=3)
    log = run(s, lambda ch: ch.map_id not in FAIL, 3000)
    counts = Counter(m for _, m, _ in log[20:])
    assert sum(counts[m] for m in FAIL) / sum(counts.values()) > 0.6
    assert all(counts[m] > 0 for _, m in POOL20["S0"])
    chs = [s.next_episode() for _ in range(12)]   # outstanding choices spread
    assert len({c.map_id for c in chs}) >= 6


def test_plr_through_curriculum_vec_env_and_state_json_round_trip():
    from curriculum_helpers import TINY_CURRICULUM, TINY_POOL, FakeMapEnv
    from train.curriculum_vec_env import CURRICULUM_INFO_KEY, CurriculumVecEnv
    from train.vec_env import SyncVecEnv
    import numpy as np
    c = copy.deepcopy(TINY_CURRICULUM)
    c["map_order"] = "plr"
    c["plr"] = {"rho": 0.3, "beta": 0.3, "window": 2, "progress_weight": 0.0}

    def make():
        return CurriculumVecEnv(SyncVecEnv([FakeMapEnv for _ in range(2)]),
                                CurriculumScheduler(c, TINY_POOL), recent_window=10)

    def drive(cv, steps):
        out, acts = [], np.zeros((cv.num_envs, 6), np.int64)
        for _ in range(steps):
            st = cv.step(acts)
            out.append(tuple(None if f is None else (f[CURRICULUM_INFO_KEY]["choice"], f["map"])
                             for f in st.final_infos))
        return out

    a = make()
    a.reset(0)
    drive(a, 5)
    assert sum(len(h) for h in a.sch.plr_outcomes["S0"]) > 0
    text = json.loads(json.dumps(a.state_json()))
    b = make()
    b.load_state_json(text)
    a.reset(9), b.reset(9)
    assert b.current == a.current and b.upcoming == a.upcoming
    assert drive(a, 12) == drive(b, 12)
    assert a.sch.state_dict() == b.sch.state_dict()
    ma, mb = a.metrics(), b.metrics()
    assert ma.keys() == mb.keys() and "plr_visited_frac_S1" in ma
    assert all((math.isnan(v) and math.isnan(mb[k])) or v == mb[k] for k, v in ma.items())


def test_vec_env_metrics_include_plr_only_in_plr_mode():
    from train.curriculum_vec_env import CurriculumVecEnv

    class Dummy:
        num_envs = 2
        single_observation_space = single_action_space = None

    shuf = CurriculumVecEnv(Dummy(), CurriculumScheduler(base_cfg(), POOL5, seed=0), 10)
    shuf._fill()
    assert not any(k.startswith("plr_") for k in shuf.metrics())
    plr = CurriculumVecEnv(Dummy(), CurriculumScheduler(plr_cfg(), POOL5, seed=0), 10)
    plr._fill()
    assert "plr_entropy_S0" in plr.metrics()
