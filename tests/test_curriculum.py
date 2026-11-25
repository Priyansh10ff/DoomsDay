"""Curriculum scheduler: promotion, replay mixing, determinism, resume, config checks."""
import copy
import json
import os
import random

import pytest

from curriculum.scheduler import CurriculumScheduler, load_config

POOL = {s: [(f"DD{i:04d}{s}", f"{s}-train-{i}") for i in range(5)] for s in ("S0", "S1", "S2", "S3")}


def cfg(**kw):
    c = copy.deepcopy(load_config())
    c["promotion"]["window"] = kw.get("window", 10)
    c["promotion"]["threshold"] = kw.get("threshold", {"S0": 0.5, "S1": 0.5, "S2": 0.5})
    c["replay"]["fraction"] = kw.get("fraction", 0.25)
    return c


def run(sch, outcome, n):
    log = []
    for _ in range(n):
        ch = sch.next_episode()
        sch.record(ch, outcome(ch))
        log.append((ch.stage, ch.map_id, ch.replay))
    return log


def test_config_labels_numbers_unmeasured():
    assert "UNMEASURED" in load_config()["_status"]


def test_no_promotion_before_window_full_or_below_threshold():
    s = CurriculumScheduler(cfg(window=10), POOL, seed=1)
    run(s, lambda ch: True, 9)
    assert s.stage == "S0"
    s2 = CurriculumScheduler(cfg(window=10), POOL, seed=1)
    flip = iter([True, False] * 100)
    run(s2, lambda ch: next(flip) and False, 50)  # all failures
    assert s2.stage == "S0"


def test_promotes_when_window_rate_crosses_threshold():
    s = CurriculumScheduler(cfg(window=10, threshold={"S0": 0.6, "S1": 0.6, "S2": 0.6}), POOL, seed=1)
    # rolling window: after 10 episodes 5/10 = 0.5 < 0.6; the 11th drops a failure -> 6/10
    seq = iter([False] * 5 + [True] * 5 + [True] * 1)
    promoted_at = None
    for k in range(11):
        ch = s.next_episode()
        if s.record(ch, next(seq)):
            promoted_at = k
    assert promoted_at == 10 and s.stage == "S1"
    assert s.history[0]["from"] == "S0" and s.history[0]["rate"] == pytest.approx(0.6)


def test_never_leaves_last_stage_and_no_replay_before_promotion():
    s = CurriculumScheduler(cfg(window=5), POOL, seed=3)
    log = run(s, lambda ch: True, 4)
    assert not any(r for _, _, r in log)
    run(s, lambda ch: True, 400)
    assert s.stage == "S3"


def test_replay_mixing_proportion_and_earlier_stages_only():
    s = CurriculumScheduler(cfg(window=5, fraction=0.3), POOL, seed=7)
    run(s, lambda ch: True, 5)          # promote to S1
    assert s.stage == "S1"
    s.cfg["promotion"]["threshold"]["S1"] = 2.0   # stay on S1
    log = run(s, lambda ch: True, 4000)
    replays = [st for st, _, r in log if r]
    assert set(replays) == {"S0"}
    assert abs(len(replays) / len(log) - 0.3) < 0.03
    assert all(st == "S1" for st, _, r in log if not r)


def test_replay_draws_uniformly_from_earlier_stages():
    s = CurriculumScheduler(cfg(window=5, fraction=0.5), POOL, seed=11)
    run(s, lambda ch: ch.stage != "S2", 30)  # reach S2, stay (fails on S2)
    assert s.stage == "S2"
    log = run(s, lambda ch: False, 4000)
    reps = [st for st, _, r in log if r]
    assert set(reps) == {"S0", "S1"}
    assert abs(reps.count("S0") / len(reps) - 0.5) < 0.05


def test_deterministic_given_seed_and_history():
    hist = random.Random(5).choices([True, False], k=600)
    def go(seed):
        s = CurriculumScheduler(cfg(window=8), POOL, seed=seed)
        it = iter(hist)
        return run(s, lambda ch: next(it), 600)
    assert go(42) == go(42)
    assert go(42) != go(43)


def test_shuffled_cycle_visits_every_map():
    s = CurriculumScheduler(cfg(window=1000), POOL, seed=2)
    log = run(s, lambda ch: False, 10)
    assert sorted(m for _, m, _ in log[:5]) == sorted(m for _, m in POOL["S0"])


def test_state_dict_resume_is_exact():
    hist = random.Random(9).choices([True, False], k=300)
    a = CurriculumScheduler(cfg(window=6), POOL, seed=4)
    it = iter(hist)
    run(a, lambda ch: next(it), 150)
    sd = json.loads(json.dumps(a.state_dict()))
    b = CurriculumScheduler(cfg(window=6), POOL, seed=4)
    b.load_state_dict(sd)
    rest = hist[150:]
    ia, ib = iter(rest), iter(rest)
    assert run(a, lambda ch: next(ia), 150) == run(b, lambda ch: next(ib), 150)


def test_bad_config_and_protocol_refused():
    c = cfg()
    del c["promotion"]["threshold"]["S1"]
    with pytest.raises(ValueError):
        CurriculumScheduler(c, POOL)
    with pytest.raises(ValueError):
        CurriculumScheduler(cfg(), {**POOL, "S2": []})
    s = CurriculumScheduler(cfg(), POOL)
    ch = s.next_episode()
    s.record(ch, True)
    with pytest.raises(RuntimeError):
        s.record(ch, True)   # already recorded


def test_parallel_envs_outstanding_choices_out_of_order():
    s = CurriculumScheduler(cfg(window=4), POOL, seed=5)
    chs = [s.next_episode() for _ in range(8)]   # 8 envs reset at once
    assert len({c.episode for c in chs}) == 8
    for c in reversed(chs):
        s.record(c, True)
    assert s.stage == "S1" and s.episodes == 8


@pytest.mark.skipif(not os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "maps", "splits.json")), reason="no splits.json")
def test_from_bundle_uses_only_train_maps():
    from maps.loader import load_splits
    sp = load_splits()
    s = CurriculumScheduler.from_bundle()
    train = {e["id"] for _, e in sp.entries("train")}
    pooled = {mid for v in s.pool.values() for _, mid in v}
    assert pooled == train
