"""C4: per-episode USE / door diagnostics (train/episode_diag.py). Logging only."""
from __future__ import annotations

import json
import math
import os

import gymnasium as gym
import numpy as np
import pytest

from maps.doors import AXIS_X, DoorSpec
from train.curriculum_vec_env import diag_metrics
from train.episode_diag import DiagMap, EpisodeDiag, EpisodeDiagEnv, diag_table

DOOR = DoorSpec(sector=7, axis=AXIS_X, a0=96.0, a1=128.0, lo=0.0, hi=64.0)
EXIT_LINE = (300.0, 0.0, 300.0, 64.0)
DMAP = DiagMap(doors=(DOOR,), on_path=(True,), exit_kind="switch", exit_use_lines=(EXIT_LINE,))


def info(x=70.0, y=32.0, angle=0.0, tic=0, gap=0.0, fresh=True, gaps=True):
    d = {"pos_x": x, "pos_y": y, "angle": angle, "tic": tic, "map": "M", "final_obs_fresh": fresh}
    if gaps:
        d["door_gaps"] = {7: gap}
    return d


def run(seq, start=None, dmap=DMAP):
    """seq: [(use, info_after)]; the pre-step info of step k is the after-info of k-1."""
    dg = EpisodeDiag(48.0)
    before = start or info()
    dg.reset(dmap, before)
    for use, after in seq:
        dg.step(use, before, after)
        before = after
    return dg.episode()


def test_press_release_press_near_a_door_counts_two_edges_and_one_reversal():
    ep = run([(1, info(tic=4, gap=8.0)),     # press at a closed door: opens
              (0, info(tic=8, gap=16.0)),    # release: still rising
              (1, info(tic=12, gap=8.0))])   # second press: the door falls again
    assert ep["use_edges"] == 2
    assert ep["door_reversals"] == 1
    assert ep["use_edges_in_range"] == 1 and ep["use_edges_in_range_door"] == 1   # 2nd: door not closed
    assert ep["door_ids_matched"] is True and ep["exit_kind"] == "switch"


def test_held_use_is_one_edge():
    ep = run([(1, info(tic=4, gap=8.0)), (1, info(tic=8, gap=16.0)), (1, info(tic=12, gap=24.0))])
    assert ep["use_edges"] == 1 and ep["door_reversals"] == 0


def test_out_of_range_and_facing_away_edges_are_not_in_range():
    far = info(x=20.0)                      # trace ends at x = 84 < 96
    ep = run([(1, info(x=20.0, tic=4)), (0, info(x=20.0, tic=8)), (1, info(x=70.0, angle=180.0, tic=12))],
             start=far)
    assert ep["use_edges"] == 2 and ep["use_edges_in_range"] == 0
    beside = info(y=100.0)                  # beyond the face's lateral extent
    assert run([(1, info(y=100.0, tic=4))], start=beside)["use_edges_in_range"] == 0


def test_exit_switch_line_in_range():
    at = info(x=250.0)
    ep = run([(1, info(x=250.0, tic=4))], start=at)
    assert ep["use_edges_in_range"] == 1 and ep["use_edges_in_range_exit"] == 1 and ep["use_edges_in_range_door"] == 0


def test_door_closing_by_itself_is_not_a_reversal():
    seq = [(1, info(tic=4, gap=8.0)), (0, info(tic=8, gap=64.0)), (0, info(tic=100, gap=64.0)),
           (0, info(tic=160, gap=56.0)), (0, info(tic=164, gap=48.0))]
    assert run(seq)["door_reversals"] == 0


def test_press_elsewhere_does_not_explain_a_reversal():
    # door opened at tic 0; much later a press 300 units away (trace misses) as it falls
    seq = [(1, info(tic=4, gap=8.0)), (0, info(x=-300.0, tic=50, gap=64.0)),
           (1, info(x=-300.0, tic=104, gap=56.0))]
    assert run(seq)["door_reversals"] == 0


def test_stale_final_info_is_ignored_and_missing_gaps_give_none():
    seq = [(1, info(tic=4, gap=8.0)), (0, info(tic=8, gap=16.0)), (1, info(tic=8, gap=0.0, fresh=False))]
    assert run(seq)["door_reversals"] == 0
    ep = run([(1, info(tic=4, gaps=False))], start=info(gaps=False))
    assert ep["door_reversals"] is None and ep["door_ids_matched"] is None
    assert ep["use_edges_in_range"] == 1     # without gaps every door face counts


def test_on_path_doors_passed_and_unknown_route():
    walk = [(0, info(x=x, tic=4 * k, gap=64.0)) for k, x in enumerate((90.0, 112.0, 140.0), 1)]
    ep = run(walk)
    assert ep["on_path_doors_passed"] == 1 and ep["on_path_doors_in_map"] == 1
    side = DiagMap(doors=(DOOR,), on_path=(False,), exit_kind="pad", exit_use_lines=())
    ep = run(walk, dmap=side)
    assert ep["on_path_doors_passed"] == 0 and ep["on_path_doors_in_map"] == 0
    unk = DiagMap(doors=(DOOR,), on_path=None, exit_kind=None, exit_use_lines=())
    ep = run(walk, dmap=unk)
    assert ep["on_path_doors_passed"] is None and ep["on_path_doors_in_map"] is None


class _FakeEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (1,))
    action_space = gym.spaces.MultiDiscrete([3, 3, 5, 2, 2, 2])

    def __init__(self):
        self.seen, self.t = [], 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(1, np.float32), info(gaps=False)

    def diag_door_gaps(self):   # like env.DoomEnv: privileged, read by the wrapper only
        return {7: 8.0 * self.t}

    def step(self, action):
        self.seen.append(action)
        self.t += 1
        end = self.t == 3
        inf = info(tic=4 * self.t, gaps=False)
        if end:
            inf["shaping_episode"] = {"doors_in_map": 1, "engine_return_unscaled": 0.0}
        return np.zeros(1, np.float32), 0.5, end, False, inf


def test_wrapper_passes_action_untouched_and_merges_into_shaping_episode():
    env = EpisodeDiagEnv(_FakeEnv(), (("M", DMAP),), 48.0)
    env.reset()
    acts = [np.array([0, 0, 2, 0, 1, 0]), np.array([1, 0, 2, 0, 0, 0]), np.array([0, 0, 2, 0, 1, 0])]
    for a in acts:
        obs, r, term, trunc, inf = env.step(a)
        assert r == 0.5 and "door_gaps" not in inf
    assert all(x is a for x, a in zip(env.unwrapped.seen, acts))
    ep = inf["shaping_episode"]
    assert ep["use_edges"] == 2 and ep["doors_in_map"] == 1 and ep["door_reversals"] == 0
    assert ep["door_ids_matched"] is True


def test_diag_metrics_per_update():
    eps = [{"use_edges": 4, "use_edges_in_range": 1, "door_reversals": 1, "doors_in_map": 2,
            "on_path_doors_passed": 1, "on_path_doors_in_map": 2},
           {"use_edges": 0, "use_edges_in_range": 0, "door_reversals": 0, "doors_in_map": 1,
            "on_path_doors_passed": 1, "on_path_doors_in_map": 1},
           {"use_edges": 2, "use_edges_in_range": 1, "door_reversals": None, "doors_in_map": 1,
            "on_path_doors_passed": None, "on_path_doors_in_map": None},
           {"use_edges": 0, "use_edges_in_range": 0, "door_reversals": 0, "doors_in_map": 0,
            "on_path_doors_passed": 0, "on_path_doors_in_map": 0}]
    m = diag_metrics(eps)
    assert m["diag_use_edges_mean"] == pytest.approx(1.5)
    assert m["diag_use_in_range_share"] == pytest.approx(2 / 6)
    assert m["diag_door_episodes"] == 2 and m["diag_reversal_episode_share"] == pytest.approx(0.5)
    assert m["diag_on_path_door_pass_rate"] == pytest.approx(2 / 3)
    assert all(math.isnan(v) for k, v in diag_metrics([]).items() if k != "diag_episodes"
               and k != "diag_door_episodes")


def test_real_bundle_route_has_every_locked_door_on_path():
    from maps.loader import REPO_ROOT
    s = json.load(open(os.path.join(REPO_ROOT, "maps", "splits.json"), encoding="utf-8"))
    b = next(x for x in s["bundles"] if x["id"] == "train-all")
    wad = os.path.join(REPO_ROOT, b["file"])
    if not os.path.exists(wad):
        pytest.skip("training bundle not built")
    tab = dict(diag_table(wad, b["lumps"]))
    n = 0
    for lump, mid in zip(b["lumps"], b["members"]):
        dm = tab[lump]
        assert dm.on_path is not None and dm.exit_kind in ("pad", "switch")
        assert (len(dm.exit_use_lines) > 0) == (dm.exit_kind == "switch")
        g = json.load(open(os.path.join(REPO_ROOT, "maps", "train", mid + ".json"), encoding="utf-8"))["graph"]
        if g.get("lock_door_sector") is not None:
            n += 1
            assert dm.on_path[[d.sector for d in dm.doors].index(g["lock_door_sector"])]
    assert n > 0
