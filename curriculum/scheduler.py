"""Curriculum scheduler: which train map (and so which stage) each episode uses.

TRAINING ONLY. It chooses maps, never actions, and is not part of the inference path:
nothing under agent/ or env/ may import it (tests/test_maps_isolation.py).

Rules (all numbers from curriculum/config.json, which labels them UNMEASURED):
  * The agent trains on its current stage. A stage's success window holds the last
    `window` outcomes of episodes played on the current stage (replay episodes are
    excluded unless `count_replay_episodes`). When the window is full and its mean is
    >= the stage's threshold, the agent is promoted and the new stage's window starts
    empty. The last stage has no threshold and is never left.
  * After at least one promotion, each episode is a replay of an earlier stage with
    probability `replay.fraction`; the earlier stage is drawn uniformly.
  * Several choices may be outstanding at once (one per parallel env); outcomes may
    arrive in any order. An outcome from a stage the agent has since left is ignored.
  * Within a stage, the map is chosen by `map_order`:
      - "shuffled_cycle" (default): every map once per cycle, in a shuffled order.
      - "plr": Prioritized Level Replay (idea from Jiang et al. 2021; written here from
        the idea, no reference code). Each stage keeps its own per-map stats:
          outcomes  last 2*plr.window successes of episodes on that map (replay and
                    left-stage outcomes included: they are outcomes on that map);
          stamp     the stage's issue counter when the map was last issued.
        Rule for one choice on stage s (n maps, clock c_s = choices issued on s so far):
          1. If any map of s was never issued, pick uniformly among those.
          2. score_i = (1 - mean(last `window` outcomes))
                       + progress_weight * |mean(last window) - mean(window before)|
             (the progress term is 0 until both windows have outcomes; a map issued
             but with no outcome yet scores 1.0, the maximum of the failure term).
          3. rank_i = 1 + number of maps with a strictly higher score (tied scores
             share a rank, so equal maps get equal weight); P_S ~ (1/rank_i)^(1/beta).
          4. staleness_i = c_s - stamp_i; P_C ~ staleness_i (uniform if all are 0).
          5. P = (1 - rho) * P_S + rho * P_C, drawn with the shared rng.
        Staleness is stamped at ISSUE time, so several outstanding choices (parallel
        envs) spread over maps instead of piling onto one.
  * One `random.Random(seed)` drives every draw, so the same seed and the same success
    history give the same sequence of choices. `state_dict()` / `load_state_dict()`
    resume it exactly. A state saved without PLR stats (shuffled_cycle) loads into a
    "plr" scheduler with fresh stats, so a run can switch to PLR on resume.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


@dataclass(frozen=True)
class EpisodeChoice:
    stage: str
    map_id: str
    lump: str          # map lump name inside the training bundle, for reset(options={"map": lump})
    replay: bool
    episode: int       # issue index; several choices may be outstanding (parallel envs)


MAP_ORDERS = ("shuffled_cycle", "plr")
CONFIG_ENV = "DOOMSDAY_CURRICULUM_CONFIG"  # optional path to another config file; unset = config.json


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class CurriculumScheduler:
    def __init__(self, config: dict, pool: dict[str, list[tuple[str, str]]], seed: int | None = None):
        """pool: stage -> [(lump, map_id), ...] (train maps only; see from_bundle)."""
        self.cfg = config
        self.stages = list(config["stages"])
        missing = [s for s in self.stages if not pool.get(s)]
        if missing:
            raise ValueError(f"no train maps for stages {missing}")
        thr = config["promotion"]["threshold"]
        for s in self.stages[:-1]:
            if s not in thr:
                raise ValueError(f"no promotion threshold configured for stage {s}")
        self.window = int(config["promotion"]["window"])
        if self.window < 1:
            raise ValueError("promotion.window must be >= 1")
        self.replay_fraction = float(config["replay"]["fraction"])
        if not 0.0 <= self.replay_fraction <= 1.0:
            raise ValueError("replay.fraction must be in [0, 1]")
        if config["replay"].get("stage_weights", "uniform") != "uniform":
            raise ValueError("only uniform replay stage weights are implemented")
        self.count_replay = bool(config["promotion"].get("count_replay_episodes", False))
        self.pool = {s: sorted(pool[s]) for s in self.stages}
        self.seed = int(config["seed"] if seed is None else seed)
        self.rng = random.Random(self.seed)
        self.stage_idx = 0
        self.results: deque = deque(maxlen=self.window)
        self.cursor = {s: 0 for s in self.stages}
        self.order = {s: self._shuffled(s) for s in self.stages}
        self.history: list[dict] = []   # promotions: {"episode", "from", "to", "rate"}
        self.episodes = 0        # outcomes recorded
        self.issued = 0          # choices handed out
        self._pending: dict[int, EpisodeChoice] = {}
        self.map_order = config.get("map_order", "shuffled_cycle")
        if self.map_order not in MAP_ORDERS:
            raise ValueError(f"map_order {self.map_order!r} not in {MAP_ORDERS}")
        if self.map_order == "plr":
            p = config.get("plr")
            if not isinstance(p, dict):
                raise ValueError('map_order "plr" needs a "plr" config block')
            self.plr_rho = float(p["rho"])
            self.plr_beta = float(p["beta"])
            self.plr_window = int(p["window"])
            self.plr_progress_weight = float(p.get("progress_weight", 0.0))
            if not 0.0 <= self.plr_rho <= 1.0:
                raise ValueError("plr.rho must be in [0, 1]")
            if not self.plr_beta > 0.0:
                raise ValueError("plr.beta must be > 0")
            if self.plr_window < 1:
                raise ValueError("plr.window must be >= 1")
            if self.plr_progress_weight < 0.0:
                raise ValueError("plr.progress_weight must be >= 0")
            self._plr_index = {s: {lump: j for j, (lump, _) in enumerate(self.pool[s])} for s in self.stages}
            self._plr_reset()

    # ---------------------------------------------------------------- PLR state
    def _plr_reset(self) -> None:
        """Fresh PLR stats. JSON-safe layout: stage -> list indexed by pool index."""
        self.plr_outcomes = {s: [[] for _ in self.pool[s]] for s in self.stages}   # bools, <= 2*window
        self.plr_stamp = {s: [None] * len(self.pool[s]) for s in self.stages}    # None = never issued
        self.plr_clock = {s: 0 for s in self.stages}

    def _plr_scores(self, stage: str) -> list[float]:
        w, out = self.plr_window, []
        for hist in self.plr_outcomes[stage]:
            recent = hist[-w:]
            if not recent:
                out.append(1.0)
                continue
            m = sum(recent) / len(recent)
            score = 1.0 - m
            older = hist[:-w][-w:]
            if older and self.plr_progress_weight > 0.0:
                score += self.plr_progress_weight * abs(m - sum(older) / len(older))
            out.append(score)
        return out

    def plr_distribution(self, stage: str) -> list[float]:
        """Sampling probabilities over pool[stage] for the next PLR draw (pure)."""
        n = len(self.pool[stage])
        stamps = self.plr_stamp[stage]
        unseen = [j for j in range(n) if stamps[j] is None]
        if unseen:
            return [1.0 / len(unseen) if stamps[j] is None else 0.0 for j in range(n)]
        scores = self._plr_scores(stage)
        inv_beta = 1.0 / self.plr_beta
        asc = sorted(scores)   # rank = 1 + count of strictly higher scores (ties share a rank)
        h = [(1.0 / (1 + n - bisect.bisect_right(asc, sc))) ** inv_beta for sc in scores]
        hs = sum(h)
        clock = self.plr_clock[stage]
        stale = [float(clock - st) for st in stamps]
        ss = sum(stale)
        pc = [x / ss for x in stale] if ss > 0 else [1.0 / n] * n
        rho = self.plr_rho
        return [(1.0 - rho) * a / hs + rho * b for a, b in zip(h, pc)]

    def _plr_take(self, stage: str) -> tuple[str, str]:
        probs = self.plr_distribution(stage)
        u, acc, pick = self.rng.random(), 0.0, None
        for j, p in enumerate(probs):
            if p <= 0.0:
                continue
            pick = j                  # last positive index: guards float round-off at the top
            acc += p
            if u < acc:
                break
        self.plr_stamp[stage][pick] = self.plr_clock[stage]
        self.plr_clock[stage] += 1
        return self.pool[stage][pick]

    def plr_metrics(self) -> dict[str, float]:
        """Per-stage PLR stats for metrics.jsonl (pure: no rng draw, no mutation).
        Empty dict unless map_order is "plr"."""
        if self.map_order != "plr":
            return {}
        out: dict[str, float] = {}
        for s in self.stages:
            n = len(self.pool[s])
            scored = [sc for sc, h in zip(self._plr_scores(s), self.plr_outcomes[s]) if h]
            probs = self.plr_distribution(s)
            ent = -sum(p * math.log(p) for p in probs if p > 0.0)
            out[f"plr_mean_score_{s}"] = float(sum(scored) / len(scored)) if scored else math.nan
            out[f"plr_visited_frac_{s}"] = float(sum(st is not None for st in self.plr_stamp[s]) / n)
            out[f"plr_entropy_{s}"] = float(ent)
            out[f"plr_entropy_norm_{s}"] = float(ent / math.log(n)) if n > 1 else 1.0
            out[f"plr_max_prob_{s}"] = float(max(probs))
        return out

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_bundle(cls, config: dict | None = None, seed: int | None = None, splits=None):
        """Pool from the verified training bundle (maps/loader.training_bundle: train
        split only, hashes checked, held-out refused)."""
        from maps.loader import BUNDLE_ENV, training_bundle
        config = config or load_config()
        # $DOOMSDAY_TRAIN_BUNDLE (if set) overrides config "bundle_id"; unset = config.
        bundle_id = os.environ.get(BUNDLE_ENV, "").strip() or config.get("bundle_id", "train-all")
        path, lumps, stages = training_bundle(bundle_id, splits)
        pool: dict[str, list] = {}
        for lump, mid in lumps.items():
            pool.setdefault(stages[lump], []).append((lump, mid))
        sch = cls(config, pool, seed)
        sch.bundle_path = path
        sch.bundle_id = bundle_id
        return sch

    # ---------------------------------------------------------------- core
    @property
    def stage(self) -> str:
        return self.stages[self.stage_idx]

    def threshold(self, stage: str) -> float | None:
        return self.cfg["promotion"]["threshold"].get(stage)

    def _shuffled(self, stage):
        idx = list(range(len(self.pool[stage])))
        self.rng.shuffle(idx)
        return idx

    def _take(self, stage) -> tuple[str, str]:
        if self.map_order == "plr":
            return self._plr_take(stage)
        if self.cursor[stage] >= len(self.order[stage]):
            self.order[stage] = self._shuffled(stage)
            self.cursor[stage] = 0
        lump, mid = self.pool[stage][self.order[stage][self.cursor[stage]]]
        self.cursor[stage] += 1
        return lump, mid

    def next_episode(self) -> EpisodeChoice:
        replay = self.stage_idx > 0 and self.rng.random() < self.replay_fraction
        stage = self.stages[self.rng.randrange(self.stage_idx)] if replay else self.stage
        lump, mid = self._take(stage)
        ch = EpisodeChoice(stage, mid, lump, replay, self.issued)
        self.issued += 1
        self._pending[ch.episode] = ch
        return ch

    def record(self, choice: EpisodeChoice, success: bool) -> bool:
        """Report the outcome of `choice`. Returns True if this caused a promotion."""
        if self._pending.get(choice.episode) != choice:
            raise RuntimeError("outcome reported for an episode that is not pending")
        del self._pending[choice.episode]
        self.episodes += 1
        if self.map_order == "plr":   # before the early return: replay/left-stage outcomes count per map
            j = self._plr_index[choice.stage].get(choice.lump)
            if j is not None:
                hist = self.plr_outcomes[choice.stage][j]
                hist.append(bool(success))
                del hist[:-2 * self.plr_window]
        if choice.stage != self.stage or (choice.replay and not self.count_replay):
            return False
        self.results.append(bool(success))
        thr = self.threshold(self.stage)
        if thr is None or len(self.results) < self.window:
            return False
        rate = sum(self.results) / len(self.results)
        if rate >= thr and self.stage_idx < len(self.stages) - 1:
            self.history.append({"episode": self.episodes, "from": self.stage,
                                 "to": self.stages[self.stage_idx + 1], "rate": rate})
            self.stage_idx += 1
            self.results.clear()
            return True
        return False

    # ---------------------------------------------------------------- resume
    def state_dict(self) -> dict:
        sd = {"seed": self.seed, "rng": self.rng.getstate(), "stage_idx": self.stage_idx,
              "results": list(self.results), "cursor": dict(self.cursor),
              "order": {k: list(v) for k, v in self.order.items()},
              "history": list(self.history), "episodes": self.episodes,
              "issued": self.issued,
              "pending": [c.__dict__ for c in self._pending.values()]}
        if self.map_order == "plr":   # shuffled_cycle state stays exactly as before
            sd["plr"] = {"maps": {s: [lump for lump, _ in self.pool[s]] for s in self.stages},
                         "outcomes": {s: [list(h) for h in v] for s, v in self.plr_outcomes.items()},
                         "stamp": {s: list(v) for s, v in self.plr_stamp.items()},
                         "clock": dict(self.plr_clock)}
        return sd

    def load_state_dict(self, sd: dict) -> None:
        st = sd["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]))
        self.seed, self.stage_idx, self.episodes = sd["seed"], sd["stage_idx"], sd["episodes"]
        self.results = deque(sd["results"], maxlen=self.window)
        self.cursor, self.order = dict(sd["cursor"]), {k: list(v) for k, v in sd["order"].items()}
        self.history = list(sd["history"])
        self.issued = sd["issued"]
        self._pending = {c["episode"]: EpisodeChoice(**c) for c in sd["pending"]}
        if self.map_order == "plr":
            p = sd.get("plr")
            if p is None:       # saved by shuffled_cycle: start PLR stats fresh
                self._plr_reset()
                return
            for s in self.stages:
                if p["maps"].get(s) != [lump for lump, _ in self.pool[s]]:
                    raise ValueError(f"PLR state for stage {s} was saved for a different map pool")
            self.plr_outcomes = {s: [[bool(x) for x in h] for h in p["outcomes"][s]] for s in self.stages}
            self.plr_stamp = {s: list(p["stamp"][s]) for s in self.stages}
            self.plr_clock = {s: int(p["clock"][s]) for s in self.stages}
