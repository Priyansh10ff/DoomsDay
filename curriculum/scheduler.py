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
  * Within a stage, maps are visited in a shuffled cycle (every map once per cycle).
  * One `random.Random(seed)` drives every draw, so the same seed and the same success
    history give the same sequence of choices. `state_dict()` / `load_state_dict()`
    resume it exactly.
"""
from __future__ import annotations

import json
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


def load_config(path: str = DEFAULT_CONFIG) -> dict:
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

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_bundle(cls, config: dict | None = None, seed: int | None = None, splits=None):
        """Pool from the verified training bundle (maps/loader.training_bundle: train
        split only, hashes checked, held-out refused)."""
        from maps.loader import training_bundle
        config = config or load_config()
        path, lumps, stages = training_bundle(config.get("bundle_id", "train-all"), splits)
        pool: dict[str, list] = {}
        for lump, mid in lumps.items():
            pool.setdefault(stages[lump], []).append((lump, mid))
        sch = cls(config, pool, seed)
        sch.bundle_path = path
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
        return {"seed": self.seed, "rng": self.rng.getstate(), "stage_idx": self.stage_idx,
                "results": list(self.results), "cursor": dict(self.cursor),
                "order": {k: list(v) for k, v in self.order.items()},
                "history": list(self.history), "episodes": self.episodes,
                "issued": self.issued,
                "pending": [c.__dict__ for c in self._pending.values()]}

    def load_state_dict(self, sd: dict) -> None:
        st = sd["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]))
        self.seed, self.stage_idx, self.episodes = sd["seed"], sd["stage_idx"], sd["episodes"]
        self.results = deque(sd["results"], maxlen=self.window)
        self.cursor, self.order = dict(sd["cursor"]), {k: list(v) for k, v in sd["order"].items()}
        self.history = list(sd["history"])
        self.issued = sd["issued"]
        self._pending = {c["episode"]: EpisodeChoice(**c) for c in sd["pending"]}
