"""Curriculum map choice per episode, as a wrapper around any `train.vec_env.VecEnv`.

TRAINING ONLY, main process only, no torch import. It chooses maps, never actions:
the stage and map id it logs go into `final_infos[i]["curriculum"]` (read by
`train.trainer.episode_record` into episodes.jsonl) and into metrics.jsonl. They are
never put into an observation, so they never reach the model.

Per env it holds two scheduler choices:
  * `current[i]`  the choice env i is playing now;
  * `upcoming[i]` the choice env i will play after its current episode ends. It is
    sent with EVERY step as that env's auto-reset options (`{"map": lump}`), because
    which step ends an episode is only known after the step.
On an episode end the outcome (`final_infos[i]["level_exit"]`) is recorded for
`current[i]`, `upcoming[i]` becomes current and a new upcoming choice is issued. So
after a promotion, each env plays at most one more episode that was issued before
it (those outcomes are ignored by the scheduler: "an outcome from a stage the agent
has since left is ignored").

`reset()` abandons every running episode. The scheduler has no "cancel", so an env
that already holds a current choice restarts on that SAME choice (still pending, its
outcome is recorded when the restarted episode ends); only envs without one get a
new choice. This keeps `issued` and the pending set from growing over the two resets
a resumed trainer performs (Trainer.__init__ and Trainer.resume_session).

`state_json()` / `load_state_json()` save and restore the scheduler together with
current/upcoming choices and the per-stage logging windows, so a resume continues
the curriculum. Load it BEFORE the trainer's first reset.
"""
from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from .vec_env import ObsBatch, ResetOptions, VecStep

CURRICULUM_INFO_KEY = "curriculum"   # added to final_infos[i] on an episode end
MAP_OPTION_KEY = "map"               # DoomEnv.reset(options={"map": lump})
ENV_MAP_INFO_KEY = "map"             # DoomEnv info: the lump actually loaded
SUCCESS_INFO_KEY = "level_exit"
SHAPING_EPISODE_KEY = "shaping_episode"  # carries train.episode_diag fields (logging only)
STATE_FORMAT = 1


class CurriculumMismatch(RuntimeError):
    """An env did not play the map the curriculum chose for it."""


def _choice_dict(ch: Any) -> dict[str, Any] | None:
    return None if ch is None else dict(ch.__dict__)


def diag_metrics(eps: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per-update aggregates of train.episode_diag fields over the episodes that ended
    since the last call. NaN where undefined. Door episodes = doors_in_map > 0 with a
    measured (non-None) door_reversals; K4 reads diag_reversal_episode_share."""
    nan = math.nan
    edges = sum(int(e["use_edges"]) for e in eps)
    inr = sum(int(e["use_edges_in_range"]) for e in eps)
    door_eps = [e for e in eps if (e.get("doors_in_map") or 0) > 0 and e.get("door_reversals") is not None]
    rev = [int(e["door_reversals"]) for e in door_eps]
    onp = [e for e in eps if e.get("on_path_doors_in_map")]
    onp_in = sum(int(e["on_path_doors_in_map"]) for e in onp)
    return {
        "diag_episodes": float(len(eps)),
        "diag_use_edges_mean": float(edges / len(eps)) if eps else nan,
        "diag_use_in_range_share": float(inr / edges) if edges else nan,
        "diag_door_episodes": float(len(door_eps)),
        "diag_reversal_episode_share": float(sum(r > 0 for r in rev) / len(rev)) if rev else nan,
        "diag_reversals_per_door_episode": float(sum(rev) / len(rev)) if rev else nan,
        "diag_on_path_door_pass_rate": float(sum(int(e["on_path_doors_passed"]) for e in onp) / onp_in) if onp_in else nan,
    }


class CurriculumVecEnv:
    """Wraps a VecEnv; the trainer sees the same VecEnv interface."""

    def __init__(self, inner: Any, scheduler: Any, recent_window: int, bundle_sha256: str | None = None) -> None:
        """`bundle_sha256`: hash of the map bundle the scheduler's lumps index into; a
        saved state is refused on a different bundle (lump names and map order would
        no longer mean the same maps)."""
        if recent_window < 1:
            raise ValueError(f"recent_window must be >= 1, got {recent_window}")
        self.inner, self.sch = inner, scheduler
        self.bundle_sha256 = bundle_sha256
        self.num_envs = inner.num_envs
        self.single_observation_space = inner.single_observation_space
        self.single_action_space = inner.single_action_space
        self.recent_window = int(recent_window)
        self.current: list[Any] = [None] * self.num_envs
        self.upcoming: list[Any] = [None] * self.num_envs
        self.recent: dict[str, deque] = {s: deque(maxlen=self.recent_window) for s in scheduler.stages}
        self.episodes_finished = 0
        self._new_promotions: list[dict[str, Any]] = []
        self.dangling_restored = 0   # restored pending choices no env holds (env count shrank)
        self._diag: list[dict[str, Any]] = []   # C4 per-episode USE/door diagnostics since the last metrics()

    # ------------------------------------------------------------ choices
    def _fill(self) -> None:
        for i in range(self.num_envs):
            if self.current[i] is None:
                self.current[i] = self.sch.next_episode()
        for i in range(self.num_envs):
            if self.upcoming[i] is None:
                self.upcoming[i] = self.sch.next_episode()

    @staticmethod
    def _opts(ch: Any) -> dict[str, Any]:
        return {MAP_OPTION_KEY: ch.lump}

    def _check_played(self, i: int, info: dict[str, Any] | None, ch: Any, when: str) -> None:
        played = (info or {}).get(ENV_MAP_INFO_KEY)
        if played is not None and played != ch.lump:
            raise CurriculumMismatch(f"env {i} {when}: played map {played!r}, curriculum chose {ch.lump!r} "
                                     f"({ch.map_id}); reset options did not reach the env")

    # ------------------------------------------------------------ VecEnv API
    def reset(self, seed: int, options: Sequence[ResetOptions] | None = None) -> tuple[ObsBatch, list[dict[str, Any]]]:
        if options is not None:
            raise ValueError("CurriculumVecEnv chooses the reset options itself")
        self._fill()
        obs, infos = self.inner.reset(seed, [self._opts(c) for c in self.current])
        for i, info in enumerate(infos):
            self._check_played(i, info, self.current[i], "at reset")
        return obs, infos

    def step(self, actions: np.ndarray, reset_options: Sequence[ResetOptions] | None = None) -> VecStep:
        if reset_options is not None:
            raise ValueError("CurriculumVecEnv chooses the reset options itself")
        st = self.inner.step(actions, [self._opts(c) for c in self.upcoming])
        ended = np.asarray(st.terminated) | np.asarray(st.truncated)
        for i in np.flatnonzero(ended):
            ch = self.current[i]
            fi = dict(st.final_infos[i] or {})
            self._check_played(int(i), fi, ch, "at episode end")
            success = bool(fi.get(SUCCESS_INFO_KEY, False))
            n_hist = len(self.sch.history)
            promoted = bool(self.sch.record(ch, success))
            if promoted:
                self._new_promotions.extend(self.sch.history[n_hist:])
            fi[CURRICULUM_INFO_KEY] = {"stage": ch.stage, "map_id": ch.map_id, "lump": ch.lump,
                                       "replay": bool(ch.replay), "choice": int(ch.episode),
                                       "promoted": promoted, "stage_after": self.sch.stage}
            st.final_infos[i] = fi
            ep = fi.get(SHAPING_EPISODE_KEY)
            if isinstance(ep, dict) and "use_edges" in ep:
                self._diag.append(ep)
            self.recent[ch.stage].append(success)
            self.episodes_finished += 1
            self.current[i] = self.upcoming[i]
            self.upcoming[i] = self.sch.next_episode()
            self._check_played(int(i), st.infos[i], self.current[i], "after auto-reset")
        return st

    def close(self) -> None:
        self.inner.close()

    def __getattr__(self, name: str) -> Any:   # process_pids, kill_engines, query, ...
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    # ------------------------------------------------------------ logging
    def drain_promotions(self) -> list[dict[str, Any]]:
        out, self._new_promotions = self._new_promotions, []
        return out

    def metrics(self) -> dict[str, float]:
        """metrics.jsonl fields. NaN where undefined (no episode yet), like
        `explained_variance`."""
        sch = self.sch
        res = list(sch.results)
        out: dict[str, float] = {
            "curriculum_stage_idx": float(sch.stage_idx),
            "curriculum_promotions": float(len(sch.history)),
            "curriculum_episodes_recorded": float(sch.episodes),
            "curriculum_choices_issued": float(sch.issued),
            "curriculum_window_fill": float(len(res)),
            "curriculum_window_success": float(sum(res) / len(res)) if res else math.nan,
            "curriculum_replay_share_running": float(sum(bool(c.replay) for c in self.current) / self.num_envs),
        }
        for s in sch.stages:
            r = self.recent[s]
            out[f"curriculum_recent_success_{s}"] = float(sum(r) / len(r)) if r else math.nan
            out[f"curriculum_recent_episodes_{s}"] = float(len(r))
            # current stage mix: share of the envs whose running episode is on stage s
            out[f"curriculum_mix_{s}"] = float(sum(c.stage == s for c in self.current) / self.num_envs)
        if getattr(sch, "map_order", None) == "plr":   # PLR per-stage stats; no keys in shuffled_cycle
            out.update(sch.plr_metrics())
        out.update(diag_metrics(self._diag))
        self._diag = []   # per update: run_loop calls metrics() once per update
        return out

    # ------------------------------------------------------------ resume
    def state_json(self) -> str:
        return json.dumps({
            "format": STATE_FORMAT, "bundle_sha256": self.bundle_sha256, "scheduler": self.sch.state_dict(),
            "current": [_choice_dict(c) for c in self.current],
            "upcoming": [_choice_dict(c) for c in self.upcoming],
            "recent": {s: list(map(bool, d)) for s, d in self.recent.items()},
            "episodes_finished": self.episodes_finished,
        })

    def load_state_json(self, text: str) -> None:
        """Restore before the first reset. With a different env count, the first
        `num_envs` saved choices are kept; extra envs get new choices, and saved choices
        no env holds stay pending in the scheduler (it has no cancel), counted in
        `dangling_restored`."""
        from curriculum.scheduler import EpisodeChoice
        d = json.loads(text)
        if d.get("format") != STATE_FORMAT:
            raise ValueError(f"curriculum state format {d.get('format')!r} != {STATE_FORMAT}")
        if d.get("bundle_sha256") != self.bundle_sha256:
            raise ValueError(f"curriculum state was saved for bundle sha256 {d.get('bundle_sha256')}, this run "
                             f"uses {self.bundle_sha256}: the map bundle changed; refusing to continue the curriculum")
        self.sch.load_state_dict(d["scheduler"])
        pending = {c["episode"] for c in d["scheduler"]["pending"]}

        def restore(items: list) -> list[Any]:
            out = [None if c is None else EpisodeChoice(**c) for c in items[:self.num_envs]]
            for c in out:
                if c is not None and c.episode not in pending:
                    raise ValueError(f"restored choice {c} is not pending in the scheduler state")
            return out + [None] * (self.num_envs - len(out))

        self.current, self.upcoming = restore(d["current"]), restore(d["upcoming"])
        held = {c.episode for c in self.current + self.upcoming if c is not None}
        self.dangling_restored = len(pending - held)
        self.recent = {s: deque(d["recent"].get(s, []), maxlen=self.recent_window) for s in self.sch.stages}
        self.episodes_finished = int(d["episodes_finished"])
