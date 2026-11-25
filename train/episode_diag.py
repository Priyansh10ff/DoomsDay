"""Per-episode USE and door diagnostics for training runs (strategy C4, kill criteria K3/K4).

TRAINING-SIDE LOGGING ONLY. Nothing here is written into an observation, read by the
model, or used to choose, filter or change an action: the wrapper passes the action to
the env untouched and only READS it afterwards. The evaluation / inference path
(`evaluate/`, `agent/`) never builds this wrapper. Privileged inputs (pose, map geometry,
the map record, engine door gaps from DoomEnv.diag_door_gaps(), read here in the
worker and never added to the returned info) are fine here, as in rewards/.

Per-episode fields, merged into `info["shaping_episode"]` on the ending step (so
train.trainer.episode_record copies them into episodes.jsonl unchanged):

  use_edges               executed USE 0 -> 1 edges (USE head of consecutive decisions;
                          the first decision of an episode counts against "released").
  use_edges_in_range      edges made while, at the START of that decision (pre-step pose
                          and door state), the USE trace (USE_RANGE units from the
                          player along the view heading) crosses a CLOSED door face
                          (gap <= DOOR_CLOSED_GAP) or a use-activated exit line. No
                          occlusion test. Without engine gaps every door face counts.
  use_edges_in_range_door / use_edges_in_range_exit   the two parts (an edge can be both).
  door_reversals          times a door that was rising (or open, gap > closed) starts
                          falling within REVERSAL_WINDOW_TICS of a player press whose USE
                          trace crossed that door's face. Doors that fall on their own
                          (DR wait ~150 tics) without such a press are not counted.
                          None when the engine's sector gaps are unavailable (obs v1,
                          telemetry off): NOT 0, so K4 never passes on missing data.
  door_ids_matched        every door sector of the map appeared in the env's door gaps
                          (engine sector id == WAD sector index); None without gaps.
  on_path_doors_passed / on_path_doors_in_map   doors on the start -> key -> exit route
                          (room graph of the map record, shortest route by corridor hops;
                          a door is on it when its corridor joins two consecutive rooms of
                          the route). None when the map's doors cannot be tied to corridors.
  exit_kind               the map record's graph.exit_kind, as recorded ("pad" = walk-over
                          exit line, "switch" = use-activated exit line).
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from maps.doors import AXIS_X, DoorSpec, door_specs
from rewards.doors import DoorProgress

USE_HEAD = 4                    # env.actions.HEAD_NAMES index of "use"
USE_RANGE = 64.0                # ZDoom PlayerPawn UseRange default (vanilla USERANGE 64)
DOOR_CLOSED_GAP = 0.5           # same as env.door_cue.DOOR_CLOSED_GAP
GAP_EPS = 0.25                  # gap change below this is "not moving"
REVERSAL_WINDOW_TICS = 35       # ~1 s at 35 tics/s
USE_EXIT_SPECIALS = frozenset({11, 51})   # S1 exit / S1 secret exit (use-activated)
INFO_EPISODE = "shaping_episode"
DIAG_FIELDS = ("use_edges", "use_edges_in_range", "use_edges_in_range_door", "use_edges_in_range_exit",
               "door_reversals", "door_ids_matched", "on_path_doors_passed", "on_path_doors_in_map",
               "exit_kind")


# ------------------------------------------------------------------ map side (built once)
@dataclass(frozen=True)
class DiagMap:
    doors: tuple[DoorSpec, ...]
    on_path: tuple[bool, ...] | None          # per door; None if not derivable
    exit_kind: str | None
    exit_use_lines: tuple[tuple[float, float, float, float], ...]


def door_faces(d: DoorSpec) -> tuple[tuple[float, float, float, float], ...]:
    """The two face segments of a door slab, world coordinates."""
    if d.axis == AXIS_X:
        return ((d.a0, d.lo, d.a0, d.hi), (d.a1, d.lo, d.a1, d.hi))
    return ((d.lo, d.a0, d.hi, d.a0), (d.lo, d.a1, d.hi, d.a1))


def _route_edges(graph: Mapping[str, Any]) -> set[frozenset] | None:
    adj: dict[int, list[int]] = {}
    for c in graph.get("corridors", []):
        adj.setdefault(c["a"], []).append(c["b"])
        adj.setdefault(c["b"], []).append(c["a"])

    def path(s: int, t: int) -> list[int] | None:
        prev, q = {s: None}, deque([s])
        while q:
            u = q.popleft()
            if u == t:
                out = [u]
                while prev[out[-1]] is not None:
                    out.append(prev[out[-1]])
                return out[::-1]
            for v in adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        return None

    stops = [graph["start_room"]] + ([graph["key_room"]] if graph.get("key_room") is not None else []) \
        + [graph["exit_room"]]
    edges: set[frozenset] = set()
    for s, t in zip(stops, stops[1:]):
        p = path(s, t)
        if p is None:
            return None
        edges |= {frozenset(e) for e in zip(p, p[1:])}
    return edges


def _door_rooms(m: dict, door_sector: int, room_of_sector: Mapping[int, int], door_secs: set[int]) -> set[int]:
    """Rooms reached from a door sector through non-room, non-door sectors (its corridor)."""
    sides = m["sidedefs"]
    nbr: dict[int, set[int]] = {}
    for ld in m["linedefs"]:
        if ld.back == 0xFFFF:
            continue
        a, b = sides[ld.front].sector, sides[ld.back].sector
        if a != b:
            nbr.setdefault(a, set()).add(b)
            nbr.setdefault(b, set()).add(a)
    rooms, seen, q = set(), {door_sector}, deque([door_sector])
    while q:
        u = q.popleft()
        for v in nbr.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v in room_of_sector:
                rooms.add(room_of_sector[v])
            elif v not in door_secs:
                q.append(v)
    return rooms


def diag_map(m: dict, record: Mapping[str, Any] | None) -> DiagMap:
    doors = door_specs(m)
    verts = m["vertices"]
    exit_lines = tuple((float(verts[ld.v1][0]), float(verts[ld.v1][1]), float(verts[ld.v2][0]), float(verts[ld.v2][1]))
                       for ld in m["linedefs"] if ld.special in USE_EXIT_SPECIALS)
    on_path, kind = None, None
    if record is not None:
        g = record["graph"]
        kind = g.get("exit_kind")
        edges = _route_edges(g)
        room_of = {int(s): i for i, s in enumerate(g.get("room_sector", []))}
        if edges is not None and room_of:
            secs = {d.sector for d in doors}
            pairs = [_door_rooms(m, d.sector, room_of, secs) for d in doors]
            if all(len(p) == 2 for p in pairs):
                on_path = tuple(frozenset(p) in edges for p in pairs)
    return DiagMap(doors, on_path, kind, exit_lines)


def _bundle_members(wad_path: str) -> dict[str, str]:
    """{lump: map id} of the bundle at `wad_path`, from maps/splits.json (no re-hashing)."""
    from maps.loader import REPO_ROOT, load_splits
    want = os.path.normcase(os.path.abspath(wad_path))
    for b in load_splits().data["bundles"]:
        if os.path.normcase(os.path.abspath(os.path.join(REPO_ROOT, b["file"]))) == want:
            return dict(zip(b["lumps"], b["members"]))
    return {}


def diag_table(wad_path: str, lumps: Iterable[str]) -> tuple[tuple[str, DiagMap], ...]:
    """((lump, DiagMap), ...) for the named maps of a generated bundle. Picklable."""
    from maps.loader import REPO_ROOT, load_splits
    from maps.qa.validate import load_map
    members = _bundle_members(wad_path)
    splits = load_splits()
    out = []
    for lump in sorted(set(lumps)):
        rec = None
        mid = members.get(lump)
        if mid is not None:
            _, entry = splits.by_id(mid)
            p = os.path.join(REPO_ROOT, os.path.splitext(entry["file"])[0] + ".json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    rec = json.load(f)
        out.append((lump, diag_map(load_map(wad_path, lump), rec)))
    return tuple(out)


# ------------------------------------------------------------------ geometry
def _seg_cross(px: float, py: float, qx: float, qy: float, s: tuple[float, float, float, float]) -> bool:
    """Segment P-Q intersects segment s (endpoints inclusive)."""
    ax, ay, bx, by = s

    def orient(x1, y1, x2, y2, x3, y3):
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    d1, d2 = orient(ax, ay, bx, by, px, py), orient(ax, ay, bx, by, qx, qy)
    d3, d4 = orient(px, py, qx, qy, ax, ay), orient(px, py, qx, qy, bx, by)
    return (d1 * d2 <= 0.0) and (d3 * d4 <= 0.0) and not (d1 == d2 == 0.0)


def use_trace(x: float, y: float, heading_deg: float, rng: float = USE_RANGE) -> tuple[float, float]:
    th = math.radians(heading_deg)
    return x + rng * math.cos(th), y + rng * math.sin(th)


# ------------------------------------------------------------------ per-episode logic
class EpisodeDiag:
    """Pure logic. reset(dmap, info), then step(use_bit, info_before, info_after) per decision."""

    def __init__(self, door_near_margin: float = 48.0) -> None:
        self.margin = float(door_near_margin)
        self.dm: DiagMap | None = None

    def reset(self, dmap: DiagMap, info: Mapping[str, Any], lump: str = "_") -> None:
        self.dm = dmap
        self.faces = [door_faces(d) for d in dmap.doors]
        self.prog = DoorProgress(((lump, dmap.doors),), 1.0, self.margin)
        self.prog.reset({**info, "map": lump})
        self.lump = lump
        self.prev_use = 0
        self.edges = self.in_range = self.in_door = self.in_exit = 0
        self.reversals = 0
        self.have_gaps = False
        self.ids_matched: bool | None = None
        n = len(dmap.doors)
        self.gap: list[float | None] = [None] * n
        self.falling = [False] * n
        self.press_tic: list[float] = [-math.inf] * n
        self._gaps(info)

    def _gaps(self, info: Mapping[str, Any]) -> None:
        g = info.get("door_gaps")
        if g is None or info.get("final_obs_fresh") is False:
            return
        if not self.have_gaps:
            self.have_gaps = True
            self.ids_matched = all(d.sector in g for d in self.dm.doors)
        tic = float(info.get("tic", 0))
        for k, d in enumerate(self.dm.doors):
            new = g.get(d.sector)
            if new is None:
                continue
            old = self.gap[k]
            if old is not None:
                if new < old - GAP_EPS:
                    if not self.falling[k] and old > DOOR_CLOSED_GAP and tic - self.press_tic[k] <= REVERSAL_WINDOW_TICS:
                        self.reversals += 1
                    self.falling[k] = True
                elif new > old + GAP_EPS:
                    self.falling[k] = False
            self.gap[k] = float(new)

    def step(self, use: int, before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
        use = 1 if int(use) else 0
        if use and not self.prev_use:
            self.edges += 1
            x, y = float(before["pos_x"]), float(before["pos_y"])
            qx, qy = use_trace(x, y, float(before["angle"]))
            hit_closed = False
            for k, fs in enumerate(self.faces):
                if any(_seg_cross(x, y, qx, qy, f) for f in fs):
                    self.press_tic[k] = float(before.get("tic", 0))
                    g = self.gap[k]
                    if g is None or g <= DOOR_CLOSED_GAP:
                        hit_closed = True
            hit_exit = any(_seg_cross(x, y, qx, qy, s) for s in self.dm.exit_use_lines)
            self.in_door += hit_closed
            self.in_exit += hit_exit
            self.in_range += bool(hit_closed or hit_exit)
        self.prev_use = use
        self._gaps(after)
        if after.get("final_obs_fresh") is not False and "pos_x" in after:
            self.prog.step({**after, "map": self.lump})

    def episode(self) -> dict[str, Any]:
        dm = self.dm
        flags = self.prog.passed_flags
        on = None if dm.on_path is None else sum(dm.on_path)
        passed = None if dm.on_path is None else sum(f and p for f, p in zip(flags, dm.on_path))
        return {"use_edges": self.edges, "use_edges_in_range": self.in_range,
                "use_edges_in_range_door": self.in_door, "use_edges_in_range_exit": self.in_exit,
                "door_reversals": self.reversals if self.have_gaps else None,
                "door_ids_matched": self.ids_matched,
                "on_path_doors_passed": passed, "on_path_doors_in_map": on, "exit_kind": dm.exit_kind}


class EpisodeDiagEnv(gym.Wrapper):
    """Reads the executed action and info; never changes the action, obs or reward."""

    def __init__(self, env: gym.Env, table: Any, door_near_margin: float = 48.0) -> None:
        super().__init__(env)
        self.table = dict(table)
        self.diag = EpisodeDiag(door_near_margin)
        self._info: Mapping[str, Any] = {}

    def _with_gaps(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """A private copy of info plus the env's door gaps; the returned info is untouched."""
        get = getattr(self.env.unwrapped, "diag_door_gaps", None)
        gaps = get() if callable(get) else None
        d = dict(info)
        if gaps is not None:
            d["door_gaps"] = gaps
        return d

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        lump = info["map"]
        mine = self._with_gaps(info)
        self.diag.reset(self.table[lump], mine, lump)
        self._info = mine
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        use = int(np.asarray(action).reshape(-1)[USE_HEAD])
        mine = self._with_gaps(info)
        self.diag.step(use, self._info, mine)
        self._info = mine
        if (terminated or truncated) and isinstance(info.get(INFO_EPISODE), dict):
            info = dict(info)
            info[INFO_EPISODE] = {**info[INFO_EPISODE], **self.diag.episode()}
        return obs, reward, terminated, truncated, info
