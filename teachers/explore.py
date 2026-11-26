"""Information-limited EXPLORATION teacher (TRAINING-TIME ONLY; see teachers/__init__.py).

Why: the route teacher (teachers/navigator.py) walks the shortest path to the exit using
the whole map file, so its labels depend on things the student cannot see (PIRLNav:
behaviour cloning of a privileged shortest-path navigator 6.4% vs 52% from human demos,
docs/research/2026-11-25-blogs-reddit-doors-and-bc-ppo.md). This teacher decides only
from what the player has SEEN this episode, so its labels are learnable from the
student's own observation and memory.

Knowledge (per episode, `Knowledge`), updated every decision from the player's pose:
  * 80 view rays over the 90-degree horizontal field of view, at the same column angles
    as the student's rays / egomap (atan((2k+1-80)/80) off the view axis), cast against
    the map's one-sided lines and the faces of every door that is closed NOW (live gap
    below VIS_OPEN_GAP). Cells along a ray up to the hit (capped at VIEW_RANGE, the
    egomap's ~450-unit sensor range) are SEEN; the cell just beyond a wall hit within
    range is KNOWN BLOCKED (the student's egomap marks the same wall cell); a door-face
    hit makes that door KNOWN. The player's own cell is always known.
  * Keys: seen when the key's cell is seen. Exit: an exit-switch line hit within range,
    or a seen exit-pad cell. Locked doors: learned ONLY from the env's "needs key" pulse
    (observation self vector, last 3 values) after a USE press, attributed to the door
    under the 64-unit use trace at the previous pose (any executed press: label, noise
    or student).
Goal (re-chosen when the current goal is reached, invalid, timed out, or a higher-
priority goal appears):
  1. the exit, if seen and reachable through known space (seen doors passable unless
     known locked without the key);
  2. a seen key not held;
  3. a known locked door whose key is held;
  4. the cheapest FRONTIER: a known free cell next to unknown space (an unknown cell with
     at least one unknown neighbour; single-cell pockets are ignored), or a passable seen
     door with unknown space next to it (opened and crossed). Cost = path cost + a
     penalty for frontiers behind the player; the choice is kept until it is resolved.
     Frontiers reached but never resolved, and goals that time out, are blacklisted.
Paths: Dijkstra over KNOWN cells only (unseen cells are never walkable). Execution: the
route controller (teachers/controller.py, unchanged) on a dynamic Route built over the
known grid: a door anchor per door the path enters, then the goal anchor. USE labels come
from teachers/use_label.py with lock knowledge limited to doors KNOWN locked (so the
teacher presses an unknown locked door once and learns it). Combat: teachers/combat.py
through the controller, unchanged (visible monsters only).
Goal-mode label (teachers/goal_modes.py, 2026-11-26): `_leg_mode` maps the committed goal
(frontier / door frontier / spin -> EXPLORE, key -> GET_KEY, lockdoor -> GO_LOCKED_DOOR,
exit -> GO_EXIT); combat, unstick and door handling come first (teachers/controller.py).
Door-loop fix (2026-11-26, ExploreConfig.crossed_max_across): a door anchor counts as
crossed only near the door's axis; see the config field.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace

import numpy as np

from teachers import goal_modes as GM
from maps.qa.validate import DOOR_SPECIALS, KEY_COLOR_OF_THING, KEYED_SPECIALS, load_map
from maps.wad import NO_SIDE
from teachers.controller import ControllerConfig, TeacherController
from teachers.navigator import (EXIT_SWITCH_SPECIAL, INF, PLAYER_THING, SQRT2, VOID, Anchor,
                                Grid, NavConfig, Route, _door_anchor, _exit_anchor, build_grid)
from teachers.use_label import (KIND_DOOR, KIND_SWITCH, USE_RANGE_ENGINE, UseLines, facing_hit,
                                lines_from_map, on_front)

N_VIEW_COLS = 80                 # env/observation.N_RAY_COLS
VIEW_RANGE = 450.0               # egomap sensor range (env/egomap.py, measured 2026-11-26)
RAY_STEP = 8.0                   # sample spacing along a view ray
VIS_OPEN_GAP = 40.0              # a door blocks sight while its gap is below this (eye ~41)
KEY_COLOURS = ("blue", "yellow", "red")   # order of the v2 key bits (env/door_cue.key_bits)
LOCK_ATTRIB_RANGE = USE_RANGE_ENGINE + 16.0


@dataclass(frozen=True)
class ExploreConfig:
    replan_every: int = 8            # decisions between routine re-plans (fields grow with knowledge)
    ahead_weight: float = 8.0        # frontier cost penalty (cells) for a frontier straight behind
    frontier_timeout: int = 160      # decisions on one frontier / door frontier before blacklisting
    goal_timeout: int = 500          # decisions on an exit / key / locked-door goal before blacklisting
    reach_radius: float = 40.0       # "at" a frontier cell
    reach_patience: int = 6          # decisions at an unresolved frontier before blacklisting it
    spin_decisions: int = 12         # turning in place when there is no goal at all
    door_cost: float = 4.0           # path-cost extra for entering a door
    wall_penalty: float = 3.0        # path-cost factor for a cell next to blocked/unknown space
    # Door loop fix (2026-11-26). Cause of the bc_v6_explore door-goal time-outs: the
    # controller's `_crossed` test is a half-plane (distance past the door centre along the
    # travel direction) with no lateral bound, so a door anchor planned from a player far
    # off to the side (S3-train-140040: 682 units away, 507 across the door axis) counts
    # as crossed at once; it completes on every re-plan and the label is a no-op until the
    # goal times out. Here a door also counts as crossed only within this lateral distance
    # of the door centre (4 cells). 0 = off (the bc_v6_explore behaviour).
    crossed_max_across: float = 128.0


# ------------------------------------------------------------------ static map
@dataclass
class ExploreMap:
    """Static geometry of one map (cached per worker). Knowledge is per episode."""
    map_id: str
    m: dict
    grid: Grid                         # TRUE grid: used only through the seen/known masks
    vis_seg: np.ndarray                # (N, 4) sight blockers
    vis_door: np.ndarray               # (N,) door sector, or -1 for a wall
    vis_switch: np.ndarray             # (N,) bool: exit switch line
    use_lines: UseLines                # true stoppers (engine diagnostics, lock attribution)
    door_sectors: tuple[int, ...]
    door_cells: dict                   # sector -> list of (c, r)
    keys: list                         # (x, y, colour)
    exit_anchor: Anchor
    exit_cells: list
    exit_is_switch: bool
    start: tuple[float, float]
    nav: NavConfig = field(default_factory=NavConfig)


def build_explore_map(m: dict, map_id: str = "", nav: NavConfig = NavConfig()) -> ExploreMap:
    grid = build_grid(m, nav.cell)
    verts, sides = m["vertices"], m["sidedefs"]
    door_secs = sorted({sides[ld.back].sector for ld in m["linedefs"]
                        if ld.back != NO_SIDE and ld.special in DOOR_SPECIALS})
    ds = set(door_secs)
    seg, vdoor, vsw = [], [], []
    for ld in m["linedefs"]:
        (x1, y1), (x2, y2) = verts[ld.v1], verts[ld.v2]
        if ld.back == NO_SIDE:
            seg.append((x1, y1, x2, y2))
            vdoor.append(-1)
            vsw.append(ld.special == EXIT_SWITCH_SPECIAL)
            continue
        s = {sides[ld.front].sector, sides[ld.back].sector} & ds
        if s:                                   # any face of a door sector: blocks when closed
            seg.append((x1, y1, x2, y2))
            vdoor.append(next(iter(s)))
            vsw.append(False)
    use_lines = lines_from_map(m, DOOR_SPECIALS, KEYED_SPECIALS, (EXIT_SWITCH_SPECIAL, 51), NO_SIDE)
    exit_anchor, exit_cells = _exit_anchor(grid, m, nav)
    keys = [(float(t.x), float(t.y), KEY_COLOR_OF_THING[t.type]) for t in m["things"]
            if t.type in KEY_COLOR_OF_THING]
    st = next(t for t in m["things"] if t.type == PLAYER_THING)
    cells = {s: [(int(c), int(r)) for c, r in zip(*np.nonzero(grid.sector == s))] for s in door_secs}
    return ExploreMap(map_id, m, grid, np.asarray(seg, np.float64).reshape(-1, 4),
                      np.asarray(vdoor, np.int64), np.asarray(vsw, bool), use_lines,
                      tuple(door_secs), cells, keys, exit_anchor, exit_cells,
                      exit_anchor.kind == "exit_switch", (float(st.x), float(st.y)), nav)


def load_explore_map(wad_path: str, map_name: str = "MAP01", map_id: str = "") -> ExploreMap:
    return build_explore_map(load_map(wad_path, map_name), map_id)


def view_angles() -> np.ndarray:
    """Column angle offsets (degrees CCW from the view axis; screen right = clockwise)."""
    k = np.arange(N_VIEW_COLS)
    return -np.degrees(np.arctan((2 * k + 1 - N_VIEW_COLS) / N_VIEW_COLS))


_VIEW_OFFSETS = view_angles()


def cast_rays(seg: np.ndarray, blocking: np.ndarray, x: float, y: float, ang_deg: float,
              max_range: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dirs (R, 2), hit distance (R,) capped at max_range, hit segment index or -1)."""
    th = np.radians(ang_deg + _VIEW_OFFSETS)
    rx, ry = np.cos(th), np.sin(th)
    R = th.shape[0]
    dist = np.full(R, max_range)
    idx = np.full(R, -1, np.int64)
    s = seg[blocking]
    if s.shape[0]:
        ex, ey = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
        wx, wy = s[:, 0] - x, s[:, 1] - y
        den = rx[:, None] * ey[None, :] - ry[:, None] * ex[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (wx[None, :] * ey[None, :] - wy[None, :] * ex[None, :]) / den
            u = (wx[None, :] * ry[:, None] - wy[None, :] * rx[:, None]) / den
        ok = (np.abs(den) > 1e-12) & (t >= 0.0) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
        t = np.where(ok, t, np.inf)
        j = np.argmin(t, axis=1)
        tj = t[np.arange(R), j]
        hit = tj <= max_range
        dist = np.where(hit, tj, max_range)
        orig = np.nonzero(blocking)[0]
        idx = np.where(hit, orig[j], -1)
    return np.stack([rx, ry], 1), dist, idx


# ------------------------------------------------------------------ per-episode knowledge
class Knowledge:
    """What the player has seen this episode (see the module docstring)."""

    def __init__(self, xm: ExploreMap) -> None:
        self.xm = xm
        W, H = xm.grid.shape
        self.seen = np.zeros((W, H), bool)       # cell observed open
        self.blocked = np.zeros((W, H), bool)    # cell known to be wall (beyond a wall hit)
        self.doors_seen: set[int] = set()
        self.locked: dict[int, str] = {}         # door sector -> colour, from the pulse only
        self.keys_seen: set[int] = set()
        self.exit_seen = False
        self.prev_pose: tuple[float, float, float] | None = None
        self.version = 0                         # bumps when anything new is known

    # ---------------------------------------------------------------- updates
    def observe(self, x: float, y: float, ang: float, door_gaps: dict) -> None:
        xm, g = self.xm, self.xm.grid
        before = (int(self.seen.sum()), len(self.doors_seen), len(self.keys_seen), self.exit_seen)
        closed = np.array([d < 0 or float(door_gaps.get(int(d), 0.0)) < VIS_OPEN_GAP
                           for d in xm.vis_door], bool) if xm.vis_door.size else np.zeros(0, bool)
        dirs, dist, idx = cast_rays(xm.vis_seg, closed, x, y, ang, VIEW_RANGE)
        n = int(math.ceil(VIEW_RANGE / RAY_STEP)) + 1
        ts = np.arange(n) * RAY_STEP
        tt = np.minimum(ts[None, :], np.maximum(dist[:, None] - 1.0, 0.0))
        px = x + dirs[:, 0:1] * tt
        py = y + dirs[:, 1:2] * tt
        c = np.floor(px / g.cell).astype(np.int64) - g.i0
        r = np.floor(py / g.cell).astype(np.int64) - g.j0
        ok = (c >= 0) & (c < g.shape[0]) & (r >= 0) & (r < g.shape[1])
        c, r = c[ok], r[ok]
        open_ = g.sector[c, r] != VOID
        self.seen[c[open_], r[open_]] = True
        dc = g.door_of[c[open_], r[open_]]
        self.doors_seen.update(int(d) for d in np.unique(dc) if d != VOID)
        for k in np.nonzero(idx >= 0)[0]:
            j = int(idx[k])
            d = int(xm.vis_door[j])
            if d >= 0:
                self.doors_seen.add(d)
                continue
            if xm.vis_switch[j]:
                self.exit_seen = True
            bx = x + dirs[k, 0] * (dist[k] + 2.0)
            by = y + dirs[k, 1] * (dist[k] + 2.0)
            bc = g.cell_of(bx, by)
            if g.inside(*bc) and not self.seen[bc]:
                self.blocked[bc] = True
        self.mark_body(x, y)
        if not self.exit_seen and not xm.exit_is_switch:
            self.exit_seen = any(self.seen[cc] for cc in xm.exit_cells)
        for i, (kx, ky, _) in enumerate(xm.keys):
            kc = g.cell_of(kx, ky)
            if i not in self.keys_seen and g.inside(*kc) and self.seen[kc]:
                self.keys_seen.add(i)
        after = (int(self.seen.sum()), len(self.doors_seen), len(self.keys_seen), self.exit_seen)
        if after != before:
            self.version += 1

    def mark_body(self, x: float, y: float) -> None:
        """The player's own cell (it stands there) is known open."""
        g = self.xm.grid
        cc = g.cell_of(x, y)
        if g.inside(*cc) and g.sector[cc] != VOID:
            self.seen[cc] = True
            self.blocked[cc] = False
            if g.door_of[cc] != VOID:
                self.doors_seen.add(int(g.door_of[cc]))

    def learn_lock(self, needs: np.ndarray, pose: tuple[float, float, float] | None) -> int | None:
        """A needs-key pulse (3 booleans b, y, r): the door under the use trace at `pose`
        (the pose the press was made from) is locked. Returns the sector or None."""
        if pose is None or not np.any(needs):
            return None
        colour = KEY_COLOURS[int(np.argmax(needs))]
        L = self.xm.use_lines
        hit = facing_hit(L, pose[0], pose[1], pose[2], LOCK_ATTRIB_RANGE)
        if hit is None or int(L.kind[hit[0]]) != KIND_DOOR:
            return None
        sec = int(L.sector[hit[0]])
        if self.locked.get(sec) != colour:
            self.locked[sec] = colour
            self.doors_seen.add(sec)
            self.version += 1
        return sec

    # ---------------------------------------------------------------- derived
    def door_cell_mask(self) -> np.ndarray:
        g = self.xm.grid
        if not self.doors_seen:
            return np.zeros(g.shape, bool)
        return np.isin(g.door_of, list(self.doors_seen))

    def known_grid(self) -> Grid:
        """Unseen cells are VOID; door cells only for seen doors."""
        g = self.xm.grid
        free = self.seen & (g.sector != VOID) & (g.door_of == VOID)
        dmask = self.door_cell_mask()
        sector = np.where(free | dmask, g.sector, VOID)
        door_of = np.where(dmask, g.door_of, VOID)
        return Grid(g.i0, g.j0, sector, door_of, g.cell)

    def unknown_mask(self) -> np.ndarray:
        """Unknown space worth exploring: not seen, not known blocked, not a known door
        cell, with at least one 4-neighbour that is also unknown (no single-cell pockets)."""
        unk = ~self.seen & ~self.blocked & ~self.door_cell_mask()
        nb = np.zeros(unk.shape, np.int64)
        nb[1:, :] += unk[:-1, :]
        nb[:-1, :] += unk[1:, :]
        nb[:, 1:] += unk[:, :-1]
        nb[:, :-1] += unk[:, 1:]
        return unk & (nb >= 1)

    def frontier_mask(self) -> np.ndarray:
        g = self.xm.grid
        free = self.seen & (g.sector != VOID) & (g.door_of == VOID)
        return free & _touches(self.unknown_mask())

    def door_frontiers(self, passable: set[int]) -> list[int]:
        """Passable seen doors with unknown space next to any of their cells."""
        u = _touches(self.unknown_mask())
        return [s for s in sorted(self.doors_seen)
                if s in passable and any(u[c] for c in self.xm.door_cells.get(s, ()))]


def _touches(mask: np.ndarray) -> np.ndarray:
    """Cells with a 4-neighbour in `mask`."""
    out = np.zeros(mask.shape, bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


# ------------------------------------------------------------------ search over the known grid
def _walk(grid: Grid, open_doors) -> np.ndarray:
    w = grid.sector != VOID
    d = grid.door_of != VOID
    if open_doors:
        return w & (~d | np.isin(grid.door_of, list(open_doors)))
    return w & ~d


def _near_blocked(walk: np.ndarray) -> np.ndarray:
    """Walkable cells with a non-walkable 8-neighbour (or the grid edge)."""
    b = np.pad(~walk, 1, constant_values=True)
    W, H = walk.shape
    out = np.zeros_like(walk)
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc or dr:
                out |= b[1 + dc:1 + dc + W, 1 + dr:1 + dr + H]
    return walk & out


_STEPS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))


def dijkstra(grid: Grid, sources: list[tuple[int, int]], open_doors, wall_penalty: float,
             door_cost: float = 0.0, reverse: bool = False):
    """Cost from the sources to every cell over walkable cells (doors in `open_doors`
    passable; entering a door cell from a non-door cell costs `door_cost` extra; leaving
    a cell next to blocked space costs `wall_penalty` times more). Returns (dist, prev).
    reverse=True gives cost-to-go toward the sources (the cost of the cell being left)."""
    walk = _walk(grid, open_doors)
    near = _near_blocked(walk)
    W, H = walk.shape
    dist = np.full((W, H), INF)
    prev = -np.ones((W, H, 2), np.int64)
    pq = []
    for c, r in sources:
        if 0 <= c < W and 0 <= r < H and walk[c, r]:
            dist[c, r] = 0.0
            pq.append((0.0, c, r))
    heapq.heapify(pq)
    dd = grid.door_of
    while pq:
        d, c, r = heapq.heappop(pq)
        if d > dist[c, r]:
            continue
        for dc, dr, step in _STEPS:
            nc, nr = c + dc, r + dr
            if not (0 <= nc < W and 0 <= nr < H and walk[nc, nr]):
                continue
            if dc and dr and not (walk[c + dc, r] and walk[c, r + dr]):
                continue
            leave = (nc, nr) if reverse else (c, r)
            cost = step * (1.0 + (wall_penalty if near[leave] else 0.0))
            if door_cost:
                a, b = ((nc, nr), (c, r)) if reverse else ((c, r), (nc, nr))
                if dd[b] != VOID and dd[a] == VOID:
                    cost += door_cost
            nd = d + cost
            if nd < dist[nc, nr]:
                dist[nc, nr] = nd
                prev[nc, nr] = (c, r)
                heapq.heappush(pq, (nd, nc, nr))
    return dist, prev


def path_to(prev: np.ndarray, end: tuple[int, int]) -> list[tuple[int, int]]:
    out = [end]
    while True:
        p = prev[out[-1]]
        if p[0] < 0:
            break
        out.append((int(p[0]), int(p[1])))
    return out[::-1]


# ------------------------------------------------------------------ controller
class ExploreController(TeacherController):
    """TeacherController whose route is re-planned over the known grid (module docstring).
    `reset(xmap)` takes an ExploreMap; `act(obs, info)` as the route teacher."""

    def __init__(self, turn_bins, cfg: ControllerConfig = ControllerConfig(), seed: int = 0,
                 xcfg: ExploreConfig = ExploreConfig()) -> None:
        super().__init__(turn_bins, cfg, seed)
        self.xcfg = xcfg
        self.spin_turn = int(np.argmax(np.abs(np.asarray(self.turn_bins))))

    # ------------------------------------------------------------ episode
    def reset(self, xmap: ExploreMap) -> None:   # type: ignore[override]
        self.xm = xmap
        self.kn = Knowledge(xmap)
        self.goal: tuple | None = None
        self.goal_steps = 0
        self.at_goal_steps = 0
        self.blacklist: set = set()
        self.last_plan_t = -10 ** 9
        self.last_plan_version = -1
        self.dirty = True
        self.replans = 0
        self.goal_log: list[tuple[int, str]] = []
        self.spin_left = 0
        self.prev_use_exec = 0
        self.locks_learned = 0
        route = Route(xmap.map_id, "", self.kn.known_grid(), [Anchor("spin", *xmap.start)],
                      xmap.start, [], xmap.nav, self._use_lines())
        super().reset(route)
        self._plan_depth = 0

    def _use_lines(self) -> UseLines:
        """Stoppers with lock knowledge limited to doors KNOWN locked."""
        L = self.xm.use_lines
        lock = tuple(self.kn.locked.get(int(L.sector[i])) if int(L.kind[i]) == KIND_DOOR else None
                     for i in range(len(L)))
        return replace(L, lock=lock)

    # ------------------------------------------------------------ main
    def act(self, obs: dict, info: dict) -> np.ndarray:
        x, y, ang = float(info["pos_x"]), float(info["pos_y"]), float(info["angle"])
        sv = obs.get("self")
        executed = np.asarray(obs["prev_action"], dtype=np.int64)
        if sv is not None and len(sv) >= 6 and int(executed[4]) == 1:
            needs = np.asarray(sv)[-3:] > 0.5
            n0 = len(self.kn.locked)
            sec = self.kn.learn_lock(needs, self.kn.prev_pose)
            if sec is None and np.any(needs):
                sec = self.kn.learn_lock(needs, (x, y, ang))
            if sec is not None:
                self.locks_learned += len(self.kn.locked) - n0
                self.events.append((self.t, f"lock:{sec}:{self.kn.locked[sec]}"))
                self.use_lines = self._use_lines()
                self.dirty = True
        self.kn.observe(x, y, ang, info["door_gaps"])
        self.kn.prev_pose = (x, y, ang)
        self._check_goal(x, y, ang, info)
        self._plan_depth = 0
        return super().act(obs, info)

    def _decide(self, x: float, y: float, ang: float, info: dict) -> np.ndarray:
        if self._plan_depth < 3 and (self.dirty or self.k >= len(self.route.anchors)):
            self._plan_depth += 1
            self._replan(x, y, ang, info)
        if self.k >= len(self.route.anchors):
            return self._action()
        an = self.anchor
        if an.kind == "spin":
            self.label_source = "nav"
            self.spin_left -= 1
            if self.spin_left <= 0:
                self.dirty = True
            return self._action(turn=self.spin_turn)
        if an.kind == "frontier":
            if self.unstick_left > 0:
                return self._unstick_step(x, y, ang)
            return self._pursue(x, y, ang, an, final_xy=(an.x, an.y), precise=False)
        return super()._decide(x, y, ang, info)

    def _leg_mode(self) -> int:
        """Goal-mode id from the committed goal (frontier / door frontier / spin / none ->
        EXPLORE). Combat, unstick and door handling are decided first by the base class."""
        kind = self.goal[0] if self.goal else None
        return {"key": GM.GET_KEY, "lockdoor": GM.GO_LOCKED_DOOR, "exit": GM.GO_EXIT}.get(kind, GM.EXPLORE)

    def _advance(self) -> None:
        super()._advance()
        if self.k >= len(self.route.anchors):
            self.dirty = True

    def _crossed(self, x: float, y: float, an: Anchor) -> bool:
        """Base half-plane test, plus a lateral bound (ExploreConfig.crossed_max_across)."""
        if not super()._crossed(x, y, an):
            return False
        lim = self.xcfg.crossed_max_across
        if lim <= 0:
            return True
        (cx, cy), (dx, dy) = an.door_center, an.door_dir
        return abs(-(x - cx) * dy + (y - cy) * dx) < lim

    # ------------------------------------------------------------ goal bookkeeping
    def _passable(self, info: dict) -> set[int]:
        held = info.get("keys_held") or {}
        return {s for s in self.kn.doors_seen
                if s not in self.kn.locked or held.get(self.kn.locked[s], False)}

    def _check_goal(self, x: float, y: float, ang: float, info: dict) -> None:
        """Invalidate the committed goal when reached / resolved / timed out, and pre-empt
        a frontier goal when a higher-priority goal becomes known."""
        xc = self.xcfg
        self.goal_steps += 1
        g = self.goal
        if g is None:
            self.dirty = True
            return
        kind = g[0]
        held = info.get("keys_held") or {}
        if kind in ("frontier", "doorf", "spin"):
            if self._priority_goal_available(info):
                self.dirty = True
                return
        if kind == "frontier":
            cc = g[1]
            if not self.kn.frontier_mask()[cc]:
                self._goal_done("resolved")
                return
            cx, cy = self.xm.grid.center(*cc)
            if math.hypot(cx - x, cy - y) < xc.reach_radius:
                self.at_goal_steps += 1
                if self.at_goal_steps >= xc.reach_patience:
                    self.blacklist.add(g)
                    self._goal_done("unresolved")
                    return
            if self.goal_steps > xc.frontier_timeout:
                self.blacklist.add(g)
                self._goal_done("timeout")
                return
        elif kind == "doorf":
            if g[1] not in self.kn.door_frontiers(self._passable(info)):
                self._goal_done("resolved")
                return
            if self.goal_steps > xc.frontier_timeout:
                self.blacklist.add(g)
                self._goal_done("timeout")
                return
        elif kind == "key":
            if held.get(self.xm.keys[g[1]][2], False):
                self._goal_done("picked")
                return
            if self.goal_steps > xc.goal_timeout:
                self.blacklist.add(g)
                self._goal_done("timeout")
                return
        elif kind == "lockdoor":
            if self.goal_steps > xc.goal_timeout:
                self.blacklist.add(g)
                self._goal_done("timeout")
                return
        elif kind == "exit" and self.goal_steps > xc.goal_timeout:
            self.blacklist.add(g)
            self._goal_done("timeout")
            return
        if self.kn.version != self.last_plan_version and self.t - self.last_plan_t >= xc.replan_every \
                and self.phase != "push" and not self._in_door_cell(x, y):
            self.dirty = True

    def _goal_done(self, why: str) -> None:
        self.events.append((self.t, f"goal:{self.goal[0]}:{why}"))
        self.goal = None
        self.dirty = True

    def _priority_goal_available(self, info: dict) -> bool:
        held = info.get("keys_held") or {}
        if self.kn.exit_seen and ("exit",) not in self.blacklist:
            return True
        if any(not held.get(self.xm.keys[i][2], False) and ("key", i) not in self.blacklist
               for i in self.kn.keys_seen):
            return True
        return any(held.get(c, False) and ("lockdoor", s) not in self.blacklist
                   for s, c in self.kn.locked.items())

    def _in_door_cell(self, x: float, y: float) -> bool:
        g = self.xm.grid
        cc = g.cell_of(x, y)
        return g.inside(*cc) and g.door_of[cc] != VOID

    # ------------------------------------------------------------ planning
    def _replan(self, x: float, y: float, ang: float, info: dict) -> None:
        xc, kn, xm = self.xcfg, self.kn, self.xm
        self.dirty = False
        self.last_plan_t = self.t
        self.last_plan_version = kn.version
        self.replans += 1
        kn.mark_body(x, y)
        kg = kn.known_grid()
        passable = self._passable(info)
        here = kg.cell_of(x, y)
        dist, prev = dijkstra(kg, [here], passable, xc.wall_penalty, xc.door_cost)
        held = info.get("keys_held") or {}
        choice = None                        # (goal tuple, end cell, terminal anchor or None)
        old = self.goal
        # 1. exit
        if kn.exit_seen and ("exit",) not in self.blacklist:
            cells = [c for c in xm.exit_cells if kg.inside(*c)]
            best = min(cells, key=lambda c: dist[c], default=None)
            if best is not None and np.isfinite(dist[best]):
                choice = (("exit",), best, xm.exit_anchor)
        # 2. a seen key not held
        if choice is None:
            cands = []
            for i in sorted(kn.keys_seen):
                kx, ky, col = xm.keys[i]
                kc = kg.cell_of(kx, ky)
                if held.get(col, False) or ("key", i) in self.blacklist or not kg.inside(*kc):
                    continue
                if np.isfinite(dist[kc]):
                    cands.append((dist[kc], i, kc))
            if cands:
                _, i, kc = min(cands)
                kx, ky, col = xm.keys[i]
                choice = (("key", i), kc, Anchor("key", kx, ky, key_color=col))
        # 3. a known locked door whose key is held
        if choice is None:
            cands = []
            for s, col in sorted(kn.locked.items()):
                if not held.get(col, False) or ("lockdoor", s) in self.blacklist:
                    continue
                cs = [c for c in xm.door_cells.get(s, ()) if np.isfinite(dist[c])]
                if cs:
                    best = min(cs, key=lambda c: dist[c])
                    cands.append((dist[best], s, best))
            if cands:
                _, s, best = min(cands)
                choice = (("lockdoor", s), best, None)
        # 4. frontier (keep the committed one while it is valid)
        if choice is None:
            choice = self._frontier_choice(kg, dist, x, y, ang, passable, old)
        if choice is None and self.blacklist:
            self.events.append((self.t, "blacklist:clear"))
            self.blacklist.clear()
            choice = self._frontier_choice(kg, dist, x, y, ang, passable, old)
        if choice is None:
            self._set_route(kg, [Anchor("spin", x, y)], ("spin",))
            self.spin_left = xc.spin_decisions
            return
        goal, end, term = choice
        path = path_to(prev, end)
        anchors: list[Anchor] = []
        for a, b in zip(path, path[1:]):
            if kg.door_of[b] != VOID and kg.door_of[a] == VOID:
                anchors.append(_door_anchor(xm.grid, xm.m, int(kg.door_of[b]), a, xm.nav))
        if term is not None:
            anchors.append(replace(term, dist=None))
        elif goal[0] == "frontier" or not anchors:
            # a frontier cell; or a door goal while already standing in / past the door
            cx, cy = kg.center(*end)
            anchors.append(Anchor("frontier", cx, cy))
        self._set_route(kg, anchors, goal, targets_last=(xm.exit_cells if goal[0] == "exit"
                                                          and not xm.exit_is_switch else None))

    def _frontier_choice(self, kg: Grid, dist: np.ndarray, x: float, y: float, ang: float,
                         passable: set[int], old):
        xc, kn = self.xcfg, self.kn
        fm = kn.frontier_mask()
        if old is not None and old[0] == "frontier" and fm[old[1]] and np.isfinite(dist[old[1]]) \
                and old not in self.blacklist:
            return (old, old[1], None)
        door_fr = kn.door_frontiers(passable)
        if old is not None and old[0] == "doorf" and old not in self.blacklist:
            cs = [c for c in kn.xm.door_cells.get(old[1], ()) if np.isfinite(dist[c])]
            if cs and old[1] in door_fr:
                return (old, min(cs, key=lambda c: dist[c]), None)
        best = None
        cells = np.argwhere(fm & np.isfinite(dist))
        for c, r in cells:
            g = ("frontier", (int(c), int(r)))
            if g in self.blacklist:
                continue
            cost = dist[c, r] + self._behind(kg, x, y, ang, int(c), int(r))
            if best is None or cost < best[0]:
                best = (cost, g, (int(c), int(r)))
        for s in door_fr:
            g = ("doorf", s)
            if g in self.blacklist:
                continue
            cs = [c for c in kn.xm.door_cells.get(s, ()) if np.isfinite(dist[c])]
            if not cs:
                continue
            e = min(cs, key=lambda c: dist[c])
            cost = dist[e] + self._behind(kg, x, y, ang, *e)
            if best is None or cost < best[0]:
                best = (cost, g, e)
        if best is None:
            return None
        return (best[1], best[2], None)

    def _behind(self, kg: Grid, x: float, y: float, ang: float, c: int, r: int) -> float:
        cx, cy = kg.center(c, r)
        if math.hypot(cx - x, cy - y) < 1e-6:
            return 0.0
        err = abs((math.degrees(math.atan2(cy - y, cx - x)) - ang + 180.0) % 360.0 - 180.0)
        return self.xcfg.ahead_weight * err / 180.0

    def _set_route(self, kg: Grid, anchors: list[Anchor], goal: tuple, targets_last=None) -> None:
        xc = self.xcfg
        walk = _walk(kg, ())
        for i, a in enumerate(anchors):
            if a.kind == "spin":
                continue
            tg = targets_last if (i == len(anchors) - 1 and targets_last) else [kg.cell_of(a.x, a.y)]
            tg = [c for c in tg if kg.inside(*c) and walk[c]]
            if not tg:                     # target point not in known open space: nearest known cell
                ok = np.argwhere(walk)
                if ok.size:
                    cx, cy = kg.cell_of(a.x, a.y)
                    j = int(np.argmin((ok[:, 0] - cx) ** 2 + (ok[:, 1] - cy) ** 2))
                    tg = [(int(ok[j, 0]), int(ok[j, 1]))]
            a.dist, _ = dijkstra(kg, tg, (), xc.wall_penalty, reverse=True)
        old_an = self.route.anchors[self.k] if self.k < len(self.route.anchors) else None
        same = (old_an is not None and anchors and old_an.kind == anchors[0].kind
                and old_an.door_sector == anchors[0].door_sector
                and (old_an.kind != "frontier" or (old_an.x, old_an.y) == (anchors[0].x, anchors[0].y)))
        self.route = Route(self.xm.map_id, "", kg, anchors, self.xm.start, [], self.xm.nav,
                           self.use_lines)
        self.k = 0
        if not same:
            self.phase, self.phase_steps = "approach", 0
        if goal != self.goal:
            self.goal = goal
            self.goal_steps = 0
            self.at_goal_steps = 0
            self.goal_log.append((self.t, "/".join(str(v) for v in goal)))
            self.events.append((self.t, "goal:" + "/".join(str(v) for v in goal)))
