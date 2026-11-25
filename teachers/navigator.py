"""Route planning for the scripted teacher (TRAINING-TIME ONLY; see teachers/__init__.py).

From a generated map's WAD geometry (and its generator record) build:
  * a cell grid on the generator's 32-unit lattice (sector per cell, scanline fill);
  * an ordered list of ANCHORS: start -> (key) -> every door the route crosses, with a
    stand-off point in front of the door line and a pass point behind it -> exit
    (pad centre, or a stand-off point in front of the exit switch);
  * per anchor, a distance field (Dijkstra over the grid, doors closed) that the
    controller descends from wherever the player currently is, so it recovers from
    pushes, noise bursts and overshoots without re-planning.

The high-level route itself is one Dijkstra over the grid with doors passable (the
teacher opens them), run start -> key -> exit. A classical planner is fine here: this
module is never imported by the agent, env, evaluate or inference code.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from maps.qa.validate import (DOOR_SPECIALS, EXIT_SPECIALS, KEY_COLOR_OF_THING, KEYED_SPECIALS,
                              load_map)
from maps.wad import NO_SIDE

VOID = -1
INF = math.inf
SQRT2 = math.sqrt(2.0)
EXIT_SWITCH_SPECIAL = 11
PLAYER_THING = 1


@dataclass(frozen=True)
class NavConfig:
    cell: int = 32                 # generator lattice (maps/generator.py CELL)
    wall_penalty: float = 3.0      # extra step cost for a cell touching a wall / closed door
    door_cost: float = 4.0         # extra cost for entering a door cell (route search only)
    door_standoff: float = 36.0    # stand-off distance in front of a door line (USE range 64)
    door_pass: float = 40.0        # pass point distance behind the far door line
    switch_standoff: float = 36.0  # stand-off distance in front of the exit switch line


@dataclass
class Anchor:
    """One step of the route. `kind` is key | door | exit_pad | exit_switch."""
    kind: str
    x: float
    y: float
    face: float | None = None              # degrees CCW, for door / exit_switch
    door_sector: int | None = None
    door_color: str | None = None          # lock colour, or None for a normal door
    door_center: tuple[float, float] | None = None
    door_dir: tuple[float, float] | None = None   # unit travel direction through the door
    pass_x: float | None = None
    pass_y: float | None = None
    key_color: str | None = None
    dist: np.ndarray | None = field(default=None, repr=False)  # cost-to-go to this anchor per cell


@dataclass
class Grid:
    i0: int                    # cell index of column 0 (x = (i0 + c) * cell)
    j0: int
    sector: np.ndarray         # (W, H) int, VOID outside the map
    door_of: np.ndarray        # (W, H) int, door sector id or VOID
    cell: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.sector.shape

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / self.cell)) - self.i0, int(math.floor(y / self.cell)) - self.j0

    def center(self, c: int, r: int) -> tuple[float, float]:
        return ((c + self.i0) * self.cell + self.cell / 2.0, (r + self.j0) * self.cell + self.cell / 2.0)

    def inside(self, c: int, r: int) -> bool:
        return 0 <= c < self.sector.shape[0] and 0 <= r < self.sector.shape[1]

    def walkable(self, c: int, r: int, open_doors: frozenset[int] = frozenset()) -> bool:
        if not self.inside(c, r) or self.sector[c, r] == VOID:
            return False
        d = self.door_of[c, r]
        return d == VOID or int(d) in open_doors


@dataclass
class Route:
    map_id: str
    stage: str
    grid: Grid
    anchors: list[Anchor]
    start: tuple[float, float]
    cell_path: list[tuple[int, int]]       # the high-level route, cell by cell
    cfg: NavConfig
    use_lines: object = None               # teachers.use_label.UseLines (view-ray stoppers)


# ------------------------------------------------------------------ grid
def build_grid(m: dict, cell: int) -> Grid:
    """Sector id per lattice cell (scanline version of maps.qa.validate.sector_at)."""
    verts = m["vertices"]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    i0, i1 = math.floor(min(xs) / cell), math.ceil(max(xs) / cell)
    j0, j1 = math.floor(min(ys) / cell), math.ceil(max(ys) / cell)
    W, H = i1 - i0, j1 - j0
    sector = np.full((W, H), VOID, dtype=np.int64)
    sides = m["sidedefs"]
    segs = []
    for ld in m["linedefs"]:
        (x1, y1), (x2, y2) = verts[ld.v1], verts[ld.v2]
        segs.append((x1, y1, x2, y2, ld.front, ld.back))
    for r in range(H):
        y = (r + j0) * cell + cell / 2.0
        hits = []
        for x1, y1, x2, y2, fr, bk in segs:
            if (y1 > y) == (y2 > y):
                continue
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            hits.append((xi, fr, bk, x1, y1, x2, y2))
        hits.sort(key=lambda h: h[0])
        k = 0
        for c in range(W):
            x = (c + i0) * cell + cell / 2.0
            while k < len(hits) and hits[k][0] <= x:
                k += 1
            if k >= len(hits):
                break
            xi, fr, bk, x1, y1, x2, y2 = hits[k]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            s = fr if cross < 0 else bk
            sector[c, r] = VOID if s == NO_SIDE else sides[s].sector
    door_secs = {sides[ld.back].sector for ld in m["linedefs"]
                 if ld.back != NO_SIDE and ld.special in DOOR_SPECIALS}
    door_of = np.where(np.isin(sector, list(door_secs)), sector, VOID) if door_secs else \
        np.full_like(sector, VOID)
    return Grid(i0, j0, sector, door_of.astype(np.int64), cell)


def _neighbours(grid: Grid, c: int, r: int, open_doors: frozenset[int]):
    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if grid.walkable(c + dc, r + dr, open_doors):
            yield c + dc, r + dr, 1.0
    for dc, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        if (grid.walkable(c + dc, r + dr, open_doors) and grid.walkable(c + dc, r, open_doors)
                and grid.walkable(c, r + dr, open_doors)):
            yield c + dc, r + dr, SQRT2


def _near_wall(grid: Grid, open_doors: frozenset[int]) -> np.ndarray:
    W, H = grid.shape
    out = np.zeros((W, H), dtype=bool)
    for c in range(W):
        for r in range(H):
            if not grid.walkable(c, r, open_doors):
                continue
            out[c, r] = any(not grid.walkable(c + dc, r + dr, open_doors)
                            for dc in (-1, 0, 1) for dr in (-1, 0, 1) if dc or dr)
    return out


def distance_field(grid: Grid, targets: list[tuple[int, int]], cfg: NavConfig,
                   open_doors: frozenset[int] = frozenset()) -> np.ndarray:
    """Dijkstra cost-to-go from every cell to the nearest target cell (INF if unreachable).
    Doors are closed (impassable) except those in `open_doors`."""
    W, H = grid.shape
    dist = np.full((W, H), INF)
    wall = _near_wall(grid, open_doors)
    pq = []
    for c, r in targets:
        if grid.inside(c, r):
            dist[c, r] = 0.0
            pq.append((0.0, c, r))
    heapq.heapify(pq)
    while pq:
        d, c, r = heapq.heappop(pq)
        if d > dist[c, r]:
            continue
        for nc, nr, step in _neighbours(grid, c, r, open_doors):
            # moving from (nc, nr) into (c, r): pay the cost of the cell being left
            nd = d + step * (1.0 + (cfg.wall_penalty if wall[nc, nr] else 0.0))
            if nd < dist[nc, nr]:
                dist[nc, nr] = nd
                heapq.heappush(pq, (nd, nc, nr))
    return dist


def shortest_cell_path(grid: Grid, src: tuple[int, int], dst_cells: list[tuple[int, int]],
                       cfg: NavConfig) -> list[tuple[int, int]]:
    """Cell path src -> nearest dst with every door passable (door cells cost extra)."""
    all_doors = frozenset(int(d) for d in np.unique(grid.door_of) if d != VOID)
    wall = _near_wall(grid, frozenset())   # closed-door geometry: keeps the path centred
    dst = set(dst_cells)
    best = {src: 0.0}
    prev: dict = {src: None}
    pq = [(0.0, src)]
    end = None
    while pq:
        d, u = heapq.heappop(pq)
        if d > best.get(u, INF):
            continue
        if u in dst:
            end = u
            break
        for nc, nr, step in _neighbours(grid, u[0], u[1], all_doors):
            cost = step * (1.0 + (cfg.wall_penalty if wall[u] else 0.0))
            if grid.door_of[nc, nr] != VOID and grid.door_of[u] == VOID:
                cost += cfg.door_cost
            nd = d + cost
            if nd < best.get((nc, nr), INF):
                best[(nc, nr)] = nd
                prev[(nc, nr)] = u
                heapq.heappush(pq, (nd, (nc, nr)))
    if end is None:
        raise RuntimeError(f"no cell path from {src} to any of {sorted(dst)[:4]}")
    path, u = [], end
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]


# ------------------------------------------------------------------ anchors
def _door_anchor(grid: Grid, m: dict, door_sector: int, before: tuple[int, int],
                 cfg: NavConfig) -> Anchor:
    cells = list(zip(*np.nonzero(grid.sector == door_sector)))
    pts = [grid.center(int(c), int(r)) for c, r in cells]
    cx = float(np.mean([p[0] for p in pts]))
    cy = float(np.mean([p[1] for p in pts]))
    span_x = max(p[0] for p in pts) - min(p[0] for p in pts)
    span_y = max(p[1] for p in pts) - min(p[1] for p in pts)
    bx, by = grid.center(*before)
    if span_x >= span_y:      # door cells run along x -> travel is along y
        d = (0.0, 1.0 if cy > by else -1.0)
    else:
        d = (1.0 if cx > bx else -1.0, 0.0)
    half = grid.cell / 2.0
    color = None
    sides = m["sidedefs"]
    for ld in m["linedefs"]:
        if ld.back != NO_SIDE and sides[ld.back].sector == door_sector and ld.special in KEYED_SPECIALS:
            color = KEYED_SPECIALS[ld.special]
    return Anchor(kind="door", x=cx - d[0] * (half + cfg.door_standoff),
                  y=cy - d[1] * (half + cfg.door_standoff),
                  face=math.degrees(math.atan2(d[1], d[0])) % 360.0, door_sector=door_sector,
                  door_color=color, door_center=(cx, cy), door_dir=d,
                  pass_x=cx + d[0] * (half + cfg.door_pass), pass_y=cy + d[1] * (half + cfg.door_pass))


def _exit_anchor(grid: Grid, m: dict, cfg: NavConfig) -> tuple[Anchor, list[tuple[int, int]]]:
    """(anchor, target cells). Pad: the pad sector's cells; switch: stand-off cell."""
    lines, sides, verts = m["linedefs"], m["sidedefs"], m["vertices"]
    sw = [ld for ld in lines if ld.special == EXIT_SWITCH_SPECIAL]
    if sw:
        # several 1-cell pieces may carry the switch: use their joint midpoint
        pts = [verts[v] for ld in sw for v in (ld.v1, ld.v2)]
        mx = float(np.mean([p[0] for p in pts]))
        my = float(np.mean([p[1] for p in pts]))
        (x1, y1), (x2, y2) = verts[sw[0].v1], verts[sw[0].v2]
        L = math.hypot(x2 - x1, y2 - y1)
        nx, ny = (y2 - y1) / L, -(x2 - x1) / L          # front normal (right of v1 -> v2)
        ax, ay = mx + nx * cfg.switch_standoff, my + ny * cfg.switch_standoff
        face = math.degrees(math.atan2(-ny, -nx)) % 360.0
        return Anchor("exit_switch", ax, ay, face=face), [grid.cell_of(ax, ay)]
    ex = [ld for ld in lines if ld.special in EXIT_SPECIALS and ld.back != NO_SIDE]
    if not ex:
        raise ValueError("map has neither an exit switch nor a two-sided exit line")
    common = None
    for ld in ex:
        s = {sides[ld.front].sector, sides[ld.back].sector}
        common = s if common is None else common & s
    if not common:
        raise ValueError("cannot identify the exit pad sector")
    # with one exit line both of its sectors qualify: the pad is the smaller (4 cells)
    pad = min(sorted(common), key=lambda s: int(np.count_nonzero(grid.sector == s)))
    cells = [(int(c), int(r)) for c, r in zip(*np.nonzero(grid.sector == pad))]
    pts = [grid.center(c, r) for c, r in cells]
    return Anchor("exit_pad", float(np.mean([p[0] for p in pts])),
                  float(np.mean([p[1] for p in pts]))), cells


def plan_route(m: dict, record: dict, map_id: str = "", cfg: NavConfig = NavConfig()) -> Route:
    grid = build_grid(m, cfg.cell)
    start_t = next(t for t in m["things"] if t.type == PLAYER_THING)
    start = (float(start_t.x), float(start_t.y))
    keys = [t for t in m["things"] if t.type in KEY_COLOR_OF_THING]
    exit_anchor, exit_cells = _exit_anchor(grid, m, cfg)
    legs: list[tuple[list[tuple[int, int]], Anchor]] = []
    cur = grid.cell_of(*start)
    if keys:
        k = keys[0]
        key_a = Anchor("key", float(k.x), float(k.y), key_color=KEY_COLOR_OF_THING[k.type])
        legs.append(([grid.cell_of(k.x, k.y)], key_a))
    legs.append((exit_cells, exit_anchor))
    anchors: list[Anchor] = []
    full: list[tuple[int, int]] = []
    for targets, goal in legs:
        path = shortest_cell_path(grid, cur, targets, cfg)
        for a, b in zip(path, path[1:]):
            if grid.door_of[b] != VOID and grid.door_of[a] == VOID:
                anchors.append(_door_anchor(grid, m, int(grid.door_of[b]), a, cfg))
        anchors.append(goal)
        full.extend(path if not full else path[1:])
        cur = path[-1]
    # per-anchor fields, doors closed: the field of anchor k leads to it from the side of
    # the previous door; door anchors are reached at their stand-off point.
    for a in anchors:
        if a.kind == "exit_pad":
            targets = exit_cells
        else:
            targets = [grid.cell_of(a.x, a.y)]
        a.dist = distance_field(grid, targets, cfg)
    stage = str(record.get("stage", "")) if record else ""
    from teachers.use_label import lines_from_map
    use_lines = lines_from_map(m, DOOR_SPECIALS, KEYED_SPECIALS, (EXIT_SWITCH_SPECIAL, 51), NO_SIDE)
    return Route(map_id, stage, grid, anchors, start, full, cfg, use_lines)


def load_route(wad_path: str, record_path: str | None = None, map_name: str = "MAP01",
               map_id: str = "", cfg: NavConfig = NavConfig()) -> Route:
    """Route for one generated map from its own WAD and JSON record (the JSON is next to
    the WAD with the same basename unless `record_path` is given)."""
    record_path = record_path or os.path.splitext(wad_path)[0] + ".json"
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    return plan_route(load_map(wad_path, map_name), record, map_id, cfg)


# ------------------------------------------------------------------ geometry helpers
def segment_clear(grid: Grid, x0: float, y0: float, x1: float, y1: float, radius: float,
                  open_doors: frozenset[int] = frozenset(), step: float = 8.0) -> bool:
    """True if a disc of `radius` can slide from (x0, y0) to (x1, y1) over walkable cells."""
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(L / step))
    if L > 1e-6:
        px, py = -(y1 - y0) / L * radius, (x1 - x0) / L * radius
    else:
        px = py = 0.0
    for k in range(n + 1):
        t = k / n
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        for ox, oy in ((0.0, 0.0), (px, py), (-px, -py)):
            if not grid.walkable(*grid.cell_of(x + ox, y + oy), open_doors):
                return False
    return True


def descend(grid: Grid, fieldv: np.ndarray, c: int, r: int, max_cells: int) -> list[tuple[int, int]]:
    """Greedy steepest-descent chain of cells on a distance field (doors closed)."""
    chain = [(c, r)]
    for _ in range(max_cells):
        best, best_v = None, fieldv[c, r]
        for nc, nr, _ in _neighbours(grid, c, r, frozenset()):
            if fieldv[nc, nr] < best_v:
                best, best_v = (nc, nr), fieldv[nc, nr]
        if best is None:
            break
        c, r = best
        chain.append(best)
    return chain


def leg_waypoints(route: Route, k: int, x: float, y: float, radius: float = 18.0,
                  max_cells: int = 400) -> list[tuple[float, float]]:
    """Smoothed walkable waypoints from (x, y) to anchor k (string-pulled field descent).
    For inspection and tests; the controller pursues the field directly."""
    a = route.anchors[k]
    g = route.grid
    chain = descend(g, a.dist, *g.cell_of(x, y), max_cells)
    pts = [g.center(*cc) for cc in chain] + [(a.x, a.y)]
    out = [(x, y)]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not segment_clear(g, out[-1][0], out[-1][1], pts[j][0], pts[j][1], radius):
            j -= 1
        out.append(pts[j])
        i = j
    return out
