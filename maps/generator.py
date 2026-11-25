"""Seeded procedural map generator (our own design, written during the hackathon).

Pipeline for one map: (stage params, split, seed) ->
  1. layout: rectangular rooms grown as a tree from room 0 along straight corridors
     on a 32-unit lattice, every space separated from every other by at least one
     void cell (so no unintended openings), plus optional loop corridors;
  2. roles: start = room 0, exit = deepest leaf; for keyed stages a locked door on
     the start->exit path and the key in a side branch off that path;
  3. heights, textures, lights (per room / corridor, from the split's themes);
  4. rasterise to cells -> sector ids, trace cell-edge boundaries into linedefs
     (one-sided against void, two-sided between sectors), merge collinear runs;
  5. things (one player-1 start, key, monsters, pickups) on interior cells;
  6. pack as a vanilla Doom-format PWAD with empty node lumps (the engine builds
     nodes on load; probe in docs/reference/map-format.md).

Reproducibility: the RNG is `random.Random(int)` with the int derived by SHA-256 from
(GEN_VERSION, split, stage, seed, attempt). No set iteration order, no `hash()`, no
time. Same inputs give a byte-identical WAD (tested across two subprocesses).

Vanilla features only: DR doors (1), DR keyed doors (26/27/28), W1 exit (52), S1
exit switch (11). No ACS, no BEHAVIOR lump, no MAPINFO.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field

from maps import themes as T
from maps.wad import (ML_BLOCKING, ML_DONTPEGBOTTOM, ML_TWOSIDED, NO_SIDE, Linedef, Sector,
                      Sidedef, Thing, pack_map)

GEN_NAME = "doomsday-mapgen"
GEN_VERSION = "0.1.1"
CELL = 32  # map units per lattice cell


class LayoutError(RuntimeError):
    """The sampled layout cannot satisfy the stage's constraints; retry with a new attempt."""


# ------------------------------------------------------------------ geometry types
@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def cells(self):
        return [(i, j) for i in range(self.x, self.x + self.w) for j in range(self.y, self.y + self.h)]

    def grow(self, d: int) -> "Rect":
        return Rect(self.x - d, self.y - d, self.w + 2 * d, self.h + 2 * d)


@dataclass
class Corridor:
    a: int                 # room index (corridor floor follows room a)
    b: int
    rect: Rect
    axis: str              # "x": runs east-west, "y": runs north-south
    a_is_low: bool         # room a is at the low-coordinate end of the corridor
    loop: bool = False
    door: str | None = None   # None, "normal", or a key colour
    door_index: int = 0       # cell index along the axis, counted from room a's end


@dataclass
class SectorSpec:
    kind: str              # room | corridor | door | pad
    floor: int
    ceil: int
    floor_flat: str
    ceil_flat: str
    light: int
    wall: str
    owner: int             # room index or corridor index
    door_special: int = 0
    jamb: str = T.DOOR_TRACK


@dataclass
class Plan:
    rooms: list[Rect]
    corridors: list[Corridor]
    start_room: int = 0
    exit_room: int = 0
    exit_kind: str = "pad"
    lock: int | None = None       # corridor index of the locked door
    lock_color: str | None = None
    key_room: int | None = None
    room_floor: list[int] = field(default_factory=list)


# ------------------------------------------------------------------ seeding
def derive_seed(split: str, stage: str, seed: int, attempt: int) -> int:
    s = f"{GEN_NAME}|{GEN_VERSION}|{split}|{stage}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little")


def _rint(rng: random.Random, lo_hi) -> int:
    lo, hi = lo_hi
    return rng.randint(int(lo), int(hi))


# ------------------------------------------------------------------ layout
def _free(occ: set, cells) -> bool:
    return not any(c in occ for c in cells)


def _corridor_ring(rect: Rect, axis: str):
    if axis == "x":
        return [(i, rect.y - 1) for i in range(rect.x, rect.x + rect.w)] + \
               [(i, rect.y + rect.h) for i in range(rect.x, rect.x + rect.w)]
    return [(rect.x - 1, j) for j in range(rect.y, rect.y + rect.h)] + \
           [(rect.x + rect.w, j) for j in range(rect.y, rect.y + rect.h)]


def grow_tree(rng: random.Random, P: dict) -> tuple[list[Rect], list[Corridor], set]:
    n_rooms = _rint(rng, P["rooms"])
    rw, rh = _rint(rng, P["room_size"]), _rint(rng, P["room_size"])
    rooms = [Rect(0, 0, rw, rh)]
    corrs: list[Corridor] = []
    occ = set(rooms[0].cells())
    tries = 0
    while len(rooms) < n_rooms and tries < P["layout_tries"]:
        tries += 1
        parent = len(rooms) - 1 if rng.random() < P["chain_bias"] else rng.randrange(len(rooms))
        d = rng.choice("ENWS")
        cw, L = _rint(rng, P["corr_width"]), _rint(rng, P["corr_len"])
        w, h = _rint(rng, P["room_size"]), _rint(rng, P["room_size"])
        pr = rooms[parent]
        if d in "EW":
            if pr.h < cw + 2 or h < cw + 2:
                continue
            cy = rng.randint(pr.y + 1, pr.y + pr.h - 1 - cw)
            ny = rng.randint(cy + cw + 1 - h, cy - 1)
            if d == "E":
                corr, nx = Rect(pr.x + pr.w, cy, L, cw), pr.x + pr.w + L
            else:
                corr, nx = Rect(pr.x - L, cy, L, cw), pr.x - L - w
            child, axis, a_low = Rect(nx, ny, w, h), "x", d == "E"
        else:
            if pr.w < cw + 2 or w < cw + 2:
                continue
            cx = rng.randint(pr.x + 1, pr.x + pr.w - 1 - cw)
            nx = rng.randint(cx + cw + 1 - w, cx - 1)
            if d == "N":
                corr, ny = Rect(cx, pr.y + pr.h, cw, L), pr.y + pr.h + L
            else:
                corr, ny = Rect(cx, pr.y - L, cw, L), pr.y - L - h
            child, axis, a_low = Rect(nx, ny, w, h), "y", d == "N"
        if not _free(occ, child.grow(1).cells()):
            continue
        if not _free(occ, corr.cells()) or not _free(occ, _corridor_ring(corr, axis)):
            continue
        rooms.append(child)
        corrs.append(Corridor(parent, len(rooms) - 1, corr, axis, a_low))
        occ.update(child.cells())
        occ.update(corr.cells())
    if len(rooms) < P["rooms"][0]:
        raise LayoutError(f"placed {len(rooms)} rooms, need at least {P['rooms'][0]}")
    return rooms, corrs, occ


def add_loops(rng: random.Random, P: dict, rooms, corrs, occ, label: list[int],
              forbid: set[int]) -> None:
    """Extra corridors between rooms that face each other across a gap, only within the
    same component label and never touching a room in `forbid`."""
    n_added = 0
    pairs = [(i, j) for i in range(len(rooms)) for j in range(len(rooms)) if i != j]
    rng.shuffle(pairs)
    linked = {(c.a, c.b) for c in corrs} | {(c.b, c.a) for c in corrs}
    for i, j in pairs:
        if n_added >= P["max_loops"]:
            break
        if (i, j) in linked or i in forbid or j in forbid or label[i] != label[j]:
            continue
        if rng.random() >= P["loop_prob"]:
            continue
        a, b = rooms[i], rooms[j]
        cw = _rint(rng, P["corr_width"])
        if a.x + a.w < b.x and b.x - (a.x + a.w) <= P["corr_len"][1]:
            lo, hi = max(a.y, b.y) + 1, min(a.y + a.h, b.y + b.h) - 1 - cw
            if hi < lo:
                continue
            cy = rng.randint(lo, hi)
            corr, axis = Rect(a.x + a.w, cy, b.x - (a.x + a.w), cw), "x"
        elif a.y + a.h < b.y and b.y - (a.y + a.h) <= P["corr_len"][1]:
            lo, hi = max(a.x, b.x) + 1, min(a.x + a.w, b.x + b.w) - 1 - cw
            if hi < lo:
                continue
            cx = rng.randint(lo, hi)
            corr, axis = Rect(cx, a.y + a.h, cw, b.y - (a.y + a.h)), "y"
        else:
            continue
        if not _free(occ, corr.cells()) or not _free(occ, _corridor_ring(corr, axis)):
            continue
        corrs.append(Corridor(i, j, corr, axis, True, loop=True))
        linked |= {(i, j), (j, i)}
        occ.update(corr.cells())
        n_added += 1


def _adjacency(n: int, corrs, skip: int | None = None):
    adj = [[] for _ in range(n)]
    for k, c in enumerate(corrs):
        if k == skip:
            continue
        adj[c.a].append((c.b, k))
        adj[c.b].append((c.a, k))
    return adj


def bfs_path(n: int, corrs, src: int, dst: int, skip: int | None = None):
    """Room path (list of room indices) from src to dst, or None. Generator bookkeeping
    and map QA only; never used by the agent."""
    adj = _adjacency(n, corrs, skip)
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v, _ in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    if dst not in prev:
        return None
    path, u = [], dst
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]


def _component(n, corrs, src, skip):
    adj = _adjacency(n, corrs, skip)
    seen, q = {src}, deque([src])
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def make_plan(rng: random.Random, P: dict) -> tuple[Plan, set]:
    rooms, corrs, occ = grow_tree(rng, P)
    n = len(rooms)
    # depth over the tree (loops not yet added)
    depth = {0: 0}
    q = deque([0])
    adj = _adjacency(n, corrs)
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    degree = [len(a) for a in adj]
    leaves = [r for r in range(1, n) if degree[r] == 1]
    if not leaves:
        raise LayoutError("no leaf room for the exit")
    dmax = max(depth[r] for r in leaves)
    exit_room = rng.choice([r for r in leaves if depth[r] == dmax])
    plan = Plan(rooms, corrs, 0, exit_room)
    label = [0] * n
    forbid = {exit_room}
    if P["keys"]:
        path = bfs_path(n, corrs, 0, exit_room)
        path_edges = [next(k for k, c in enumerate(corrs) if {c.a, c.b} == {path[i], path[i + 1]})
                      for i in range(len(path) - 1)]
        rng.shuffle(path_edges)
        chosen = None
        for k in path_edges:
            near = _component(n, corrs, 0, skip=k)
            c = corrs[k]
            near_room = c.a if c.a in near else c.b
            to_lock = bfs_path(n, corrs, 0, near_room, skip=k)
            cands = sorted(r for r in near if r not in to_lock and r != 0)
            if cands:
                leaf_c = [r for r in cands if degree[r] == 1]
                chosen = (k, rng.choice(leaf_c or cands), near)
                break
        if chosen is None:
            raise LayoutError("no side branch for the key off the start->lock path")
        k, key_room, near = chosen
        plan.lock, plan.key_room = k, key_room
        plan.lock_color = rng.choice(P["key_colors"])
        corrs[k].door = plan.lock_color
        label = [0 if r in near else 1 for r in range(n)]
        forbid = {exit_room}
    plan.exit_kind = rng.choice(P["exit_kinds"])
    # floors over the tree, then loops only where the step is walkable
    floor = [0] * n
    seen = {0}
    q = deque([0])
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if v not in seen:
                seen.add(v)
                floor[v] = floor[u] + rng.choice(P["floor_steps"])
                q.append(v)
    plan.room_floor = floor
    n_tree = len(corrs)
    add_loops(rng, P, rooms, corrs, occ, label, forbid)
    corrs[n_tree:] = [c for c in corrs[n_tree:] if abs(floor[c.a] - floor[c.b]) <= 24]
    # doors
    for k, c in enumerate(corrs):
        if c.door is None and rng.random() < P["door_prob"]:
            c.door = "normal"
    if P["door_prob"] > 0 and not any(c.door == "normal" for c in corrs):
        free = [c for c in corrs if c.door is None]   # never overwrite the locked door
        if free:
            rng.choice(free).door = "normal"
    for c in corrs:
        length = c.rect.w if c.axis == "x" else c.rect.h
        c.door_index = rng.randrange(length)
    return plan, occ


# ------------------------------------------------------------------ raster
def _corridor_cells_by_index(c: Corridor):
    """Corridor cells grouped by index along the axis, index 0 at room a's end."""
    r = c.rect
    length = r.w if c.axis == "x" else r.h
    out = []
    for k in range(length):
        kk = k if c.a_is_low else length - 1 - k
        if c.axis == "x":
            out.append([(r.x + kk, j) for j in range(r.y, r.y + r.h)])
        else:
            out.append([(i, r.y + kk) for i in range(r.x, r.x + r.w)])
    return out


def rasterise(rng: random.Random, P: dict, plan: Plan, theme_names: tuple[str, ...]):
    theme = T.THEMES[rng.choice(theme_names)]
    alt_theme = T.THEMES[rng.choice(theme_names)]
    cells: dict = {}
    secs: list[SectorSpec] = []

    def pick(kind):
        th = alt_theme if rng.random() < P["alt_theme_prob"] else theme
        return rng.choice(th[kind])

    room_sec = []
    for i, r in enumerate(plan.rooms):
        f = plan.room_floor[i]
        secs.append(SectorSpec("room", f, f + rng.choice(P["room_heights"]), pick("floors"),
                               pick("ceils"), _rint(rng, P["light"]), pick("walls"), i))
        room_sec.append(len(secs) - 1)
        for cc in r.cells():
            cells[cc] = room_sec[-1]
    for k, c in enumerate(plan.corridors):
        fa, fb = plan.room_floor[c.a], plan.room_floor[c.b]
        ceil = max(fa, fb) + rng.choice(P["corr_heights"])
        light, wall, ff, cf = _rint(rng, P["light"]), pick("walls"), pick("floors"), pick("ceils")
        groups = _corridor_cells_by_index(c)
        if c.door is None:
            parts = [(groups, "corridor")]
        else:
            parts = [(groups[:c.door_index], "corridor"), ([groups[c.door_index]], "door"),
                     (groups[c.door_index + 1:], "corridor")]
        for gl, kind in parts:
            if not gl:
                continue
            if kind == "door":
                keyed = c.door != "normal"
                spec = SectorSpec("door", fa, fa, ff, cf, light, rng.choice(T.DOOR_TEXTURES), k,
                                  door_special=T.SPECIAL_KEYED[c.door] if keyed else T.SPECIAL_DOOR,
                                  jamb=T.KEY_JAMB[c.door] if keyed else T.DOOR_TRACK)
            else:
                spec = SectorSpec("corridor", fa, ceil, ff, cf, light, wall, k)
            secs.append(spec)
            for g in gl:
                for cc in g:
                    cells[cc] = len(secs) - 1
    # exit pad / switch inside the exit room
    er = plan.rooms[plan.exit_room]
    switch_edges: dict = {}
    pad_cells: list = []
    if plan.exit_kind == "pad":
        px = rng.randint(er.x + 1, er.x + er.w - 3)
        py = rng.randint(er.y + 1, er.y + er.h - 3)
        base = secs[room_sec[plan.exit_room]]
        secs.append(SectorSpec("pad", base.floor, base.ceil, rng.choice(T.EXIT_PAD_FLATS),
                               base.ceil_flat, min(255, base.light + 32), base.wall,
                               plan.exit_room))
        pad_cells = [(px, py), (px + 1, py), (px, py + 1), (px + 1, py + 1)]
        for cc in pad_cells:
            cells[cc] = len(secs) - 1
    else:
        sides = []
        for side in "NSEW":
            if side in "NS":
                yy = er.y + er.h - 1 if side == "N" else er.y
                out = yy + 1 if side == "N" else yy - 1
                for x in range(er.x + 1, er.x + er.w - 2):
                    if (x, out) not in cells and (x + 1, out) not in cells:
                        sides.append((side, x, yy))
            else:
                xx = er.x + er.w - 1 if side == "E" else er.x
                out = xx + 1 if side == "E" else xx - 1
                for y in range(er.y + 1, er.y + er.h - 2):
                    if (out, y) not in cells and (out, y + 1) not in cells:
                        sides.append((side, xx, y))
        if not sides:
            raise LayoutError("no free wall for the exit switch")
        side, a, b = rng.choice(sides)
        for t in (0, 1):
            if side == "N":
                switch_edges[("H", b + 1, a + t)] = "exit_switch"
            elif side == "S":
                switch_edges[("H", b, a + t)] = "exit_switch"
            elif side == "E":
                switch_edges[("V", a + 1, b + t)] = "exit_switch"
            else:
                switch_edges[("V", a, b + t)] = "exit_switch"
    return cells, secs, room_sec, switch_edges, pad_cells


# ------------------------------------------------------------------ trace
def trace(cells: dict, secs: list[SectorSpec], switch_edges: dict):
    """Cell-edge boundaries -> merged linedefs. Returns (vertices, linedefs, sidedefs)."""
    units = []  # (axis, coord, pos, front, back, special, override)
    for (x, y) in sorted(cells):
        s = cells[(x, y)]
        e, w, nn, so = cells.get((x + 1, y)), cells.get((x - 1, y)), cells.get((x, y + 1)), cells.get((x, y - 1))
        if e != s:
            units.append(("V", x + 1, y, s, e))       # A = west (s), B = east (e)
        if w is None:
            units.append(("V", x, y, None, s))
        if nn != s:
            units.append(("H", y + 1, x, s, nn))      # A = south (s), B = north (nn)
        if so is None:
            units.append(("H", y, x, None, s))
    recs = []
    for axis, coord, pos, a, b in units:
        if a is None or b is None:
            front, back = (a if b is None else b), None
        elif secs[b].kind == "door":
            front, back = a, b
        elif secs[a].kind == "door":
            front, back = b, a
        else:
            front, back = min(a, b), max(a, b)
        front_is_A = front == a
        special, override = 0, switch_edges.get((axis, coord, pos))
        if back is not None and secs[back].kind == "door":
            special = secs[back].door_special
        elif back is not None and "pad" in (secs[front].kind, secs[back].kind):
            special = T.SPECIAL_EXIT_W1
        if override == "exit_switch":
            special = T.SPECIAL_EXIT_S1
        recs.append(((axis, coord, front, back, special, override, front_is_A), pos))
    # merge contiguous runs with identical keys
    recs.sort(key=lambda r: (r[0][0], r[0][1], str(r[0][2:]), r[1]))
    runs = []
    for key, pos in recs:
        if runs and runs[-1][0] == key and runs[-1][2] == pos:
            runs[-1][2] = pos + 1
        else:
            runs.append([key, pos, pos + 1])
    # split runs at vertices used by other runs (no T-junctions)
    def ends(run):
        (axis, coord, *_), p0, p1 = run
        return [(p0, coord), (p1, coord)] if axis == "H" else [(coord, p0), (coord, p1)]
    endpoints = set()
    for r in runs:
        endpoints.update(ends(r))
    pieces = []
    for key, p0, p1 in runs:
        axis, coord = key[0], key[1]
        cuts = [p0] + [p for p in range(p0 + 1, p1)
                       if ((p, coord) if axis == "H" else (coord, p)) in endpoints] + [p1]
        for i in range(len(cuts) - 1):
            pieces.append((key, cuts[i], cuts[i + 1]))
    vid: dict = {}
    verts, lines, sides = [], [], []

    def v(pt):
        if pt not in vid:
            vid[pt] = len(verts)
            verts.append((pt[0] * CELL, pt[1] * CELL))
        return vid[pt]

    def side(sec_i, up, lo, mid):
        sides.append(Sidedef(0, 0, up, lo, mid, sec_i))
        return len(sides) - 1

    for (axis, coord, front, back, special, override, front_is_A), p0, p1 in pieces:
        # orientation: front sector on the right of v1 -> v2
        if axis == "V":   # A = west, B = east; front west -> go south, front east -> go north
            a_pt, b_pt = (coord, p0), (coord, p1)
            v1, v2 = (b_pt, a_pt) if front_is_A else (a_pt, b_pt)
        else:             # A = south, B = north; front south -> go east, front north -> go west
            a_pt, b_pt = (p0, coord), (p1, coord)
            v1, v2 = (a_pt, b_pt) if front_is_A else (b_pt, a_pt)
        fs = secs[front]
        if back is None:
            if override == "exit_switch":
                mid = T.EXIT_SWITCHES[0]
            elif fs.kind == "door":
                mid = fs.jamb
            else:
                mid = fs.wall
            flags = ML_BLOCKING | (ML_DONTPEGBOTTOM if fs.kind == "door" else 0)
            lines.append(Linedef(v(v1), v(v2), flags, special, 0, side(front, "-", "-", mid)))
        else:
            bs = secs[back]
            f_up = bs.wall if bs.kind == "door" else fs.wall
            fsd = side(front, f_up, fs.wall, "-")
            bsd = side(back, bs.wall, bs.wall, "-")
            lines.append(Linedef(v(v1), v(v2), ML_TWOSIDED, special, 0, fsd, bsd))
    return verts, lines, sides


# ------------------------------------------------------------------ things
def place_things(rng: random.Random, P: dict, plan: Plan, pad_cells, stage_has_monsters: bool):
    used: set = set(pad_cells)
    things: list[Thing] = []
    expected: list[dict] = []

    def interior(ri):
        r = plan.rooms[ri]
        return [(i, j) for i in range(r.x + 1, r.x + r.w - 1) for j in range(r.y + 1, r.y + r.h - 1)]

    def take(ri):
        free = [c for c in interior(ri)
                if all((c[0] + dx, c[1] + dy) not in used for dx in (-1, 0, 1) for dy in (-1, 0, 1))]
        if not free:
            raise LayoutError(f"no free cell in room {ri}")
        c = rng.choice(free)
        used.add(c)
        return c

    def put(c, ttype, role, room):
        ang = 45 * rng.randrange(8)
        things.append(Thing(c[0] * CELL + CELL // 2, c[1] * CELL + CELL // 2, ang, ttype, 7))
        expected.append({"type": ttype, "class": T.THING_CLASS[ttype], "role": role, "room": room,
                         "x": things[-1].x, "y": things[-1].y})

    put(take(plan.start_room), T.THING_PLAYER1, "start", plan.start_room)
    if plan.key_room is not None:
        card, skull = T.KEY_THINGS[plan.lock_color]
        put(take(plan.key_room), skull if rng.random() < P["skull_key_prob"] else card, "key",
            plan.key_room)
    if stage_has_monsters:
        others = [r for r in range(len(plan.rooms)) if r != plan.start_room]
        names = sorted(P["monster_weights"])
        weights = [P["monster_weights"][k] for k in names]
        for _ in range(_rint(rng, P["monsters"])):
            name = rng.choices(names, weights)[0]
            ri = rng.choice(others)
            put(take(ri), T.MONSTERS[name], "monster", ri)
    pk_names = sorted(P["pickup_weights"])
    pk_w = [P["pickup_weights"][k] for k in pk_names]
    for _ in range(_rint(rng, P["pickups"])):
        name = rng.choices(pk_names, pk_w)[0]
        ri = rng.randrange(len(plan.rooms))
        put(take(ri), T.PICKUPS[name], "pickup", ri)
    return things, expected


# ------------------------------------------------------------------ top level
def stage_params(gen_cfg: dict, split: str, stage: str) -> dict:
    """Merged parameters: defaults <- stage <- split overrides <- split/stage overrides."""
    P = json.loads(json.dumps(gen_cfg["defaults"]))
    P.update(gen_cfg["stages"][stage])
    sp = gen_cfg["splits"][split]
    P.update(sp.get("params", {}))
    P.update(sp.get("stage_params", {}).get(stage, {}))
    return P


def generate(gen_cfg: dict, split: str, stage: str, seed: int, map_name: str = "MAP01"):
    """Return (lumps, record). Raises LayoutError after `max_attempts` failed attempts,
    with every attempt's error in the message."""
    P = stage_params(gen_cfg, split, stage)
    theme_names = tuple(gen_cfg["splits"][split]["themes"])
    errors = []
    for attempt in range(P["max_attempts"]):
        rng = random.Random(derive_seed(split, stage, seed, attempt))
        try:
            plan, _ = make_plan(rng, P)
            cells, secs, room_sec, switch_edges, pad_cells = rasterise(rng, P, plan, theme_names)
            verts, lines, sides = trace(cells, secs, switch_edges)
            things, expected = place_things(rng, P, plan, pad_cells, P["monsters"][1] > 0)
        except LayoutError as e:
            errors.append(f"attempt {attempt}: {e}")
            continue
        sectors = [Sector(s.floor, s.ceil, s.floor_flat, s.ceil_flat, s.light) for s in secs]
        lumps = pack_map(map_name, verts, lines, sides, sectors, things)
        record = {
            "generator": GEN_NAME, "generator_version": GEN_VERSION,
            "split": split, "stage": stage, "seed": seed, "attempt": attempt,
            "params": P, "themes": list(theme_names),
            "mods": [],
            "counts": {"rooms": len(plan.rooms), "corridors": len(plan.corridors),
                       "loops": sum(c.loop for c in plan.corridors),
                       "doors": sum(c.door == "normal" for c in plan.corridors),
                       "locked_doors": sum(c.door not in (None, "normal") for c in plan.corridors),
                       "sectors": len(sectors), "linedefs": len(lines), "things": len(things),
                       "monsters": sum(e["role"] == "monster" for e in expected),
                       "pickups": sum(e["role"] == "pickup" for e in expected)},
            "graph": {
                "rooms": [asdict(r) for r in plan.rooms],
                "corridors": [{"a": c.a, "b": c.b, "door": c.door, "loop": c.loop}
                              for c in plan.corridors],
                "start_room": plan.start_room, "exit_room": plan.exit_room,
                "exit_kind": plan.exit_kind, "lock": plan.lock, "lock_color": plan.lock_color,
                "key_room": plan.key_room,
                "room_sector": room_sec,
                "lock_door_sector": next((i for i, s in enumerate(secs)
                                          if s.kind == "door" and s.owner == plan.lock), None),
            },
            "things": expected,
            "failed_attempts": errors,
        }
        return lumps, record
    raise LayoutError(f"{split}/{stage} seed {seed}: all {P['max_attempts']} attempts failed: "
                      + "; ".join(errors))
