"""Per-map validity checks (map QA only; see maps/qa/__init__.py).

1. structure(): WAD lumps parse; one player-1 start; every sector is closed (each
   vertex has even degree among that sector's sides); sidedef/sector indices valid;
   texture and flat names exist in the IWAD; an exit special exists; every keyed door
   has its key.
2. wad_search(): breadth-first search over (sector, keys held) built from the emitted
   WAD alone (heights, door specials, key things, exit lines). Independent of the
   generator's own plan.
3. plan_search(): the same question over the generator's room graph (record["graph"]).
   Both must agree; for keyed stages the exit must be unreachable without the key and
   the key room must lie off the start->lock path.
4. engine_check(): ONE ViZDoom instance: loads the PWAD, player spawns at the start
   thing, sector count and object classes match, the locked door refuses without its
   key and opens with it, and the exit ends the level. Uses console cheats (warp,
   give) because this is QA, not training or evaluation.
"""
from __future__ import annotations

import math
from collections import Counter, deque

from maps import themes as T
from maps.wad import NO_SIDE, read_wad, unpack_map

KEY_COLOR_OF_THING = {5: "blue", 40: "blue", 6: "yellow", 39: "yellow", 13: "red", 38: "red"}
KEYED_SPECIALS = {26: "blue", 27: "yellow", 28: "red", 32: "blue", 33: "red", 34: "yellow"}
DOOR_SPECIALS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
EXIT_SPECIALS = {11, 51, 52, 124, 197, 198}
STEP, PLAYER_H = 24, 56


def load_map(path: str, map_name: str = "MAP01") -> dict:
    return unpack_map(read_wad(path), map_name)


# ------------------------------------------------------------------ 1. structure
def structure(m: dict, resources: dict | None = None) -> list[str]:
    P = []
    things, lines, sides, verts, secs = m["things"], m["linedefs"], m["sidedefs"], m["vertices"], m["sectors"]
    starts = [t for t in things if t.type == 1]
    if len(starts) != 1:
        P.append(f"{len(starts)} player-1 starts (need exactly 1)")
    deg = Counter()
    for i, ld in enumerate(lines):
        if ld.v1 >= len(verts) or ld.v2 >= len(verts):
            P.append(f"line {i} vertex out of range")
            continue
        for s in (ld.front, ld.back):
            if s == NO_SIDE:
                continue
            if s >= len(sides):
                P.append(f"line {i} sidedef {s} out of range")
                continue
            sec = sides[s].sector
            if sec >= len(secs):
                P.append(f"side {s} sector {sec} out of range")
                continue
            deg[(sec, ld.v1)] += 1
            deg[(sec, ld.v2)] += 1
        if ld.front == NO_SIDE:
            P.append(f"line {i} has no front side")
        if (ld.back != NO_SIDE) != bool(ld.flags & 0x0004):
            P.append(f"line {i} two-sided flag does not match back side")
    odd = [k for k, c in deg.items() if c % 2]
    if odd:
        P.append(f"{len(odd)} open sector boundaries, e.g. sector/vertex {odd[:3]}")
    used = {sides[s].sector for ld in lines for s in (ld.front, ld.back) if s != NO_SIDE and s < len(sides)}
    if used != set(range(len(secs))):
        P.append(f"sectors without sides: {sorted(set(range(len(secs))) - used)[:5]}")
    if not any(ld.special in EXIT_SPECIALS for ld in lines):
        P.append("no exit line or switch")
    held = {KEY_COLOR_OF_THING[t.type] for t in things if t.type in KEY_COLOR_OF_THING}
    for ld in lines:
        c = KEYED_SPECIALS.get(ld.special)
        if c and c not in held:
            P.append(f"{c} locked door without a {c} key")
            break
    if resources is not None:
        tex = {n for sd in sides for n in (sd.upper, sd.lower, sd.middle) if n != "-"}
        flats = {n for s in secs for n in (s.floor_flat, s.ceil_flat)}
        if tex - resources["textures"]:
            P.append(f"unknown textures {sorted(tex - resources['textures'])}")
        if flats - resources["flats"]:
            P.append(f"unknown flats {sorted(flats - resources['flats'])}")
    return P


# ------------------------------------------------------------------ 2. WAD search
def sector_at(m: dict, x: float, y: float) -> int | None:
    """Sector containing (x, y): nearest line hit by a ray towards +x, then its side."""
    best, best_sec = math.inf, None
    for ld in m["linedefs"]:
        (x1, y1), (x2, y2) = m["vertices"][ld.v1], m["vertices"][ld.v2]
        if (y1 > y) == (y2 > y):
            continue
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        if xi <= x or xi >= best:
            continue
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        s = ld.front if cross < 0 else ld.back   # right of v1->v2 (cross < 0) = front
        best = xi
        best_sec = None if s == NO_SIDE else m["sidedefs"][s].sector
    return best_sec


def wad_search(m: dict, use_keys: bool = True) -> dict:
    lines, sides, secs = m["linedefs"], m["sidedefs"], m["sectors"]
    nbr = {i: set() for i in range(len(secs))}
    for ld in lines:
        if ld.back != NO_SIDE:
            a, b = sides[ld.front].sector, sides[ld.back].sector
            nbr[a].add(b)
            nbr[b].add(a)
    door_secs = {sides[ld.back].sector for ld in lines if ld.back != NO_SIDE and ld.special in DOOR_SPECIALS}
    ceil = {}
    for i, s in enumerate(secs):
        if i in door_secs:
            others = [secs[j].ceil for j in nbr[i] if j not in door_secs]
            ceil[i] = (min(others) - 4) if others else s.ceil
        else:
            ceil[i] = s.ceil
    edges = {i: [] for i in range(len(secs))}  # (dst, required key or None)
    for ld in lines:
        if ld.back == NO_SIDE:
            continue
        f, b = sides[ld.front].sector, sides[ld.back].sector
        for src, dst, via_front in ((f, b, True), (b, f, False)):
            need = None
            if dst in door_secs:
                if not via_front or ld.special not in DOOR_SPECIALS:
                    continue  # doors open only from the front side of a door line
                need = KEYED_SPECIALS.get(ld.special)
            if secs[dst].floor - secs[src].floor > STEP:
                continue
            if min(ceil[src], ceil[dst]) - max(secs[src].floor, secs[dst].floor) < PLAYER_H:
                continue
            edges[src].append((dst, need))
    exit_secs = set()
    for ld in lines:
        if ld.special in EXIT_SPECIALS:
            exit_secs.add(sides[ld.front].sector)
            if ld.back != NO_SIDE:
                exit_secs.add(sides[ld.back].sector)
    key_at = {}
    for t in m["things"]:
        if t.type in KEY_COLOR_OF_THING:
            key_at.setdefault(sector_at(m, t.x, t.y), set()).add(KEY_COLOR_OF_THING[t.type])
    start = [t for t in m["things"] if t.type == 1]
    s0 = sector_at(m, start[0].x, start[0].y) if start else None
    if s0 is None:
        return {"start_sector": None, "exit_reachable": False, "keys_collected": [], "reached": 0}
    init = (s0, frozenset(key_at.get(s0, set())) if use_keys else frozenset())
    seen, q = {init}, deque([init])
    reached, exit_ok, keys_seen = set(), False, set()
    while q:
        sec, keys = q.popleft()
        reached.add(sec)
        keys_seen |= keys
        if sec in exit_secs:
            exit_ok = True
        for dst, need in edges[sec]:
            if need and need not in keys:
                continue
            nk = keys | (key_at.get(dst, set()) if use_keys else set())
            st = (dst, frozenset(nk))
            if st not in seen:
                seen.add(st)
                q.append(st)
    return {"start_sector": s0, "exit_reachable": exit_ok, "keys_collected": sorted(keys_seen),
            "reached": len(reached), "n_sectors": len(secs)}


# ------------------------------------------------------------------ 3. plan search
def plan_search(graph: dict) -> dict:
    n = len(graph["rooms"])
    adj = {i: [] for i in range(n)}
    for c in graph["corridors"]:
        need = c["door"] if c["door"] not in (None, "normal") else None
        adj[c["a"]].append((c["b"], need))
        adj[c["b"]].append((c["a"], need))
    key_room = graph.get("key_room")
    color = graph.get("lock_color")

    def search(keys_allowed: bool):
        k0 = frozenset([color]) if keys_allowed and key_room == graph["start_room"] else frozenset()
        seen = {(graph["start_room"], k0)}
        q = deque(seen)
        rooms = set()
        while q:
            r, keys = q.popleft()
            rooms.add(r)
            for v, need in adj[r]:
                if need and need not in keys:
                    continue
                nk = keys | ({color} if keys_allowed and v == key_room else set())
                st = (v, frozenset(nk))
                if st not in seen:
                    seen.add(st)
                    q.append(st)
        return rooms

    with_keys, without = search(True), search(False)
    out = {"exit_reachable": graph["exit_room"] in with_keys,
           "exit_reachable_without_key": graph["exit_room"] in without,
           "rooms_reachable": len(with_keys), "rooms": n}
    if graph.get("lock") is not None:
        lock = graph["corridors"][graph["lock"]]
        near = lock["a"] if lock["a"] in without else lock["b"]
        # shortest start->near-side path without crossing the lock
        prev, q = {graph["start_room"]: None}, deque([graph["start_room"]])
        while q:
            u = q.popleft()
            for v, need in adj[u]:
                if need is None and v not in prev:
                    prev[v] = u
                    q.append(v)
        path, u = [], near
        while u is not None:
            path.append(u)
            u = prev.get(u)
        out["key_off_start_to_lock_path"] = key_room not in path
    return out


def check_record(m: dict, record: dict, resources: dict | None) -> list[str]:
    """All static checks for one generated map. Returns the list of problems."""
    P = structure(m, resources)
    w = wad_search(m, use_keys=True)
    w0 = wad_search(m, use_keys=False)
    p = plan_search(record["graph"])
    if not w["exit_reachable"]:
        P.append("WAD search: exit not reachable")
    if not p["exit_reachable"]:
        P.append("plan search: exit not reachable")
    if w["exit_reachable"] != p["exit_reachable"]:
        P.append("WAD search and plan search disagree")
    if w["reached"] != w["n_sectors"]:
        P.append(f"WAD search: only {w['reached']} of {w['n_sectors']} sectors reachable")
    keyed = record["graph"].get("lock") is not None
    if keyed:
        if w0["exit_reachable"]:
            P.append("WAD search: exit reachable WITHOUT the key")
        if p["exit_reachable_without_key"]:
            P.append("plan search: exit reachable WITHOUT the key")
        if not p.get("key_off_start_to_lock_path", False):
            P.append("key room lies on the start->lock path (no backtracking needed)")
    return P


# ------------------------------------------------------------------ 4. engine
def engine_check(wad_path: str, m: dict, record: dict, iwad: str) -> dict:
    """One engine instance, closed in `finally`. Returns a result dict with `ok`."""
    import numpy as np
    import vizdoom as vzd

    GV = vzd.GameVariable
    g = vzd.DoomGame()
    g.set_doom_game_path(iwad)
    g.set_doom_scenario_path(wad_path)
    g.set_doom_map("MAP01")
    g.set_doom_skill(3)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_objects_info_enabled(True)
    g.set_sectors_info_enabled(True)
    g.set_notifications_buffer_enabled(True)
    g.set_notifications_buffer_size(4)
    g.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.USE, vzd.Button.TURN_LEFT_RIGHT_DELTA])
    g.set_available_game_variables([GV.POSITION_X, GV.POSITION_Y, GV.ANGLE, GV.HEALTH])
    g.set_map_exit_reward(1.0)
    g.set_episode_timeout(6000)
    g.add_game_args("+sv_cheats 1")  # QA only: warp / give
    res: dict = {"ok": False, "problems": []}
    P = res["problems"]

    def pos():
        return (g.get_game_variable(GV.POSITION_X), g.get_game_variable(GV.POSITION_Y),
                g.get_game_variable(GV.ANGLE))

    def stand_before(ld, dist=40.0):
        (x1, y1), (x2, y2) = m["vertices"][ld.v1], m["vertices"][ld.v2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        L = math.hypot(x2 - x1, y2 - y1)
        nx, ny = (y2 - y1) / L, -(x2 - x1) / L  # front normal
        g.send_game_command(f"warp {mx + nx * dist:.0f} {my + ny * dist:.0f}")
        g.make_action([0, 0, 0], 2)
        face = math.degrees(math.atan2(-ny, -nx)) % 360
        a0 = pos()[2]
        g.make_action([0, 0, -(((face - a0) + 180) % 360 - 180)], 1)

    try:
        g.init()
        g.new_episode()
        st = g.get_state()
        if st is None:
            P.append("no state after new_episode")
            return res
        start = next(t for t in m["things"] if t.type == 1)
        x, y, _ = pos()
        res["spawn"] = (x, y)
        if abs(x - start.x) > 1 or abs(y - start.y) > 1:
            P.append(f"spawned at {(x, y)}, start thing at {(start.x, start.y)}")
        if len(st.sectors) != len(m["sectors"]):
            P.append(f"engine has {len(st.sectors)} sectors, WAD {len(m['sectors'])}")
        got = Counter(o.name for o in st.objects)
        want = Counter(T.THING_CLASS[t.type] for t in m["things"])
        if got != want:
            P.append(f"objects differ: engine {dict(got)} vs placed {dict(want)}")
        res["objects"] = dict(got)
        g.make_action([0, 0, 0], 10)  # past the 7-tic turn dead time (engine-probes 5)
        gr = record["graph"]
        if gr.get("lock_door_sector") is not None:
            ds = gr["lock_door_sector"]
            sides = m["sidedefs"]
            face = next(ld for ld in m["linedefs"] if ld.back != NO_SIDE
                        and sides[ld.back].sector == ds and ld.special in KEYED_SPECIALS
                        and sides[ld.front].sector in _near_side_sectors(m, gr))
            stand_before(face)
            h0 = g.get_state().sectors[ds].ceiling_height
            notes = []
            for k in range(6):
                g.make_action([0, 1 if k == 0 else 0, 0], 4)
                notes.append(g.get_state().notifications_buffer or "")
            h1 = g.get_state().sectors[ds].ceiling_height
            res["locked_without_key"] = {"ceil_before": h0, "ceil_after": h1,
                                         "message": _readable(notes)}
            if h1 != h0:
                P.append("locked door opened without the key")
            g.send_game_command(f"give {gr['lock_color']}card")
            g.make_action([0, 0, 0], 4)
            for k in range(6):
                g.make_action([0, 1 if k == 0 else 0, 0], 4)
            h2 = g.get_state().sectors[ds].ceiling_height
            res["locked_with_key_ceil"] = h2
            if h2 <= h0 + 16:
                P.append("locked door did not open with the key")
        # exit
        exits = [ld for ld in m["linedefs"] if ld.special in EXIT_SPECIALS]
        if gr["exit_kind"] == "pad":
            pts = [m["vertices"][v] for ld in exits for v in (ld.v1, ld.v2)]
            cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            g.send_game_command(f"warp {cx:.0f} {cy:.0f}")
            g.make_action([0, 0, 0], 2)
            for _ in range(20):
                g.make_action([1, 0, 0], 4)
                if g.is_episode_finished():
                    break
        else:
            stand_before(exits[0])
            for k in range(6):
                g.make_action([0, 1 if k == 0 else 0, 0], 4)
                if g.is_episode_finished():
                    break
        res["exit_finished"] = g.is_episode_finished()
        res["exit_reward"] = g.get_total_reward()
        if not g.is_episode_finished() or g.is_player_dead() or res["exit_reward"] < 1.0:
            P.append(f"exit did not end the level (finished={g.is_episode_finished()}, "
                     f"reward={res['exit_reward']})")
    finally:
        g.close()
    res["ok"] = not P
    return res


def _readable(notes) -> str:
    """Plain-text lines of the notifications buffer (drops console separator glyphs)."""
    out = []
    for n in notes:
        for line in n.splitlines():
            t = "".join(ch for ch in line if ch.isprintable()).strip()
            if len(t) > 3 and t not in out:
                out.append(t)
    return " | ".join(out)


def _near_side_sectors(m: dict, gr: dict) -> set:
    """Sectors on the start side of the locked door (WAD search without keys)."""
    lines, sides = m["linedefs"], m["sidedefs"]
    reached = set()
    start = next(t for t in m["things"] if t.type == 1)
    s0 = sector_at(m, start.x, start.y)
    door_lock = gr["lock_door_sector"]
    nbr = {}
    for ld in lines:
        if ld.back != NO_SIDE:
            a, b = sides[ld.front].sector, sides[ld.back].sector
            nbr.setdefault(a, set()).add(b)
            nbr.setdefault(b, set()).add(a)
    q = deque([s0])
    reached.add(s0)
    while q:
        u = q.popleft()
        for v in nbr.get(u, ()):
            if v != door_lock and v not in reached:
                reached.add(v)
                q.append(v)
    return reached
