"""USE labels from what the player is looking at (TRAINING-TIME ONLY; see teachers/__init__.py).

Plan change C3 (docs/research/2026-11-25-strategy-level-completion.md). The teacher
labels USE = 1 exactly when a press would do something useful from the CURRENT state:

  * the facing ray (the view axis, which the engine's use trace follows from the
    player's centre) first hits, within the label range, either
      - the FRONT face of a CLOSED door (door sector ceiling - floor <= CLOSED_GAP), whose
        lock (if any) the player can open with the keys held, or
      - the front of the EXIT SWITCH line;
  * and the previous EXECUTED action had USE up (a press is a 0 -> 1 edge; the wrapper
    holds USE for the whole decision and the engine acts on the edge only).

Never a press while the first hit is a door face whose gap is > 0 (the door is moving or
open: a press there reverses it, engine probe R2.2), and never two presses in a row
(the release comes from the previous-executed-use condition).

The condition depends only on the state (pose, live door gaps, keys held, previous
executed action), never on the teacher's route progress, so the same function labels
teacher-driven and student-driven (DAgger) states. The student sees the same thing in
observation v2: the door channel lights the centre columns at short depth for a closed
door face; the rays show a near wall straight ahead for a switch.

Use range: vanilla / ZDoom player use range is 64 map units along the view direction
from the player's centre (ZDoom `Player.UseRange` default 64). The label uses a margin
below that (USE_RANGE_LABEL) so momentum or quantisation cannot turn a label into a
miss; collect.py counts INEFFECTIVE presses (labelled, executed, door still closed on
the next decision) to check the margin against the engine.

Ray stoppers follow the engine's use trace: one-sided lines (walls, jambs, the switch
wall) and every door face (a line with a door special stops the trace whether or not
the door is open). Two-sided lines without a use special do not stop it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

USE_RANGE_ENGINE = 64.0     # engine use range (map units, from the player's centre)
USE_RANGE_LABEL = 58.0      # label range: margin below the engine range
CLOSED_GAP = 0.5            # ceiling - floor at or below this = closed (probe: exactly 0)

KIND_WALL, KIND_DOOR, KIND_SWITCH = 0, 1, 2


@dataclass(frozen=True)
class UseLines:
    """Ray stoppers of one map. `seg` (N, 4) x1, y1, x2, y2; `normal` (N, 2) unit normal
    toward the FRONT side; `kind` (N,) KIND_*; `sector` (N,) door sector or -1;
    `lock` per line: key colour or None."""
    seg: np.ndarray
    normal: np.ndarray
    kind: np.ndarray
    sector: np.ndarray
    lock: tuple = field(default=())

    def __len__(self) -> int:
        return int(self.seg.shape[0])


def make_use_lines(rows: list[tuple]) -> UseLines:
    """rows: (x1, y1, x2, y2, kind, sector, lock). Front = right of v1 -> v2 (Doom side 0)."""
    if not rows:
        return UseLines(np.zeros((0, 4)), np.zeros((0, 2)), np.zeros(0, np.int64),
                        np.zeros(0, np.int64), ())
    seg = np.array([r[:4] for r in rows], dtype=np.float64)
    d = seg[:, 2:4] - seg[:, 0:2]
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    normal = np.stack([d[:, 1] / L, -d[:, 0] / L], axis=1)
    return UseLines(seg, normal, np.array([r[4] for r in rows], np.int64),
                    np.array([r[5] for r in rows], np.int64), tuple(r[6] for r in rows))


def facing_hit(lines: UseLines, x: float, y: float, ang_deg: float,
               max_range: float) -> tuple[int, float] | None:
    """(line index, distance) of the first stopper the view ray from (x, y) at heading
    `ang_deg` (degrees CCW) meets within `max_range`, or None."""
    if len(lines) == 0:
        return None
    th = math.radians(ang_deg)
    rx, ry = math.cos(th), math.sin(th)
    s = lines.seg
    ex, ey = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    wx, wy = s[:, 0] - x, s[:, 1] - y
    den = rx * ey - ry * ex                       # cross(r, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * ey - wy * ex) / den             # distance along the ray
        u = (wx * ry - wy * rx) / den             # position along the segment
    ok = (np.abs(den) > 1e-12) & (t >= 0.0) & (t <= max_range) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    if not ok.any():
        return None
    idx = np.nonzero(ok)[0]
    i = int(idx[np.argmin(t[idx])])
    return i, float(t[i])


def on_front(lines: UseLines, i: int, x: float, y: float) -> bool:
    n = lines.normal[i]
    return (x - lines.seg[i, 0]) * n[0] + (y - lines.seg[i, 1]) * n[1] > 0.0


def press_target(lines: UseLines, x: float, y: float, ang_deg: float, door_gaps: dict,
                 keys_held: dict | None, max_range: float = USE_RANGE_LABEL) -> tuple[str, int] | None:
    """("door", sector) or ("switch", -1) if a press now would open a closed door or hit
    the exit switch; None otherwise. Ignores the previous action (see wants_press)."""
    hit = facing_hit(lines, x, y, ang_deg, max_range)
    if hit is None:
        return None
    i, _ = hit
    k = int(lines.kind[i])
    if k == KIND_WALL or not on_front(lines, i, x, y):
        return None
    if k == KIND_SWITCH:
        return ("switch", -1)
    sec = int(lines.sector[i])
    if sec not in door_gaps:
        raise KeyError(f"no live gap for door sector {sec} (door_gaps must cover every door)")
    if float(door_gaps[sec]) > CLOSED_GAP:
        return None                               # moving or open: a press would reverse it
    lock = lines.lock[i] if lines.lock else None
    if lock and not (keys_held or {}).get(lock, False):
        return None
    return ("door", sec)


def wants_press(lines: UseLines, x: float, y: float, ang_deg: float, door_gaps: dict,
                keys_held: dict | None, prev_use_executed: int,
                max_range: float = USE_RANGE_LABEL) -> bool:
    """The USE label: a fresh press edge at a pressable target."""
    if int(prev_use_executed) != 0:
        return False
    return press_target(lines, x, y, ang_deg, door_gaps, keys_held, max_range) is not None


def lines_from_map(m: dict, door_specials, keyed_specials: dict, exit_switch_specials,
                   no_side: int) -> UseLines:
    """Stoppers of an unpacked WAD map (maps.wad.unpack_map). Exit switches: specials 11
    (exit) and 51 (secret exit), both S1 use lines."""
    verts, sides = m["vertices"], m["sidedefs"]
    rows = []
    for ld in m["linedefs"]:
        (x1, y1), (x2, y2) = verts[ld.v1], verts[ld.v2]
        if ld.special in door_specials and ld.back != no_side:
            rows.append((x1, y1, x2, y2, KIND_DOOR, sides[ld.back].sector,
                         keyed_specials.get(ld.special)))
        elif ld.special in exit_switch_specials:
            rows.append((x1, y1, x2, y2, KIND_SWITCH, -1, None))
        elif ld.back == no_side:
            rows.append((x1, y1, x2, y2, KIND_WALL, -1, None))
    return make_use_lines(rows)


def lines_from_grid(grid, door_locks: dict | None = None) -> UseLines:
    """Stoppers derived from a cell grid (synthetic routes in tests): walls between a
    walkable-or-door cell and void, door faces between a door cell and a non-door
    walkable cell (front toward the non-door cell)."""
    from teachers.navigator import VOID
    door_locks = door_locks or {}
    W, H = grid.shape
    cs = grid.cell
    rows = []

    def open_cell(c, r):
        return grid.inside(c, r) and grid.sector[c, r] != VOID

    for c in range(W):
        for r in range(H):
            if not open_cell(c, r):
                continue
            x0, y0 = (c + grid.i0) * cs, (r + grid.j0) * cs
            # (neighbour offset, segment v1 -> v2 with the CURRENT cell on its right)
            edges = (((1, 0), (x0 + cs, y0 + cs, x0 + cs, y0)),
                     ((-1, 0), (x0, y0, x0, y0 + cs)),
                     ((0, 1), (x0, y0 + cs, x0 + cs, y0 + cs)),
                     ((0, -1), (x0 + cs, y0, x0, y0)))
            here_door = grid.door_of[c, r] != VOID
            for (dc, dr), (a, b, p, q) in edges:
                nc, nr = c + dc, r + dr
                if not open_cell(nc, nr):
                    rows.append((a, b, p, q, KIND_WALL, -1, None))
                elif here_door and grid.door_of[nc, nr] == VOID:
                    sec = int(grid.door_of[c, r])
                    # front faces the non-door neighbour: reverse so it is on the right
                    rows.append((p, q, a, b, KIND_DOOR, sec, door_locks.get(sec)))
    return make_use_lines(rows)


def door_gaps_of(game) -> dict:
    """{sector id: ceiling - floor} for every sector of the engine's current state
    (Sector.id is the WAD sector index, probe round 5). PRIVILEGED, labelling only.
    Raises if the engine gives no sector list (sector info off or +viz_nocheat)."""
    st = game.get_state()
    if st is None:
        raise RuntimeError("no engine state (episode finished)")
    if not st.sectors:
        raise RuntimeError("engine returned no sectors: the teacher needs sector info enabled")
    return {int(s.id): float(s.ceiling_height - s.floor_height) for s in st.sectors}
