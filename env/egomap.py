"""Observation v3: the ego-centric OCCUPANCY MAP (`egomap`), spatial memory across rooms.

OUTPUT (the only thing that leaves this module toward the policy)
  `egomap`: uint8 (5, 32, 32), values 0/1. Cell size 48 map units, so the crop spans
  1536 x 1536 units. The agent sits at the centre cell (row 16, col 16) facing UP:
  row decreasing = straight ahead, column increasing = to the agent's right. The crop is
  re-rotated with the agent's heading every decision.
    channel 0  seen free space   (a depth ray passed through the cell)
    channel 1  seen wall/obstacle (a depth ray ended in the cell, not on a door)
    channel 2  visited            (the agent's own position occupied the cell this episode)
    channel 3  closed door seen   (a ray ended on a column the v2 door channel lights)
    channel 4  key seen, not yet picked up (a visible key entity's estimated position)
  Memory persists for the whole episode and is cleared at reset.

WHY THIS IS ALLOWED (rules 4.1 / 4.4, CLAUDE.md working interpretation)
  * It is a SENSORY ENCODING: everything written into the map is what the agent itself
    perceived through the depth buffer (its own field of view, line of sight only) or
    where its own body went. Nothing unseen is ever written; no automap, no sector or
    line geometry, no object list is read here.
  * It is EGO-CENTRIC: internally the map is kept in a SPAWN-RELATIVE frame (origin =
    the agent's position at episode start, x axis = its heading at start). The agent's
    pose in that frame comes from pose deltas (the engine's position / angle minus the
    spawn pose: perfect odometry relative to spawn). The output is a crop of that grid
    re-expressed in the agent's CURRENT frame. Absolute coordinates, map ids, sector ids
    and line ids never appear in the output: shifting the same layout elsewhere on the
    map gives an identical egomap (tested), and the spawn-frame pose itself is never
    output either.
  * It PRE-COMPUTES NO ACTION: cells only say "free / wall / door / been here / a key was
    seen here". There is
    no path, no goal direction, no frontier target, no turn or fire suggestion; deciding
    where to go from the map is left to the network.

HOW THE MAP IS FILLED (every decision, numpy-vectorised)
  * Per screen column (80 columns, 2 px each, the same columns as `rays`), the reading is
    the nearest depth over the middle ray band (horizon +-4 rows) over WORLD pixels only:
    pixels with a non-zero labels-buffer value (any thing: monsters, items, decorations,
    the player's own weapon) or depth 0 are excluded, so moving things are never stamped
    as walls. A column with no world pixel has no reading and marks nothing.
  * Column angle: atan((2k+1-80)/80) off the view axis (horizontal FOV 90 degrees,
    probe R5.2), screen right = clockwise.
  * Distance: perpendicular distance = (level - DEPTH_OFFSET) / DEPTH_PER_UNIT (probe R5.2
    fit, level = 0.13979 x units - 1.134); range along the ray = perpendicular / cos(angle).
  * Cells along the ray up to the hit -> free (capped at RAY_MAX_UNITS); the cell just
    beyond the hit point -> wall, or door when the v2 door channel lights the column.
  * SENSOR RANGE (measured 2026-11-26, synthetic corridor): world depth is linear only up
    to level 62 (~450 units); every farther surface reads 63 (or 255). A saturated column
    marks free space out to ~455 units (perpendicular) and no wall. So walls are mapped
    within ~450 units of where the agent has been; RAY_MAX_UNITS (1500) never binds.
    WALL_MAX_PERP_UNITS (800) is kept as a second guard: beyond ~820 units floor/ceiling
    pixels in the +-4-row band would be nearer than any wall (probe R5.2 / door_cue).
  * A free pass through a cell clears its door bit (a door seen open is no longer a closed
    door); a door hit clears the cell's wall bit. Walls are never cleared.
  * Key seen (channel 4): for each visible key entity row of the observation (entity_type
    key-blue / key-yellow / key-red; the labels-buffer line-of-sight entity set), its
    bearing from the row's screen-x (u = tan of the angle off the view axis, FOV 90) and
    its distance from the row's own-pixel depth via the same fit as the rays; one internal
    cell is marked, in a per-colour bit. A saturated (>= 63) or missing depth marks nothing.
    When the keys-held bit of a colour turns on (v2 key bits), every key-seen cell of THAT
    colour is cleared, and that colour is no longer marked this episode. Output channel 4 =
    OR of the three colour bits.
  * Visited: the agent's cell, with the segment from the previous pose, every decision.
  * Internal grid: INT_CELL = 24 units (half an output cell), INT_GRID = 512 cells per side
    (+-6144 units around spawn); writes outside it (and on its outer ring) are dropped. Each output cell ORs a 2x2
    grid of sample points (the centres of the four 24-unit internal cells it covers when
    axis-aligned; nearest internal cell per point after rotation).
  * Without depth/labels (e.g. `+viz_nocheat 1`) nothing is written: the map stays zeros.

OBSERVATION v4 (EgoMap(locked_channel=True)): one more output channel, (6, 32, 32).
    channel 5  locked door known  (the agent pressed USE on a closed door and the engine
               answered "needs <colour> key")
  * Input: the v2 "needs key" pulse bits only (3 booleans for this decision; derived in
    the env from the vanilla locked-door message, the text never reaches this module). The
    engine prints that message only when the key of that colour is NOT held, so colour
    attribution is implicit; the channel itself is colour-blind (one bit for any colour).
  * Mark: on a pulse, the door cells that channel 3 marks for the centre screen columns
    (COL_CENTRE: the view ray, where vanilla USE acts), if those columns hit a door face
    lit by the v2 door channel within LOCK_USE_RANGE_UNITS. Nothing else is marked (no
    door lit at the centre -> nothing).
  * Persist: for the episode, in the spawn-frame grid like every other channel.
  * Clear ("seen open / passed"): each mark remembers its hit point p and the view
    direction n at marking time (spawn frame, private). A mark is cleared when a ray of a
    later decision that did NOT stop on a door face crosses the plane through p (normal n)
    within LOCK_LATERAL_UNITS of p and runs on at least LOCK_BEYOND_UNITS past it (a
    closed door stops every ray at its face, so only an open door lets one through,
    whatever the viewing angle), or when the agent's own trail gets LOCK_BEYOND_UNITS / 2
    past the plane within the same lateral window. The v3 free-space pass does NOT clear
    it (grazing rays in front of a closed face would).
"""
from __future__ import annotations

import numpy as np

from .door_cue import DEPTH_OFFSET, DEPTH_PER_UNIT
from .observation import ENTITY_TYPES, N_RAY_COLS, SCREEN_W

# ---------------------------------------------------------------- contract (fixed)
EGOMAP_CHANNELS = 5                   # observation v3
EGOMAP_SIZE = 32
EGOMAP_SHAPE = (EGOMAP_CHANNELS, EGOMAP_SIZE, EGOMAP_SIZE)
EGOMAP_CHANNELS_V4 = 6                # observation v4: + locked door known
EGOMAP_SHAPE_V4 = (EGOMAP_CHANNELS_V4, EGOMAP_SIZE, EGOMAP_SIZE)
CH_LOCKED = 5


def egomap_shape(obs_version: int) -> tuple[int, int, int] | None:
    """(C, H, W) of `egomap` under an observation version; None before v3."""
    if obs_version >= 4:
        return EGOMAP_SHAPE_V4
    return EGOMAP_SHAPE if obs_version >= 3 else None
EGOMAP_CELL_UNITS = 48.0
EGOMAP_AGENT_CELL = (16, 16)          # (row, col); facing up (row decreasing)
CH_FREE, CH_WALL, CH_VISITED, CH_DOOR, CH_KEY = 0, 1, 2, 3, 4

# ---------------------------------------------------------------- internals
INT_CELL = 24.0
INT_GRID = 512
_HALF = INT_GRID // 2
RAY_MAX_UNITS = 1500.0
WALL_MAX_PERP_UNITS = 800.0
# Measured 2026-11-26 (synthetic corridor, side walls at known distance, vizdoom 1.3.3):
# world depth is linear in perpendicular distance only up to level 62; from about 455
# units every wall reads exactly 63 (and a few far pixels 255). A reading >= 63 is
# therefore "nothing within ~455 units": free up to there, no wall stamped.
SATURATED_LEVEL = 63
SATURATED_PERP_UNITS = (SATURATED_LEVEL - 0.5 - DEPTH_OFFSET) / DEPTH_PER_UNIT   # ~455
FREE_STEP_UNITS = 12.0                # sample spacing along a ray (half an internal cell)
WALL_NUDGE_UNITS = 3.0                # hit point pushed this far into the surface
FREE_STOP_UNITS = 12.0                # free samples stop this short of the hit point

_BIT = {CH_FREE: 1, CH_WALL: 2, CH_VISITED: 4, CH_DOOR: 8}
# Key colours get their own internal bits (blue, yellow, red); output channel 4 ORs them.
KEY_COLOURS = ("blue", "yellow", "red")
_KEY_BITS = (16, 32, 64)
_KEY_ANY = 16 | 32 | 64
_KEY_TYPE_IDS = tuple(ENTITY_TYPES.index(f"key-{c}") for c in KEY_COLOURS)
assert _KEY_TYPE_IDS == tuple(range(_KEY_TYPE_IDS[0], _KEY_TYPE_IDS[0] + 3))
_KEY_LO, _KEY_HI = _KEY_TYPE_IDS[0], _KEY_TYPE_IDS[-1]
U_COL, D_COL = 0, 4           # entity feature columns: u (screen x in [-1, 1]), d (depth / 255)

_COL_X = np.arange(N_RAY_COLS, dtype=np.float64) * (SCREEN_W // N_RAY_COLS) + (SCREEN_W // N_RAY_COLS) / 2.0
_COL_TAN = (_COL_X - SCREEN_W / 2.0) / (SCREEN_W / 2.0)     # + = right of the view axis
_COL_ANG = np.arctan(_COL_TAN)                               # radians, + = right
_COL_INV_COS = np.sqrt(1.0 + _COL_TAN ** 2)

_N_FREE = int(np.ceil(RAY_MAX_UNITS / FREE_STEP_UNITS)) + 1
_FREE_T = np.arange(_N_FREE, dtype=np.float64) * FREE_STEP_UNITS          # (S,)

# output crop sample points in the agent frame: forward (up), left
_r = np.arange(EGOMAP_SIZE, dtype=np.float64)
_ROW_FWD = (EGOMAP_AGENT_CELL[0] - _r) * EGOMAP_CELL_UNITS                 # row -> forward
_COL_LEFT = (EGOMAP_AGENT_CELL[1] - _r) * EGOMAP_CELL_UNITS                # col -> left
# 2x2 sample points per output cell at the centres of its four internal cells (offsets
# +-12 never fall on an internal cell boundary, also after 90-degree rotations)
_SUB = np.array([-INT_CELL / 2, INT_CELL / 2])
_GRID4 = (EGOMAP_SIZE, EGOMAP_SIZE, 2, 2)    # row, col, sub-forward, sub-left
_FWD = np.broadcast_to(_ROW_FWD[:, None, None, None] + _SUB[None, None, :, None], _GRID4).reshape(-1)
_LEFT = np.broadcast_to(_COL_LEFT[None, :, None, None] + _SUB[None, None, None, :], _GRID4).reshape(-1)
_N_SUB = 4
_FWD_C, _LEFT_C = _FWD / INT_CELL, _LEFT / INT_CELL     # in internal-cell units
_SHIFTS = np.arange(4, dtype=np.uint8)[:, None, None]

# v4 locked-door channel (internal bit 128, unused by the v3 packing)
_LOCKED_BIT = 128
COL_CENTRE = (N_RAY_COLS // 2 - 1, N_RAY_COLS // 2)     # the two columns either side of the view axis
LOCK_USE_RANGE_UNITS = 100.0    # vanilla USERANGE 64 + depth quantisation (~7 units/level) margin
LOCK_LATERAL_UNITS = 40.0       # a crossing ray / the trail must pass this close to the mark
LOCK_BEYOND_UNITS = 32.0        # ... and get this far past the door plane


def world_column_levels(depth: np.ndarray, labels_buf: np.ndarray, band: tuple[int, int]) -> np.ndarray:
    """(80,) float64 raw depth levels per column, nearest WORLD pixel of the middle ray
    band rows [r0, r1) (EnvConfig.ray_bands[1]); 0.0 = no world pixel (no reading)."""
    r0, r1 = band
    d = depth[r0:r1].astype(np.int16)
    world = (labels_buf[r0:r1] == 0) & (d > 0)
    d = np.where(world, d, np.int16(1000))
    m = d.reshape(r1 - r0, N_RAY_COLS, SCREEN_W // N_RAY_COLS).min(axis=(0, 2))
    return np.where(m >= 1000, 0, m).astype(np.float64)


class EgoMap:
    """Per-episode occupancy memory. Private state: the packed spawn-frame grid and the
    agent's spawn-relative pose. Only `crop()` output reaches the observation."""

    def __init__(self, locked_channel: bool = False):
        self.grid = np.zeros((INT_GRID, INT_GRID), dtype=np.uint8)   # bit-packed channels
        self.locked_channel = bool(locked_channel)                    # observation v4
        self.shape = EGOMAP_SHAPE_V4 if self.locked_channel else EGOMAP_SHAPE
        self.reset()

    def reset(self):
        self.grid.fill(0)
        self._spawn = None        # (x0, y0, heading0_deg): private, never output
        self._pose = None         # (x, y, heading_rad) in the spawn frame
        self._prev_xy = None
        self._held = [False, False, False]   # keys held at the last update (b, y, r)
        self._locks: list = []    # v4, private: (cells, px, py, nx, ny) in the spawn frame

    # ------------------------------------------------------------ pose
    def _set_pose(self, px: float, py: float, heading_deg: float):
        if self._spawn is None:
            self._spawn = (float(px), float(py), float(heading_deg))
        x0, y0, h0 = self._spawn
        t0 = np.radians(h0)
        c0, s0 = np.cos(t0), np.sin(t0)
        dx, dy = float(px) - x0, float(py) - y0
        self._pose = (dx * c0 + dy * s0, -dx * s0 + dy * c0, np.radians(float(heading_deg) - h0))

    # ------------------------------------------------------------ writes
    def _cells(self, x: np.ndarray, y: np.ndarray):
        ix = np.floor(x / INT_CELL).astype(np.int64) + _HALF
        iy = np.floor(y / INT_CELL).astype(np.int64) + _HALF
        # the outermost ring of cells is never written, so crop() can clip instead of mask
        ok = (ix >= 1) & (ix < INT_GRID - 1) & (iy >= 1) & (iy < INT_GRID - 1)
        return iy[ok] * INT_GRID + ix[ok]

    def update(self, px: float, py: float, heading_deg: float, levels: np.ndarray,
               door_cols: np.ndarray | None, entities: np.ndarray | None = None,
               entity_type: np.ndarray | None = None, entity_mask: np.ndarray | None = None,
               keys_held=None, needs_key=None) -> None:
        """One decision. levels: (80,) raw depth levels (0 = no reading); door_cols: (80,)
        v2 door channel (1 = door face) or None. entities / entity_type / entity_mask: the
        observation's entity set (only key rows are used); keys_held: 3 booleans (b, y, r),
        the v2 keys-held bits; needs_key: 3 booleans, the v2 "needs key" pulse (used only
        with locked_channel, observation v4)."""
        self._set_pose(px, py, heading_deg)
        x, y, h = self._pose
        flat = self.grid.reshape(-1)

        # visited: segment from the previous pose to this one
        if self._prev_xy is None:
            vx, vy = np.array([x]), np.array([y])
        else:
            n = int(np.ceil(np.hypot(x - self._prev_xy[0], y - self._prev_xy[1]) / FREE_STEP_UNITS)) + 1
            n = min(n, 64)
            t = np.linspace(0.0, 1.0, n)
            vx = self._prev_xy[0] + t * (x - self._prev_xy[0])
            vy = self._prev_xy[1] + t * (y - self._prev_xy[1])
        self._prev_xy = (x, y)

        lv = np.asarray(levels, dtype=np.float64)
        has = lv > 0
        lock_rays = None
        if has.any():
            sat = lv[has] >= SATURATED_LEVEL
            perp = np.where(sat, SATURATED_PERP_UNITS, (lv[has] - DEPTH_OFFSET) / DEPTH_PER_UNIT)
            rng = perp * _COL_INV_COS[has]
            ang = h - _COL_ANG[has]                       # screen right = clockwise
            ca, sa = np.cos(ang), np.sin(ang)
            # free: samples strictly before the hit (saturated: up to the sensor range), capped
            stop = np.minimum(np.where(sat, rng, rng - FREE_STOP_UNITS), RAY_MAX_UNITS)
            ft = _FREE_T[:int(np.max(stop) // FREE_STEP_UNITS) + 1] if stop.size else _FREE_T[:0]
            m = ft[None, :] <= stop[:, None]
            fx = x + (ca[:, None] * ft[None, :])[m]
            fy = y + (sa[:, None] * ft[None, :])[m]
            free_idx = self._cells(fx, fy)
            flat[free_idx] = (flat[free_idx] | _BIT[CH_FREE]) & np.uint8(~_BIT[CH_DOOR] & 0xFF)
            # hits: wall or door
            hit = (~sat) & (perp <= WALL_MAX_PERP_UNITS)
            if hit.any():
                hr = rng[hit] + WALL_NUDGE_UNITS
                hx, hy = x + ca[hit] * hr, y + sa[hit] * hr
                if door_cols is not None:
                    is_door = np.asarray(door_cols)[has][hit] > 0.5
                else:
                    is_door = np.zeros(hr.shape, dtype=bool)
                w_idx = self._cells(hx[~is_door], hy[~is_door])
                flat[w_idx] |= _BIT[CH_WALL]
                d_idx = self._cells(hx[is_door], hy[is_door])
                flat[d_idx] = (flat[d_idx] & np.uint8(~_BIT[CH_WALL] & 0xFF)) | _BIT[CH_DOOR]
            if self.locked_channel:
                lock_rays = (has, sat, rng, ca, sa, door_cols)
        v_idx = self._cells(vx, vy)
        flat[v_idx] |= _BIT[CH_VISITED]
        self._update_keys(flat, x, y, h, entities, entity_type, entity_mask, keys_held)
        if self.locked_channel:
            self._update_locks(flat, x, y, h, vx, vy, lock_rays, needs_key)

    # ------------------------------------------------------------ v4: locked doors
    def _update_locks(self, flat, x, y, h, vx, vy, rays, needs_key):
        pulse = needs_key is not None and bool(np.any(needs_key))
        if not self._locks and not pulse:
            return
        # 1) clear marks whose door was seen open (a ray that did not stop on a door face
        #    ran through the mark's plane) or passed (the trail got beyond the plane)
        if self._locks:
            rr = None
            if rays is not None:
                has, sat, rng, ca, sa, door_cols = rays
                door = ((np.asarray(door_cols)[has] > 0.5) if door_cols is not None
                        else np.zeros(rng.shape, dtype=bool))
                through = sat | ~door
                rx, ry, rr = ca[through], sa[through], rng[through]
            keep = []
            for mark in self._locks:
                cells, px, py, nx, ny = mark
                f = (vx - px) * nx + (vy - py) * ny
                lat = np.abs(-(vx - px) * ny + (vy - py) * nx)
                gone = bool(np.any((f > LOCK_BEYOND_UNITS / 2) & (lat <= LOCK_LATERAL_UNITS)))
                if not gone and rr is not None and rr.size:
                    dn = rx * nx + ry * ny                       # ray direction . plane normal
                    ok = dn > 0.1
                    s_cross = np.where(ok, ((px - x) * nx + (py - y) * ny) / np.where(ok, dn, 1.0), -1.0)
                    clat = np.abs(-(x + rx * s_cross - px) * ny + (y + ry * s_cross - py) * nx)
                    beyond = (rr - s_cross) * dn                 # how far past the plane the ray ends
                    gone = bool(np.any(ok & (s_cross > 0) & (clat <= LOCK_LATERAL_UNITS)
                                       & (beyond >= LOCK_BEYOND_UNITS)))
                if gone:
                    flat[cells] &= np.uint8(~_LOCKED_BIT & 0xFF)
                else:
                    keep.append(mark)
            if len(keep) != len(self._locks):
                self._locks = keep
                for mark in keep:                                # overlapping marks stay set
                    flat[mark[0]] |= np.uint8(_LOCKED_BIT)
        # 2) mark: a needs-key pulse this decision, centre columns on a lit door face in USE range
        if not pulse or rays is None:
            return
        has, sat, rng, ca, sa, door_cols = rays
        if door_cols is None:
            return
        cols = np.flatnonzero(has)
        sel = np.isin(cols, COL_CENTRE) & (np.asarray(door_cols)[cols] > 0.5) & ~sat
        sel &= rng <= LOCK_USE_RANGE_UNITS
        if not sel.any():
            return
        hr = rng[sel] + WALL_NUDGE_UNITS
        hx, hy = x + ca[sel] * hr, y + sa[sel] * hr
        # the 2x2 block of internal cells whose centres surround each hit point: a 48-unit
        # square always holds one of the crop's 24-unit sample points at any rotation, so
        # the mark never drops out of the crop (a single 24-unit cell can).
        bx = np.concatenate([hx - INT_CELL / 2, hx + INT_CELL / 2, hx - INT_CELL / 2, hx + INT_CELL / 2])
        by = np.concatenate([hy - INT_CELL / 2, hy - INT_CELL / 2, hy + INT_CELL / 2, hy + INT_CELL / 2])
        cells = np.unique(self._cells(bx, by))
        if cells.size == 0:
            return
        flat[cells] |= np.uint8(_LOCKED_BIT)
        px = float(np.mean(x + ca[sel] * rng[sel]))
        py = float(np.mean(y + sa[sel] * rng[sel]))
        nx, ny = self._door_normal(cols, np.asarray(door_cols), sat, rng, ca, sa, x, y, h)
        for i, mark in enumerate(self._locks):   # the same door pressed again: one mark
            if np.hypot(mark[1] - px, mark[2] - py) <= INT_CELL:
                self._locks[i] = (np.union1d(mark[0], cells),) + tuple(mark[1:])
                return
        self._locks.append((cells, px, py, nx, ny))

    @staticmethod
    def _door_normal(cols, door_cols, sat, rng, ca, sa, x, y, h):
        """Unit normal of the door face under the view ray (spawn frame, pointing away from
        the agent): a line fit through the hit points of the lit door columns contiguous
        with the centre; the view heading when fewer than 4 columns / under 16 units span."""
        fallback = (float(np.cos(h)), float(np.sin(h)))
        lit = np.zeros(N_RAY_COLS, dtype=bool)
        pos = np.full(N_RAY_COLS, -1)
        ok = (door_cols[cols] > 0.5) & ~sat
        lit[cols[ok]] = True
        pos[cols] = np.arange(cols.size)
        lo, hi = COL_CENTRE
        if not (lit[lo] or lit[hi]):
            return fallback
        lo, hi = (lo, hi) if lit[lo] else (hi, hi)
        hi = hi if lit[hi] else lo
        while lo > 0 and lit[lo - 1]:
            lo -= 1
        while hi < N_RAY_COLS - 1 and lit[hi + 1]:
            hi += 1
        idx = pos[lo:hi + 1]
        if idx.size < 4:
            return fallback
        pts = np.stack([x + ca[idx] * rng[idx], y + sa[idx] * rng[idx]], axis=1)
        d = pts - pts.mean(axis=0)
        if np.hypot(*(pts[-1] - pts[0])) < 16.0:
            return fallback
        tx, ty = np.linalg.svd(d, full_matrices=False)[2][0]     # face direction
        nx, ny = -ty, tx
        if nx * fallback[0] + ny * fallback[1] < 0:
            nx, ny = -nx, -ny
        return float(nx), float(ny)

    def _update_keys(self, flat, x, y, h, ents, types, mask, keys_held):
        if keys_held is not None:
            for i, held in enumerate(keys_held):
                held = bool(held)
                if held and not self._held[i]:       # picked up: forget that colour's cells
                    self.grid &= np.uint8(~_KEY_BITS[i] & 0xFF)
                self._held[i] = held
        if ents is None or types is None or mask is None:
            return
        types = np.asarray(types)
        sel = np.asarray(mask, dtype=bool) & (types >= _KEY_LO) & (types <= _KEY_HI)
        if not sel.any():
            return
        e = np.asarray(ents, dtype=np.float64)[sel]
        lv = e[:, D_COL] * 255.0
        ok = (lv > 0) & (lv < SATURATED_LEVEL)
        if not ok.any():
            return
        u, lv, t = e[ok, U_COL], lv[ok], types[sel][ok]
        perp = (lv - DEPTH_OFFSET) / DEPTH_PER_UNIT
        ang = h - np.arctan(u)                                 # screen right = clockwise
        kx, ky = x + np.cos(ang) * perp * np.sqrt(1.0 + u * u), y + np.sin(ang) * perp * np.sqrt(1.0 + u * u)
        for i, tid in enumerate(_KEY_TYPE_IDS):
            if self._held[i]:
                continue
            m = t == tid
            if m.any():
                flat[self._cells(kx[m], ky[m])] |= np.uint8(_KEY_BITS[i])

    # ------------------------------------------------------------ read
    def crop(self) -> np.ndarray:
        """(5, 32, 32) uint8 ego-centric crop ((6, 32, 32) with locked_channel, v4);
        zeros before the first update."""
        if self._pose is None:
            return np.zeros(self.shape, dtype=np.uint8)
        x, y, h = self._pose
        c, s = np.cos(h), np.sin(h)
        sx = (x / INT_CELL + _HALF) + (_FWD_C * c - _LEFT_C * s)
        sy = (y / INT_CELL + _HALF) + (_FWD_C * s + _LEFT_C * c)
        ix = np.clip(np.floor(sx), 0, INT_GRID - 1).astype(np.intp)   # border ring is all zero
        iy = np.clip(np.floor(sy), 0, INT_GRID - 1).astype(np.intp)
        vals = self.grid[iy, ix]
        packed = np.bitwise_or.reduce(vals.reshape(EGOMAP_SIZE, EGOMAP_SIZE, _N_SUB), axis=2)
        out = np.empty(self.shape, dtype=np.uint8)
        out[:4] = (packed[None] >> _SHIFTS) & 1
        out[CH_KEY] = (packed & _KEY_ANY) != 0
        if self.locked_channel:
            out[CH_LOCKED] = (packed & _LOCKED_BIT) != 0
        return out
