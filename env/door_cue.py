"""Observation v2 sensor cues: the line-of-sight DOOR channel and the KEY bits.

Both produce only finished, ego-centric arrays for the policy; nothing else leaves this
module toward the observation.

DOOR CHANNEL (80,) float32 in {0, 1}, one value per 2-pixel screen column (the same
columns as the depth rays): 1.0 where the nearest visible surface at eye level IS the
face of a closed or partly-closed door, else 0.0. It describes the scene ("door surface
here"), never what to do.

How it is computed (probe round 5, docs/reference/engine-probes.md):
  * Door identity, from the unmodified engine's sector info: a closed door is a sector
    whose ceiling height equals its floor height. Every sector of the level (not only
    those in view) is checked, and one found at gap 0 at any time in the episode is
    remembered (private, per episode, cleared on reset; never output), so the cue stays
    on while the door starts to rise. Only its two-sided, non-blocking lines (the two
    faces) are used; the blocking lines are the jambs.
    Limitation (audit 02 F10b): sector info in vizdoom 1.3.3 carries no line specials,
    tags or textures (probe R5.1), so a secret door or a closet with ceiling == floor
    cannot be told from a door and is marked as one when its face is the visible surface.
    Our generated maps have none; the line-of-sight rule below still applies to them.
  * A remembered door is cast against only while its gap (ceiling - floor) is at most
    DOOR_CAST_MAX_GAP: from gap 48 on, the eye-level band already sees the passage
    beyond, so the face is no longer the visible surface there (probe).
  * Line of sight: each column's ray is cast in the player's frame against the face
    segments (2-D, perpendicular distance), converted to depth-buffer levels with the
    probe's fit, and the column is marked ONLY if that agrees with the column's actual
    middle-band depth reading (the masked band minimum the rays use) within the
    tolerance below, AND (occlusion gate, audit 02 F1) no line of the level that blocks
    the eye-level band (one-sided wall, or two-sided line whose current opening does not
    clear the band) is crossed by the column's ray before the face. The depth check alone
    could not separate a hidden face from a wall 16-30 units in front of it; the gate
    does so from geometry. Residual: blockers nearer than ~11.7 units (depth level 0,
    masked) are left to the depth check; view bob is not modelled, so a two-sided line
    whose opening edge lies within ~8 units of the band edge and within the tolerance
    distance of the face can still pass. A door behind a wall, behind the player, or outside the field
    of view never lights. Columns with no depth reading are 0.
  * The player's pose is used inside this function only, to put the face segments in the
    player's frame (ego-centric sensory encoding). Pose, coordinates, sector ids and
    line ids are never output: the result depends only on the scene relative to the
    player (rigid-motion invariant, tested).

Range limit (derived from geometry, not measured; probe pairs reach ~400 units): the
middle band spans 4 rows either side of the horizon; beyond about 820 map units (depth
level ~113) floor/ceiling pixels in the band should be nearer than the door, so far doors
would stop lighting.

KEY BITS (6,) float32: keys held (blue, yellow, red), and "needs key" (blue, yellow, red),
a one-decision pulse set when the engine printed a locked door/switch message for that
colour during this decision. Only the parsed booleans enter the observation, never the
notification text (it can carry the map name).
"""
from __future__ import annotations

import re

import numpy as np

from .observation import N_KEY_BITS, N_RAY_COLS, SCREEN_W

KEY_COLOURS = ("blue", "yellow", "red")

# --- probe round 5 fit: depth level = DEPTH_PER_UNIT * perpendicular distance + DEPTH_OFFSET
# (7,787 closed-door column pairs on 3 generated TRAIN maps; residual 1st-99th percentile
# -2.3 .. +0.75 levels, max +0.81). About 7.15 map units per level.
DEPTH_PER_UNIT = 0.13979
DEPTH_OFFSET = -1.134
# --- tolerance on (reading - predicted), in depth levels.
# Upper: the fit never read farther than +0.81; a reading beyond the face means the ray
# passes the door (an open gap), so it is kept tight.
DOOR_TOL_FAR = 1.5
# Lower: readings up to ~2 levels nearer than the fit at close range, and a cluster at -3
# to -6 beyond 150 units whose cause was not established (probe). Occluders gave residuals
# of -20 and below (median -37).
# Re-measured 2026-11-26 with the occlusion gate (7 generated TRAIN maps, 998 frames):
# faces that ARE the first blocker in the geometry read within -0.57 .. +0.75 up to 150
# units, but beyond 150 units 16 of 2,076 still read -2 .. -6 (the cluster persists among
# truly visible faces, so it is not occlusion). The relative term is kept to keep them;
# occlusion is now carried by the gate, not by this tolerance.
DOOR_TOL_NEAR_ABS = 2.0
DOOR_TOL_NEAR_REL = 0.05
DOOR_CLOSED_GAP = 0.5         # ceiling - floor at or below this = closed (probe: exactly 0)
DOOR_CAST_MAX_GAP = 44.0      # face fills the eye band up to gap 40; passage seen from 48
# --- occlusion gate (audit 02 F1 / R9): the depth check alone let a face hidden behind a
# surface up to ~16-30 units nearer light. The gate requires the face to be the first
# eye-level-blocking line on the column's ray in the level geometry.
EYE_HEIGHT = 41.0             # Doom VIEWHEIGHT above the player's floor z
BAND_SLOPE = 0.044            # band half-height per unit of perpendicular distance: the
                              # middle band is rows h-4 .. h+3, pixel centres 3.5 rows from the
                              # horizon; 3.5 / 80 px focal = 0.044 (3.5 / 96 = 0.036 if Doom's
                              # 1.2 pixel aspect applies). Derived, not measured; the upper end
                              # is taken. View bob (up to +-8 units) is NOT added: on MAP01 a
                              # +8 slack switched off 3x more truly visible columns (93 vs 32 of
                              # 4,566) and a blocker that bob pushes into the band is read by the
                              # band minimum, which fails the depth check unless it lies within
                              # the tolerance of the face (residual, see the module docstring).
OCC_EPS = 1.0                 # a blocker counts only if nearer than the face by > 1 unit
                              # (the face itself is a closed two-sided line at the face)
OCC_MIN_F = (0.5 - DEPTH_OFFSET) / DEPTH_PER_UNIT   # ~11.7 units: nearer surfaces read depth
                              # level 0, which the rays mask out, so the band minimum cannot
                              # see them; the depth check alone decides there

# Column bin centres (pixels 1, 3, ..., 159) -> horizontal angle off the view axis (radians,
# positive = right of centre). Horizontal FOV 90 degrees: tan(angle) = (x - 80) / 80.
_HALF_W = SCREEN_W / 2.0
_COL_X = np.arange(N_RAY_COLS, dtype=np.float64) * (SCREEN_W // N_RAY_COLS) + (SCREEN_W // N_RAY_COLS) / 2.0
_COL_TAN = (_COL_X - _HALF_W) / _HALF_W              # tan of the offset angle
_COL_COS = 1.0 / np.sqrt(1.0 + _COL_TAN ** 2)        # cos of the offset angle


def _to_player_frame(seg: np.ndarray, px: float, py: float, heading_deg: float):
    """(f1, l1, f2, l2): segment endpoints in the player's frame (f = forward, l = left)."""
    th = np.radians(heading_deg)
    c, s = np.cos(th), np.sin(th)
    dx1, dy1 = seg[:, 0] - px, seg[:, 1] - py
    dx2, dy2 = seg[:, 2] - px, seg[:, 3] - py
    return dx1 * c + dy1 * s, -dx1 * s + dy1 * c, dx2 * c + dy2 * s, -dx2 * s + dy2 * c


def _cast(f1, l1, f2, l2, cols: np.ndarray):
    """Cast the given columns' view rays against segments already in the player's frame.
    Returns (f, ok), each (len(cols), N): f = forward (perpendicular, view-plane) distance of
    the crossing, ok = the ray crosses the segment in front of the player."""
    # column ray: direction (1, -tan); its forward component is the perpendicular distance.
    k = -_COL_TAN[cols][:, None]               # left per unit forward for each column
    ef, el = (f2 - f1)[None, :], (l2 - l1)[None, :]
    # solve f*(1, k) = P1 + u*E  ->  f - u*ef = f1 ; f*k - u*el = l1
    den = ef * k - el                          # determinant of [[1, -ef], [k, -el]]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (-f1[None, :] * el + ef * l1[None, :]) / den
        u = (l1[None, :] - k * f1[None, :]) / den
    ok = (np.abs(den) > 1e-12) & (f > 1e-6) & (u >= 0.0) & (u <= 1.0)
    return f, ok


def _occluded(cols: np.ndarray, f_door: np.ndarray, occ, px, py, heading_deg, eye_z) -> np.ndarray:
    """OCCLUSION GATE. (len(cols),) bool: True where some line of the level's geometry that
    blocks the eye-level band is crossed by the column's ray nearer than f_door - OCC_EPS.

    occ = (seg (M, 4), lo (M,), hi (M,)): every line of the level; [lo, hi] is the vertical
    opening through it (one-sided: empty, lo = +inf, hi = -inf; two-sided: highest floor to
    lowest ceiling of its two sectors, current heights). A line blocks at forward distance f
    unless its opening clears the band [eye - m, eye + m], m = BAND_SLOPE * f,
    i.e. unless the whole +-4-row band could see through it. Erring towards "blocks" can
    only switch a column off, never on. Heights enter only as differences from eye_z."""
    seg, lo, hi = occ
    if seg.shape[0] == 0:
        return np.zeros(cols.shape, dtype=bool)
    f1, l1, f2, l2 = _to_player_frame(seg, px, py, heading_deg)
    # only lines with some part in front of the player and nearer than the farthest door hit
    keep = (np.maximum(f1, f2) > 0.0) & (np.minimum(f1, f2) < float(f_door.max()))
    if not keep.any():
        return np.zeros(cols.shape, dtype=bool)
    f1, l1, f2, l2 = f1[keep], l1[keep], f2[keep], l2[keep]
    rlo, rhi = lo[keep] - eye_z, hi[keep] - eye_z        # opening relative to the eye
    f, ok = _cast(f1, l1, f2, l2, cols)
    m = BAND_SLOPE * f
    blocks = (rlo[None, :] > -m) | (rhi[None, :] < m)
    return (ok & blocks & (f >= OCC_MIN_F) & (f < f_door[:, None] - OCC_EPS)).any(axis=1)


def door_channel(faces: np.ndarray, px: float, py: float, heading_deg: float,
                 mid_reading: np.ndarray, occ=None, eye_z: float = 0.0) -> np.ndarray:
    """(80,) float32 door cue. Pure function of arrays.

    faces: (N, 4) float segments (x1, y1, x2, y2) of door faces to test (may be empty).
    px, py, heading_deg: the player's pose, used only to express the faces in the player's
      frame (heading: degrees counter-clockwise; screen right = clockwise).
    mid_reading: (80,) the middle band's depth reading in raw levels (0 = no reading).
    occ, eye_z: the level's lines and their openings (DoorTracker.scene_now) and the eye
      height (player z + EYE_HEIGHT), for the occlusion gate. The env always passes them;
      occ=None (no gate) exists only for engine-free tests of the depth check alone.
    A column lights iff (1) a door face's predicted depth agrees with the reading AND
    (2) no eye-level-blocking line is crossed before that face (the face is the first
    blocking surface on the ray in the level geometry).
    """
    out = np.zeros((N_RAY_COLS,), dtype=np.float32)
    faces = np.asarray(faces, dtype=np.float64).reshape(-1, 4)
    if faces.shape[0] == 0:
        return out
    f, ok = _cast(*_to_player_frame(faces, px, py, heading_deg), np.arange(N_RAY_COLS))
    pred = np.clip(DEPTH_PER_UNIT * f + DEPTH_OFFSET, 1.0, 255.0)
    rd = np.asarray(mid_reading, dtype=np.float64)[:, None]
    match = ok & (rd > 0) & (rd <= pred + DOOR_TOL_FAR) & \
        (rd >= pred - (DOOR_TOL_NEAR_ABS + DOOR_TOL_NEAR_REL * pred))
    lit = match.any(axis=1)
    if occ is not None and lit.any():
        cols = np.nonzero(lit)[0]
        f_door = np.where(match[cols], f[cols], np.inf).min(axis=1)   # nearest matching face
        lit[cols[_occluded(cols, f_door, occ, px, py, heading_deg, eye_z)]] = False
    out[lit] = 1.0
    return out


class DoorTracker:
    """Per-episode memory of which sectors were seen closed, and their face segments.
    Private to the env: nothing here is output except through door_channel()."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._faces: dict[int, np.ndarray] = {}
        self._gaps: dict[int, float] = {}
        self._saw_sectors = False
        # occlusion-gate line table, built once per episode (geometry is static; only the
        # sector heights change): unique segments and their one or two sectors' row indices
        self._sec_ids: list | None = None
        self._seg = np.zeros((0, 4))
        self._side = np.zeros((0, 2), dtype=np.int64)

    def _build_lines(self, sectors):
        """Unique line segments over all sectors' line lists. A two-sided line appears in
        both of its sectors' lists, a one-sided line in one (probe: freedoom2 MAP01, 705 + 57
        segments twice, 510 + 2 once; none listed twice by the same sector). is_blocking is
        NOT used: it marks impassable two-sided lines too (57 on MAP01), which can be seen
        through. A segment listed once is treated as one-sided (opaque)."""
        sides: dict[tuple, list[int]] = {}
        for r, sec in enumerate(sectors):
            for ln in sec.lines:
                a, b = (ln.x1, ln.y1), (ln.x2, ln.y2)
                key = (a, b) if a <= b else (b, a)
                lst = sides.setdefault(key, [])
                if r not in lst:
                    lst.append(r)
        keys = list(sides)
        self._seg = np.array([(a[0], a[1], b[0], b[1]) for a, b in keys], dtype=np.float64).reshape(-1, 4)
        self._side = np.array([(v[0], v[1] if len(v) == 2 else -1) for v in sides.values()],
                              dtype=np.int64).reshape(-1, 2)

    def scene_now(self, sectors):
        """(faces, occ) for door_channel: the castable door faces (as faces_now) and the
        occlusion table (seg (M,4), lo (M,), hi (M,)) with each line's current vertical
        opening (one-sided: lo=+inf, hi=-inf). One pass over the sector list. Private;
        never output."""
        if not sectors:
            return np.zeros((0, 4)), None
        hts = [(sec.id, sec.floor_height, sec.ceiling_height) for sec in sectors]
        faces = self.faces_now(sectors, hts)
        ids = [h[0] for h in hts]
        if self._sec_ids != ids:                    # first call of the episode (or reorder)
            self._build_lines(sectors)
            self._sec_ids = ids
        fc = np.empty((len(hts) + 1, 2))
        fc[:-1] = [(h[1], h[2]) for h in hts]       # rows follow the list order
        fc[-1] = (np.inf, -np.inf)                  # row -1: "no sector" -> empty opening
        a, b = self._side[:, 0], self._side[:, 1]
        lo = np.maximum(fc[a, 0], fc[b, 0])
        hi = np.minimum(fc[a, 1], fc[b, 1])
        return faces, (self._seg, lo, hi)

    def gaps(self) -> dict[int, float] | None:
        """PRIVILEGED (training diagnostics only, via DoomEnv.diag_door_gaps): {engine
        sector id: gap} of every remembered door at the last faces_now() call; None if no
        sector list was seen this episode. Never part of an observation or of info."""
        return dict(self._gaps) if self._saw_sectors else None

    def faces_now(self, sectors, hts=None) -> np.ndarray:
        """(N, 4) face segments of remembered doors whose gap is <= DOOR_CAST_MAX_GAP.
        hts: optional [(id, floor, ceiling)] in list order, already read by scene_now."""
        if not sectors:
            return np.zeros((0, 4))
        self._saw_sectors = True
        if hts is None:
            hts = [(sec.id, sec.floor_height, sec.ceiling_height) for sec in sectors]
        out = []
        for i, (sid, fl, ce) in enumerate(hts):
            gap = ce - fl
            fc = self._faces.get(sid)
            if fc is not None:
                self._gaps[sid] = float(gap)
            if gap > DOOR_CAST_MAX_GAP:
                continue
            if fc is None:
                if gap > DOOR_CLOSED_GAP:
                    continue                  # never seen closed: not known as a door
                fc = np.array([(l.x1, l.y1, l.x2, l.y2) for l in sectors[i].lines if not l.is_blocking],
                              dtype=np.float64).reshape(-1, 4)
                self._faces[sid] = fc
                self._gaps[sid] = float(gap)
            out.append(fc)
        return np.concatenate(out, axis=0) if out else np.zeros((0, 4))


# ---------------------------------------------------------------- keys
_NEED_LINE = re.compile(r"\bneed", re.IGNORECASE)
_COLOUR = re.compile(r"\b(blue|yellow|red)\b", re.IGNORECASE)


def keys_needed(text: str) -> set[str]:
    """Colours named on a locked door/switch message line in a notifications string.
    Captured engine text (probe R2.3, round 5): "Yellow key needed for this door.",
    "Blue key needed.". Doom wording "You need a blue key ..." also matches. Pickup
    text ("Blue passcard secured!") has no "need" and never matches."""
    out: set[str] = set()
    for line in (text or "").splitlines():
        if _NEED_LINE.search(line):
            out.update(m.lower() for m in _COLOUR.findall(line))
    return out


def key_bits(held: dict, needed: set[str]) -> np.ndarray:
    """(6,) float32: held blue, yellow, red; needs blue, yellow, red (this decision)."""
    out = np.zeros((N_KEY_BITS,), dtype=np.float32)
    for i, col in enumerate(KEY_COLOURS):
        out[i] = 1.0 if held.get(col, False) else 0.0
        out[3 + i] = 1.0 if col in needed else 0.0
    return out
