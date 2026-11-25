"""Observation v2 sensor cues: the line-of-sight DOOR channel and the KEY bits.

Both produce only finished, ego-centric arrays for the policy; nothing else leaves this
module toward the observation.

DOOR CHANNEL (80,) float32 in {0, 1}, one value per 2-pixel screen column (the same
columns as the depth rays): 1.0 where the nearest visible surface at eye level IS the
face of a closed or partly-closed door, else 0.0. It describes the scene ("door surface
here"), never what to do.

How it is computed (probe round 5, docs/reference/engine-probes.md):
  * Door identity, from the unmodified engine's sector info: a closed door is a sector
    whose ceiling height equals its floor height. A sector seen closed at any time in the
    episode is remembered (private, per episode, cleared on reset; never output), so the
    cue stays on while the door starts to rise. Only its two-sided, non-blocking lines
    (the two faces) are used; the blocking lines are the jambs.
  * A remembered door is cast against only while its gap (ceiling - floor) is at most
    DOOR_CAST_MAX_GAP: from gap 48 on, the eye-level band already sees the passage
    beyond, so the face is no longer the visible surface there (probe).
  * Line of sight: each column's ray is cast in the player's frame against the face
    segments (2-D, perpendicular distance), converted to depth-buffer levels with the
    probe's fit, and the column is marked ONLY if that agrees with the column's actual
    middle-band depth reading (the masked band minimum the rays use) within the
    tolerance below. A door behind a wall, behind the player, or outside the field of
    view never lights. Columns with no depth reading are 0.
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
DOOR_TOL_NEAR_ABS = 2.0
DOOR_TOL_NEAR_REL = 0.05
DOOR_CLOSED_GAP = 0.5         # ceiling - floor at or below this = closed (probe: exactly 0)
DOOR_CAST_MAX_GAP = 44.0      # face fills the eye band up to gap 40; passage seen from 48

# Column bin centres (pixels 1, 3, ..., 159) -> horizontal angle off the view axis (radians,
# positive = right of centre). Horizontal FOV 90 degrees: tan(angle) = (x - 80) / 80.
_HALF_W = SCREEN_W / 2.0
_COL_X = np.arange(N_RAY_COLS, dtype=np.float64) * (SCREEN_W // N_RAY_COLS) + (SCREEN_W // N_RAY_COLS) / 2.0
_COL_TAN = (_COL_X - _HALF_W) / _HALF_W              # tan of the offset angle
_COL_COS = 1.0 / np.sqrt(1.0 + _COL_TAN ** 2)        # cos of the offset angle


def door_channel(faces: np.ndarray, px: float, py: float, heading_deg: float,
                 mid_reading: np.ndarray) -> np.ndarray:
    """(80,) float32 door cue. Pure function of arrays.

    faces: (N, 4) float segments (x1, y1, x2, y2) of door faces to test (may be empty).
    px, py, heading_deg: the player's pose, used only to express the faces in the player's
      frame (heading: degrees counter-clockwise; screen right = clockwise).
    mid_reading: (80,) the middle band's depth reading in raw levels (0 = no reading).
    """
    out = np.zeros((N_RAY_COLS,), dtype=np.float32)
    faces = np.asarray(faces, dtype=np.float64).reshape(-1, 4)
    if faces.shape[0] == 0:
        return out
    th = np.radians(heading_deg)
    c, s = np.cos(th), np.sin(th)
    # segment endpoints in the player's frame: f = forward, l = left
    dx1, dy1 = faces[:, 0] - px, faces[:, 1] - py
    dx2, dy2 = faces[:, 2] - px, faces[:, 3] - py
    f1, l1 = dx1 * c + dy1 * s, -dx1 * s + dy1 * c
    f2, l2 = dx2 * c + dy2 * s, -dx2 * s + dy2 * c
    # column ray in the player's frame: direction (1, -tan) scaled; the forward component
    # of the hit point is directly the perpendicular (view-plane) distance.
    k = -_COL_TAN[:, None]                    # left per unit forward for each column
    ef, el = (f2 - f1)[None, :], (l2 - l1)[None, :]
    # solve f*(1, k) = P1 + u*E  ->  f - u*ef = f1 ; f*k - u*el = l1
    den = ef * k - el                          # determinant of [[1, -ef], [k, -el]]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (-f1[None, :] * el + ef * l1[None, :]) / den
        u = (l1[None, :] - k * f1[None, :]) / den
    ok = (np.abs(den) > 1e-12) & (f > 1e-6) & (u >= 0.0) & (u <= 1.0)
    pred = np.clip(DEPTH_PER_UNIT * f + DEPTH_OFFSET, 1.0, 255.0)
    rd = np.asarray(mid_reading, dtype=np.float64)[:, None]
    match = ok & (rd > 0) & (rd <= pred + DOOR_TOL_FAR) & \
        (rd >= pred - (DOOR_TOL_NEAR_ABS + DOOR_TOL_NEAR_REL * pred))
    out[match.any(axis=1)] = 1.0
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

    def gaps(self) -> dict[int, float] | None:
        """PRIVILEGED (training diagnostics only, via DoomEnv.diag_door_gaps): {engine
        sector id: gap} of every remembered door at the last faces_now() call; None if no
        sector list was seen this episode. Never part of an observation or of info."""
        return dict(self._gaps) if self._saw_sectors else None

    def faces_now(self, sectors) -> np.ndarray:
        """(N, 4) face segments of remembered doors whose gap is <= DOOR_CAST_MAX_GAP."""
        if not sectors:
            return np.zeros((0, 4))
        self._saw_sectors = True
        out = []
        for sec in sectors:
            gap = sec.ceiling_height - sec.floor_height
            if sec.id in self._faces:
                self._gaps[sec.id] = float(gap)
            if gap > DOOR_CAST_MAX_GAP:
                continue
            sid = sec.id
            fc = self._faces.get(sid)
            if fc is None:
                if gap > DOOR_CLOSED_GAP:
                    continue                  # never seen closed: not known as a door
                fc = np.array([(l.x1, l.y1, l.x2, l.y2) for l in sec.lines if not l.is_blocking],
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
