"""Policy observation extraction (ADR 0001 contract). Pure functions only.

Inputs are restricted to: the depth buffer, the labels buffer, the label list's
screen-box fields (x, y, width, height, value, id, name, category), the screen
buffer, and the agent's own game variables passed in by the env. World-frame
fields of labels are never read here; a source-level test enforces it.
Ego-motion rotates the agent's own velocity into its body frame (ruling d).
"""
from __future__ import annotations

import math

import numpy as np

SCREEN_H, SCREEN_W = 120, 160
N_RAY_COLS = 80
N_WEAPONS = 10
SELF_DIM = 28          # observation v1 self vector (unchanged)
N_KEY_BITS = 6         # v2 appends: keys held (b, y, r), needs key (b, y, r); env/door_cue.py
# v2 = v1 + door ray channel (last row of `rays`) + key bits in `self`
# v3 = v2 + `egomap` (5, 32, 32) uint8 ego-centric occupancy memory (env/egomap.py)
OBS_VERSIONS = (1, 2, 3)


def check_obs_version(v: int) -> int:
    if v not in OBS_VERSIONS:
        raise ValueError(f"obs_version {v!r} not in {OBS_VERSIONS}")
    return v


def ray_channels(n_bands: int, obs_version: int) -> int:
    """Rows of `rays`: the depth bands, plus the door channel under v2."""
    return n_bands + (1 if check_obs_version(obs_version) >= 2 else 0)


def self_dim(obs_version: int) -> int:
    return SELF_DIM + (N_KEY_BITS if check_obs_version(obs_version) >= 2 else 0)

# entity_type ids (fixed table, ADR 0001; "hazard" appended by ADR 0006 so that
# existing ids do not move). Models must size their type embedding by len().
ENTITY_TYPES = (
    "pad", "monster", "projectile", "health", "armor", "ammo", "weapon",
    "key-blue", "key-yellow", "key-red", "other-pickup", "decoration", "unknown",
    "hazard",
)
T = {n: i for i, n in enumerate(ENTITY_TYPES)}

# Engine object_category -> type. Categories seen on Freedoom 1/2 maps (probes
# `categories`, `projectiles`). "Explosive" is deliberately absent: the engine puts
# every missile AND impact effects (BulletPuff, RevenantTracerSmoke) in it, so the
# category alone cannot say what the thing is; such actors are typed by class name
# below, and unlisted "Explosive" actors fall to "unknown".
_CATEGORY_TABLE = {
    "Monster": T["monster"],
    "Projectile": T["projectile"],
    "Health": T["health"],
    "Armor": T["armor"],
    "Ammo": T["ammo"],
    "Weapon": T["weapon"],
    "Powerup": T["other-pickup"],
    "Decoration": T["decoration"],
    "LightSource": T["decoration"],
    "Vegetation": T["decoration"],
    "Gore": T["decoration"],
    "Gibs": T["decoration"],
    "SFX": T["decoration"],         # TeleportFog, ItemFog, RocketSmokeTrail
    "Hazard": T["hazard"],          # ExplosiveBarrel
}
# Missiles (all report category "Explosive"; probe `projectiles`).
_PROJECTILE_NAMES = frozenset({
    "DoomImpBall", "BaronBall", "CacodemonBall", "Rocket", "PlasmaBall", "BFGBall",
    "ArachnotronPlasma", "FatShot", "RevenantTracer",
})
# Visual impact/trail effects that also report "Explosive": scenery-like, not
# something that moves toward anyone, so they share the decoration id.
_EFFECT_NAMES = frozenset({"BulletPuff", "RevenantTracerSmoke", "RocketSmokeTrail"})
SELF_NAMES = frozenset({"DoomPlayer"})


def entity_type_id(name: str, category: str) -> int:
    if category == "Key":
        low = name.lower()
        if "blue" in low:
            return T["key-blue"]
        if "yellow" in low:
            return T["key-yellow"]
        if "red" in low:
            return T["key-red"]
        return T["unknown"]
    if name in _PROJECTILE_NAMES:
        return T["projectile"]
    if name in _EFFECT_NAMES:
        return T["decoration"]
    return _CATEGORY_TABLE.get(category, T["unknown"])


# ---------------------------------------------------------------- rays
def rays(depth: np.ndarray, bands, own: np.ndarray | None = None) -> np.ndarray:
    """(3, 80) float32: nearest WORLD depth per (row band, 2-px column bin), in [0,1].

    Depth polarity (probe): larger value = farther, about 8 map units per level.
    Pixels in `own` (invalid_depth_mask: the player's weapon, its firing overlay,
    and depth-0 artefacts of translucent sprites) are excluded from the min.
    Encoding:
      * a real reading is max(depth, 1) / 255, so it is never exactly 0;
      * NO_READING (exactly 0.0) means no pixel of the cell has a valid depth (in
        practice: the weapon covers it): no world measurement this step. The same
        code is used when telemetry is absent.
    """
    out = np.empty((len(bands), N_RAY_COLS), dtype=np.float32)
    for i, (r0, r1) in enumerate(bands):
        blk = depth[r0:r1].astype(np.int16)
        if own is not None:
            blk = np.where(own[r0:r1], _MASKED, blk)
        m = blk.reshape(r1 - r0, N_RAY_COLS, SCREEN_W // N_RAY_COLS).min(axis=(0, 2))
        out[i] = np.where(m == _MASKED, 0, np.maximum(m, 1))
    out *= 1.0 / 255.0
    return out


NO_READING = 0.0
_MASKED = np.int16(1000)  # larger than any uint8 depth


def invalid_depth_mask(labels, labels_buf: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """(H, W) bool: pixels whose depth value is not a measurement of the world.

    Engine behaviour (probes `own_pixels`, `zero_depth_breakdown`, `overlay_sources`,
    `label_zero_depth`, docs/reference/engine-probes.md R3):
      * the player's weapon carries the Self label's value (255) and depth 0;
      * while firing, more depth-0 pixels appear in the weapon area with a labels
        value no listed label claims (never seen without attacking);
      * the translucent BulletPuff sprite has some depth-0 pixels although it sits on
        a wall several levels away; no other class showed any depth-0 pixel;
      * world geometry is never nearer than the player radius (16 units = 2 levels).
    So depth 0 never measures the world. Invalid = depth 0, plus every Self-label
    pixel (all observed at depth 0; included explicitly in case one is not).
    """
    bad = depth == 0
    self_vals = [l.value for l in labels if l.object_category == "Self" or l.object_name in SELF_NAMES]
    if self_vals:
        bad |= np.isin(labels_buf, self_vals)
    return bad


# ---------------------------------------------------------------- entities
class EntityTracker:
    """Keeps (u, d) per engine label id from the previous decision for du/dd."""

    def __init__(self):
        self.prev: dict[int, tuple[float, float]] = {}

    def reset(self):
        self.prev = {}


def entities(labels, labels_buf: np.ndarray, depth: np.ndarray, tracker: EntityTracker,
             max_entities: int):
    """Return (entities (N,8) f32, entity_type (N,) i64, entity_mask (N,) bool).

    Only labels with at least one pixel of their own value in the labels buffer get a
    row: a label with none is not visible (hidden behind the weapon or a nearer
    sprite, probe R4.2) and is dropped, as if absent from the list.
    `d` is the mean depth of the pixels carrying this entity's own label value AND a
    valid (non-zero) depth, so weapon/overlay pixels and the depth-0 artefacts of
    translucent sprites never enter it. If the entity has own pixels but none with a
    valid depth (possible for the translucent BulletPuff), d = NO_READING (0.0) and
    dd = 0; real readings are >= 1/255. An entity that drops out (hidden) and comes
    back is reported again with is_new = 1 and du = dd = 0 (tracking keeps only the
    previous decision's rows).
    """
    feats = np.zeros((max_entities, 8), dtype=np.float32)
    types = np.zeros((max_entities,), dtype=np.int64)
    mask = np.zeros((max_entities,), dtype=bool)
    rows = []
    for lab in labels:
        name = lab.object_name
        if name in SELF_NAMES or lab.object_category == "Self":
            continue
        x, y, w, h = lab.x, lab.y, lab.width, lab.height
        if w <= 0 or h <= 0:
            continue
        box_d = depth[y:y + h, x:x + w]
        own = labels_buf[y:y + h, x:x + w] == lab.value
        if not own.any():
            # Probe R4.2: the engine lists sprites that are fully hidden behind a nearer
            # sprite (weapon, firing overlay, another thing). Not visible -> no row.
            continue
        sel = box_d[own & (box_d > 0)]
        if sel.size:
            d = max(float(sel.mean()), 1.0) / 255.0
        else:
            d = NO_READING  # no pixel of this entity is visible: no measurement
        u = (x + 0.5 * w) / SCREEN_W * 2.0 - 1.0
        v = (y + 0.5 * h) / SCREEN_H * 2.0 - 1.0
        rows.append((d, lab.object_id, u, v, w / SCREEN_W, h / SCREEN_H,
                     entity_type_id(name, lab.object_category)))
    if len(rows) > max_entities:
        # sensor-range limit: keep the nearest, then restore label order
        keep = sorted(range(len(rows)),
                      key=lambda i: rows[i][0] if rows[i][0] > 0 else np.inf)[:max_entities]
        rows = [rows[i] for i in sorted(keep)]
    new_prev = {}
    for i, (d, oid, u, v, w, h, t) in enumerate(rows):
        p = tracker.prev.get(oid)
        if p is None:
            du = dd = 0.0
            is_new = 1.0
        else:
            du, is_new = u - p[0], 0.0
            dd = d - p[1] if (d > 0 and p[1] > 0) else 0.0
        feats[i] = (u, v, w, h, d, du, dd, is_new)
        types[i] = t
        mask[i] = True
        new_prev[oid] = (u, d)
    tracker.prev = new_prev
    return feats, types, mask


# ---------------------------------------------------------------- self
def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def ego_motion(vx: float, vy: float, vz: float, heading_deg: float,
               prev_heading_deg: float | None) -> np.ndarray:
    """Body-frame (forward, left, up) speed and turn delta, scaled. Proprioception only.

    The heading is used for the rotation and the wrapped difference, never output.
    """
    th = math.radians(heading_deg)
    c, s = math.cos(th), math.sin(th)
    fwd = vx * c + vy * s
    left = -vx * s + vy * c
    turn = 0.0 if prev_heading_deg is None else wrap_deg(heading_deg - prev_heading_deg)
    return np.clip(np.array([fwd / 20.0, left / 20.0, vz / 20.0, turn / 180.0],
                            dtype=np.float32), -1.0, 1.0)


def self_vector(health, armor, sel_ammo, damage_since_last, selected_weapon,
                weapons_owned, ego: np.ndarray) -> np.ndarray:
    out = np.zeros((SELF_DIM,), dtype=np.float32)
    out[0] = health / 100.0
    out[1] = armor / 100.0
    out[2] = min(max(sel_ammo, 0.0), 200.0) / 200.0
    out[3] = min(max(damage_since_last, 0.0), 200.0) / 100.0
    sw = int(selected_weapon)
    if 0 <= sw < N_WEAPONS:
        out[4 + sw] = 1.0
    out[14:24] = (np.asarray(weapons_owned, dtype=np.float32) > 0).astype(np.float32)
    out[24:28] = ego
    return out


# ---------------------------------------------------------------- screen
def screen_small(screen_chw: np.ndarray) -> np.ndarray:
    """(3,120,160) uint8 -> (3,60,80) uint8 by 2x2 area average."""
    s = screen_chw.reshape(3, 60, 2, 80, 2).astype(np.uint16).sum(axis=(2, 4))
    return ((s + 2) >> 2).astype(np.uint8)
