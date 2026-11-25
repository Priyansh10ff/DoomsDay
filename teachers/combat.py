"""Combat labelling for the scripted teacher (TRAINING-TIME ONLY; see teachers/__init__.py).

Aiming here uses trigonometry on PRIVILEGED world positions (atan2 toward a monster).
That is an aimbot, which the rules forbid in the decision path; it is allowed here only
because this package is never imported by agent/, env/ or evaluate/
(tests/test_teachers_isolation.py, direct and transitive). The student never sees any
of this: it learns from the labels, and its own observation is the env's entity rows.

Visibility follows the env's own entity rule (env/observation.py `entities`): a monster
counts only if its label is in the engine's label list AND at least one pixel of the
labels buffer carries its value. So the teacher reacts only to monsters the student can
see; the label's world position is then used to aim.

Liveness (probe `python -m teachers.probe_combat corpses`, 2026-11-26, two kills on
S3-train-103004): on the decision after KILLCOUNT rises, the killed monster's label is
no longer in the "Monster" category (it vanished from the category-filtered list on all
six decisions watched), so "category Monster with an own pixel" already means a live,
visible monster. No kill bookkeeping is needed for target choice.

Attack (probe `python -m teachers.probe_combat fire`, pistol, frame skip 4, 60
decisions): a HELD attack keeps firing (16 shots, one every 3-4 decisions, the refire
cycle); every tap pattern fired less (1-0: 12, 1-0-0: 13, 1-1-0: 15). So the label
holds attack while aligned. (The shotgun is assumed to behave the same when held: same
refire mechanism, not probed.) Refire shots of the pistol carry a spread, which only
matters at range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MONSTER_CATEGORY = "Monster"
# Engine class -> collision radius (vanilla thing table); unknown monsters use 20.
MONSTER_RADIUS = {"Zombieman": 20.0, "ShotgunGuy": 20.0, "DoomImp": 20.0, "Demon": 30.0,
                  "Spectre": 30.0, "ChaingunGuy": 20.0, "LostSoul": 16.0, "Cacodemon": 31.0}
MELEE_ONLY = frozenset({"Demon", "Spectre", "LostSoul"})
PROJECTILE_NAMES = frozenset({"DoomImpBall", "BaronBall", "CacodemonBall", "Rocket", "PlasmaBall",
                              "BFGBall", "ArachnotronPlasma", "FatShot", "RevenantTracer"})
PLAYER_RADIUS = 16.0


@dataclass(frozen=True)
class Sighting:
    """One visible thing this decision (world frame: PRIVILEGED, teacher only)."""
    id: int
    name: str
    x: float
    y: float
    vx: float
    vy: float
    pix: int          # own pixels in the labels buffer (> 0 = visible to the student)
    w: int            # screen box
    h: int


def _sightings(state, want) -> list[Sighting]:
    labels = getattr(state, "labels", None)
    buf = getattr(state, "labels_buffer", None)
    if not labels or buf is None:
        return []
    out = []
    for lab in labels:
        if not want(lab):
            continue
        x, y, w, h = lab.x, lab.y, lab.width, lab.height
        if w <= 0 or h <= 0:
            continue
        pix = int(np.count_nonzero(buf[y:y + h, x:x + w] == lab.value))
        if pix == 0:
            continue          # env rule: no own pixel -> not visible, no entity row
        out.append(Sighting(int(lab.object_id), str(lab.object_name), float(lab.object_position_x),
                            float(lab.object_position_y), float(lab.object_velocity_x),
                            float(lab.object_velocity_y), pix, int(w), int(h)))
    return out


def sightings_of(game) -> tuple[list[Sighting], list[Sighting]]:
    """(visible monsters, visible projectiles) of the engine's current state. A fake game
    without a label list (tests) gives two empty lists."""
    st = game.get_state()
    if st is None:
        return [], []
    mons = _sightings(st, lambda l: l.object_category == MONSTER_CATEGORY)
    proj = _sightings(st, lambda l: l.object_name in PROJECTILE_NAMES)
    return mons, proj


@dataclass(frozen=True)
class CombatConfig:
    engage_range: float = 1400.0      # ignore visible monsters farther than this
    aim_tol_min_deg: float = 2.5      # turn quantum is 4 deg/decision: never demand < 2
    fire_turn_max_deg: float = 4.5    # fire with a turn only if it is a fine correction
    melee_backoff_range: float = 140.0   # back off from a melee-only monster this close
    close_range: float = 96.0         # any monster this close: back off while shooting
    dodge_horizon_tics: float = 40.0  # projectile closest approach within this many tics ...
    dodge_miss_dist: float = 44.0     # ... and nearer than this = incoming
    dodge_probe: tuple[float, ...] = (24.0, 48.0)   # lateral free-space probe for the dodge side
    melee_priority: float = 0.6       # distance factor for melee-only monsters (charge first)
    max_engage_decisions: int = 80    # give up on one target after this many engaged decisions
    back_probe: float = 40.0          # backing off only if this far behind is walkable


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def choose_target(mons: list[Sighting], x: float, y: float, cfg: CombatConfig) -> Sighting | None:
    """Most threatening live visible monster: nearest, melee-only monsters weighted closer."""
    best, best_s = None, math.inf
    for m in mons:
        d = math.hypot(m.x - x, m.y - y)
        if d > cfg.engage_range:
            continue
        s = d * (cfg.melee_priority if m.name in MELEE_ONLY else 1.0)
        if s < best_s - 1e-9 or (abs(s - best_s) <= 1e-9 and best is not None and m.id < best.id):
            best, best_s = m, s
    return best


def aim_tolerance(m: Sighting, dist: float, cfg: CombatConfig) -> float:
    r = MONSTER_RADIUS.get(m.name, 20.0)
    return max(cfg.aim_tol_min_deg, math.degrees(math.atan2(r * 0.8, max(dist, 1.0))))


def incoming(proj: list[Sighting], x: float, y: float, cfg: CombatConfig):
    """The most urgent projectile on a near-collision course: (time, closest-point offset
    vector from the player) or None. Player assumed still over the horizon."""
    best = None
    for p in proj:
        v2 = p.vx * p.vx + p.vy * p.vy
        if v2 < 1e-6:
            continue
        rx, ry = p.x - x, p.y - y
        t = -(rx * p.vx + ry * p.vy) / v2
        if not 0.0 < t <= cfg.dodge_horizon_tics:
            continue
        ox, oy = rx + p.vx * t, ry + p.vy * t
        if math.hypot(ox, oy) > cfg.dodge_miss_dist:
            continue
        if best is None or t < best[0]:
            best = (t, ox, oy, p.vx, p.vy)
    return best
