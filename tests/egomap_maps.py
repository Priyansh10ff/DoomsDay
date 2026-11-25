"""Test-only synthetic maps for the observation-v3 egomap tests (env-engineer).

A straight corridor 128 units wide (y in [-64, 64]) running east from x = 0 to x = 1536,
with a closed DR door sector (ceiling == floor, special 1 on both faces) across it at
x in [DOOR_X0, DOOR_X0 + 16]. Player 1 starts at (PLAYER_X, 0) facing east. `shift`
translates every vertex and thing by the same integer offset: the same local scene at a
different absolute place on the map (translation-invariance test). Written with our own
maps.wad packer into a temp folder; the engine builds the nodes itself.
"""
from __future__ import annotations

import os

from maps.wad import (ML_BLOCKING, ML_TWOSIDED, Linedef, Sector, Sidedef, Thing, pack_map,
                      write_wad)

HALF_W = 64
LENGTH = 1536
DOOR_X0 = 640
DOOR_T = 16
PLAYER_X = 64
CEIL = 128
WALL, DOOR_TEX, TRACK = "STARTAN2", "BIGDOOR2", "DOORTRAK"


BLUE_KEYCARD = 5


def corridor_wad(path: str, shift: tuple[int, int] = (0, 0), with_door: bool = True,
                 player_x: int = PLAYER_X, key_x: int | None = None) -> str:
    """key_x: put a blue keycard on the corridor axis at that x (None = no key)."""
    sx, sy = shift
    x1, x2 = DOOR_X0, DOOR_X0 + DOOR_T   # with_door=False: same lines, sector fully open
    verts = [(0, -HALF_W), (0, HALF_W), (x1, HALF_W), (x1, -HALF_W),
             (x2, HALF_W), (x2, -HALF_W), (LENGTH, HALF_W), (LENGTH, -HALF_W)]
    verts = [(x + sx, y + sy) for x, y in verts]
    sectors = [Sector(0, CEIL, "FLOOR0_1", "CEIL3_5", 192),                     # 0: A
               Sector(0, 0 if with_door else CEIL, "FLOOR0_1", "CEIL3_5", 192),  # 1: door
               Sector(0, CEIL, "FLOOR0_1", "CEIL3_5", 192)]                     # 2: B
    sides: list[Sidedef] = []
    lines: list[Linedef] = []

    def side(sec, upper="-", lower="-", mid="-"):
        sides.append(Sidedef(0, 0, upper, lower, mid, sec))
        return len(sides) - 1

    def one(v1, v2, sec, tex=WALL):
        lines.append(Linedef(v1, v2, ML_BLOCKING, 0, 0, side(sec, mid=tex)))

    def two(v1, v2, front, back, special):
        lines.append(Linedef(v1, v2, ML_TWOSIDED, special, 0,
                             side(front, upper=DOOR_TEX), side(back)))

    # sector A (clockwise: interior on the right)
    one(1, 2, 0); one(3, 0, 0); one(0, 1, 0)
    two(2, 3, 0, 1, 1)                     # A | door face, front = A
    # door sector jambs
    one(2, 4, 1, TRACK); one(5, 3, 1, TRACK)
    two(5, 4, 2, 1, 1)                     # B | door face, front = B
    # sector B
    one(4, 6, 2); one(6, 7, 2); one(7, 5, 2)
    things = [Thing(player_x + sx, 0 + sy, 0, 1, 7)]
    if key_x is not None:
        things.append(Thing(key_x + sx, 0 + sy, 0, BLUE_KEYCARD, 7))
    lumps = pack_map("MAP01", verts, lines, sides, sectors, things)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_wad(path, lumps)
    return path


def corridor_config(wad_path: str, obs_version: int = 3):
    """Non-vanilla engine-test config on the synthetic corridor (map_wad in a temp folder,
    freedoom2 IWAD for textures). Screen off, timer off."""
    from env.config import TRAIN_CONFIG, VANILLA_ENGINE_REWARDS
    return TRAIN_CONFIG.replace(name="egomap-corridor-test", scenario_wad=None, map_wad=wad_path,
                                maps=("MAP01",), skill=3, episode_timeout_tics=4200,
                                engine_rewards=VANILLA_ENGINE_REWARDS, obs_version=obs_version,
                                screen_enabled=False, win_timer_ms=None)
