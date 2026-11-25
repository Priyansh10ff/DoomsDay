"""Synthetic corridor route and a tiny kinematic fake of DoomEnv for teacher tests
(no engine). The corridor runs east along x: cells 0..11, rows 0..2; a door sector
fills column 5; the exit pad is the last two columns. The fake engine models the door's
gap like a vanilla DR door: a fresh USE edge whose view ray hits a door face within 64
units starts it opening (closed) or reverses it (moving / open); 32 units per decision,
fully open at 72, passable from 56."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from teachers.navigator import (VOID, Anchor, Grid, NavConfig, Route, _door_anchor,
                                distance_field)
from teachers.use_label import (KIND_DOOR, USE_RANGE_ENGINE, facing_hit, lines_from_grid,
                                on_front)

TURN_BINS = (-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0)
CELL = 32
W, H, DOOR_COL, ROOM_SEC, DOOR_SEC, PAD_SEC = 12, 3, 5, 0, 1, 2


def corridor_route(door_color: str | None = None) -> Route:
    sector = np.full((W, H), ROOM_SEC, dtype=np.int64)
    sector[DOOR_COL, :] = DOOR_SEC
    sector[W - 2:, :] = PAD_SEC
    door_of = np.where(sector == DOOR_SEC, DOOR_SEC, VOID).astype(np.int64)
    g = Grid(0, 0, sector, door_of, CELL)
    cfg = NavConfig()
    fake_map = {"linedefs": [], "sidedefs": []}
    door = _door_anchor(g, fake_map, DOOR_SEC, (DOOR_COL - 1, 1), cfg)
    door.door_color = door_color
    pad_cells = [(c, r) for c in range(W - 2, W) for r in range(H)]
    pts = [g.center(c, r) for c, r in pad_cells]
    pad = Anchor("exit_pad", float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
    door.dist = distance_field(g, [g.cell_of(door.x, door.y)], cfg)
    pad.dist = distance_field(g, pad_cells, cfg)
    lines = lines_from_grid(g, {DOOR_SEC: door_color} if door_color else None)
    return Route("synthetic", "S1", g, [door, pad], (48.0, 48.0), [], cfg, lines)


GAP_STEP, GAP_OPEN, GAP_PASS = 32.0, 72.0, 56.0


@dataclass
class _Sector:
    id: int
    floor_height: float
    ceiling_height: float


class _State:
    def __init__(self, sectors):
        self.sectors = sectors


class _Game:
    def __init__(self, env: "FakeDoomEnv") -> None:
        self.env = env

    def get_state(self):
        return _State([_Sector(ROOM_SEC, 0.0, 128.0), _Sector(DOOR_SEC, 0.0, self.env.gap),
                       _Sector(PAD_SEC, 0.0, 128.0)])


@dataclass
class _Cfg:
    turn_bins: tuple = TURN_BINS
    frame_skip: int = 4
    name: str = "fake"


class FakeDoomEnv:
    """Kinematic stand-in: forward 30 units/decision (run 50), back 20, strafe 20, turn
    -4 * bin degrees; walls are void cells; the door opens 2 decisions after a USE edge
    within 80 units of its centre; entering the pad ends the episode as a level exit."""

    def __init__(self, route: Route, start_angle: float = 0.0, max_steps: int = 300) -> None:
        self.route, self.start_angle, self.max_steps = route, start_angle, max_steps
        self.cfg = _Cfg()
        self.action_space = spaces.MultiDiscrete([3, 3, len(TURN_BINS), 2, 2, 2])
        self.observation_space = spaces.Dict({
            "rays": spaces.Box(0.0, 1.0, (2, 4), np.float32),
            "entity_mask": spaces.Box(0, 1, (3,), np.bool_),
            "prev_action": spaces.Box(0, 6, (6,), np.int64),
        })

    def _obs(self) -> dict:
        return {"rays": np.full((2, 4), self.t / 100.0, np.float32),
                "entity_mask": np.zeros(3, bool), "prev_action": self.prev.copy()}

    def _info(self) -> dict:
        return {"pos_x": self.x, "pos_y": self.y, "angle": self.ang, "map": self.map,
                "keys_held": {"blue": False, "yellow": False, "red": False}}

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.map = (options or {}).get("map", "DD0001")
        self.x, self.y = self.route.start
        self.ang = self.start_angle
        self.prev = np.zeros(6, np.int64)
        self.t = 0
        self.gap, self.gap_dir = 0.0, 0
        self.presses, self.reversals = 0, 0
        self.game = _Game(self)
        return self._obs(), self._info()

    def _free(self, x: float, y: float) -> bool:
        g = self.route.grid
        cc = g.cell_of(x, y)
        if not g.inside(*cc) or g.sector[cc] == VOID:
            return False
        if g.door_of[cc] != VOID:
            return self.gap >= GAP_PASS
        return True

    def step(self, action):
        a = np.asarray(action, np.int64)
        move, strafe, turn, _, use, speed = (int(v) for v in a)
        use_ang = self.ang - TURN_BINS[turn]          # the engine turns (one tic) before USE
        self.ang = (self.ang - 4 * TURN_BINS[turn]) % 360.0
        r = math.radians(self.ang)
        fwd = {0: 0.0, 1: 50.0 if speed else 30.0, 2: -20.0}[move]
        side = {0: 0.0, 1: 20.0, 2: -20.0}[strafe]          # left = +90 degrees
        nx = self.x + fwd * math.cos(r) + side * math.cos(r + math.pi / 2)
        ny = self.y + fwd * math.sin(r) + side * math.sin(r + math.pi / 2)
        if use == 1 and self.prev[4] == 0:
            L = self.route.use_lines
            hit = facing_hit(L, self.x, self.y, use_ang, USE_RANGE_ENGINE)
            if hit and int(L.kind[hit[0]]) == KIND_DOOR and on_front(L, hit[0], self.x, self.y):
                self.presses += 1
                if self.gap <= 0.0:
                    self.gap_dir = 1
                else:                                    # vanilla: a press reverses it
                    self.reversals += 1
                    self.gap_dir = -1 if self.gap_dir >= 0 else 1
        self.gap = min(max(self.gap + GAP_STEP * self.gap_dir, 0.0), GAP_OPEN)
        if self.gap in (0.0, GAP_OPEN):
            self.gap_dir = 0
        if self._free(nx, ny):
            self.x, self.y = nx, ny
        self.prev = a.copy()
        self.t += 1
        g = self.route.grid
        on_pad = g.sector[g.cell_of(self.x, self.y)] == PAD_SEC
        trunc = self.t >= self.max_steps
        info = self._info()
        if on_pad:
            info["level_exit"] = True
        return self._obs(), (1.0 if on_pad else 0.0), bool(on_pad), bool(trunc and not on_pad), info
