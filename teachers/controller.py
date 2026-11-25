"""Scripted waypoint-following controller (TRAINING-TIME ONLY; see teachers/__init__.py).

Each decision it reads the absolute pose from the env `info` dict (privileged) and the
executed previous action from `obs["prev_action"]`, and returns one action in the env's
six-head space (move 3, strafe 3, turn 7 bins, attack 2, use 2, speed 2), so every
label is an action the student can take.

Behaviour per anchor (teachers/navigator.py):
  * key / exit_pad: pursue the anchor's distance field (pure pursuit on the farthest
    descent cell in line of sight), run when aligned and far.
  * door: driven by the door's LIVE gap (info["door_gaps"], read from the engine's
    sector heights by collect.py): closed -> APPROACH the stand-off point (inside USE
    range) -> ALIGN (turn to face the door squarely, stand still); open or moving ->
    PUSH forward to the pass point. A re-closed door is approached and faced again.
  * exit_switch: approach the stand-off point, align, stand still.
  * stuck (commanded forward but barely moved over `stuck_window` decisions): back off
    and strafe for a few decisions, then resume.
USE (change C3): the USE head of EVERY label, whatever produced the rest of it, is set by
teachers.use_label.wants_press from the current state only: 1 exactly when the view ray
first hits a closed (gap 0) openable door face or the exit switch within USE range and
the previous EXECUTED action had USE up (one fresh press edge, then a release). Never a
press while that door is moving or open (a press would reverse it, probe R2.2).
Attack is always 0 (no combat teacher yet).

Turning: TURN_LEFT_RIGHT_DELTA is degrees per tic, positive turns right (ANGLE, which
is counter-clockwise, decreases); one decision holds it for `frame_skip` tics, so bin
value b changes the heading by -frame_skip * b degrees.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from teachers.navigator import VOID, Anchor, Route, descend, segment_clear
from teachers.use_label import (CLOSED_GAP, USE_RANGE_LABEL, lines_from_grid, press_target,
                                wants_press)

MOVE_NONE, MOVE_FWD, MOVE_BACK = 0, 1, 2
STRAFE_NONE, STRAFE_LEFT, STRAFE_RIGHT = 0, 1, 2
H_MOVE, H_STRAFE, H_TURN, H_ATTACK, H_USE, H_SPEED = range(6)


@dataclass(frozen=True)
class ControllerConfig:
    frame_skip: int = 4
    move_align_deg: float = 35.0       # move forward only when heading error is below this
    run_align_deg: float = 15.0        # run (speed head) only when this well aligned ...
    run_min_dist: float = 160.0        # ... and the aim point is at least this far
    face_tol_deg: float = 6.0          # door / switch alignment tolerance
    lookahead_cells: int = 14          # pure-pursuit horizon along the field descent
    clear_radius: float = 17.0         # player radius (16) + margin for line-of-sight tests
    key_radius: float = 18.0           # "at the key" distance (pickup also ends it)
    standoff_tol: float = 22.0         # "at the stand-off point" distance
    door_crossed_margin: float = 20.0  # past the far door face by this much = crossed
    use_range: float = USE_RANGE_LABEL  # view-ray distance within which a press is labelled
    stuck_window: int = 8              # decisions of forward command to judge progress
    stuck_min_disp: float = 14.0       # less displacement than this over the window = stuck
    unstick_back: int = 3              # decisions backing off
    unstick_strafe: int = 3            # decisions strafing (with forward) after backing off
    align_timeout: int = 30            # decisions trying to face a door before re-approaching
    drift_factor: float = 1.6          # leave align/press when this many tolerances away
    # DAgger labelling (the student drives, the teacher only labels). The executed action
    # differs from the label most of the time, so: (1) an override is not treated as a
    # noise burst (stuck history and an unstick in progress are kept); (2) unstick is
    # entered only when the teacher's own steering label would push forward into the
    # blockage (otherwise the steering label, e.g. "turn toward the route", is the
    # correction); (3) the unstick strafe side is the side with more free grid space
    # (a function of the state, not a coin flip).
    student_driven: bool = False
    side_probe: tuple[float, ...] = (24.0, 48.0, 72.0)   # lateral probe distances for (3)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def turn_bin(err_deg: float, turn_bins: tuple[float, ...], frame_skip: int) -> int:
    """Index of the turn bin whose heading change (-frame_skip * value) best cancels
    `err_deg` (positive = target is counter-clockwise / to the left)."""
    best, best_e = 0, math.inf
    for i, v in enumerate(turn_bins):
        e = abs(err_deg - (-frame_skip * v))
        if e < best_e - 1e-9:
            best, best_e = i, e
    return best


class TeacherController:
    """One episode's controller. Call `reset(route)` then `act(obs, info)` each decision."""

    def __init__(self, turn_bins: tuple[float, ...], cfg: ControllerConfig = ControllerConfig(),
                 seed: int = 0) -> None:
        self.turn_bins = tuple(turn_bins)
        self.zero_turn = turn_bin(0.0, self.turn_bins, cfg.frame_skip)
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.route: Route | None = None

    # ------------------------------------------------------------ episode
    def reset(self, route: Route) -> None:
        self.route = route
        self.k = 0                     # current anchor index
        self.phase = "approach"
        self.phase_steps = 0
        self.last_use_exec = 0
        self.use_lines = route.use_lines if route.use_lines is not None else lines_from_grid(route.grid)
        self.press_target = None       # what the last USE=1 label aimed at, for logging
        self.pos_hist: list[tuple[float, float, bool]] = []
        self.unstick_left = 0
        self.unstick_side = STRAFE_LEFT
        self.last_label: np.ndarray | None = None
        self.key_close_steps = 0
        self.events: list[tuple[int, str]] = []
        self.t = 0
        self.label_source = "nav"      # "nav" or "unstick": what produced the last label

    @property
    def anchor(self) -> Anchor:
        return self.route.anchors[min(self.k, len(self.route.anchors) - 1)]

    def _action(self, move=MOVE_NONE, strafe=STRAFE_NONE, turn=None, use=0, speed=0) -> np.ndarray:
        return np.array([move, strafe, self.zero_turn if turn is None else turn, 0, use, speed],
                        dtype=np.int64)

    def _set_phase(self, p: str) -> None:
        if p != self.phase:
            self.events.append((self.t, f"a{self.k}:{p}"))
        self.phase, self.phase_steps = p, 0

    def _advance(self) -> None:
        self.events.append((self.t, f"a{self.k}:done"))
        self.k += 1
        self.phase, self.phase_steps = "approach", 0
        self.pos_hist.clear()

    # ------------------------------------------------------------ main
    def act(self, obs: dict, info: dict) -> np.ndarray:
        c = self.cfg
        x, y, ang = float(info["pos_x"]), float(info["pos_y"]), float(info["angle"])
        executed = np.asarray(obs["prev_action"], dtype=np.int64)
        use_exec = int(executed[H_USE])
        self.last_use_exec = use_exec
        overridden = self.last_label is not None and not np.array_equal(executed, self.last_label)
        if overridden and not c.student_driven:
            self.pos_hist.clear()      # a noise burst moved us: do not call it stuck
            self.unstick_left = 0
        self.pos_hist.append((x, y, int(executed[H_MOVE]) == MOVE_FWD))
        del self.pos_hist[:-c.stuck_window]
        self.t += 1
        self.phase_steps += 1
        self.label_source = "nav"
        a = self._decide(x, y, ang, info)
        a = self._use_overlay(a, x, y, ang, info, use_exec)
        self.last_label = a
        return a

    def _use_overlay(self, a: np.ndarray, x: float, y: float, ang: float, info: dict,
                     use_exec: int) -> np.ndarray:
        """Set the USE head from the current state only (teachers.use_label). The engine
        turns before it runs the use trace within a tic, so if this label's turn would
        take the view ray off the target on the first tic, the press is labelled with no
        turn instead."""
        gaps, keys = info["door_gaps"], info.get("keys_held")
        a = a.copy()
        press = wants_press(self.use_lines, x, y, ang, gaps, keys, use_exec, self.cfg.use_range)
        a[H_USE] = 1 if press else 0
        self.press_target = None
        if press:
            self.press_target = press_target(self.use_lines, x, y, ang, gaps, keys, self.cfg.use_range)
            if a[H_TURN] != self.zero_turn:
                ang1 = ang - self.turn_bins[int(a[H_TURN])]     # heading after the first tic
                if press_target(self.use_lines, x, y, ang1, gaps, keys, self.cfg.use_range) is None:
                    a[H_TURN] = self.zero_turn
            self.events.append((self.t, f"a{self.k}:press:{self.press_target[0]}{self.press_target[1]}"))
        return a

    def _decide(self, x: float, y: float, ang: float, info: dict) -> np.ndarray:
        c = self.cfg
        if self.k >= len(self.route.anchors):
            return self._action()
        if self.unstick_left > 0:
            return self._unstick_step(x, y, ang)
        an = self.anchor
        if an.kind == "key":
            held = bool((info.get("keys_held") or {}).get(an.key_color, False))
            d = math.hypot(an.x - x, an.y - y)
            self.key_close_steps = self.key_close_steps + 1 if d < c.key_radius else 0
            if held or self.key_close_steps >= 3:
                self._advance()
                return self._decide(x, y, ang, info)
            return self._pursue(x, y, ang, an, final_xy=(an.x, an.y), precise=True)
        if an.kind == "exit_pad":
            return self._pursue(x, y, ang, an, final_xy=(an.x, an.y), precise=False)
        if an.kind in ("door", "exit_switch"):
            return self._use_target(x, y, ang, an, info)
        raise ValueError(an.kind)

    # ------------------------------------------------------------ behaviours
    def _steer(self, x: float, y: float, ang: float, tx: float, ty: float, *, allow_run: bool,
               align_deg: float | None = None) -> np.ndarray:
        c = self.cfg
        bearing = math.degrees(math.atan2(ty - y, tx - x))
        err = wrap_deg(bearing - ang)
        turn = turn_bin(err, self.turn_bins, c.frame_skip)
        dist = math.hypot(tx - x, ty - y)
        lim = c.move_align_deg if align_deg is None else align_deg
        move = MOVE_FWD if abs(err) < lim else MOVE_NONE
        speed = 1 if (allow_run and move == MOVE_FWD and abs(err) < c.run_align_deg
                      and dist > c.run_min_dist) else 0
        return self._action(move=move, turn=turn, speed=speed)

    def _aim_point(self, x: float, y: float, an: Anchor, final_xy: tuple[float, float]):
        """Farthest point along the anchor's field descent that is in clear line of sight."""
        g = self.route.grid
        cc = g.cell_of(x, y)
        if not g.walkable(*cc) or not np.isfinite(an.dist[cc]):
            return None
        if segment_clear(g, x, y, final_xy[0], final_xy[1], self.cfg.clear_radius):
            return final_xy
        chain = descend(g, an.dist, cc[0], cc[1], self.cfg.lookahead_cells)
        for cell in reversed(chain[1:]):
            px, py = g.center(*cell)
            if segment_clear(g, x, y, px, py, self.cfg.clear_radius):
                return px, py
        return g.center(*chain[1]) if len(chain) > 1 else final_xy

    def _pursue(self, x: float, y: float, ang: float, an: Anchor, final_xy, precise: bool) -> np.ndarray:
        aim = self._aim_point(x, y, an, final_xy)
        if aim is None:
            return self._recover(x, y, ang)
        if not self.cfg.student_driven and self._stuck():
            return self._start_unstick(x, y, ang)
        d_final = math.hypot(final_xy[0] - x, final_xy[1] - y)
        a = self._steer(x, y, ang, aim[0], aim[1], allow_run=not precise or d_final > self.cfg.run_min_dist)
        if self.cfg.student_driven and a[H_MOVE] == MOVE_FWD and self._stuck():
            return self._start_unstick(x, y, ang)
        return a

    def _use_target(self, x: float, y: float, ang: float, an: Anchor, info: dict) -> np.ndarray:
        c = self.cfg
        is_door = an.kind == "door"
        if is_door and self._crossed(x, y, an):
            self._advance()
            return self._decide(x, y, ang, info)
        d_stand = math.hypot(an.x - x, an.y - y)
        near = d_stand < c.standoff_tol * c.drift_factor
        if is_door:
            gap = float(info["door_gaps"][an.door_sector])
            if gap > CLOSED_GAP:                 # open or moving: go through
                if self.phase == "push" or near or self._in_door_or_beyond(x, y, an):
                    self._set_phase("push")
                    return self._push(x, y, ang, an)
                self._set_phase("approach")
                return self._pursue(x, y, ang, an, final_xy=(an.x, an.y), precise=True)
            if self._in_door_or_beyond(x, y, an):
                self._set_phase("push")          # in the doorway as it shuts: keep going
                return self._push(x, y, ang, an)
            if self.phase == "push":
                self._set_phase("approach")      # the door closed again before we crossed
        if self.phase == "align" and not near:
            self._set_phase("approach")          # slid away (momentum, a push, noise)
        if self.phase != "align":
            if d_stand >= c.standoff_tol:
                if self.phase != "approach":
                    self._set_phase("approach")
                return self._pursue(x, y, ang, an, final_xy=(an.x, an.y), precise=True)
            self._set_phase("align")
        # align: face the door / switch squarely and stand inside USE range; the press
        # itself comes from the state-based USE overlay in act()
        err = wrap_deg(an.face - ang)
        if self.phase_steps > c.align_timeout:
            self._set_phase("approach")
            return self._action()
        if abs(err) > c.face_tol_deg:
            return self._action(turn=turn_bin(err, self.turn_bins, c.frame_skip))
        return self._action()

    def _push(self, x: float, y: float, ang: float, an: Anchor) -> np.ndarray:
        return self._steer(x, y, ang, an.pass_x, an.pass_y, allow_run=False, align_deg=25.0)

    def _crossed(self, x: float, y: float, an: Anchor) -> bool:
        (cx, cy), (dx, dy) = an.door_center, an.door_dir
        half = self.route.grid.cell / 2.0
        return (x - cx) * dx + (y - cy) * dy > half + self.cfg.door_crossed_margin

    def _in_door_or_beyond(self, x: float, y: float, an: Anchor) -> bool:
        (cx, cy), (dx, dy) = an.door_center, an.door_dir
        half = self.route.grid.cell / 2.0
        along = (x - cx) * dx + (y - cy) * dy
        across = abs(-(x - cx) * dy + (y - cy) * dx)
        return along > -half and across < 2 * self.route.grid.cell

    # ------------------------------------------------------------ recovery
    def _stuck(self) -> bool:
        c = self.cfg
        h = self.pos_hist
        if len(h) < c.stuck_window or not all(f for _, _, f in h):
            return False
        return math.hypot(h[-1][0] - h[0][0], h[-1][1] - h[0][1]) < c.stuck_min_disp

    def _start_unstick(self, x: float, y: float, ang: float) -> np.ndarray:
        self.events.append((self.t, f"a{self.k}:unstick"))
        self.unstick_left = self.cfg.unstick_back + self.cfg.unstick_strafe
        self.unstick_side = (self._freer_side(x, y, ang) if self.cfg.student_driven
                             else self.rng.choice((STRAFE_LEFT, STRAFE_RIGHT)))
        self.pos_hist.clear()
        return self._unstick_step(x, y, ang)

    def _freer_side(self, x: float, y: float, ang: float) -> int:
        """STRAFE_LEFT or STRAFE_RIGHT: the side with more walkable probe points (ties: left).
        Left of heading `ang` (degrees CCW) is ang + 90."""
        g = self.route.grid
        score = {}
        for side, off in ((STRAFE_LEFT, 90.0), (STRAFE_RIGHT, -90.0)):
            r = math.radians(ang + off)
            n = 0
            for d in self.cfg.side_probe:
                cc = g.cell_of(x + d * math.cos(r), y + d * math.sin(r))
                if not (g.inside(*cc) and g.walkable(*cc)):
                    break
                n += 1
            score[side] = n
        return STRAFE_RIGHT if score[STRAFE_RIGHT] > score[STRAFE_LEFT] else STRAFE_LEFT

    def _unstick_step(self, x: float, y: float, ang: float) -> np.ndarray:
        self.label_source = "unstick"
        self.unstick_left -= 1
        if self.unstick_left >= self.cfg.unstick_strafe:
            return self._action(move=MOVE_BACK, strafe=self.unstick_side)
        return self._action(move=MOVE_FWD, strafe=self.unstick_side)

    def _recover(self, x: float, y: float, ang: float) -> np.ndarray:
        """Off the current anchor's field (e.g. pushed back through a door, or standing in
        a doorway). Re-target the nearest anchor (prefer forward) reachable from here."""
        g = self.route.grid
        cc = g.cell_of(x, y)
        if g.inside(*cc) and g.door_of[cc] != VOID:
            ds = int(g.door_of[cc])
            for i, a in enumerate(self.route.anchors):
                if a.kind == "door" and a.door_sector == ds and i >= self.k - 1:
                    self.k = i
                    self._set_phase("push")
                    return self._push(x, y, ang, a)
            return self._action(move=MOVE_BACK)
        if g.walkable(*cc):
            anchors = self.route.anchors
            # never skip past a key not yet collected
            limit = next((i for i in range(self.k, len(anchors)) if anchors[i].kind == "key"),
                         len(anchors) - 1)
            order = sorted(range(limit + 1), key=lambda i: (i < self.k, abs(i - self.k)))
            for i in order:
                if np.isfinite(self.route.anchors[i].dist[cc]):
                    self.events.append((self.t, f"a{self.k}:retarget->a{i}"))
                    self.k = i
                    self.phase, self.phase_steps = "approach", 0
                    an = self.route.anchors[i]
                    return self._pursue(x, y, ang, an, (an.x, an.y), precise=True)
        return self._action(move=MOVE_BACK, strafe=self.rng.choice((STRAFE_LEFT, STRAFE_RIGHT)))
