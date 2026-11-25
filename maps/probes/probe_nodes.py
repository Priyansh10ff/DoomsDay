"""Probe: does ViZDoom 1.3.3 load a PWAD with only THINGS/LINEDEFS/SIDEDEFS/VERTEXES/
SECTORS (SEGS, SSECTORS, NODES, REJECT, BLOCKMAP empty, or absent)?

Map: two 256x256 rooms joined by a 64-wide door (special 1, DR) and a walk-over exit
line (special 52) at the far end of room 2. One engine instance at a time.
Run: python -m maps.probes.probe_nodes  (writes into the scratch dir given as argv[1])
"""
from __future__ import annotations

import os
import sys

import vizdoom as vzd

from maps.wad import (ML_BLOCKING, ML_TWOSIDED, NO_SIDE, Linedef, Sector, Sidedef, Thing,
                      pack_map, write_wad)
from env.config import iwad_path


def two_rooms(absent_lumps: bool):
    # sector 0: room A x 0..256; sector 1: door x 256..272 (closed); sector 2: room B 272..528
    # sector 3: exit strip inside room B x 480..496 (walk-over lines special 52)
    V = [(0, 0), (256, 0), (256, 96), (256, 160), (256, 256), (0, 256),
         (272, 0), (272, 96), (272, 160), (272, 256), (528, 0), (528, 256)]
    secs = [Sector(0, 128, "FLAT1", "CEIL3_5", 192), Sector(0, 0, "FLAT1", "CEIL3_5", 160),
            Sector(0, 128, "FLOOR0_1", "CEIL3_5", 160)]
    sides, lines = [], []

    def side(sec, up="-", lo="-", mid="-"):
        sides.append(Sidedef(0, 0, up, lo, mid, sec))
        return len(sides) - 1

    def one(a, b, sec, tex="STARTAN2"):
        lines.append(Linedef(a, b, ML_BLOCKING, 0, 0, side(sec, mid=tex)))

    def two(a, b, front, back, special=0, up="STARTAN2"):
        lines.append(Linedef(a, b, ML_TWOSIDED, special, 0, side(front, up=up, lo="STARTAN2"),
                             side(back, up=up, lo="STARTAN2")))
    # room A (front faces inside: walk clockwise in Doom's y-up coords)
    one(0, 5, 0); one(5, 4, 0); one(4, 3, 0); one(2, 1, 0); one(1, 0, 0)
    two(3, 2, 0, 1, special=1, up="DOOR3")           # door face seen from A
    # door jambs (one-sided, belong to door sector)
    one(7, 2, 1, "DOORTRAK"); one(3, 8, 1, "DOORTRAK")
    two(7, 8, 2, 1, special=1, up="DOOR3")           # door face seen from B
    # room B
    one(6, 7, 2); one(8, 9, 2); one(9, 11, 2); one(11, 10, 2); one(10, 6, 2)
    # exit: a W1 exit line across room B at x=480 (two-sided, both sides room B)
    V += [(480, 64), (480, 192)]
    lines.append(Linedef(13, 12, ML_TWOSIDED, 52, 0, side(2), side(2)))
    # The exit line splits room B; that is legal (both sides same sector).
    things = [Thing(128, 128, 0, 1), Thing(400, 128, 180, 5)]  # player 1, blue keycard
    lumps = pack_map("MAP01", V, lines, sides, secs, things)
    if absent_lumps:
        lumps = [l for l in lumps if l[0] in ("MAP01", "THINGS", "LINEDEFS", "SIDEDEFS",
                                              "VERTEXES", "SECTORS")]
    return lumps


def run(wad: str) -> dict:
    g = vzd.DoomGame()
    g.set_doom_game_path(iwad_path("freedoom2"))
    g.set_doom_scenario_path(wad)
    g.set_doom_map("MAP01")
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_sectors_info_enabled(True)
    g.set_objects_info_enabled(True)
    g.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.USE,
                             vzd.Button.TURN_LEFT_RIGHT_DELTA])
    g.set_available_game_variables([vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y])
    g.set_map_exit_reward(1.0)
    g.set_episode_timeout(2000)
    out = {}
    try:
        g.init()
        g.new_episode()
        st = g.get_state()
        out["spawn"] = (g.get_game_variable(vzd.GameVariable.POSITION_X),
                        g.get_game_variable(vzd.GameVariable.POSITION_Y))
        out["n_sectors"] = len(st.sectors)
        out["objects"] = sorted(o.name for o in st.objects)
        out["door_closed_at_start"] = st.sectors[1].ceiling_height == st.sectors[1].floor_height
        # Walk east; at decision 11 (player pressed against the door face) press USE ONCE.
        # USE held from spawn never opens the door (edge-triggered: the press happens out
        # of range), and pressing every other decision makes the door oscillate.
        for i in range(140):
            g.make_action([1, 1 if i == 11 else 0, 0], 4)
            if g.is_episode_finished():
                break
            if i == 18:
                out["door_ceiling_after_use"] = g.get_state().sectors[1].ceiling_height
        out["finished"] = g.is_episode_finished()
        out["exit_reward"] = g.get_total_reward()
        out["dead"] = g.is_player_dead()
        out["timeout"] = g.is_episode_timeout_reached() if out["finished"] else None
        out["tics"] = g.get_episode_time()
    finally:
        g.close()
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    os.makedirs(d, exist_ok=True)
    for absent in (False, True):
        p = os.path.join(d, f"probe_two_rooms_{'absent' if absent else 'empty'}.wad")
        write_wad(p, two_rooms(absent))
        try:
            print(os.path.basename(p), run(p))
        except Exception as e:  # report, do not hide
            print(os.path.basename(p), "FAILED:", type(e).__name__, e)
