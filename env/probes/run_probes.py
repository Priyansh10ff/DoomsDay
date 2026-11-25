"""Engine probes for ADR 0001 / 0002 (run one section at a time, one DoomGame at a time).

Usage: python -m env.probes.run_probes <section>
Sections: scan, nocheat, depth, depth2, depth_calib, keys, hud_crops, turn, use, use_test,
notif, hud_shift, categories, tic1_screen, horizon, projectiles, final_state, wadlines,
locked, locked_screen, held_use_door, switch, own_pixels, zero_depth_breakdown,
zero_depth_where, zero_label0_shape, overlay_sources, label_zero_depth, damage_after_end,
mwh_lumps, mwh_goal, same_seed, sky_depth, pwad_map_fallback, occlusion, occlusion_obs, ini_path

This is probe tooling, not the policy's observation path. It may read absolute
positions and use cheat console commands (sv_cheats, summon, give, warp) because
it only measures engine behaviour. Nothing here is imported by env.doom_env.
"""
from __future__ import annotations

import json
import os
import sys
import time

import tempfile

import numpy as np
import vizdoom as vzd

OUT_DIR = tempfile.gettempdir()  # probe artifacts never go into the repo

PKG = os.path.dirname(vzd.__file__)
FD1 = os.path.join(PKG, "freedoom1.wad")
FD2 = os.path.join(PKG, "freedoom2.wad")
GV = vzd.GameVariable
B = vzd.Button

PROBE_VARS = [
    GV.HEALTH, GV.ARMOR, GV.SELECTED_WEAPON, GV.SELECTED_WEAPON_AMMO, GV.DAMAGE_TAKEN,
    GV.POSITION_X, GV.POSITION_Y, GV.POSITION_Z, GV.ANGLE,
    GV.VELOCITY_X, GV.VELOCITY_Y, GV.VELOCITY_Z, GV.KILLCOUNT, GV.ITEMCOUNT,
    GV.DEAD, GV.WEAPON2, GV.AMMO2,
]
BUTTONS = [B.MOVE_FORWARD, B.MOVE_BACKWARD, B.MOVE_LEFT, B.MOVE_RIGHT,
           B.TURN_LEFT_RIGHT_DELTA, B.ATTACK, B.USE, B.SPEED]


def make_game(wad=FD2, doom_map="map01", args=(), objects=True, sectors=True, hud=True,
              notifications=True, res=vzd.ScreenResolution.RES_160X120, messages=True):
    g = vzd.DoomGame()
    g.set_doom_game_path(wad)
    g.set_doom_map(doom_map)
    g.set_screen_resolution(res)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_render_hud(hud)
    g.set_render_messages(messages)
    g.set_depth_buffer_enabled(True)
    g.set_labels_buffer_enabled(True)
    g.set_objects_info_enabled(objects)
    g.set_sectors_info_enabled(sectors)
    g.set_notifications_buffer_enabled(notifications)
    if notifications:
        g.set_notifications_buffer_size(4)  # covers a 4-tic make_action (default is 1 tic)
    g.set_available_buttons(BUTTONS)
    g.set_available_game_variables(PROBE_VARS)
    g.set_mode(vzd.Mode.PLAYER)
    g.set_episode_start_time(1)
    g.set_seed(123)
    for a in args:
        g.add_game_args(a)
    g.init()
    return g


def gv(state):
    return {str(v).split(".")[-1]: float(x) for v, x in zip(PROBE_VARS, state.game_variables)}


def act(fwd=0, back=0, left=0, right=0, turn=0.0, attack=0, use=0, speed=0):
    return [fwd, back, left, right, turn, attack, use, speed]


def scripted_actions(n, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(act(fwd=int(rng.integers(2)), left=int(rng.integers(2)),
                       turn=float(rng.choice([-10, -4, 0, 4, 10])), attack=int(rng.integers(2))))
    return out


# ---------------------------------------------------------------------------
def scan():
    """List key objects on Freedoom maps (objects info, whole-level; probe only)."""
    res = {}
    for wad, maps in ((FD1, ["E1M1", "E1M2", "E1M3"]), (FD2, ["MAP01", "MAP02", "MAP03"])):
        g = make_game(wad, maps[0], hud=False, sectors=False, notifications=False)
        try:
            for m in maps:
                g.set_doom_map(m)
                g.new_episode()
                s = g.get_state()
                keys = [(o.name, round(o.position_x), round(o.position_y)) for o in s.objects
                        if "card" in o.name.lower() or "skull" in o.name.lower()]
                p = gv(s)
                res[f"{os.path.basename(wad)}:{m}"] = {
                    "n_objects": len(s.objects), "keys": keys,
                    "player": (round(p["POSITION_X"]), round(p["POSITION_Y"]), p["ANGLE"])}
        finally:
            g.close()
    print(json.dumps(res, indent=1))


def summarize_buffers(s):
    d = s.depth_buffer
    lb = s.labels_buffer
    out = {"depth": None if d is None else (list(d.shape), str(d.dtype), int(d.min()), int(d.max()),
                                            round(float(d.mean()), 2)),
           "labels_buf": None if lb is None else (list(lb.shape), int(len(np.unique(lb)))),
           "labels_list": None if s.labels is None else len(s.labels)}
    if s.labels:
        out["label_names"] = sorted({l.object_name for l in s.labels})
    return {**out,
        "objects": None if s.objects is None else len(s.objects),
        "sectors": None if s.sectors is None else len(s.sectors),
        "notifications": repr(s.notifications_buffer)[:200],
    }


def nocheat():
    """Same seed, same scripted actions, with and without +viz_nocheat 1."""
    acts = scripted_actions(60)
    runs = {}
    for name, args in (("plain", ()), ("nocheat", ("+viz_nocheat 1",))):
        g = make_game(FD2, "map01", args=args)
        try:
            g.new_episode()
            snaps = []
            for i, a in enumerate(acts):
                if i in (0, 30, 59):
                    s = g.get_state()
                    cp = lambda b: None if b is None else b.copy()
                    snaps.append({"tic": s.tic, "buf": summarize_buffers(s), "gv": gv(s),
                                  "depth": cp(s.depth_buffer), "labels": cp(s.labels_buffer),
                                  "screen": cp(s.screen_buffer)})
                g.make_action(a, 4)
            runs[name] = snaps
        finally:
            g.close()
    for i in range(3):
        a, b = runs["plain"][i], runs["nocheat"][i]
        print(f"--- snapshot {i} tic {a['tic']} vs {b['tic']}")
        print(" plain  :", a["buf"])
        print(" nocheat:", b["buf"])
        print(" gv plain  :", a["gv"])
        print(" gv nocheat:", b["gv"])
        if a["depth"] is not None and b["depth"] is not None:
            dd = np.abs(a["depth"].astype(int) - b["depth"].astype(int))
            print(" depth |diff| mean/max, frac!=0:", dd.mean(), dd.max(), (dd > 0).mean())
        if a["labels"] is not None and b["labels"] is not None:
            print(" labels frac!=0:", (a["labels"] != b["labels"]).mean())
        print(" screen |diff| mean:", np.abs(a["screen"].astype(int) - b["screen"].astype(int)).mean())


def depth():
    """Depth polarity/range, horizon row, status-bar rows, labels value of walls/floor."""
    g = make_game(FD2, "map01")
    try:
        g.new_episode()
        out = []
        for i in range(12):
            s = g.get_state()
            d = s.depth_buffer.astype(int)
            lb = s.labels_buffer
            # per-row stats
            rows = {r: (int(d[r].min()), int(d[r].max()), float(d[r].mean())) for r in
                    (0, 20, 40, 50, 55, 58, 60, 62, 64, 70, 80, 90, 95, 96, 100, 104, 110, 119)}
            out.append({"i": i, "rows": rows, "label_vals_rows_100_119": sorted(np.unique(lb[100:]).tolist())[:10],
                        "screen_row_110_mean": float(s.screen_buffer[110].mean())})
            g.make_action(act(turn=30.0), 1)
        for o in out[:4]:
            print(o)
        # find the row of min depth column-wise (horizon estimate): depth is largest at horizon if far = high
        s = g.get_state()
        d = s.depth_buffer.astype(int)
        print("row argmax of column-mean depth:", int(d.mean(axis=1).argmax()),
              "row means:", [round(float(x), 1) for x in d.mean(axis=1)])
        # walk into wall: depth should shrink/grow
        vals = []
        for i in range(40):
            g.make_action(act(fwd=1), 4)
            s = g.get_state()
            vals.append(int(s.depth_buffer[60, 80]))
        print("centre depth while walking forward:", vals)
    finally:
        g.close()


def depth2():
    """Status-bar extent, horizon row, depth units (back away from a wall)."""
    g = make_game(FD2, "map01", objects=False, sectors=False)
    try:
        g.new_episode()
        g.make_action(act(), 2)
        s = g.get_state()
        d = s.depth_buffer
        zero_rows = [r for r in range(d.shape[0]) if d[r].max() == 0]
        print("rows with depth all-zero:", zero_rows[:3], "...", zero_rows[-1:] if zero_rows else None,
              "count", len(zero_rows))
        # with HUD off
        # walk into wall then back off, track centre column depth at rows 40..60 and position
        for _ in range(15):
            g.make_action(act(fwd=1), 4)
        p0 = gv(g.get_state())
        recs = []
        for _ in range(30):
            g.make_action(act(back=1), 2)
            s = g.get_state()
            p = gv(s)
            dist = float(np.hypot(p["POSITION_X"] - p0["POSITION_X"], p["POSITION_Y"] - p0["POSITION_Y"]))
            col = s.depth_buffer[:, 80].astype(int)
            recs.append((round(dist, 1), [int(col[r]) for r in (30, 40, 45, 48, 50, 52, 55, 60)]))
        for r in recs[::3]:
            print("moved back", r[0], "units; depth col80 rows 30,40,45,48,50,52,55,60:", r[1])
        print("start angle", p0["ANGLE"], "end angle", gv(g.get_state())["ANGLE"])
    finally:
        g.close()
    g = make_game(FD2, "map01", objects=False, sectors=False, hud=False)
    try:
        g.new_episode()
        g.make_action(act(), 2)
        d = g.get_state().depth_buffer
        print("HUD off: rows all-zero:", [r for r in range(d.shape[0]) if d[r].max() == 0][:5])
    finally:
        g.close()


def _ray_hit(px, py, ang_deg, lines):
    """Distance along heading to the nearest blocking line (probe-only ground truth)."""
    dx, dy = np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))
    best = np.inf
    for (x1, y1, x2, y2) in lines:
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((x1 - px) * ey - (y1 - py) * ex) / den
        u = ((x1 - px) * dy - (y1 - py) * dx) / den
        if t > 0 and 0 <= u <= 1:
            best = min(best, t)
    return best


def depth_calib():
    """Regress depth[horizon,80] against true wall distance; also turn dead time."""
    g = make_game(FD2, "map01", objects=False, sectors=True, hud=True)
    try:
        g.new_episode()
        angs = []
        for i in range(20):
            g.make_action(act(turn=1.0), 1)
            angs.append(round(gv(g.get_state())["ANGLE"], 2))
        print("ANGLE per tic with turn=+1 each tic from start:", angs, "tic", g.get_state().tic)
        s = g.get_state()
        lines = [(l.x1, l.y1, l.x2, l.y2) for sec in s.sectors for l in sec.lines if l.is_blocking]
        rng = np.random.default_rng(0)
        pairs = []
        for i in range(150):
            g.make_action(act(fwd=int(rng.integers(2)), turn=float(rng.choice([-10, 0, 10]))), 4)
            s = g.get_state()
            p = gv(s)
            if p["VELOCITY_X"] ** 2 + p["VELOCITY_Y"] ** 2 > 0.01:
                continue  # only use still frames to avoid lag
            true = _ray_hit(p["POSITION_X"], p["POSITION_Y"], p["ANGLE"], lines)
            for r in (48, 49, 50, 51):
                pairs.append((r, true, int(s.depth_buffer[r, 80])))
        pairs = np.array([(r, t, d) for r, t, d in pairs if np.isfinite(t)])
        for r in (48, 49, 50, 51):
            sel = pairs[pairs[:, 0] == r]
            if len(sel) > 3:
                ratio = sel[:, 1] / np.maximum(sel[:, 2], 1)
                print(f"row {r}: n={len(sel)} true/depth median={np.median(ratio):.2f} "
                      f"iqr=({np.percentile(ratio, 25):.2f},{np.percentile(ratio, 75):.2f})",
                      "samples(true,depth):", [(round(a), int(b)) for a, b in sel[:8, 1:]])
        print("max depth seen in run:", int(pairs[:, 2].max()))
    finally:
        g.close()


def keys():
    """Summon key items in front of the player; check labels, HUD, notifications."""
    g = make_game(FD2, "map01", args=("+sv_cheats 1",))
    try:
        g.new_episode()
        g.make_action(act(), 2)
        results = {}
        for cls in ("BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull"):
            g.send_game_command(f"summon {cls} 96")  # 96 units ahead
            g.make_action(act(), 4)
            s = g.get_state()
            labs = [(l.object_name, l.object_category, l.value, l.x, l.y, l.width, l.height)
                    for l in s.labels if l.object_name != "DoomPlayer"]
            results[cls] = labs
            print(cls, labs)
        np.save(os.path.join(OUT_DIR, "_hud_before.npy"), g.get_state().screen_buffer)
        # pick them up: walk forward
        notes = []
        for i in range(12):
            g.make_action(act(fwd=1), 4)
            s = g.get_state()
            if s.notifications_buffer:
                notes.append((s.tic, s.notifications_buffer))
        print("notifications on pickup:", notes)
        s = g.get_state()
        np.save(os.path.join(OUT_DIR, "_hud_after.npy"), s.screen_buffer)
        print("labels after walk:", [(l.object_name, l.object_category) for l in s.labels])
        print("objects named *card/*skull still present:",
              [o.name for o in s.objects if "Card" in o.name or "Skull" in o.name])
        # monsters / other categories
        for cls in ("Zombieman", "Shotgun", "Medikit", "Clip", "GreenArmor", "DoomImpBall"):
            g.send_game_command(f"summon {cls} 128")
            g.make_action(act(), 2)
        s = g.get_state()
        print("misc labels:", sorted({(l.object_name, l.object_category) for l in s.labels}))
    finally:
        g.close()


def hud_crops():
    """Render the saved HUD screens to PNG via numpy (no PIL dependency required)."""
    import zlib, struct
    def png(path, arr):
        h, w, _ = arr.shape
        raw = b"".join(b"\x00" + arr[r].tobytes() for r in range(h))
        def chunk(t, d):
            c = struct.pack(">I", len(d)) + t + d
            return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    here = OUT_DIR
    for n in ("_hud_before", "_hud_after"):
        a = np.load(os.path.join(here, n + ".npy"))
        big = np.repeat(np.repeat(a, 4, 0), 4, 1)
        png(os.path.join(sys.argv[2] if len(sys.argv) > 2 else here, n + ".png"), big)
        small = a.reshape(60, 2, 80, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
        png(os.path.join(sys.argv[2] if len(sys.argv) > 2 else here, n + "_60x80.png"),
            np.repeat(np.repeat(small, 8, 0), 8, 1))


def turn():
    """Turn calibration: heading change per decision for each bin, with/without SPEED."""
    g = make_game(FD2, "map01", objects=False, sectors=False)
    try:
        g.new_episode()
        g.make_action(act(), 2)
        rows = []
        for speed in (0, 1):
            for t in (-10, -4, -1, 0, 1, 4, 10, 0.5, 2.5):
                a0 = gv(g.get_state())["ANGLE"]
                g.make_action(act(turn=float(t), speed=speed), 4)
                a1 = gv(g.get_state())["ANGLE"]
                dlt = (a1 - a0 + 180) % 360 - 180
                rows.append((speed, t, round(dlt, 3)))
        for r in rows:
            print("speed=%d turn=%s -> dANGLE over 4 tics=%s" % r)
    finally:
        g.close()


def use():
    """USE edge-trigger probe: find the nearest door (sector whose ceiling==floor),
    warp beside it, compare held USE vs first-tic USE, watch ceiling height."""
    g = make_game(FD2, "map01", args=("+sv_cheats 1",))
    try:
        g.new_episode()
        s = g.get_state()
        p = gv(s)
        doors = []
        for i, sec in enumerate(s.sectors):
            if abs(sec.ceiling_height - sec.floor_height) < 1:
                xs = [l.x1 for l in sec.lines] + [l.x2 for l in sec.lines]
                ys = [l.y1 for l in sec.lines] + [l.y2 for l in sec.lines]
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                doors.append((float(np.hypot(cx - p["POSITION_X"], cy - p["POSITION_Y"])), i, cx, cy,
                              min(xs), max(xs), min(ys), max(ys)))
        doors.sort()
        print("player", p["POSITION_X"], p["POSITION_Y"], p["ANGLE"])
        print("closed sectors (ceil==floor), nearest 5:", [tuple(round(v) for v in d) for d in doors[:5]])
        print("total sectors", len(s.sectors))
        print(json.dumps({"doors": [list(map(float, d)) for d in doors[:5]]}))
    finally:
        g.close()


def use_test(door_idx, x, y, angle, mode, n=20):
    """mode 'held': USE all 4 tics every decision; 'first': USE only on first tic."""
    g = make_game(FD2, "map01", args=("+sv_cheats 1",))
    try:
        g.new_episode()
        g.make_action(act(), 2)
        g.send_game_command(f"warp {x} {y}")
        g.make_action(act(), 2)
        # face the door: set heading by turning with delta (probe only)
        a0 = gv(g.get_state())["ANGLE"]
        dlt = (angle - a0 + 180) % 360 - 180
        g.make_action(act(turn=float(dlt)), 1)
        s = g.get_state()
        print("pos", gv(s)["POSITION_X"], gv(s)["POSITION_Y"], gv(s)["ANGLE"])
        trace = []
        for i in range(n):
            if mode == "held":
                g.make_action(act(use=1), 4)
            else:
                g.make_action(act(use=1), 1)
                g.make_action(act(use=0), 3)
            s = g.get_state()
            sec = s.sectors[door_idx]
            trace.append(round(sec.ceiling_height - sec.floor_height, 1))
        print(mode, "door gap per decision:", trace)
    finally:
        g.close()


def notif():
    """What the notifications buffer holds on the bundled notifications scenario."""
    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, "basic_notifications.cfg"))
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_notifications_buffer_size(4)
    g.init()
    try:
        g.new_episode()
        seen = []
        for i in range(60):
            g.make_action([0, 0, 1], 1)
            if g.is_episode_finished():
                break
            s = g.get_state()
            if s.notifications_buffer:
                seen.append((s.tic, s.notifications_buffer))
        print("notifications seen:", seen[:10], "type:", type(seen[0][1]) if seen else None)
    finally:
        g.close()


# =========================================================================
# Sections added for ADR 0006 (and the three probes run inline in round 1)
# =========================================================================
def hud_shift():
    """(Round-1 inline probe, reconstructed.) Vertical shift between HUD-on and
    HUD-off depth images at the same pose."""
    bufs = {}
    for hud in (True, False):
        g = make_game(FD2, "map01", objects=False, sectors=False, hud=hud, notifications=False)
        try:
            g.new_episode()
            g.make_action(act(), 12)
            for _ in range(6):
                g.make_action(act(fwd=1, turn=-6.0), 4)
            bufs[hud] = g.get_state().depth_buffer.astype(float)
        finally:
            g.close()
    on, off = bufs[True], bufs[False]
    best = []
    for sh in range(-15, 16):
        rows = np.arange(0, 100)
        src = np.clip(rows + sh, 0, 119)
        best.append((round(float(np.abs(on[:100] - off[src]).mean()), 4), sh))
    best.sort()
    print("best (mean |diff|, shift: HUD-on row r == HUD-off row r+shift):", best[:4])


def categories():
    """(Round-1 inline probe, reconstructed.) object_category vocabulary (objects info)."""
    from collections import Counter
    c = Counter()
    for wad, maps in ((FD1, ["E1M1", "E1M2", "E1M3", "E1M4"]),
                      (FD2, ["MAP01", "MAP02", "MAP03", "MAP04", "MAP05"])):
        g = make_game(wad, maps[0], hud=False, sectors=False, notifications=False)
        try:
            for m in maps:
                g.set_doom_map(m)
                g.new_episode()
                for o in g.get_state().objects:
                    c[(o.category, o.name)] += 1
        finally:
            g.close()
    cats = {}
    for (cat, n), _ in sorted(c.items()):
        cats.setdefault(cat, []).append(n)
    for cat, ns in cats.items():
        print(cat, ":", " ".join(ns))


def tic1_screen():
    """(Round-1 inline probe, reconstructed.) Save the tic-1 and tic-41 screens for a
    visual check that the level name is not drawn; render with hud_crops."""
    g = make_game(FD2, "map01", objects=False, sectors=False)
    try:
        g.new_episode()
        s = g.get_state()
        print("tic", s.tic, "notifications:", repr(s.notifications_buffer)[-60:])
        np.save(os.path.join(OUT_DIR, "_hud_before.npy"), s.screen_buffer)
        for _ in range(10):
            g.make_action(act(), 4)
        np.save(os.path.join(OUT_DIR, "_hud_after.npy"), g.get_state().screen_buffer)
    finally:
        g.close()


def horizon(dmin=4, offset=0.0):
    """Direct horizon measurement. For floor and ceiling pixels, perspective gives
    1/depth linear in row, zero at the horizon. Fit per column on HUD on and off,
    over several poses, using labels-buffer floor/ceiling pixels with depth dmin..60.
    `offset` is added to the integer depth (0.5 = assume rounding-down quantization)."""
    cols = range(0, 160, 4)
    if len(sys.argv) > 2:
        dmin, offset = int(sys.argv[2]), float(sys.argv[3])
    if len(sys.argv) > 4:  # centre-column half width
        hw = int(sys.argv[4])
        cols = range(80 - hw, 80 + hw)
    print(f"dmin={dmin} offset={offset} columns {cols.start}..{cols.stop - 1}")
    for hud in (True, False):
        g = make_game(FD2, "map01", objects=False, sectors=False, hud=hud, notifications=False)
        try:
            g.new_episode()
            g.make_action(act(), 10)
            ests = {"floor": [], "ceil": []}
            vals_seen = set()
            for pose in range(16):
                g.make_action(act(fwd=pose % 2, turn=22.5), 4)
                s = g.get_state()
                d = s.depth_buffer.astype(float)
                lb = s.labels_buffer
                vals_seen |= set(np.unique(lb).tolist())
                view = 100 if hud else 120
                for col in cols:
                    for part, rows in (("floor", range(view // 2 + 2, view)),
                                       ("ceil", range(0, view // 2 - 2))):
                        rr = [r for r in rows if lb[r, col] == 1 and dmin <= d[r, col] <= 60]
                        if len(rr) < 6:
                            continue
                        y = 1.0 / (d[rr, col] + offset)
                        x = np.array(rr, float) + 0.5
                        # require a single plane: residual small
                        k, b = np.polyfit(x, y, 1)
                        if abs(k) < 1e-6:
                            continue
                        res = np.abs(np.polyval([k, b], x) - y).max()
                        if res > 0.02:
                            continue
                        ests[part].append(-b / k)
            for part in ("floor", "ceil"):
                e = np.array(ests[part])
                if len(e):
                    print(f"HUD {'on ' if hud else 'off'} {part}: n={len(e)} horizon (pixel-edge "
                          f"coordinate) median={np.median(e):.2f} iqr=({np.percentile(e, 25):.2f},"
                          f"{np.percentile(e, 75):.2f})")
                else:
                    print(f"HUD {hud} {part}: no fits")
            print("labels values seen (0=wall?,1=floor/ceiling?):", sorted(vals_seen)[:6])
        finally:
            g.close()


def projectiles():
    """object_category of projectiles, puffs and effects; the BulletPuff of our own shots."""
    g = make_game(FD2, "map01", args=("+sv_cheats 1",), sectors=False)
    try:
        g.new_episode()
        g.make_action(act(), 10)
        seen = set()
        for _ in range(6):  # shoot the wall ahead
            g.make_action(act(attack=1), 4)
            s = g.get_state()
            seen |= {(l.object_name, l.object_category) for l in s.labels}
        print("labels while shooting a wall:", sorted(seen))
        cats = {}
        for cls in ("DoomImpBall", "CacodemonBall", "BaronBall", "Rocket", "PlasmaBall",
                    "ArachnotronPlasma", "FatShot", "RevenantTracer", "BFGBall",
                    "BulletPuff", "Blood", "TeleportFog", "ItemFog", "ExplosiveBarrel"):
            before = {o.id for o in g.get_state().objects}
            g.send_game_command(f"summon {cls} 200")
            g.make_action(act(), 1)
            s = g.get_state()
            new = [(o.name, o.category) for o in s.objects if o.id not in before]
            cats[cls] = new[:2]
        for k, v in cats.items():
            print(f"summon {k}: {v}")
    finally:
        g.close()


def final_state():
    """Is a state (and its notifications buffer) available after the episode ends?"""
    g = make_game(FD2, "map01", objects=False, sectors=False)
    g.set_episode_timeout  # noqa (timeout must be set before init; re-create below)
    g.close()
    g = vzd.DoomGame()
    g.set_doom_game_path(FD2)
    g.set_doom_map("map01")
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_notifications_buffer_enabled(True)
    g.set_available_buttons(BUTTONS)
    g.set_episode_timeout(20)
    g.init()
    try:
        g.new_episode()
        while not g.is_episode_finished():
            g.make_action(act(), 4)
        s = g.get_state()
        print("after finish: get_state() ->", None if s is None else type(s),
              "| is_new_episode", g.is_new_episode(), "| last reward", g.get_last_reward())
    finally:
        g.close()


# ---------------------------------------------------------------- WAD parsing (probe only)
def _wad_lumps(path):
    import struct
    with open(path, "rb") as f:
        data = f.read()
    ident, n, off = struct.unpack_from("<4sii", data, 0)
    lumps = []
    for i in range(n):
        lo, sz, name = struct.unpack_from("<ii8s", data, off + 16 * i)
        lumps.append((name.rstrip(b"\0").decode("ascii", "replace"), lo, sz))
    return data, lumps


def _map_lumps(path, mapname):
    import struct
    data, lumps = _wad_lumps(path)
    idx = [i for i, l in enumerate(lumps) if l[0] == mapname.upper()][0]
    out = {}
    for name, lo, sz in lumps[idx + 1: idx + 12]:
        out[name] = data[lo:lo + sz]
    verts = [struct.unpack_from("<hh", out["VERTEXES"], i) for i in range(0, len(out["VERTEXES"]), 4)]
    lines = [struct.unpack_from("<HHHHHHH", out["LINEDEFS"], i) for i in range(0, len(out["LINEDEFS"]), 14)]
    sides = [struct.unpack_from("<hh8s8s8sH", out["SIDEDEFS"], i) for i in range(0, len(out["SIDEDEFS"]), 30)]
    secs = [struct.unpack_from("<hh8s8shhh", out["SECTORS"], i) for i in range(0, len(out["SECTORS"]), 26)]
    return verts, lines, sides, secs


LOCKED = {26: "DR blue", 27: "DR yellow", 28: "DR red", 32: "D1 blue", 33: "D1 red", 34: "D1 yellow",
          99: "SR blue fast", 133: "S1 blue fast", 134: "SR red fast", 135: "S1 red fast",
          136: "SR yellow fast", 137: "S1 yellow fast"}
SWITCH = {7, 9, 11, 14, 15, 18, 20, 21, 23, 29, 41, 49, 50, 51, 55, 62, 63, 64, 65, 66, 67, 68, 69,
          70, 71, 101, 102, 103, 111, 112, 113, 114, 115, 116, 122, 123, 127, 131, 132, 133, 140,
          158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
          176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
          194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212}


def wadlines():
    """List locked-door and switch linedefs on freedoom2 MAP02/MAP03 (from the WAD file)."""
    for m in ("MAP02", "MAP03"):
        verts, lines, sides, secs = _map_lumps(FD2, m)
        print(m, "lines", len(lines), "sectors", len(secs))
        for i, (v1, v2, fl, sp, tag, fs, bs) in enumerate(lines):
            if sp in LOCKED or sp in (11, 103, 62, 70, 102):
                (x1, y1), (x2, y2) = verts[v1], verts[v2]
                print(f"  line {i} special {sp} {LOCKED.get(sp, 'switch/other')} tag {tag} "
                      f"({x1},{y1})-({x2},{y2}) two-sided={bs != 0xFFFF}")


def _stand_before_line(g, verts, line, dist=40.0, back=False):
    """Warp in front of a linedef's front side and face it (probe-only geometry)."""
    v1, v2 = line[0], line[1]
    (x1, y1), (x2, y2) = verts[v1], verts[v2]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy))
    nx, ny = dy / L, -dx / L  # front (right-hand) side normal
    if back:
        nx, ny = -nx, -ny
    px, py = mx + nx * dist, my + ny * dist
    g.send_game_command(f"warp {px:.0f} {py:.0f}")
    g.make_action(act(), 2)
    face = float(np.angle(complex(-nx, -ny), deg=True)) % 360  # direction toward the line
    a0 = gv(g.get_state())["ANGLE"]
    # positive TURN delta = clockwise = ANGLE decreases
    g.make_action(act(turn=-(((face - a0) + 180) % 360 - 180)), 1)
    p = gv(g.get_state())
    return (round(p["POSITION_X"]), round(p["POSITION_Y"]), round(p["ANGLE"], 1)), (px, py, face)


def _sector_heights(s):
    return np.array([(sec.floor_height, sec.ceiling_height) for sec in s.sectors], float)


def _usable_game(m):
    g = make_game(FD2, m, args=("+sv_cheats 1",), objects=False, sectors=True)
    g.new_episode()
    g.make_action(act(), 10)  # past the 7-tic turn dead time
    return g


def locked():
    """Locked door (freedoom2 MAP02 line 215, DR yellow) and locked switch (line 379,
    S1 blue): notification text without and with the key; held USE.
    Optional argv: <line> [back] to try another line or the back side."""
    m = "MAP02"
    verts, lines, sides, secs = _map_lumps(FD2, m)
    targets = [215, 379] if len(sys.argv) < 3 else [int(sys.argv[2])]
    use_back = len(sys.argv) > 3 and sys.argv[3] == "back"
    for i in targets:
        line = lines[i]
        if True:
            colour = LOCKED[line[3]].split()[1]
            g = _usable_game(m)
            try:
                # sector alignment check: WAD sector order vs state.sectors
                s = g.get_state()
                h = _sector_heights(s)
                wad_h = np.array([(a[0], a[1]) for a in secs], float)
                aligned = h.shape == wad_h.shape and np.allclose(h, wad_h)
                pose, tgt = _stand_before_line(g, verts, line, back=use_back)
                # door: the sector behind the line; locked switch: the line's tagged sector
                if line[6] != 0xFFFF and line[4] == 0:
                    back_sec = sides[line[6]][5]
                else:
                    tagged = [k for k, sc in enumerate(secs) if sc[6] == line[4]]
                    back_sec = tagged[0] if tagged else 0
                gap0 = h[back_sec, 1] - h[back_sec, 0]
                notes = []
                for k in range(6):
                    g.make_action(act(use=1 if k < 3 else 0), 4)
                    nb = g.get_state().notifications_buffer
                    if nb:
                        notes.append(nb)
                h1 = _sector_heights(g.get_state())
                print(f"{m} line {i} ({LOCKED[line[3]]}) stand {pose} target {tuple(round(t) for t in tgt)}"
                      f" sectors aligned={aligned} door sector {back_sec} gap {gap0} -> "
                      f"{h1[back_sec, 1] - h1[back_sec, 0]} (no key)")
                print("   notifications without key:", notes)
                g.send_game_command(f"give {colour}card")
                g.send_game_command(f"give {colour}skull")
                g.make_action(act(), 4)
                give_note = g.get_state().notifications_buffer
                notes = []
                for k in range(8):
                    g.make_action(act(use=1 if k < 3 else 0), 4)
                    nb = g.get_state().notifications_buffer
                    if nb:
                        notes.append(nb)
                h2 = _sector_heights(g.get_state())
                print(f"   after 'give': notification {give_note!r}; door gap -> "
                      f"{h2[back_sec, 1] - h2[back_sec, 0]}; notifications {notes}")
            finally:
                g.close()


def locked_screen():
    """Per-tic notifications and the screen after one USE on the locked MAP02 door
    (line 215), plus a console `echo` to see what the buffer does capture."""
    verts, lines, sides, secs = _map_lumps(FD2, "MAP02")
    g = _usable_game("MAP02")
    try:
        print("notifications buffer size (tics):", g.get_notifications_buffer_size())
        print("stand:", _stand_before_line(g, verts, lines[215])[0])
        notes = []
        for k in range(4):
            g.make_action(act(use=1 if k == 0 else 0), 1)
            notes.append(g.get_state().notifications_buffer)
        np.save(os.path.join(OUT_DIR, "_hud_before.npy"), g.get_state().screen_buffer)
        for _ in range(6):
            g.make_action(act(), 1)
            notes.append(g.get_state().notifications_buffer)
        print("per-tic notifications after USE on locked door:", notes)
        g.send_game_command("echo probe-echo")
        g.make_action(act(), 1)
        print("console echo ->", repr(g.get_state().notifications_buffer))
        np.save(os.path.join(OUT_DIR, "_hud_after.npy"), g.get_state().screen_buffer)
    finally:
        g.close()


def held_use_door():
    """Held USE (ADR 0006 encoding) on the MAP01 door used in round 1, and a 1,0,1 pattern."""
    for pattern in ("1" * 12, "1" + "0" * 3 + "1" + "0" * 7, "101" + "0" * 9):
        g = _usable_game("map01")
        try:
            g.send_game_command("warp 670 -160")
            g.make_action(act(), 2)
            a0 = gv(g.get_state())["ANGLE"]
            g.make_action(act(turn=-(((0 - a0) + 180) % 360 - 180)), 1)
            trace = []
            for ch in pattern:
                g.make_action(act(use=int(ch)), 4)  # held for all 4 tics
                sec = g.get_state().sectors[27]
                trace.append(round(sec.ceiling_height - sec.floor_height))
            print(f"use pattern {pattern}: door gap {trace}")
        finally:
            g.close()


def switch():
    """Held USE on a switch: find an S1/SR switch line on MAP02/MAP03 and watch every
    sector's heights before and after a 3-decision held press."""
    for m in ("MAP02", "MAP03"):
        verts, lines, sides, secs = _map_lumps(FD2, m)
        cands = [(i, l) for i, l in enumerate(lines)
                 if l[3] in SWITCH and l[3] not in LOCKED and l[3] != 11 and l[4] != 0][:3]
        for i, line in cands:
            g = _usable_game(m)
            try:
                tagged = [k for k, sc in enumerate(secs) if sc[6] == line[4]]
                pose, _ = _stand_before_line(g, verts, line, dist=36.0)
                h0 = _sector_heights(g.get_state())
                for k in range(12):
                    g.make_action(act(use=1 if k < 3 else 0), 4)
                h1 = _sector_heights(g.get_state())
                changed = np.nonzero(np.any(np.abs(h1 - h0) > 0.5, axis=1))[0].tolist()
                print(f"{m} line {i} special {line[3]} tag {line[4]} stand {pose}: tagged sectors "
                      f"{tagged}; sectors changed {changed}; "
                      f"heights {[(tuple(h0[k]), tuple(h1[k])) for k in tagged[:2]]}")
            finally:
                g.close()


def own_pixels():
    """Player's own (weapon) pixels in depth/labels: how the engine reports them,
    with and without firing, on MAP01 and MAP02. Per step: Self label values,
    depth of Self-labelled pixels, zero-depth pixels in the 3D view that are NOT
    Self-labelled, Self-pixel row span, ray cells (ADR 0006 bands, 2-px columns)
    that are entirely Self pixels, and entity boxes whose label pixels are fully
    covered by Self pixels (entity-depth fallback case)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from env.config import ray_bands_for
    bands = ray_bands_for(True)
    rng = np.random.default_rng(3)
    for m in ("MAP01", "MAP02"):
        for firing in (False, True):
            g = make_game(FD2, m, objects=False, sectors=False)
            try:
                g.new_episode()
                stats = {"steps": 0, "self_values": set(), "self_label_count": [],
                         "self_depth_values": set(), "nonself_zero_depth_px": 0,
                         "self_rows": [999, -1], "self_px_mean": [],
                         "all_own_cells": [0, 0, 0], "steps_with_all_own": 0,
                         "entities_fully_covered": 0, "entities_seen": 0,
                         "self_px_outside_self_box": 0}
                for i in range(150):
                    a = act(fwd=int(rng.integers(2)), turn=float(rng.choice([-4, 0, 4])),
                            attack=int(firing or rng.random() < 0.2), speed=int(rng.integers(2)))
                    g.make_action(a, 4)
                    if g.is_episode_finished():
                        g.new_episode()
                        continue
                    s = g.get_state()
                    d, lb = s.depth_buffer, s.labels_buffer
                    selfs = [l for l in s.labels if l.object_category == "Self"]
                    stats["steps"] += 1
                    stats["self_label_count"].append(len(selfs))
                    own = np.zeros(lb.shape, bool)
                    for l in selfs:
                        stats["self_values"].add(l.value)
                        own |= lb == l.value
                        box = np.zeros(lb.shape, bool)
                        box[l.y:l.y + l.height, l.x:l.x + l.width] = True
                        stats["self_px_outside_self_box"] += int((own & ~box).sum())
                    view = slice(0, 100)
                    stats["self_px_mean"].append(int(own[view].sum()))
                    if own.any():
                        rows = np.nonzero(own.any(axis=1))[0]
                        stats["self_rows"] = [min(stats["self_rows"][0], int(rows.min())),
                                              max(stats["self_rows"][1], int(rows.max()))]
                        stats["self_depth_values"] |= set(np.unique(d[own]).tolist())
                    stats["nonself_zero_depth_px"] += int(((d[view] == 0) & ~own[view]).sum())
                    any_all = False
                    for bi, (r0, r1) in enumerate(bands):
                        cells = own[r0:r1].reshape(r1 - r0, 80, 2).all(axis=(0, 2))
                        stats["all_own_cells"][bi] += int(cells.sum())
                        any_all |= bool(cells.any())
                    stats["steps_with_all_own"] += int(any_all)
                    for l in s.labels:
                        if l.object_category == "Self":
                            continue
                        stats["entities_seen"] += 1
                        sel = lb[l.y:l.y + l.height, l.x:l.x + l.width] == l.value
                        if not sel.any():
                            stats["entities_fully_covered"] += 1
                stats["self_values"] = sorted(stats["self_values"])
                stats["self_depth_values"] = sorted(stats["self_depth_values"])
                stats["self_label_count"] = sorted(set(stats["self_label_count"]))
                stats["self_px_mean"] = round(float(np.mean(stats["self_px_mean"])), 1)
                print(m, "firing" if firing else "mostly-not-firing", stats)
            finally:
                g.close()


def zero_depth_breakdown():
    """Where do zero-depth pixels in the 3D view (rows 0-99) come from? Groups them by
    labels value and by whether a Self label is in the label list that step; also
    reports the label values present when no Self label is listed."""
    from collections import Counter
    rng = np.random.default_rng(3)
    g = make_game(FD2, "MAP01", objects=False, sectors=False)
    try:
        g.new_episode()
        by_val = Counter()
        no_self_steps = 0
        no_self_has_255 = 0
        no_self_255_depth = Counter()
        lab_vals_listed = Counter()
        for i in range(150):
            g.make_action(act(fwd=int(rng.integers(2)), turn=float(rng.choice([-4, 0, 4])),
                              attack=int(rng.random() < 0.5)), 4)
            if g.is_episode_finished():
                g.new_episode()
                continue
            s = g.get_state()
            d, lb = s.depth_buffer[:100], s.labels_buffer[:100]
            has_self = any(l.object_category == "Self" for l in s.labels)
            for l in s.labels:
                lab_vals_listed[(l.object_category, l.value == 255)] += 1
            z = d == 0
            for v, c in zip(*np.unique(lb[z], return_counts=True)):
                by_val[(int(v), has_self)] += int(c)
            if not has_self:
                no_self_steps += 1
                if (lb == 255).any():
                    no_self_has_255 += 1
                    no_self_255_depth.update(np.unique(d[lb == 255]).tolist())
        print("zero-depth px in view by (labels value, self label listed):", dict(by_val))
        print("steps without a Self label:", no_self_steps, "of which with label-255 pixels:",
              no_self_has_255, "their depth values:", dict(no_self_255_depth))
        print("listed labels (category, value==255):", dict(lab_vals_listed))
    finally:
        g.close()


def zero_depth_where():
    """Locate zero-depth view pixels whose labels value is not 255: rows/cols span,
    inside the Self label box or not, attack on/off, and which listed label (if any)
    has that labels value."""
    from collections import Counter, defaultdict
    rng = np.random.default_rng(3)
    g = make_game(FD2, "MAP01", objects=False, sectors=False)
    try:
        g.new_episode()
        info = defaultdict(lambda: {"px": 0, "in_self_box": 0, "rows": [999, -1], "cols": [999, -1],
                                    "attack_steps": 0, "steps": 0, "owners": Counter()})
        for i in range(150):
            atk = int(rng.random() < 0.5)
            g.make_action(act(fwd=int(rng.integers(2)), turn=float(rng.choice([-4, 0, 4])),
                              attack=atk), 4)
            if g.is_episode_finished():
                g.new_episode()
                continue
            s = g.get_state()
            d, lb = s.depth_buffer[:100], s.labels_buffer[:100]
            selfbox = np.zeros(lb.shape, bool)
            for l in s.labels:
                if l.object_category == "Self":
                    selfbox[l.y:min(l.y + l.height, 100), l.x:l.x + l.width] = True
            owners = {l.value: (l.object_name, l.object_category) for l in s.labels}
            for v in (0, 127):
                m = (d == 0) & (lb == v)
                if not m.any():
                    continue
                e = info[v]
                rr, cc = np.nonzero(m)
                e["px"] += int(m.sum())
                e["in_self_box"] += int((m & selfbox).sum())
                e["rows"] = [min(e["rows"][0], int(rr.min())), max(e["rows"][1], int(rr.max()))]
                e["cols"] = [min(e["cols"][0], int(cc.min())), max(e["cols"][1], int(cc.max()))]
                e["attack_steps"] += atk
                e["steps"] += 1
                e["owners"][owners.get(v, ("<not listed>", ""))] += 1
        for v, e in info.items():
            e["owners"] = dict(e["owners"])
            print("labels value", v, dict(e))
    finally:
        g.close()


def zero_label0_shape():
    """For zero-depth, labels-0 view pixels (non-Self): is each affected column zero over
    a contiguous wall span (a surface nearer than one depth level), and what are the
    depths just next to the zero region?  Also walks straight into a wall to see
    whether depth reaches 0 there."""
    rng = np.random.default_rng(3)
    g = make_game(FD2, "MAP01", objects=False, sectors=False)
    try:
        g.new_episode()
        shown = 0
        for i in range(150):
            g.make_action(act(fwd=int(rng.integers(2)), turn=float(rng.choice([-4, 0, 4])),
                              attack=int(rng.random() < 0.5)), 4)
            if g.is_episode_finished():
                g.new_episode()
                continue
            s = g.get_state()
            d, lb = s.depth_buffer[:100].astype(int), s.labels_buffer[:100]
            m = (d == 0) & (lb == 0)
            if m.sum() < 20 or shown >= 4:
                continue
            shown += 1
            cols = np.nonzero(m.any(axis=0))[0]
            c = int(cols[len(cols) // 2])
            print(f"step {i}: zero/label0 px={int(m.sum())} cols {int(cols.min())}-{int(cols.max())}; "
                  f"col {c} depth rows 0..99 = {d[:, c].tolist()}")
            print(f"   labels col {c}: {lb[:, c].tolist()}")
        # walk into a wall head-on
        g.new_episode()
        g.make_action(act(), 10)
        trace = []
        for _ in range(12):
            g.make_action(act(fwd=1, speed=1), 4)
            s = g.get_state()
            trace.append(int(s.depth_buffer[30, 80]))
        print("depth row 30 col 80 walking into wall:", trace)
    finally:
        g.close()


def overlay_sources():
    """Zero-depth view pixels that are NOT labelled as the Self label nor any listed
    thing label ("unclaimed zeros"), split by region (rows 0-19 vs 40-99) under
    render_messages on/off and attack never/always. Also: brightness of those pixels
    in the screen (muzzle flash is bright) and whether the Self label box contains them."""
    for messages in (True, False):
        for attack in (0, 1):
            rng = np.random.default_rng(3)
            g = make_game(FD2, "MAP01", objects=False, sectors=False, messages=messages)
            try:
                g.new_episode()
                top = low = low_in_selfbox = steps_with = 0
                bright = []
                for i in range(100):
                    g.make_action(act(fwd=int(rng.integers(2)), turn=float(rng.choice([-4, 0, 4])),
                                      attack=attack), 4)
                    if g.is_episode_finished():
                        g.new_episode()
                        continue
                    s = g.get_state()
                    d, lb = s.depth_buffer[:100], s.labels_buffer[:100]
                    claimed = np.zeros(lb.shape, bool)
                    selfbox = np.zeros(lb.shape, bool)
                    for l in s.labels:
                        claimed |= lb == l.value
                        if l.object_category == "Self":
                            selfbox[l.y:min(l.y + l.height, 100), l.x:l.x + l.width] = True
                    m = (d == 0) & ~claimed
                    if m.any():
                        steps_with += 1
                    top += int(m[:20].sum())
                    low += int(m[40:].sum())
                    low_in_selfbox += int((m[40:] & selfbox[40:]).sum())
                    if m[40:].any():
                        bright.append(float(s.screen_buffer[:100][m].mean()))
                print(f"messages={messages} attack={attack}: steps with unclaimed zeros {steps_with}/100; "
                      f"rows0-19 px {top}; rows40-99 px {low} (inside Self box {low_in_selfbox}); "
                      f"mean RGB of low ones {np.mean(bright) if bright else None}")
            finally:
                g.close()


def label_zero_depth():
    """Per listed (non-Self) label class: pixels with depth 0 vs depth > 0, and the
    median depth of its non-zero pixels. Random play with frequent attack on MAP01 and
    MAP02, plus summoned items close in front of the player."""
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0, []])
    rng = np.random.default_rng(5)
    for m in ("MAP01", "MAP02"):
        g = make_game(FD2, m, args=("+sv_cheats 1",), objects=False, sectors=False)
        try:
            g.new_episode()
            g.make_action(act(), 10)
            for cls in ("Stimpack", "Clip", "BlueCard", "Medikit"):
                g.send_game_command(f"summon {cls} 40")
            for i in range(150):
                g.make_action(act(fwd=int(rng.random() < 0.3), turn=float(rng.choice([-4, 0, 4])),
                                  attack=int(rng.random() < 0.6)), 4)
                if g.is_episode_finished():
                    g.new_episode()
                    continue
                s = g.get_state()
                d, lb = s.depth_buffer[:100], s.labels_buffer[:100]
                for l in s.labels:
                    if l.object_category == "Self":
                        continue
                    px = d[lb == l.value]
                    a = acc[(l.object_name, l.object_category)]
                    a[0] += int((px == 0).sum())
                    a[1] += int((px > 0).sum())
                    if (px > 0).any():
                        a[2].append(int(np.median(px[px > 0])))
        finally:
            g.close()
    for k, (z, nz, med) in sorted(acc.items()):
        print(f"{k}: depth0 px {z}, depth>0 px {nz}, median nonzero depth "
              f"{int(np.median(med)) if med else None}")


def damage_after_end():
    """Is DAMAGE_TAKEN readable via get_game_variable after the episode has ended, and
    does console `kill` count as damage? Case A: `kill`. Case B: lethal real damage
    from a barrel exploded next to a 1-health player (`summon` + shoot, cheats)."""
    for case in ("kill", "barrel"):
        g = make_game(FD2, "MAP01", args=("+sv_cheats 1",), objects=False, sectors=False)
        try:
            g.new_episode()
            g.make_action(act(), 10)
            before = (g.get_game_variable(GV.DAMAGE_TAKEN), g.get_game_variable(GV.HEALTH))
            if case == "kill":
                g.send_game_command("kill")
                g.make_action(act(), 4)
            else:
                g.send_game_command("set sv_cheats 1")
                g.send_game_command("summon ExplosiveBarrel 48")
                g.make_action(act(), 2)
                for _ in range(10):
                    g.make_action(act(attack=1), 4)
                    if g.is_episode_finished():
                        break
            after = (g.get_game_variable(GV.DAMAGE_TAKEN), g.get_game_variable(GV.HEALTH))
            print(f"{case}: finished={g.is_episode_finished()} dead={g.is_player_dead()} "
                  f"(DAMAGE_TAKEN, HEALTH) before {before} after {after}")
        finally:
            g.close()


MWH_WAD = os.path.join(vzd.scenarios_path, "my_way_home.wad")


def mwh_lumps():
    """my_way_home.wad: lump list and the ACS source text if a SCRIPTS lump exists."""
    data, lumps = _wad_lumps(MWH_WAD)
    print("lumps:", [(n, sz) for n, _, sz in lumps])
    for name, lo, sz in lumps:
        if name == "SCRIPTS":
            print("---- SCRIPTS lump (ACS source) ----")
            print(data[lo:lo + sz].decode("latin-1"))


def mwh_goal(map_exit_reward=None):
    """Measure the bundled reward: per-tic living reward, then warp onto the goal item
    (largest-id non-player object other than decorations, printed for inspection) and
    record per-decision rewards, episode end and its cause. Optional argv[2]: a
    map_exit_reward to add (to check for double counting)."""
    if len(sys.argv) > 2:
        map_exit_reward = float(sys.argv[2])
    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, "my_way_home.cfg"))
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_objects_info_enabled(True)
    g.add_game_args("+sv_cheats 1")
    if map_exit_reward is not None:
        g.set_map_exit_reward(map_exit_reward)
    g.set_available_game_variables([GV.POSITION_X, GV.POSITION_Y])
    g.init()
    try:
        print("cfg rewards: living", g.get_living_reward(), "map_exit", g.get_map_exit_reward(),
              "death", g.get_death_penalty(), "timeout", g.get_episode_timeout(),
              "start", g.get_episode_start_time())
        g.new_episode()
        s = g.get_state()
        objs = [(o.id, o.name, round(o.position_x), round(o.position_y)) for o in s.objects]
        print("objects:", objs)
        nb = len(g.get_available_buttons())
        r_idle = [g.make_action([0] * nb, 1) for _ in range(5)]
        print("idle per-tic rewards:", r_idle)
        goal = [o for o in s.objects if o.name not in ("DoomPlayer",)]
        tgt = goal[0]
        g.send_game_command(f"warp {tgt.position_x:.0f} {tgt.position_y:.0f}")
        rs = []
        fwd = [0] * nb
        fwd[2] = 1  # MOVE_FORWARD in the cfg's button order (touch needs movement)
        for k in range(8):
            r = g.make_action(fwd, 1)
            rs.append(round(r, 6))
            if g.is_episode_finished():
                break
        print(f"after warp onto {tgt.name}: per-tic rewards {rs}; finished={g.is_episode_finished()} "
              f"timeout={g.is_episode_timeout_reached()} dead={g.is_player_dead()} "
              f"total={g.get_total_reward():.6f}")
    finally:
        g.close()


def same_seed():
    """ADR 0008: does DoomEnv.reset(seed=s) reproduce the episode (spawn pose, first
    observations under a fixed action sequence) and do different seeds differ? Runs
    on TRAIN (my_way_home: ACS random start among 17 points) and EVAL (freedoom2
    MAP01, monsters). Per config: one instance with resets seed A, seed A, seed B, then
    the instance is closed and a fresh one resets with seed A. One engine at a time."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from env import EVAL_CONFIG, TRAIN_CONFIG, DoomEnv

    def episode(env, seed, n=30):
        obs, info = env.reset(seed=seed)
        pose = (round(info["pos_x"], 2), round(info["pos_y"], 2), round(info["angle"], 2))
        seq = [obs]
        arng = np.random.default_rng(99)  # identical action sequence for every episode
        for _ in range(n):
            a = [int(arng.integers(3)), int(arng.integers(3)), int(arng.integers(7)),
                 int(arng.integers(2)), 0, int(arng.integers(2))]
            obs, r, te, tr, info = env.step(a)
            seq.append(obs)
            if te or tr:
                break
        final = (round(info["pos_x"], 2), round(info["pos_y"], 2), round(info["angle"], 2))
        return pose, final, seq

    def same(s1, s2):
        if len(s1) != len(s2):
            return False
        return all(np.array_equal(a[k], b[k]) for a, b in zip(s1, s2) for k in a)

    for cfg in (TRAIN_CONFIG.replace(win_timer_ms=None), EVAL_CONFIG):
        env = DoomEnv(cfg)
        try:
            A1 = episode(env, 7)
            A2 = episode(env, 7)
            B = episode(env, 8)
        finally:
            env.close()
        env = DoomEnv(cfg)
        try:
            A3 = episode(env, 7)
        finally:
            env.close()
        print(f"[{cfg.name}] spawn A1 {A1[0]} A2 {A2[0]} B {B[0]} A3(fresh) {A3[0]}")
        print(f"   final pose A1 {A1[1]} A2 {A2[1]} B {B[1]} A3 {A3[1]}")
        print(f"   obs sequences identical: A1==A2 {same(A1[2], A2[2])}  A1==A3(fresh) "
              f"{same(A1[2], A3[2])}  A1==B {same(A1[2], B[2])}  (len {len(A1[2])})")


def sky_depth():
    """Depth and labels value of sky pixels (outdoor area of MAP01 seen at spawn)."""
    g = make_game(FD2, "map01", objects=False, sectors=False)
    try:
        g.new_episode()
        vals = {}
        for k in range(24):
            g.make_action(act(turn=15.0), 1)
            s = g.get_state()
            top = s.depth_buffer[:10]
            vals[k] = (int(top.min()), int(top.max()), sorted(np.unique(s.labels_buffer[:10]).tolist())[:4])
        print("top-10-rows depth (min,max,labels) per yaw step:", vals)
    finally:
        g.close()


# ---------------------------------------------------------------- round 4 (audit E1, R9, ini)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _level_fingerprint(g):
    """Sector count, object count/names and the tic-1 notifications text (level name)."""
    s = g.get_state()
    names = sorted({o.name for o in s.objects})
    return {"sectors": len(s.sectors), "objects": len(s.objects), "names": names[:6],
            "notif": " ".join((s.notifications_buffer or "").split())[:60]}


def pwad_map_fallback():
    """Audit E1: which map loads when a map name absent from the PWAD is requested?
    PWADs my_way_home.wad (one map, MAP01) and the generated bundle (DD0001..) over the
    freedoom2 IWAD; MAP02 asked at init and via set_doom_map + new_episode (the reset
    path). Fingerprints compared with freedoom2 MAP02 loaded with no PWAD."""
    bundle = os.path.join(REPO, "maps", "bundles", "train-all.wad")

    def run(pwad, first, then=()):
        g = vzd.DoomGame()
        g.set_doom_game_path(FD2)
        if pwad:
            g.set_doom_scenario_path(pwad)
        g.set_doom_map(first)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_notifications_buffer_enabled(True)
        g.set_mode(vzd.Mode.PLAYER)
        tag = os.path.basename(pwad) if pwad else "IWAD only"
        try:
            g.init()
        except Exception as e:
            print(f"  init {tag} map {first}: {type(e).__name__}: {e}")
            g.close()
            return
        try:
            print(f"  init {tag} map {first}: {_level_fingerprint(g)}")
            for m in then:
                try:
                    g.set_doom_map(m)
                    g.new_episode()
                    print(f"    set_doom_map({m!r}) + new_episode: {_level_fingerprint(g)}")
                except Exception as e:
                    print(f"    set_doom_map({m!r}) + new_episode: {type(e).__name__}: {e}")
        finally:
            g.close()

    case = sys.argv[2] if len(sys.argv) > 2 else "reset"
    if case == "reset":
        print("reference: freedoom2 with no PWAD")
        run(None, "MAP01", then=("MAP02",))
        print("my_way_home.wad over freedoom2")
        run(MWH_WAD, "MAP01", then=("MAP02", "map03", "MAP01", "MAP99"))
    elif case == "init":  # observed to hang: run under a timeout
        run(MWH_WAD, "MAP02")
    elif case == "bundle":
        print("generated bundle over freedoom2")
        run(bundle, "DD0001", then=("MAP02", "dd0002"))


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1, o2 = orient(ax, ay, bx, by, cx, cy), orient(ax, ay, bx, by, dx, dy)
    o3, o4 = orient(cx, cy, dx, dy, ax, ay), orient(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def occlusion(n_steps=300):
    """Audit R9: does state.labels ever list an object hidden behind a wall?
    Random walk with turning on freedoom2 MAP01-03 and four generated S3 maps. Per
    step, per non-player object (whole-level objects info, probe only):
      * in_cone: bearing within +-40 deg of the heading (horizontal FOV is 90 deg);
      * walled: the segment player->object centre crosses a blocking line;
      * labelled: its id is in state.labels.
    Every non-Self label with zero pixels of its own value in the labels buffer is
    classified by what covers its box: the Self (weapon) value, a depth-0 pixel
    (firing overlay), or neither. Optional argv[2] = steps per map."""
    import math
    if len(sys.argv) > 2:
        n_steps = int(sys.argv[2])
    tdir = os.path.join(REPO, "maps", "train")
    gen = sorted(os.path.join(tdir, f) for f in os.listdir(tdir) if f.startswith("S3-") and f.endswith(".wad"))[:4]
    targets = [(None, "map01"), (None, "map02"), (None, "map03")] + [(w, "MAP01") for w in gen]
    rng = np.random.default_rng(5)
    keys = ("objects_seen", "walled_in_cone", "walled_in_cone_labelled", "walled_labelled_any",
            "walled_labelled_with_px", "clear_in_cone", "clear_in_cone_labelled", "labels",
            "labels_zero_px", "zero_px_self_cover", "zero_px_overlay", "zero_px_other", "zero_px_walled",
            "zero_px_other_sprite_cover", "zero_px_other_geometry_only")
    tot = dict.fromkeys(keys, 0)
    examples = []
    for pwad, m in targets:
        g = vzd.DoomGame()
        g.set_doom_game_path(FD2)
        if pwad:
            g.set_doom_scenario_path(pwad)
        g.set_doom_map(m)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_render_hud(True)
        g.set_depth_buffer_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_available_buttons(BUTTONS)
        g.set_available_game_variables(PROBE_VARS)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_seed(11)
        g.add_game_args("+sv_cheats 1")
        g.init()
        per = dict.fromkeys(keys, 0)
        tag = os.path.basename(pwad) if pwad else m
        try:
            g.new_episode()
            g.send_game_command("god")
            lines = None
            for k in range(n_steps):
                if g.is_episode_finished():
                    g.new_episode()
                    g.send_game_command("god")
                    lines = None
                s = g.get_state()
                if lines is None or k % 25 == 0:  # doors move
                    lines = list({(l.x1, l.y1, l.x2, l.y2) for sec in s.sectors for l in sec.lines
                                  if l.is_blocking})
                v = s.game_variables
                px, py, ang = float(v[5]), float(v[6]), float(v[8])
                lab_by_id = {l.object_id: l for l in s.labels}
                lb, dep = s.labels_buffer, s.depth_buffer
                walled_ids = set()
                for o in s.objects:
                    if o.name == "DoomPlayer":
                        continue
                    dx, dy = o.position_x - px, o.position_y - py
                    if dx * dx + dy * dy < 1.0:
                        continue
                    per["objects_seen"] += 1
                    bear = math.degrees(math.atan2(dy, dx))
                    in_cone = abs((bear - ang + 180.0) % 360.0 - 180.0) <= 40.0
                    walled = any(_segments_cross(px, py, o.position_x, o.position_y, *ln) for ln in lines)
                    labelled = o.id in lab_by_id
                    if walled:
                        walled_ids.add(o.id)
                    if walled and labelled:
                        per["walled_labelled_any"] += 1
                        lab = lab_by_id[o.id]
                        npx = int((lb == lab.value).sum())
                        per["walled_labelled_with_px"] += int(npx > 0)
                        if len(examples) < 10:
                            examples.append(("walled+labelled", tag, k, o.name, round(math.hypot(dx, dy)),
                                             in_cone, npx, (lab.x, lab.y, lab.width, lab.height)))
                    if in_cone:
                        key = "walled_in_cone" if walled else "clear_in_cone"
                        per[key] += 1
                        per[key + "_labelled"] += int(labelled)
                self_vals = [l.value for l in s.labels if l.object_category == "Self"]
                for lab in s.labels:
                    if lab.object_category == "Self":
                        continue
                    per["labels"] += 1
                    if int((lb == lab.value).sum()) == 0:
                        per["labels_zero_px"] += 1
                        box_l = lb[lab.y:lab.y + lab.height, lab.x:lab.x + lab.width]
                        box_d = dep[lab.y:lab.y + lab.height, lab.x:lab.x + lab.width]
                        if self_vals and np.isin(box_l, self_vals).any():
                            per["zero_px_self_cover"] += 1
                        elif (box_d == 0).any():
                            per["zero_px_overlay"] += 1
                        else:
                            per["zero_px_other"] += 1
                            other_vals = set(np.unique(box_l).tolist()) & {l.value for l in s.labels}
                            per["zero_px_other_sprite_cover"] += int(bool(other_vals))
                            per["zero_px_other_geometry_only"] += int(not other_vals)
                            if len(examples) < 24:
                                examples.append(("zero_px_other", tag, k, lab.object_name,
                                                 (lab.x, lab.y, lab.width, lab.height), lab.object_id in walled_ids))
                        per["zero_px_walled"] += int(lab.object_id in walled_ids)
                a = act(fwd=int(rng.random() < 0.7), turn=float(rng.choice([-10, -4, 0, 4, 10])),
                        attack=int(rng.random() < 0.15), use=int(rng.random() < 0.2))
                g.make_action(a, 4)
        finally:
            g.close()
        print(f"{tag}: {per}")
        for kk in tot:
            tot[kk] += per[kk]
    print("TOTAL:", tot)
    for e in examples:
        print("  example:", e)


def ini_path():
    """Audit item 3: set_doom_config_path. Init a game with a fresh per-instance ini
    path in a temp dir; report when the file is created, whether the cwd _vizdoom.ini
    is touched, and diff it against the cwd file (ignoring the timestamp comment)."""
    import difflib
    import shutil
    cwd_ini = os.path.abspath("_vizdoom.ini")
    before = os.path.getmtime(cwd_ini) if os.path.exists(cwd_ini) else None
    d = tempfile.mkdtemp(prefix="dd_ini_probe_")
    p = os.path.join(d, "instance.ini")
    g = vzd.DoomGame()
    g.set_doom_game_path(FD2)
    g.set_doom_config_path(p)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_mode(vzd.Mode.PLAYER)
    g.init()
    try:
        print("after init: exists", os.path.exists(p), "| get_doom_config_path", g.get_doom_config_path())
        g.new_episode()
        g.make_action([0] * len(g.get_available_buttons()), 4)
    finally:
        g.close()
    time.sleep(0.5)
    after = os.path.getmtime(cwd_ini) if os.path.exists(cwd_ini) else None
    print("after close: exists", os.path.exists(p), "| cwd ini mtime changed:", after != before)

    def body(path):
        with open(path, encoding="latin-1") as f:
            return [l for l in f.read().splitlines() if not l.startswith("# This file was generated")]
    if os.path.exists(p) and os.path.exists(cwd_ini):
        a, b = body(cwd_ini), body(p)
        diff = list(difflib.unified_diff(a, b, "cwd", "instance", lineterm="", n=0))
        print(f"lines cwd {len(a)} instance {len(b)}; diff lines: {len(diff)}")
        for l in diff[:40]:
            print("   ", l)
    shutil.rmtree(d, ignore_errors=True)



def occlusion_obs(n_steps=300):
    """R4.2 follow-up: after the fix, does env.observation.entities() still emit a row
    for any label with zero own pixels? Same random walk as `occlusion` on freedoom2
    MAP01 and S3-train-103004 (the maps with most zero-pixel labels). Per step: raw
    zero-own-pixel labels, rows emitted, and rows expected (= non-Self labels with own
    pixels, capped at 24). Also counts emitted rows with d = NO_READING."""
    sys.path.insert(0, REPO)
    from env import observation as O
    if len(sys.argv) > 2:
        n_steps = int(sys.argv[2])
    rng = np.random.default_rng(5)
    s3 = os.path.join(REPO, "maps", "train", "S3-train-103004.wad")
    for pwad, m in ((None, "map01"), (s3, "MAP01")):
        g = vzd.DoomGame()
        g.set_doom_game_path(FD2)
        if pwad:
            g.set_doom_scenario_path(pwad)
        g.set_doom_map(m)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_render_hud(True)
        g.set_depth_buffer_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_available_buttons(BUTTONS)
        g.set_available_game_variables(PROBE_VARS)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_seed(11)
        g.add_game_args("+sv_cheats 1")
        g.init()
        c = dict(raw_zero_px=0, rows=0, expected=0, mismatch_steps=0, rows_no_reading=0)
        tr = O.EntityTracker()
        try:
            g.new_episode()
            g.send_game_command("god")
            for k in range(n_steps):
                if g.is_episode_finished():
                    g.new_episode()
                    g.send_game_command("god")
                s = g.get_state()
                lb = s.labels_buffer
                vis = 0
                for lab in s.labels:
                    if lab.object_category == "Self" or lab.width <= 0 or lab.height <= 0:
                        continue
                    if (lb[lab.y:lab.y + lab.height, lab.x:lab.x + lab.width] == lab.value).any():
                        vis += 1
                    else:
                        c["raw_zero_px"] += 1
                f, t, msk = O.entities(s.labels, lb, s.depth_buffer, tr, 24)
                exp = min(vis, 24)
                c["rows"] += int(msk.sum())
                c["expected"] += exp
                c["mismatch_steps"] += int(int(msk.sum()) != exp)
                c["rows_no_reading"] += int((f[msk, 4] == O.NO_READING).sum())
                a = act(fwd=int(rng.random() < 0.7), turn=float(rng.choice([-10, -4, 0, 4, 10])),
                        attack=int(rng.random() < 0.15), use=int(rng.random() < 0.2))
                g.make_action(a, 4)
        finally:
            g.close()
        print(f"{os.path.basename(pwad) if pwad else m}: {c}")


if __name__ == "__main__":
    sec = sys.argv[1]
    t0 = time.time()
    if sec == "use_test":
        use_test(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    else:
        globals()[sec]()
    print(f"[{sec} done in {time.time() - t0:.1f}s]")
