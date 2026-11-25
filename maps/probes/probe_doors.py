"""Round 5 probe: how a door sector looks in the unmodified engine's sector info, and
how the depth buffer's middle band relates to a 2-D ray cast against door lines.

TRAINING-TIME DIAGNOSTIC (one engine), written by env-engineer for docs/reference/engine-probes.md round 5; kept under maps/probes/ because it imports teachers (no env/ module may). Drives the scripted teacher (teachers/) on a few
generated TRAIN maps. Never imported by env/, agent/, evaluate/ or inference code.

    python -m maps.probes.probe_doors [pairs_out.npy]
"""
from __future__ import annotations

import math
import sys

import numpy as np


def _cfg(sectors=True):
    from configs.a0_env import generated_maps_env_config
    from env.config import TRAIN_CONFIG
    from maps.loader import training_bundle
    bundle, lump_to_id, stages = training_bundle()
    base = TRAIN_CONFIG.replace(screen_enabled=False)
    cfg = generated_maps_env_config(base, bundle, tuple(sorted(lump_to_id)))
    return cfg.replace(sectors_info=sectors), lump_to_id, stages


def column_angles_deg():
    x = np.arange(80) * 2 + 1.0          # bin centre in pixels (0..159)
    return np.degrees(np.arctan((x - 80.0) / 80.0))


def cast(px, py, heading, lines):
    """Euclidean hit distance per column against segments (N,4); inf where no hit."""
    off = column_angles_deg()
    ang = np.radians(heading - off)
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x1, y1, x2, y2 = (lines[:, i][None, :] for i in range(4))
    ex, ey = x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    wx, wy = x1 - px, y1 - py
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * ey - wy * ex) / den
        u = (wx * dy - wy * dx) / den
    ok = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    return t.min(axis=1), off


PAIRS = []


def main():
    from env import DoomEnv
    from maps.doors import door_specs
    from maps.qa.validate import load_map
    from maps.loader import resolve
    from teachers.collect import Job
    from teachers.controller import ControllerConfig, TeacherController
    from teachers.navigator import load_route
    from teachers.use_label import door_gaps_of
    cfg, lump_to_id, stages = _cfg()
    id_to_lump = {v: k for k, v in lump_to_id.items()}
    env = DoomEnv(cfg)
    try:
        for mid, drop_key in (("S1-train-101001", False), ("S1-train-101004", False),
                              ("S2-train-102001", True), ("S2-train-102001", False)):
            lump = id_to_lump[mid]
            wad = resolve(mid, "train")
            doors = door_specs(load_map(wad, "MAP01"))
            route = load_route(wad, map_id=mid)
            if drop_key:
                route.anchors = [a for a in route.anchors if a.kind != "key"]
            print(f"\n=== {mid} ({lump}) drop_key={drop_key} doors(WAD)={[d.sector for d in doors]}")
            obs, info = env.reset(seed=1, options={"map": lump})
            st = env.game.get_state()
            secs = st.sectors
            print("n sectors", len(secs), "ids", [s.id for s in secs][:30])
            for s in secs:
                gap = s.ceiling_height - s.floor_height
                tag = " <- WAD door" if s.id in {d.sector for d in doors} else ""
                if gap < 72 or tag:
                    print(f"  sector id={s.id} floor={s.floor_height} ceil={s.ceiling_height} gap={gap} "
                          f"lines={len(s.lines)} blocking={[l.is_blocking for l in s.lines]}{tag}")
            ctrl = TeacherController(cfg.turn_bins, ControllerConfig(), seed=1)
            ctrl.reset(route)
            info = dict(info, door_gaps=door_gaps_of(env.game))
            door_ids = {d.sector for d in doors}
            locked_presses = 0
            for t in range(900):
                a = ctrl.act(obs, info)
                an = ctrl.anchor
                if (drop_key and ctrl.phase == "align" and a[2] == ctrl.zero_turn and an.kind == "door"
                        and an.door_color):
                    a = a.copy(); a[4] = t % 2          # press USE at the locked door ourselves
                    locked_presses += 1
                    if locked_presses > 40:
                        break
                obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    print("episode end at", t, "exit", info.get("level_exit"))
                    break
                info = dict(info, door_gaps=door_gaps_of(env.game))
                st = env.game.get_state()
                nb = st.notifications_buffer
                if nb and nb.strip():
                    print(f"  t={t} notif={nb.strip()!r}")
                gaps = {s.id: s.ceiling_height - s.floor_height for s in st.sectors if s.id in door_ids}
                x, y, h = info["pos_x"], info["pos_y"], info["angle"]
                lines = np.array([[l.x1, l.y1, l.x2, l.y2] for s in st.sectors if s.id in door_ids
                                  for l in s.lines], dtype=np.float64)
                if lines.size == 0:
                    continue
                hit, off = cast(x, y, h, lines)
                band = st.depth_buffer[46:54].astype(np.int16)
                own = band == 0
                band = np.where(own, 1000, band).reshape(8, 80, 2).min(axis=(0, 2))
                lit = np.isfinite(hit)
                perp_all = hit * np.cos(np.radians(off))
                for c in np.where(lit)[0]:
                    PAIRS.append((float(perp_all[c]), int(band[c]), float(max(gaps.values()))))
                if t % 3 == 0 and lit.any() and np.nanmin(hit[lit]) < 700:
                    perp = hit * np.cos(np.radians(off))
                    cols = np.where(lit)[0]
                    pick = cols[:: max(1, len(cols) // 6)]
                    rows = [(int(c), round(float(hit[c]) / 8, 1), round(float(perp[c]) / 8, 1),
                             int(band[c])) for c in pick]
                    print(f"  t={t} phase={ctrl.phase} anchor={ctrl.anchor.kind} gaps={gaps} "
                          f"(col, euclid/8, perp/8, depth) {rows}")
                elif t % 10 == 0:
                    print(f"  t={t} phase={ctrl.phase} gaps={gaps}")
    finally:
        env.close()
    if len(sys.argv) > 1:
        np.save(sys.argv[1], np.array(PAIRS))


if __name__ == "__main__":
    sys.exit(main())
