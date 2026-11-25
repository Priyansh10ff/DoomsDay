"""Extended map-format probes (map-forge). One engine instance at a time.

  A. multi-map PWAD: a bundle of 3 generated maps named DD0000..DD0002 plus one MAP33,
     each loaded by set_doom_map in ONE engine session; and through DoomEnv with
     reset(options={"map": ...}).
  B. monsters with empty REJECT/BLOCKMAP: warp near a placed monster on an S3 map and
     check that it moves and/or damages the player.
  C. level exit inside a bundle ends the ViZDoom episode (does not continue to the
     next map).
Run: python -m maps.probes.probe_extended <scratch_dir>
"""
from __future__ import annotations

import json
import math
import os
import sys

import vizdoom as vzd

from env.config import iwad_path
from maps.bundle import map_lumps
from maps.generator import generate
from maps.wad import read_wad, unpack_map, write_wad

GV = vzd.GameVariable


def game(wad, m, sectors=False, objects=True):
    g = vzd.DoomGame()
    g.set_doom_game_path(iwad_path("freedoom2"))
    g.set_doom_scenario_path(wad)
    g.set_doom_map(m)
    g.set_doom_skill(3)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_objects_info_enabled(objects)
    g.set_sectors_info_enabled(sectors)
    g.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.USE, vzd.Button.TURN_LEFT_RIGHT_DELTA])
    g.set_available_game_variables([GV.POSITION_X, GV.POSITION_Y, GV.HEALTH])
    g.set_map_exit_reward(1.0)
    g.set_episode_timeout(4000)
    g.add_game_args("+sv_cheats 1")
    g.init()
    return g


def main(d):
    os.makedirs(d, exist_ok=True)
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "gen_config.json")))
    paths, recs = [], []
    for st, seed in (("S0", 100011), ("S1", 100012), ("S3", 100013)):
        lumps, rec = generate(cfg, "train", st, seed)
        while st == "S0" and rec["graph"]["exit_kind"] != "pad":  # C needs a walk-over exit
            seed += 1
            lumps, rec = generate(cfg, "train", st, seed)
        p = os.path.join(d, f"ext_{st}.wad")
        write_wad(p, lumps)
        paths.append(p)
        recs.append(rec)
    # A: bundle with DD0000..DD0002 and MAP33 (copy of the first)
    lumps = []
    names = ["DD0000", "DD0001", "DD0002", "MAP33"]
    for name, p in zip(names, paths + [paths[0]]):
        ml = map_lumps(p)
        lumps += [(name, b"")] + ml[1:]
    bundle = os.path.join(d, "ext_bundle.wad")
    write_wad(bundle, lumps)
    g = game(bundle, "DD0000", sectors=True)
    try:
        for name, rec in zip(names, recs + [recs[0]]):
            g.set_doom_map(name)
            g.new_episode()
            st = g.get_state()
            m = unpack_map(read_wad(bundle), name)
            start = next(t for t in m["things"] if t.type == 1)
            print(f"A {name}: state={st is not None} sectors engine={len(st.sectors)} wad={len(m['sectors'])} "
                  f"spawn=({g.get_game_variable(GV.POSITION_X):.0f},{g.get_game_variable(GV.POSITION_Y):.0f}) "
                  f"start=({start.x},{start.y}) objects={sorted(o.name for o in st.objects)}")
        # C: exit inside the bundle (DD0000 is a pad or switch map; use warp to the exit line)
        g.set_doom_map("DD0000")
        g.new_episode()
        g.make_action([0, 0, 0], 10)
        m = unpack_map(read_wad(bundle), "DD0000")
        ex = [ld for ld in m["linedefs"] if ld.special in (11, 52)]
        pts = [m["vertices"][v] for ld in ex for v in (ld.v1, ld.v2)]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        g.send_game_command(f"warp {cx:.0f} {cy:.0f}")
        for _ in range(20):
            g.make_action([1, 0, 0], 4)
            if g.is_episode_finished():
                break
        if g is not None:
            print(f"C exit in bundle: finished={g.is_episode_finished()} reward={g.get_total_reward()} "
                  f"timeout={g.is_episode_timeout_reached()}")
    finally:
        if g is not None:
            g.close()
    # B: monster activity on the S3 map
    m = unpack_map(read_wad(paths[2]), "MAP01")
    mons = [t for t in recs[2]["things"] if t["role"] == "monster"]
    g = game(paths[2], "MAP01")
    try:
        g.new_episode()
        g.make_action([0, 0, 0], 10)
        tgt = mons[0]
        g.send_game_command(f"warp {tgt['x'] + 96} {tgt['y']}")
        g.make_action([0, 0, 0], 2)
        before = {o.id: (o.position_x, o.position_y) for o in g.get_state().objects if o.name != "DoomPlayer"}
        h0 = g.get_game_variable(GV.HEALTH)
        for _ in range(60):
            g.make_action([0, 0, 0], 4)
            if g.is_episode_finished():
                break
        st = g.get_state()
        after = {o.id: (o.position_x, o.position_y) for o in st.objects} if st else {}
        moved = sum(1 for k, p in before.items() if k in after and math.dist(p, after[k]) > 8)
        print(f"B monsters: target {tgt['class']} warp-dist 96; monsters moved={moved}/{len(before)} "
              f"health {h0} -> {g.get_game_variable(GV.HEALTH)} dead={g.is_player_dead()}")
    finally:
        g.close()
    # A2: through DoomEnv with reset(options={"map": ...})
    from env import DoomEnv
    from env.config import TRAIN_CONFIG, VANILLA_ENGINE_REWARDS
    cfgE = TRAIN_CONFIG.replace(name="probe-bundle", scenario_wad=None, map_wad=bundle,
                                maps=("DD0000", "DD0001", "DD0002"), skill=3,
                                engine_rewards=VANILLA_ENGINE_REWARDS, episode_timeout_tics=2100)
    env = DoomEnv(cfgE)
    try:
        for name in ("DD0002", "DD0000", "DD0001"):
            obs, info = env.reset(seed=1, options={"map": name})
            print(f"A2 DoomEnv reset map={name}: ok, keys={sorted(obs)}")
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1])
