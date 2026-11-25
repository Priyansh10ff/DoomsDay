"""Visual check of observation v3 `egomap` on S2-train-102001 (generated TRAIN map, key +
locked door). The scripted teacher (training-time tool, test/check side only) drives;
at 3 moments the 5 egomap channels are saved next to the screen.

Run: python reports/runs/2026-11-26-egomap-check/make_egomap_png.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from configs.a0_env import generated_maps_env_config  # noqa: E402
from env import DoomEnv  # noqa: E402
from env.config import TRAIN_CONFIG  # noqa: E402
from maps.loader import resolve, training_bundle  # noqa: E402
from teachers.controller import ControllerConfig, TeacherController  # noqa: E402
from teachers.navigator import load_route  # noqa: E402
from teachers.use_label import door_gaps_of  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
MAP_ID = "S2-train-102001"
NAMES = ("0 free", "1 wall", "2 visited", "3 closed door", "4 key seen")


def save(obs, info, t, tag):
    e = obs["egomap"]
    fig, ax = plt.subplots(1, 7, figsize=(21, 3.4))
    ax[0].imshow(np.transpose(obs["screen"], (1, 2, 0)))
    ax[0].set_title(f"screen t={t} ({tag})")
    for c in range(5):
        ax[c + 1].imshow(e[c], cmap="gray", vmin=0, vmax=1)
        ax[c + 1].plot(16, 16, "r^", ms=6)
        ax[c + 1].set_title(NAMES[c])
    rgb = np.zeros((32, 32, 3))
    rgb[..., 1] = 0.35 * e[0]
    rgb[e[1] > 0] = (0.9, 0.9, 0.9)
    rgb[e[2] > 0] = (0.2, 0.4, 1.0)
    rgb[e[3] > 0] = (1.0, 0.5, 0.0)
    rgb[e[4] > 0] = (1.0, 0.0, 1.0)
    ax[6].imshow(rgb)
    ax[6].plot(16, 16, "r^", ms=6)
    ax[6].set_title("composite (agent faces up)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    path = os.path.join(OUT, f"egomap_{MAP_ID}_t{t:03d}_{tag}.png")
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print(path, "pose(test-side only)", round(info["pos_x"]), round(info["pos_y"]), round(info["angle"]),
          "keys", info["keys_held"], "cells", e.reshape(5, -1).sum(1).tolist())


def main():
    bundle, lump_to_id, _ = training_bundle()
    lump = {v: k for k, v in lump_to_id.items()}[MAP_ID]
    base = TRAIN_CONFIG.replace(screen_enabled=True, obs_version=3, win_timer_ms=None)
    cfg = generated_maps_env_config(base, bundle, tuple(sorted(lump_to_id)))
    env = DoomEnv(cfg)
    try:
        route = load_route(resolve(MAP_ID, "train"), map_id=MAP_ID)
        ctrl = TeacherController(env.cfg.turn_bins, ControllerConfig(), seed=1)
        obs, info = env.reset(seed=1, options={"map": lump})
        ctrl.reset(route)
        save(obs, info, 0, "start")
        got_key_at = None
        for t in range(1, 1000):
            a = ctrl.act(obs, dict(info, door_gaps=door_gaps_of(env.game)))
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                print("episode ended at", t, "exit" if info.get("level_exit") else "")
                break
            if got_key_at is None and obs["egomap"][4].any() and t > 5 and t % 10 == 0:
                save(obs, info, t, "key-in-memory")
                got_key_at = -1
            if got_key_at == -1 and info["keys_held"]["blue"]:
                got_key_at = t
            if got_key_at and got_key_at > 0 and t == got_key_at + 40:
                save(obs, info, t, "after-key")
                break
        else:
            save(obs, info, t, "end")
    finally:
        env.close()


if __name__ == "__main__":
    main()
