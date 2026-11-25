"""Seeded synthetic states over real S0-S2 TRAIN routes for the teacher's no-monster
golden (tests/fixtures/teacher_labels_pre_combat_maps.json, captured before the combat
teacher existed). No engine: poses are sampled along the planned cell path with jitter,
headings / door gaps / keys / previous executed actions at random; each map is one
controller episode of N_STEPS decisions."""
from __future__ import annotations

import numpy as np

MAP_IDS = ("S0-train-100001", "S0-train-100002", "S1-train-101001", "S1-train-101002",
           "S2-train-102001", "S2-train-102002")
N_STEPS = 120


def synthetic_labels(map_id: str, n_steps: int = N_STEPS, monsters: bool = False) -> list:
    from maps.loader import resolve
    from teachers.controller import TeacherController
    from teachers.navigator import load_route
    route = load_route(resolve(map_id, "train"), map_id=map_id)
    rng = np.random.default_rng(sum(map(ord, map_id)))
    ctrl = TeacherController((-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0), seed=7)
    ctrl.reset(route)
    doors = sorted({int(a.door_sector) for a in route.anchors if a.kind == "door"}
                   | {int(s) for s in np.unique(route.grid.door_of) if s >= 0})
    path = route.cell_path
    labels = []
    i = 0
    for t in range(n_steps):
        i = min(max(i + int(rng.integers(-1, 4)), 0), len(path) - 1)
        x, y = route.grid.center(*path[i])
        x += float(rng.uniform(-12, 12))
        y += float(rng.uniform(-12, 12))
        prev = np.array([rng.integers(3), rng.integers(3), rng.integers(7), 0, rng.integers(2),
                         rng.integers(2)], np.int64)
        if t % 3:
            prev = labels[-1] if labels else prev
            prev = np.asarray(prev, np.int64)
        info = {"pos_x": x, "pos_y": y, "angle": float(rng.uniform(0, 360)),
                "door_gaps": {s: float(rng.choice([0.0, 0.0, 20.0, 72.0])) for s in doors},
                "keys_held": {"blue": bool(t > 60), "yellow": bool(t > 60), "red": bool(t > 60)}}
        if monsters:
            info["monsters"] = []
        a = ctrl.act({"prev_action": prev}, info)
        labels.append([int(v) for v in a])
    return labels
