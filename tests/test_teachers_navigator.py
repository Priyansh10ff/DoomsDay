"""Route planning on real TRAIN maps (no engine): grid, anchor order, doors, reachability."""
import json
import math
import os

import numpy as np
import pytest

from maps.loader import resolve
from maps.qa.validate import load_map, sector_at
from teachers.navigator import VOID, leg_waypoints, load_route, segment_clear

MAPS = {
    "S0": ["S0-train-100001", "S0-train-100005", "S0-train-100012"],
    "S1": ["S1-train-101002", "S1-train-101003", "S1-train-101010"],
    "S2": ["S2-train-102001", "S2-train-102004", "S2-train-102006"],
}
ALL = [(s, m) for s, ids in MAPS.items() for m in ids]


def _route(map_id: str):
    path = resolve(map_id, "train")      # refuses held-out / val maps
    with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
        rec = json.load(f)
    return load_route(path, map_id=map_id), rec, load_map(path)


@pytest.mark.parametrize("stage,map_id", ALL)
def test_grid_matches_sector_at(stage: str, map_id: str) -> None:
    r, _, m = _route(map_id)
    g = r.grid
    W, H = g.shape
    for c in range(0, W, 2):
        for rr in range(0, H, 2):
            s = sector_at(m, *g.center(c, rr))
            assert g.sector[c, rr] == (VOID if s is None else s), (c, rr)


@pytest.mark.parametrize("stage,map_id", ALL)
def test_anchor_order_and_kinds(stage: str, map_id: str) -> None:
    r, rec, _ = _route(map_id)
    kinds = [a.kind for a in r.anchors]
    gr = rec["graph"]
    assert kinds[-1] == ("exit_pad" if gr["exit_kind"] == "pad" else "exit_switch")
    if stage == "S0":
        assert kinds == [kinds[-1]], kinds
    if stage == "S2":
        assert kinds.count("key") == 1
        key_i = kinds.index("key")
        locked = [i for i, a in enumerate(r.anchors) if a.kind == "door" and a.door_color]
        assert len(locked) == 1, "the route crosses the locked door exactly once"
        assert key_i < locked[0], "key before the locked door"
        assert r.anchors[locked[0]].door_color == gr["lock_color"]
        assert r.anchors[key_i].key_color == gr["lock_color"]
        assert r.anchors[locked[0]].door_sector == gr["lock_door_sector"]
    else:
        assert "key" not in kinds
        assert not any(a.door_color for a in r.anchors)
    # one door anchor per door entered by the cell path
    g = r.grid
    entries = sum(1 for a, b in zip(r.cell_path, r.cell_path[1:])
                  if g.door_of[b] != VOID and g.door_of[a] == VOID)
    assert kinds.count("door") == entries


@pytest.mark.parametrize("stage,map_id", ALL)
def test_anchors_reachable_in_sequence(stage: str, map_id: str) -> None:
    """Doors closed, each anchor is reachable from the previous anchor's end point
    (the start, the key, or the pass point behind a door)."""
    r, _, _ = _route(map_id)
    g = r.grid
    x, y = r.start
    for a in r.anchors:
        cc = g.cell_of(x, y)
        assert g.walkable(*cc), (a.kind, x, y)
        assert np.isfinite(a.dist[cc]), f"{a.kind} anchor unreachable from {(x, y)}"
        if a.kind == "door":
            (cx, cy), (dx, dy) = a.door_center, a.door_dir
            assert g.walkable(*g.cell_of(a.x, a.y)) and g.walkable(*g.cell_of(a.pass_x, a.pass_y))
            assert ((a.x - cx) * dx + (a.y - cy) * dy) < 0 < ((a.pass_x - cx) * dx + (a.pass_y - cy) * dy)
            assert math.isclose(a.face, math.degrees(math.atan2(dy, dx)) % 360.0)
            x, y = a.pass_x, a.pass_y
        else:
            x, y = a.x, a.y


@pytest.mark.parametrize("stage,map_id", [ALL[0], ALL[3], ALL[6]])
def test_leg_waypoints_are_walkable(stage: str, map_id: str) -> None:
    r, _, _ = _route(map_id)
    pts = leg_waypoints(r, 0, *r.start)
    assert len(pts) >= 2
    a0 = r.anchors[0]
    assert math.hypot(pts[-1][0] - a0.x, pts[-1][1] - a0.y) < 1e-6
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert segment_clear(r.grid, x0, y0, x1, y1, radius=8.0)


def test_route_is_deterministic() -> None:
    a, _, _ = _route("S2-train-102001")
    b, _, _ = _route("S2-train-102001")
    assert [(x.kind, x.x, x.y, x.face) for x in a.anchors] == [(x.kind, x.x, x.y, x.face) for x in b.anchors]
    for p, q in zip(a.anchors, b.anchors):
        assert np.array_equal(p.dist, q.dist)
