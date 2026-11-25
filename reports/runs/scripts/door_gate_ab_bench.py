"""A/B wrapper around tools.env_bench: GATE=0 -> old behaviour (no occlusion table, no gate),
GATE=1 -> new. Also times the door-cue Python code per decision."""
import os, sys, time, atexit, json
sys.path.insert(0, r"D:\DoomsDay")
from env import door_cue as D
GATE = os.environ.get("GATE", "1") == "1"
T = {"n": 0, "s": 0.0}
_dc, _sn = D.door_channel, D.DoorTracker.scene_now
def scene_now(self, sectors):
    t = time.perf_counter()
    r = _sn(self, sectors) if GATE else (self.faces_now(sectors), None)
    T["s"] += time.perf_counter() - t
    return r
def door_channel(*a, **k):
    t = time.perf_counter()
    if not GATE:
        a = a[:5]
    r = _dc(*a, **k)
    T["s"] += time.perf_counter() - t; T["n"] += 1
    return r
D.door_channel = door_channel
D.DoorTracker.scene_now = scene_now
atexit.register(lambda: print("DOORCODE", json.dumps({"gate": GATE, "calls": T["n"], "ms_per_call": round(1e3 * T["s"] / max(T["n"], 1), 4)})))
from tools import env_bench
sys.argv = ["env_bench"] + sys.argv[1:]
env_bench.main()
