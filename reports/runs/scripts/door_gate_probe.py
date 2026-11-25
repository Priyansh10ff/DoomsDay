"""Old (depth check only) vs new (depth check + occlusion gate) door cue on real frames,
and residuals of geometrically-first door faces. Teacher drives (test helpers)."""
import sys, json
sys.path.insert(0, r"D:\DoomsDay"); sys.path.insert(0, r"D:\DoomsDay\tests")
import numpy as np, os
from env import door_cue as D
D.VIEW_BOB = 0.0  # (bob slack removed from env/door_cue.py; kept for the record)
D.BAND_SLOPE = float(os.environ.get("SLOPE", D.BAND_SLOPE))
import test_env_obs_v2_engine as T

REC = []
DIAG = []
_orig = D.door_channel
def rec(faces, px, py, h, mid, occ=None, eye_z=0.0):
    out = _orig(faces, px, py, h, mid, occ, eye_z)
    REC.append((np.array(faces, float).reshape(-1, 4), px, py, h, np.array(mid), None if occ is None else tuple(np.array(a) for a in occ), eye_z, out))
    return out
D.door_channel = rec

maps = sys.argv[1].split(",")
steps = int(sys.argv[2])
stats = dict(frames=0, old_cols=0, new_cols=0, new_not_subset=0, removed=0)
res_near, res_far, removed_res = [], [], []
def episodes():
    """Yields (map tag) after filling REC. MAP01 mode: vanilla freedoom2 MAP01 (engine-test
    config), random actions biased forward, plus a pass near door 27."""
    if maps == ["MAP01"]:
        from env import DoomEnv
        from env_test_helpers import FD2_ENGINE_TEST_CONFIG as FD2
        cfg = FD2.replace(maps=("MAP01",), game_args=("+sv_cheats 1",), episode_timeout_tics=100000,
                          obs_version=2, screen_enabled=False)
        env = DoomEnv(cfg)
        try:
            for seed in (1, 2, 3):
                REC.clear()
                env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                env.action_space.seed(seed)
                if seed == 3:
                    env.step([0, 0, 3, 0, 0, 0]); env.game.send_game_command("warp 500 -160")
                for t in range(steps):
                    a = env.action_space.sample()
                    a[0] = 1 if rng.random() < 0.6 else a[0]
                    _, _, te, tr, _ = env.step(a)
                    if te or tr:
                        break
                yield "MAP01"
        finally:
            env.close()
        return
    with T.make(2) as env:
        for mid_ in maps:
            for seed in (1, 2):
                REC.clear()
                T._drive(env, mid_, steps, lambda *a: None, seed=seed)
                yield mid_

for mid_ in episodes():
    if True:
        if True:
            for faces, px, py, h, mid, occ, ez, new in REC:
                stats["frames"] += 1
                old = _orig(faces, px, py, h, mid)
                stats["old_cols"] += int(old.sum()); stats["new_cols"] += int(new.sum())
                stats["new_not_subset"] += int(((new > 0) & (old == 0)).sum())
                stats["removed"] += int(((old > 0) & (new == 0)).sum())
                if faces.shape[0] == 0 or occ is None:
                    continue
                cols = np.arange(80)
                f, ok = D._cast(*D._to_player_frame(faces, px, py, h), cols)
                fd = np.where(ok, f, np.inf).min(axis=1)
                has = np.isfinite(fd) & (mid > 0)
                if not has.any():
                    continue
                c = cols[has]
                occd = D._occluded(c, fd[has], occ, px, py, h, ez)
                pred = np.clip(D.DEPTH_PER_UNIT * fd[has] + D.DEPTH_OFFSET, 1, 255)
                r = mid[has] - pred
                vis = ~occd
                res_near += r[vis & (fd[has] <= 150)].tolist()
                res_far += r[vis & (fd[has] > 150)].tolist()
                rm = (old[c] > 0) & occd
                removed_res += list(zip(r[rm].round(2).tolist(), fd[has][rm].round(0).tolist()))
                rm_v = rm & (r > -1.5)
                if rm_v.any() and len(DIAG) < 12:
                    rm = rm_v
                    seg, lo, hi = occ
                    f1, l1, f2, l2 = D._to_player_frame(seg, px, py, h)
                    cc = c[rm][:1]
                    ff, okk = D._cast(f1, l1, f2, l2, cc)
                    m = D.BAND_SLOPE * ff + D.VIEW_BOB
                    bl = (lo - ez > -m[0]) | (hi - ez < m[0])
                    idx = np.nonzero(okk[0] & bl & (ff[0] < fd[has][rm][0] - D.OCC_EPS))[0]
                    DIAG.append(dict(map=mid_, col=int(cc[0]), fd=round(float(fd[has][rm][0]),1), resid=round(float(r[rm][0]),2), eye=ez,
                        blockers=[(round(float(ff[0][i]),1), float(lo[i]-ez), float(hi[i]-ez), seg[i].tolist()) for i in idx[:4]]))
def pct(a):
    a = np.array(a)
    return {} if a.size == 0 else {"n": int(a.size), **{f"p{q}": round(float(np.percentile(a, q)), 2) for q in (0.5, 1, 5, 50, 99)},
                                   "min": round(float(a.min()), 2), "n_in_-6_-2": int(((a >= -6) & (a < -2)).sum()),
                                   "n_in_-2_1.5": int(((a >= -2) & (a <= 1.5)).sum())}
print(json.dumps({"stats": stats, "visible_first_face_resid_le150": pct(res_near),
                  "visible_first_face_resid_gt150": pct(res_far),
                  "removed_examples(resid,f)": removed_res[:25], "n_removed": len(removed_res), "n_removed_resid_gt_-1.5": sum(1 for r, f in removed_res if r > -1.5), "removed_visible": [x for x in removed_res if x[0] > -1.5][:20], "diag": DIAG}, indent=1))
