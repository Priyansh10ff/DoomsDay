"""Batch 2: ADD train maps (never touches batch 1, val or held-out).

    python -m maps.build_batch2 gen --stages S0 S1 --count 200 [--max-seeds 400]
    python -m maps.build_batch2 finalize
    python -m maps.build_batch2 regen            # rebuild any missing listed WAD, verify hashes

Unlike `python -m maps.build` (which deletes and rewrites every split), this script
is additive:
  * `gen` generates TRAIN maps from `maps/gen_config_b2.json` (gen_config.json with
    widened train ranges; gen_config.json itself stays byte-unchanged so batch-1, val
    and held-out maps still regenerate exactly). Seeds per stage come from the
    config's `batch.seed_base` block (110000/120000/130000/140000 + k), disjoint from
    batch 1 (100001-103060), val (5xxxxx) and held-out (9xxxxx). Each map gets the same
    QA as batch 1 (static structure, WAD + plan search, key off the start->lock path,
    one-engine check). Passing maps are written as maps/train/<id>.wad|json; every
    generation failure and rejection is appended to
    reports/runs/<date>-maps-batch2-<stage>.jsonl with its seed and error. Re-running
    resumes: seeds already logged are skipped.
  * `finalize` appends the passing batch-2 maps to splits.json's train split (a map
    whose hash equals an existing map's is dropped and reported), packs the bundle
    `train-all-b2` = batch-1 train maps in their train-all order, then batch 2, and
    asserts that the val/held-out sections, the 80 batch-1 entries and the `train-all`
    bundle entry and file are unchanged.
One engine per `gen` process; the caller runs at most two processes at once.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import sys
import time

from maps import generator as G
from maps.bundle import pack_bundle
from maps.loader import REPO_ROOT, SPLITS_PATH, check_integrity, sha256_file
from maps.wad import iwad_resources, write_wad

CFG_REL = "maps/gen_config_b2.json"
CFG_PATH = os.path.join(REPO_ROOT, CFG_REL)
TRAIN_DIR = "maps/train"
BUNDLE_ID = "train-all-b2"
BUNDLE_REL = f"maps/bundles/{BUNDLE_ID}.wad"
BATCH = 2
STAGES = ("S0", "S1", "S2", "S3")


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


def log_path(stage: str, date: str) -> str:
    return os.path.join(REPO_ROOT, "reports", "runs", f"{date}-maps-batch2-{stage}.jsonl")


def all_logs(stage: str) -> list[dict]:
    """Events of every batch-2 generation log of `stage`, whatever day it was written."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "reports", "runs", f"*-maps-batch2-{stage}.jsonl"))):
        out.extend(_read_log(p))
    return out


def _read_log(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def gen(stages, count: int, max_seeds: int, date: str, no_engine: bool = False) -> None:
    cfg = load_cfg()
    from env.config import iwad_path
    from maps.qa.validate import check_record, engine_check, load_map
    iwad = iwad_path("freedoom2")
    resources = iwad_resources(iwad)
    for stage in stages:
        lp = log_path(stage, date)
        done = _read_log(lp)
        seen = {d["seed"] for d in done}
        passed = sum(d["result"] == "pass" for d in done)
        base = cfg["batch"]["seed_base"][stage]
        t0 = time.time()
        k = 0
        while passed < count and k < max_seeds:
            k += 1
            seed = base + k
            if seed in seen:
                continue
            mid = f"{stage}-train-{seed}"
            ev = {"id": mid, "stage": stage, "seed": seed}
            try:
                lumps, rec = G.generate(cfg, "train", stage, seed)
            except G.LayoutError as e:
                ev.update(result="generation_failure", error=str(e))
            else:
                path = os.path.join(REPO_ROOT, TRAIN_DIR, f"{mid}.wad")
                write_wad(path, lumps)
                m = load_map(path)
                static = check_record(m, rec, resources)
                eng = None if (static or no_engine) else engine_check(path, m, rec, iwad)
                problems = static + (eng["problems"] if eng else [])
                rec["validation"] = {"static_problems": static, "engine": eng}
                rec["batch"] = BATCH
                rec["config_file"] = CFG_REL
                if problems or (eng is None and not no_engine):
                    os.remove(path)
                    ev.update(result="rejected", problems=problems)
                else:
                    with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
                        json.dump(rec, f, indent=1, sort_keys=True)
                    ev.update(result="pass", attempt=rec["attempt"], counts=rec["counts"],
                              engine_checked=eng is not None)
                    passed += 1
            with open(lp, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
        print(f"{stage}: {passed} passed after {k} seeds ({time.time() - t0:.0f} s)", flush=True)


def _batch2_records(date: str) -> list[tuple[dict, dict]]:
    """(log event, sidecar record) for every passing batch-2 map, stage then seed order."""
    out = []
    for stage in STAGES:
        for ev in sorted(all_logs(stage), key=lambda d: d["seed"]):
            if ev["result"] != "pass":
                continue
            side = os.path.join(REPO_ROOT, TRAIN_DIR, f"{ev['id']}.json")
            with open(side, encoding="utf-8") as f:
                out.append((ev, json.load(f)))
    return out


def finalize(date: str) -> dict:
    with open(SPLITS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    before = copy.deepcopy(data)
    old_bundle = next(b for b in data["bundles"] if b["id"] == "train-all")
    old_bundle_hash = sha256_file(os.path.join(REPO_ROOT, old_bundle["file"]))
    assert old_bundle_hash == old_bundle["sha256"], "train-all.wad changed before finalize"
    batch1_ids = [e["id"] for e in data["splits"]["train"] if e.get("batch", 1) == 1]
    data["splits"]["train"] = [e for e in data["splits"]["train"] if e.get("batch", 1) == 1]
    data["bundles"] = [b for b in data["bundles"] if b["id"] != BUNDLE_ID]
    hashes = {e["sha256"]: e["id"] for s in data["splits"].values() for e in s}
    dropped = []
    cfg_sha = sha256_file(CFG_PATH)
    for ev, rec in _batch2_records(date):
        rel = f"{TRAIN_DIR}/{ev['id']}.wad"
        h = sha256_file(os.path.join(REPO_ROOT, rel))
        if h in hashes:
            dropped.append({"id": ev["id"], "seed": ev["seed"], "reason": f"duplicate of {hashes[h]}"})
            continue
        hashes[h] = ev["id"]
        data["splits"]["train"].append({
            "id": ev["id"], "file": rel, "sha256": h, "stage": ev["stage"],
            "stage_name": rec["params"]["stage_name"], "seed": ev["seed"], "attempt": rec["attempt"],
            "map": "MAP01", "generator": G.GEN_NAME, "generator_version": G.GEN_VERSION,
            "config_file": CFG_REL, "config_sha256": cfg_sha, "mods": [], "counts": rec["counts"],
            "exit_kind": rec["graph"]["exit_kind"], "lock_color": rec["graph"]["lock_color"],
            "batch": BATCH})
    by_id = {e["id"]: e for e in data["splits"]["train"]}
    members = list(old_bundle["members"]) + [e["id"] for e in data["splits"]["train"]
                                             if e.get("batch") == BATCH]
    assert set(old_bundle["members"]) == set(batch1_ids)
    lumps = pack_bundle([os.path.join(REPO_ROOT, by_id[m]["file"]) for m in members],
                        os.path.join(REPO_ROOT, BUNDLE_REL))
    assert all(len(n) <= 8 and n.isascii() and n.isalnum() and n == n.upper() for n in lumps), lumps
    assert len(set(lumps)) == len(lumps)
    data["bundles"].append({"id": BUNDLE_ID, "split": "train", "file": BUNDLE_REL,
                            "sha256": sha256_file(os.path.join(REPO_ROOT, BUNDLE_REL)),
                            "lumps": lumps, "members": members,
                            "note": "batch-1 train maps (train-all order, DD0000-DD0079) then batch 2"})
    data["written"] = datetime.date.today().isoformat()
    data.setdefault("generator_configs", {})["batch2"] = {"config_file": CFG_REL, "config_sha256": cfg_sha}
    check_integrity(data)
    prev_b2 = {e["id"] for e in before["splits"]["train"] if e.get("batch") == BATCH}
    new_b2 = {e["id"] for e in data["splits"]["train"] if e.get("batch") == BATCH}
    if not prev_b2 <= new_b2:
        raise RuntimeError(f"finalize would drop {len(prev_b2 - new_b2)} listed batch-2 maps "
                           f"(logs missing?); refusing to write splits.json")
    # nothing outside batch-2 changed
    for s in ("val", "heldout"):
        assert data["splits"][s] == before["splits"][s], s
    assert data["splits"]["train"][:len(batch1_ids)] == [e for e in before["splits"]["train"]
                                                          if e.get("batch", 1) == 1]
    assert next(b for b in data["bundles"] if b["id"] == "train-all") == old_bundle
    assert sha256_file(os.path.join(REPO_ROOT, old_bundle["file"])) == old_bundle_hash
    assert data["external_eval"] == before["external_eval"]
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    summary = {s: sum(e["stage"] == s and e.get("batch") == BATCH for e in data["splits"]["train"])
               for s in STAGES}
    rep = {"date": date, "bundle": BUNDLE_ID, "bundle_maps": len(members), "batch2_by_stage": summary,
           "dropped_duplicates": dropped,
           "logs": {s: _summ(all_logs(s)) for s in STAGES}}
    print(json.dumps(rep, indent=1))
    return rep


def _summ(evs: list[dict]) -> dict:
    out = {"seeds_tried": len(evs)}
    for r in ("pass", "generation_failure", "rejected"):
        out[r] = sum(e["result"] == r for e in evs)
    return out


def regen() -> int:
    """Rebuild every listed generated map whose file is missing; verify every hash."""
    from maps.loader import _abs
    with open(SPLITS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    cfgs: dict = {}
    bad = 0
    for split, entries in data["splits"].items():
        for e in entries:
            p = _abs(e["file"])
            if not os.path.isfile(p):
                cf = e.get("config_file", "maps/gen_config.json")
                if cf not in cfgs:
                    with open(_abs(cf), encoding="utf-8") as f:
                        cfgs[cf] = json.load(f)
                lumps, rec = G.generate(cfgs[cf], split, e["stage"], e["seed"])
                os.makedirs(os.path.dirname(p), exist_ok=True)
                write_wad(p, lumps)
                print(f"rebuilt {e['file']}")
            if sha256_file(p) != e["sha256"]:
                print(f"HASH MISMATCH {e['file']}")
                bad += 1
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--stages", nargs="+", required=True, choices=STAGES)
    g.add_argument("--count", type=int, default=200)
    g.add_argument("--max-seeds", type=int, default=400)
    g.add_argument("--no-engine", action="store_true")
    g.add_argument("--date", default=datetime.date.today().isoformat())
    f = sub.add_parser("finalize")
    f.add_argument("--date", default=datetime.date.today().isoformat())
    sub.add_parser("regen")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        gen(a.stages, a.count, a.max_seeds, a.date, a.no_engine)
    elif a.cmd == "finalize":
        finalize(a.date)
    else:
        return regen()
    return 0


if __name__ == "__main__":
    sys.exit(main())
