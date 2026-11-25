"""Batch build: generate, validate, split, bundle, record.

    python -m maps.build --counts 20,5,5 [--no-engine]

For each stage and split, seeds are taken in order from the split's seed range
(offset by 1000 per stage) until the target count of maps has PASSED validation.
Every generation failure and every rejected map is recorded with its seed and error
in the run report (nothing is dropped silently). Engine checks run one ViZDoom
instance at a time.

Writes:
  maps/train/*.wad|json, maps/val/*.wad|json, heldout_maps/generated/*.wad|json
  maps/bundles/train-all.wad      all train maps, lumps DD0000.. (maps/bundle.py)
  maps/splits.json                every map: file, sha256, stage, seed, generator
  heldout_maps/MANIFEST.json      held-out files: sha256, source, seed, date (ADR 0008)
  reports/runs/<date>-maps-batch.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time

from maps import generator as G
from maps.bundle import pack_bundle
from maps.loader import HELDOUT_DIR, REPO_ROOT, sha256_file
from maps.wad import iwad_resources, read_wad, write_wad

STAGES = ("S0", "S1", "S2", "S3")
SPLIT_DIRS = {"train": "maps/train", "val": "maps/val", "heldout": f"{HELDOUT_DIR}/generated"}
CFG_PATH = os.path.join(REPO_ROOT, "maps", "gen_config.json")
STAGE_TO_ROLE_STAGE = {"S0": "1 nav-only", "S1": "3 doors", "S2": "4 keycards+locked doors",
                       "S3": "5 full levels"}


def freedoom_entries():
    import vizdoom as vzd
    out = []
    for name, split, phase in (("freedoom1", "val", 1), ("freedoom2", "heldout", 2)):
        p = os.path.join(os.path.dirname(vzd.__file__), f"{name}.wad")
        maps = [n for n, _ in read_wad(p) if (n.startswith("MAP") and len(n) == 5 and n[3:].isdigit())
                or (len(n) == 4 and n[0] == "E" and n[2] == "M" and n[1].isdigit() and n[3].isdigit())]
        out.append({"id": f"freedoom-phase{phase}", "source": "freedoom", "split": split,
                    "iwad": name, "file": p.replace("\\", "/"), "sha256": sha256_file(p),
                    "maps": maps, "mods": [],
                    "provenance": f"bundled with the vizdoom {vzd.__version__} wheel (not downloaded)",
                    "note": ("Phase 1: external evaluation proxy, val only (never train; stricter "
                             "than ADR 0005, which allows it in the last rung)") if phase == 1 else
                            "Phase 2: standing held-out set (ADR 0005); never trained on"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="20,5,5", help="maps per stage for train,val,heldout")
    ap.add_argument("--no-engine", action="store_true", help="skip the engine load check")
    ap.add_argument("--max-seeds", type=int, default=60, help="seeds tried per stage/split cap")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    counts = dict(zip(("train", "val", "heldout"), (int(x) for x in a.counts.split(","))))
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    from env.config import iwad_path
    from maps.qa.validate import check_record, engine_check, load_map
    iwad = iwad_path("freedoom2")
    resources = iwad_resources(iwad)
    today = datetime.date.today().isoformat()
    splits = {s: [] for s in counts}
    failures, rejected, summary = [], [], {}
    t0 = time.time()
    for split, target in counts.items():
        d = os.path.join(REPO_ROOT, SPLIT_DIRS[split])
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(d):  # previous batch output (ours, regenerable)
            if fn.endswith((".wad", ".json")) and fn.split("-")[0] in STAGES:
                os.remove(os.path.join(d, fn))
        lo = cfg["splits"][split]["seed_range"][0]
        for si, stage in enumerate(STAGES):
            passed = tried = 0
            seed = lo + 1000 * si
            while passed < target and tried < a.max_seeds:
                tried += 1
                seed += 1
                mid = f"{stage}-{split}-{seed}"
                try:
                    lumps, rec = G.generate(cfg, split, stage, seed)
                except G.LayoutError as e:
                    failures.append({"id": mid, "split": split, "stage": stage, "seed": seed,
                                     "error": str(e)})
                    continue
                rel = f"{SPLIT_DIRS[split]}/{mid}.wad"
                path = os.path.join(REPO_ROOT, rel)
                write_wad(path, lumps)
                m = load_map(path)
                static = check_record(m, rec, resources)
                eng = None
                if not static and not a.no_engine:
                    eng = engine_check(path, m, rec, iwad)
                problems = static + (eng["problems"] if eng else [])
                rec["validation"] = {"static_problems": static, "engine": eng}
                if problems:
                    rejected.append({"id": mid, "split": split, "stage": stage, "seed": seed,
                                     "problems": problems})
                    os.remove(path)
                    continue
                with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=1, sort_keys=True)
                splits[split].append({
                    "id": mid, "file": rel, "sha256": sha256_file(path), "stage": stage,
                    "stage_name": rec["params"]["stage_name"], "seed": seed,
                    "attempt": rec["attempt"], "map": "MAP01", "generator": G.GEN_NAME,
                    "generator_version": G.GEN_VERSION, "config_file": "maps/gen_config.json",
                    "config_sha256": sha256_file(CFG_PATH), "mods": [], "counts": rec["counts"],
                    "exit_kind": rec["graph"]["exit_kind"], "lock_color": rec["graph"]["lock_color"],
                })
                passed += 1
            summary[f"{split}/{stage}"] = {"target": target, "passed": passed, "seeds_tried": tried}
    # bundle of all train maps
    os.makedirs(os.path.join(REPO_ROOT, "maps", "bundles"), exist_ok=True)
    members = [e["id"] for e in splits["train"]]
    brel = "maps/bundles/train-all.wad"
    lumps = pack_bundle([os.path.join(REPO_ROOT, e["file"]) for e in splits["train"]],
                        os.path.join(REPO_ROOT, brel))
    data = {
        "format": 1, "written": today,
        "generator": {"name": G.GEN_NAME, "version": G.GEN_VERSION, "module": "maps/generator.py",
                      "config_file": "maps/gen_config.json", "config_sha256": sha256_file(CFG_PATH),
                      "python": platform.python_version(), "iwad": "freedoom2"},
        "stage_map": STAGE_TO_ROLE_STAGE,
        "splits": splits,
        "bundles": [{"id": "train-all", "split": "train", "file": brel,
                     "sha256": sha256_file(os.path.join(REPO_ROOT, brel)),
                     "lumps": lumps, "members": members}],
        "external_eval": freedoom_entries(),
    }
    from maps.loader import check_integrity
    check_integrity(data)
    with open(os.path.join(REPO_ROOT, "maps", "splits.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    manifest = {"written": today, "note": "Held-out maps (ADR 0008). Never loaded by a training "
                "config; WAD files are git-ignored and regenerated byte-identically by "
                "python -m maps.build.",
                "files": [{"file": e["file"], "sha256": e["sha256"], "source": G.GEN_NAME + " "
                           + G.GEN_VERSION, "stage": e["stage"], "seed": e["seed"], "date": today}
                          for e in splits["heldout"]],
                "external": [{"id": e["id"], "sha256": e["sha256"], "maps": len(e["maps"])}
                             for e in data["external_eval"] if e["split"] == "heldout"]}
    with open(os.path.join(REPO_ROOT, HELDOUT_DIR, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    report = {"date": today, "seconds": round(time.time() - t0, 1), "engine_checked": not a.no_engine,
              "summary": summary, "generation_failures": failures, "rejected": rejected,
              "counts": {s: len(v) for s, v in splits.items()},
              "stage_distribution": {s: {st: sum(e["stage"] == st for e in v) for st in STAGES}
                                     for s, v in splits.items()}}
    rp = a.report or os.path.join(REPO_ROOT, "reports", "runs", f"{today}-maps-batch1.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("seconds", "summary", "counts")}, indent=1))
    print(f"generation failures: {len(failures)}, rejected by validation: {len(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
