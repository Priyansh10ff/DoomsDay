"""Map loader with split enforcement and hash verification.

Every map path the training side uses must come through `resolve()` (or
`training_bundle()`), which:
  * refuses, LOUDLY, any held-out map (split "heldout", a path under heldout_maps/, or
    a file whose SHA-256 equals any held-out file's hash) when mode == "train";
  * refuses val maps in "train" mode (they are for model selection only);
  * refuses external Freedoom entries in "train" mode;
  * verifies the file's SHA-256 against maps/splits.json before returning the path.

`load_splits()` also checks that every map appears in exactly one split, by id AND
by hash. The env side enforces the same rule independently
(env/config.py: EnvConfig.assert_trainable refuses heldout_maps/); a test checks the
two agree on every listed map.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_PATH = os.path.join(REPO_ROOT, "maps", "splits.json")
HELDOUT_DIR = "heldout_maps"          # must equal env.config.HELDOUT_MAP_DIR (tested)
SPLITS = ("train", "val", "heldout")
MODES = ("train", "val", "eval")      # which splits each mode may load:
ALLOWED = {"train": {"train"}, "val": {"train", "val"}, "eval": {"train", "val", "heldout"}}


class HeldOutAccessError(AssertionError):
    """A held-out (or otherwise forbidden) map was requested in training mode."""


class SplitIntegrityError(AssertionError):
    """splits.json is inconsistent, or a file on disk does not match its recorded hash."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _abs(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def under_heldout_dir(path: str) -> bool:
    p = os.path.normcase(os.path.realpath(_abs(path)))
    root = os.path.normcase(os.path.realpath(os.path.join(REPO_ROOT, HELDOUT_DIR)))
    try:
        return os.path.commonpath([p, root]) == root
    except ValueError:
        return False


@dataclass(frozen=True)
class Splits:
    data: dict

    def entries(self, split: str | None = None):
        for s in SPLITS if split is None else (split,):
            for e in self.data["splits"][s]:
                yield s, e

    def by_id(self, map_id: str) -> tuple[str, dict]:
        for s, e in self.entries():
            if e["id"] == map_id:
                return s, e
        for e in self.data.get("external_eval", []):
            if e["id"] == map_id:
                return e["split"], e
        raise KeyError(map_id)

    def heldout_hashes(self) -> set[str]:
        hs = {e["sha256"] for _, e in self.entries("heldout")}
        hs |= {e["sha256"] for e in self.data.get("external_eval", []) if e["split"] == "heldout"}
        return hs


def check_integrity(data: dict) -> None:
    """Each map in exactly one split, by id and by hash; held-out paths only under
    heldout_maps/, and no train/val path there."""
    seen_id, seen_hash = {}, {}
    for s in SPLITS:
        for e in data["splits"][s]:
            for key, table in ((e["id"], seen_id), (e["sha256"], seen_hash)):
                if key in table:
                    raise SplitIntegrityError(f"{key} appears in {table[key]} and {s}")
                table[key] = s
            if (s == "heldout") != under_heldout_dir(e["file"]):
                raise SplitIntegrityError(f"{e['id']} ({s}) has path {e['file']}; held-out maps "
                                          f"must be under {HELDOUT_DIR}/ and only they")
    for e in data.get("external_eval", []):
        if e["split"] not in ("val", "heldout"):
            raise SplitIntegrityError(f"external map set {e['id']} must be val or heldout, "
                                      f"not {e['split']}")
    for b in data.get("bundles", []):
        for mid in b["members"]:
            if seen_id.get(mid) != b["split"]:
                raise SplitIntegrityError(f"bundle {b['id']} member {mid} is not in {b['split']}")


def load_splits(path: str = SPLITS_PATH) -> Splits:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    check_integrity(data)
    return Splits(data)


def _verify(entry: dict) -> str:
    p = _abs(entry["file"])
    if not os.path.isfile(p):
        raise SplitIntegrityError(f"{entry['id']}: file {entry['file']} missing (regenerate with "
                                  f"python -m maps.build)")
    h = sha256_file(p)
    if h != entry["sha256"]:
        raise SplitIntegrityError(f"{entry['id']}: {entry['file']} sha256 {h} != recorded "
                                  f"{entry['sha256']} (file changed since splits.json was written)")
    return p


def _assert_allowed(split: str, entry: dict, mode: str, splits: Splits) -> None:
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if mode == "train":
        if split == "heldout" or under_heldout_dir(entry.get("file", "")):
            raise HeldOutAccessError(
                f"HELD-OUT MAP REQUESTED IN TRAINING MODE: {entry['id']} ({entry.get('file')}). "
                f"Held-out maps are never loaded for training (ADR 0005, 0008).")
        if entry.get("source") == "freedoom":
            raise HeldOutAccessError(f"Freedoom map set {entry['id']} is evaluation-only; "
                                     f"refused in training mode")
        if entry.get("sha256") in splits.heldout_hashes():
            raise HeldOutAccessError(f"{entry['id']} has the SHA-256 of a held-out map; refused "
                                     f"in training mode")
    if split not in ALLOWED[mode]:
        if split == "heldout":
            raise HeldOutAccessError(f"held-out map {entry['id']} requested in {mode!r} mode")
        raise PermissionError(f"{split} map {entry['id']} not loadable in {mode!r} mode")


def resolve(map_id: str, mode: str, splits: Splits | None = None) -> str:
    """Absolute path of a generated map WAD, after the split assert and hash check."""
    splits = splits or load_splits()
    split, entry = splits.by_id(map_id)
    _assert_allowed(split, entry, mode, splits)
    if entry.get("source") == "freedoom":
        raise ValueError(f"{map_id} is an IWAD map set; use its iwad/maps fields")
    return _verify(entry)


def training_bundle(bundle_id: str = "train-all", splits: Splits | None = None):
    """(bundle path, {lump name: map id}, {lump name: stage}) for training. Verifies the
    bundle hash and every member's hash, and that every member is a train map."""
    splits = splits or load_splits()
    b = next(x for x in splits.data["bundles"] if x["id"] == bundle_id)
    if b["split"] != "train":
        raise HeldOutAccessError(f"bundle {bundle_id} is {b['split']}, not train")
    lumps, stages = {}, {}
    for lump, mid in zip(b["lumps"], b["members"]):
        split, entry = splits.by_id(mid)
        _assert_allowed(split, entry, "train", splits)
        _verify(entry)
        lumps[lump], stages[lump] = mid, entry["stage"]
    if under_heldout_dir(b["file"]):
        raise HeldOutAccessError(f"bundle {bundle_id} lives under {HELDOUT_DIR}/")
    return _verify(b), lumps, stages
