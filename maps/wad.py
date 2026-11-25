"""Minimal WAD reader/writer and Doom-format map lump packing (our own code).

Doom (vanilla binary) map format only:
  THINGS   10 bytes: x, y, angle, type, flags            (int16 x5)
  LINEDEFS 14 bytes: v1, v2, flags, special, tag, front, back (int16/uint16)
  SIDEDEFS 30 bytes: xoff, yoff, upper[8], lower[8], middle[8], sector
  VERTEXES  4 bytes: x, y
  SECTORS  26 bytes: floor, ceil, floorflat[8], ceilflat[8], light, special, tag

The writer emits the map marker followed by THINGS, LINEDEFS, SIDEDEFS, VERTEXES,
SEGS, SSECTORS, NODES, SECTORS, REJECT, BLOCKMAP in vanilla order. SEGS, SSECTORS,
NODES, REJECT and BLOCKMAP are written EMPTY: the engine's built-in node builder
must build them (verified for ViZDoom 1.3.3, docs/reference/map-format.md).

Output is deterministic: no timestamps, fixed lump order, fixed padding.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

NO_SIDE = 0xFFFF  # "no sidedef" in a LINEDEF (stored as -1 / 65535)

MAP_LUMP_ORDER = ("THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS", "SSECTORS",
                  "NODES", "SECTORS", "REJECT", "BLOCKMAP")


def _name8(s: str) -> bytes:
    b = s.upper().encode("ascii")
    if len(b) > 8:
        raise ValueError(f"lump/texture name {s!r} longer than 8 characters")
    return b.ljust(8, b"\0")


def _unname8(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("ascii", errors="replace").upper()


def write_wad(path: str, lumps: list[tuple[str, bytes]], kind: bytes = b"PWAD") -> None:
    """Write lumps as a WAD. Layout: header, lump data in order, directory."""
    data = bytearray()
    directory = []
    offset = 12
    for name, payload in lumps:
        directory.append((offset if payload else 0, len(payload), _name8(name)))
        data += payload
        offset += len(payload)
    header = struct.pack("<4sii", kind, len(lumps), offset)
    out = bytearray(header) + data
    for off, size, name in directory:
        out += struct.pack("<ii8s", off, size, name)
    with open(path, "wb") as f:
        f.write(bytes(out))


def read_wad(path: str) -> list[tuple[str, bytes]]:
    """Read a WAD's lumps (name, bytes) in directory order. Data only, nothing run."""
    with open(path, "rb") as f:
        raw = f.read()
    kind, n, diroff = struct.unpack_from("<4sii", raw, 0)
    if kind not in (b"IWAD", b"PWAD"):
        raise ValueError(f"{path}: not a WAD (magic {kind!r})")
    out = []
    for i in range(n):
        off, size, name = struct.unpack_from("<ii8s", raw, diroff + 16 * i)
        out.append((_unname8(name), raw[off:off + size]))
    return out


# ------------------------------------------------------------------ map records
@dataclass
class Thing:
    x: int
    y: int
    angle: int
    type: int
    flags: int = 7  # easy | medium | hard; not multiplayer-only


@dataclass
class Linedef:
    v1: int
    v2: int
    flags: int
    special: int
    tag: int
    front: int
    back: int = NO_SIDE


@dataclass
class Sidedef:
    xoff: int
    yoff: int
    upper: str
    lower: str
    middle: str
    sector: int


@dataclass
class Sector:
    floor: int
    ceil: int
    floor_flat: str
    ceil_flat: str
    light: int
    special: int = 0
    tag: int = 0


# Linedef flag bits (vanilla).
ML_BLOCKING = 0x0001
ML_TWOSIDED = 0x0004
ML_DONTPEGTOP = 0x0008
ML_DONTPEGBOTTOM = 0x0010


def pack_map(name: str, vertices, linedefs, sidedefs, sectors, things) -> list[tuple[str, bytes]]:
    t = b"".join(struct.pack("<hhhhh", th.x, th.y, th.angle, th.type, th.flags) for th in things)
    l = b"".join(struct.pack("<HHHHHHH", ld.v1, ld.v2, ld.flags, ld.special, ld.tag,
                             ld.front, ld.back) for ld in linedefs)
    s = b"".join(struct.pack("<hh8s8s8sH", sd.xoff, sd.yoff, _name8(sd.upper), _name8(sd.lower),
                             _name8(sd.middle), sd.sector) for sd in sidedefs)
    v = b"".join(struct.pack("<hh", x, y) for x, y in vertices)
    sec = b"".join(struct.pack("<hh8s8shhh", se.floor, se.ceil, _name8(se.floor_flat),
                               _name8(se.ceil_flat), se.light, se.special, se.tag)
                   for se in sectors)
    by_name = {"THINGS": t, "LINEDEFS": l, "SIDEDEFS": s, "VERTEXES": v, "SECTORS": sec}
    return [(name, b"")] + [(k, by_name.get(k, b"")) for k in MAP_LUMP_ORDER]


def unpack_map(lumps: list[tuple[str, bytes]], map_name: str) -> dict:
    """Parse the five geometry lumps of one map in a lump list (for QA and tests)."""
    idx = [i for i, (n, _) in enumerate(lumps) if n == map_name.upper()]
    if not idx:
        raise KeyError(map_name)
    i = idx[0] + 1
    rec = {}
    while i < len(lumps) and lumps[i][0] in MAP_LUMP_ORDER + ("BEHAVIOR", "SCRIPTS"):
        rec[lumps[i][0]] = lumps[i][1]
        i += 1
    things = [Thing(*struct.unpack_from("<hhhhh", rec["THINGS"], o))
              for o in range(0, len(rec["THINGS"]), 10)]
    lines = [Linedef(*struct.unpack_from("<HHHHHHH", rec["LINEDEFS"], o))
             for o in range(0, len(rec["LINEDEFS"]), 14)]
    sides = []
    for o in range(0, len(rec["SIDEDEFS"]), 30):
        xo, yo, up, lo, mid, sec = struct.unpack_from("<hh8s8s8sH", rec["SIDEDEFS"], o)
        sides.append(Sidedef(xo, yo, _unname8(up), _unname8(lo), _unname8(mid), sec))
    verts = [struct.unpack_from("<hh", rec["VERTEXES"], o) for o in range(0, len(rec["VERTEXES"]), 4)]
    secs = []
    for o in range(0, len(rec["SECTORS"]), 26):
        f, c, ff, cf, li, sp, tg = struct.unpack_from("<hh8s8shhh", rec["SECTORS"], o)
        secs.append(Sector(f, c, _unname8(ff), _unname8(cf), li, sp, tg))
    return {"things": things, "linedefs": lines, "sidedefs": sides, "vertices": verts,
            "sectors": secs, "lump_names": list(rec)}


# ------------------------------------------------------------------ IWAD resources
def iwad_resources(iwad_path: str) -> dict:
    """Wall texture names (TEXTURE1/TEXTURE2) and flat names (F_START..F_END) in an IWAD."""
    lumps = read_wad(iwad_path)
    textures: set[str] = set()
    flats: set[str] = set()
    in_flats = False
    for name, data in lumps:
        if name in ("TEXTURE1", "TEXTURE2"):
            (n,) = struct.unpack_from("<i", data, 0)
            for k in range(n):
                (off,) = struct.unpack_from("<i", data, 4 + 4 * k)
                textures.add(_unname8(data[off:off + 8]))
        if name in ("F_START", "FF_START"):
            in_flats = True
            continue
        if name in ("F_END", "FF_END"):
            in_flats = False
            continue
        if in_flats and data:
            flats.add(name)
    return {"textures": textures, "flats": flats}
