"""Door geometry of generated maps, for the TRAINING-ONLY door-progress reward.

Read from the WAD itself (the bundle the engine plays), never from a plan: a door is
a sector that is the back side of a linedef carrying a door special (DR / keyed DR,
maps.qa.validate.DOOR_SPECIALS). Generated maps are axis-aligned on a 32-unit
lattice, so every door sector is a rectangle with its two door-special faces
perpendicular to the travel axis.

PRIVILEGED: this geometry is used only by rewards/ to pay a training reward. It is
never part of an observation and never reaches the model. No torch, no vizdoom.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from maps.qa.validate import DOOR_SPECIALS, load_map
from maps.wad import NO_SIDE

AXIS_X, AXIS_Y = 0, 1


@dataclass(frozen=True)
class DoorSpec:
    """One door sector as a slab across a corridor.

    `axis` is the travel axis (AXIS_X: the door faces are vertical lines x = a0 and
    x = a1; AXIS_Y: horizontal lines y = a0, y = a1). `lo`, `hi` bound the opening
    on the other axis."""
    sector: int
    axis: int
    a0: float
    a1: float
    lo: float
    hi: float

    def local(self, x: float, y: float) -> tuple[float, float]:
        """(along, lateral) coordinates of a point in this door's frame."""
        return (x, y) if self.axis == AXIS_X else (y, x)


def door_specs(m: dict) -> tuple[DoorSpec, ...]:
    """Every door sector of an unpacked map (maps.wad.unpack_map), sorted by sector."""
    verts, sides = m["vertices"], m["sidedefs"]
    faces: dict[int, list] = {}
    every_line: dict[int, list] = {}
    for ld in m["linedefs"]:
        for s in (ld.front, ld.back):
            if s != NO_SIDE:
                every_line.setdefault(sides[s].sector, []).append(ld)
        if ld.back != NO_SIDE and ld.special in DOOR_SPECIALS:
            faces.setdefault(sides[ld.back].sector, []).append(ld)
    out = []
    for sec in sorted(faces):
        pts = [verts[v] for ld in every_line[sec] for v in (ld.v1, ld.v2)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        f = faces[sec][0]
        vertical = verts[f.v1][0] == verts[f.v2][0]   # face is x = const -> travel along x
        for ld in faces[sec]:
            if (verts[ld.v1][0] == verts[ld.v2][0]) != vertical:
                raise ValueError(f"door sector {sec}: door faces are not parallel")
        if vertical:
            out.append(DoorSpec(sec, AXIS_X, float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))))
        else:
            out.append(DoorSpec(sec, AXIS_Y, float(min(ys)), float(max(ys)), float(min(xs)), float(max(xs))))
    return tuple(out)


DoorTable = tuple[tuple[str, tuple[DoorSpec, ...]], ...]


def door_table(wad_path: str, lumps: Iterable[str]) -> DoorTable:
    """((lump, doors), ...) for the named maps of one WAD (e.g. the TRAIN bundle).
    A tuple (not a dict) so a frozen factory holding it stays hashable and picklable."""
    return tuple((lump, door_specs(load_map(wad_path, lump))) for lump in sorted(set(lumps)))
