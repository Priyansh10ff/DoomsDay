"""Pack several single-map WADs into one multi-map PWAD (training bundles).

DoomEnv binds one PWAD at engine start but already switches maps per episode via
`reset(options={"map": name})` / `EnvConfig.maps`. A bundle therefore lets the
curriculum change map (and stage) every episode with no env change and no engine
restart. Lump names are "DD" + 4 digits (e.g. DD0007): not MAP01-MAP32, so the IWAD's
MAPINFO entries (level names, MAP07 boss specials, intermission texts) never apply.
Map lumps are copied byte for byte; only the marker name changes.
"""
from __future__ import annotations

from maps.wad import MAP_LUMP_ORDER, read_wad, write_wad


def bundle_lump_name(i: int) -> str:
    return f"DD{i:04d}"


def map_lumps(path: str, map_name: str = "MAP01") -> list[tuple[str, bytes]]:
    lumps = read_wad(path)
    i = next(k for k, (n, _) in enumerate(lumps) if n == map_name)
    out = [lumps[i]]
    for n, d in lumps[i + 1:]:
        if n not in MAP_LUMP_ORDER:
            break
        out.append((n, d))
    return out


def pack_bundle(members: list[str], out_path: str) -> list[str]:
    """members: single-map WAD paths (MAP01 each). Returns the lump names, in order."""
    lumps, names = [], []
    for i, p in enumerate(members):
        ml = map_lumps(p)
        name = bundle_lump_name(i)
        lumps.append((name, b""))
        lumps.extend(ml[1:])
        names.append(name)
    write_wad(out_path, lumps)
    return names
