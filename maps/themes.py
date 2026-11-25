"""Texture/flat themes and thing types for the generator (vanilla Doom 2 format,
names checked against freedoom2.wad by tests/test_maps_generator.py).

Themes are PARTITIONED by split (THEMES_BY_SPLIT): no wall/flat theme used for train
maps appears in val or held-out maps, so a held-out map cannot be recognised by its
look. Door, switch and key-door textures are shared: they are the vanilla cues for
"this is a door" and must look the same everywhere.
"""
from __future__ import annotations

THEMES: dict[str, dict[str, tuple[str, ...]]] = {
    "tech": {
        "walls": ("STARTAN2", "STARTAN3", "STARG1", "STARG2", "STARGR1", "TEKWALL4", "STARBR2"),
        "floors": ("FLOOR0_1", "FLOOR0_3", "FLOOR4_8", "FLAT14", "FLOOR7_1"),
        "ceils": ("CEIL3_5", "CEIL5_1", "FLAT1", "CEIL3_3", "TLITE6_4"),
    },
    "brick": {
        "walls": ("BRICK1", "BRICK2", "BRICK4", "BRICK5", "BRICK7", "BIGBRIK1", "BIGBRIK2"),
        "floors": ("FLAT5_4", "FLOOR6_1", "FLAT18", "FLAT1_1"),
        "ceils": ("CEIL1_1", "FLAT10", "CEIL3_1"),
    },
    "wood": {
        "walls": ("WOOD1", "WOOD3", "WOOD5", "WOOD9", "WOOD12", "WOODVERT"),
        "floors": ("FLAT5_1", "FLAT5_2", "FLOOR5_1", "FLOOR5_2"),
        "ceils": ("CEIL1_1", "FLAT5_1", "CEIL1_3"),
    },
    "cement": {
        "walls": ("CEMENT1", "CEMENT2", "CEMENT4", "CEMENT6", "GRAY1", "GRAY4", "GRAY5", "GRAY7"),
        "floors": ("FLOOR1_1", "FLOOR0_2", "FLAT19", "FLAT3"),
        "ceils": ("CEIL3_1", "FLAT19", "CEIL5_2"),
    },
    "brown": {
        "walls": ("BROWN1", "BROWN96", "BROWNGRN", "BRONZE1", "BRONZE2", "BROWNHUG"),
        "floors": ("FLOOR3_3", "FLAT1_2", "FLAT1_3", "FLOOR7_2"),
        "ceils": ("CEIL4_1", "CEIL4_2", "FLAT2"),
    },
    "rock": {
        "walls": ("ROCK1", "ROCK2", "ROCK3", "ROCK4", "SP_ROCK1", "STONE2", "STONE3"),
        "floors": ("FLAT5_7", "GRNROCK", "RROCK09", "RROCK11", "FLOOR6_2"),
        "ceils": ("RROCK03", "FLAT5_7", "RROCK10"),
    },
    # ---- val-only
    "metal": {
        "walls": ("METAL2", "METAL3", "METAL4", "METAL5", "METAL6", "SUPPORT3", "SHAWN2"),
        "floors": ("FLOOR4_5", "FLOOR4_6", "FLAT23", "FLAT8"),
        "ceils": ("CEIL5_2", "FLAT9", "FLAT17"),
    },
    "aqua": {
        "walls": ("AQCONC01", "AQCONC04", "AQCONC08", "AQPANL01", "AQRUST01", "AQMETL05"),
        "floors": ("AQF001", "AQF012", "AQF025", "AQF040"),
        "ceils": ("AQF050", "AQF060", "AQF070"),
    },
    # ---- held-out only
    "marble": {
        "walls": ("MARBLE1", "MARBLE2", "MARBLE3", "MARBGRAY", "GSTONE1", "GSTONE2"),
        "floors": ("FLOOR5_3", "FLOOR5_4", "DEM1_5", "DEM1_6"),
        "ceils": ("DEM1_1", "DEM1_2", "FLAT5_3"),
    },
    "flesh": {
        "walls": ("SKIN2", "SKINMET1", "SKINMET2", "SP_HOT1", "SKINTEK1", "SKSNAKE1"),
        "floors": ("SFLR6_1", "SFLR6_4", "SFLR7_1", "MFLR8_1"),
        "ceils": ("SFLR7_4", "MFLR8_3", "CEIL5_1"),
    },
    "ashwall": {
        "walls": ("ASHWALL2", "ASHWALL3", "ASHWALL4", "ASHWALL6", "ASHWALL7", "TANROCK3"),
        "floors": ("RROCK16", "RROCK17", "RROCK18", "RROCK19"),
        "ceils": ("RROCK13", "RROCK14", "RROCK20"),
    },
}

THEMES_BY_SPLIT: dict[str, tuple[str, ...]] = {
    "train": ("tech", "brick", "wood", "cement", "brown", "rock"),
    "val": ("metal", "aqua"),
    "heldout": ("marble", "flesh", "ashwall"),
}

DOOR_TEXTURES = ("BIGDOOR1", "BIGDOOR2", "BIGDOOR3", "BIGDOOR4", "DOOR1", "DOOR3", "SPCDOOR1")
DOOR_TRACK = "DOORTRAK"
KEY_JAMB = {"blue": "DOORBLU", "yellow": "DOORYEL", "red": "DOORRED"}
EXIT_SWITCHES = ("SW1EXIT",)
EXIT_PAD_FLATS = ("GATE1", "GATE2", "GATE3", "GATE4")

# Vanilla linedef specials used.
SPECIAL_DOOR = 1                     # DR door open/wait/close (any player, use)
SPECIAL_KEYED = {"blue": 26, "yellow": 27, "red": 28}   # DR keyed doors
SPECIAL_EXIT_W1 = 52                 # W1 exit level
SPECIAL_EXIT_S1 = 11                 # S1 exit level (switch)

# Thing type numbers (Doom 2 / Freedoom Phase 2). Engine class names verified in
# docs/reference/map-format.md.
THING_PLAYER1 = 1
KEY_THINGS = {"blue": (5, 40), "yellow": (6, 39), "red": (13, 38)}  # (card, skull)
MONSTERS = {"zombieman": 3004, "shotgunguy": 9, "imp": 3001, "demon": 3002}
PICKUPS = {
    "stimpack": 2011, "medikit": 2012, "healthbonus": 2014, "greenarmor": 2018,
    "clip": 2007, "shells": 2008, "shotgun": 2001,
}
# Engine (ZDoom) actor class names for the thing types above; used by QA to compare
# objects in the engine with what the generator placed.
THING_CLASS = {
    1: "DoomPlayer", 5: "BlueCard", 6: "YellowCard", 13: "RedCard", 40: "BlueSkull",
    39: "YellowSkull", 38: "RedSkull", 3004: "Zombieman", 9: "ShotgunGuy", 3001: "DoomImp",
    3002: "Demon", 2011: "Stimpack", 2012: "Medikit", 2014: "HealthBonus", 2018: "GreenArmor",
    2007: "Clip", 2008: "Shell", 2001: "Shotgun",
}


def all_texture_names() -> set[str]:
    s = {DOOR_TRACK, *DOOR_TEXTURES, *KEY_JAMB.values(), *EXIT_SWITCHES}
    for t in THEMES.values():
        s.update(t["walls"])
    return s


def all_flat_names() -> set[str]:
    s = set(EXIT_PAD_FLATS)
    for t in THEMES.values():
        s.update(t["floors"])
        s.update(t["ceils"])
    return s
