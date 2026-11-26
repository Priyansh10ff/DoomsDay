"""Goal-mode ids of the teacher datasets (`goal_mode` int64 (T,) in each episode npz).

A copy of teachers/goal_modes.py for the model side, which must not import `teachers/`.
tests/test_teachers_goal_modes.py proves the two tables are identical: change both or
neither. Used as a TRAINING TARGET (sub-goal prediction head), never as a model input.
"""
from __future__ import annotations

EXPLORE = 0
GET_KEY = 1
GO_LOCKED_DOOR = 2
GO_EXIT = 3
FIGHT = 4
OPEN_DOOR = 5
RECOVER = 6

GOAL_MODES: tuple[str, ...] = ("explore", "get_key", "go_locked_door", "go_exit", "fight",
                               "open_door", "recover")
N_GOAL_MODES = len(GOAL_MODES)
