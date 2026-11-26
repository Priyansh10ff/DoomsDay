"""Goal-mode label table for the scripted teachers (TRAINING-TIME labels; torch-free).

Each teacher decision reports one id from this fixed table (`ctrl.goal_mode`), saved per
step as `goal_mode` int64 (T,) in every episode npz (teachers/collect.py). It feeds an
auxiliary sub-goal prediction head: a TARGET for training, never a model input.

The model side must not import `teachers/` (tests/test_teachers_isolation.py), so the same
table is written again in configs/goal_modes.py; tests/test_teachers_goal_modes.py proves
the two are identical. Change both or neither. Ids are part of the dataset contract.
"""
from __future__ import annotations

EXPLORE = 0          # explore teacher: heading for a frontier (or spinning with no goal)
GET_KEY = 1          # going to a key not yet held
GO_LOCKED_DOOR = 2   # going to a locked door whose key is held
GO_EXIT = 3          # going to the exit (pad or switch, including the switch press)
FIGHT = 4            # combat label (a visible live monster engaged)
OPEN_DOOR = 5        # at a door: facing / pressing it, or pushing through it while it opens
RECOVER = 6          # unstick manoeuvre (back off and strafe)

GOAL_MODES: tuple[str, ...] = ("explore", "get_key", "go_locked_door", "go_exit", "fight",
                               "open_door", "recover")
N_GOAL_MODES = len(GOAL_MODES)
