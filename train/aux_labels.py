"""Training-only auxiliary labels (ADR 0003). Never inputs to the policy.

Labels with no reliable source are marked invalid and masked out of the loss.
No torch import.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

KEY_NAMES: tuple[str, ...] = ("blue", "yellow", "red")


def keys_from_infos(infos: Sequence[dict[str, Any]], key_names: tuple[str, ...] = KEY_NAMES) -> tuple[np.ndarray, np.ndarray]:
    """(N, n_keys) float32 from info["keys_held"], and (N,) validity."""
    n = len(infos)
    out = np.zeros((n, len(key_names)), np.float32)
    valid = np.zeros(n, bool)
    for i, info in enumerate(infos):
        held = info.get("keys_held") if info is not None else None
        if isinstance(held, dict):
            out[i] = [float(bool(held.get(k, False))) for k in key_names]
            valid[i] = True
    return out, valid


def visible_monsters(entity_type: np.ndarray, entity_mask: np.ndarray, monster_type_id: int,
                     cap: int) -> np.ndarray:
    """(N,) min(visible monsters, cap) / cap, from the TRUE (unmasked) entities (ADR 0007)."""
    assert entity_type.shape == entity_mask.shape and entity_type.ndim == 2, \
        f"entity_type/mask expected (N, E), got {entity_type.shape}, {entity_mask.shape}"
    assert cap > 0, "monster_count_cap must be positive"
    count = ((entity_type == monster_type_id) & entity_mask.astype(bool)).sum(axis=1)
    return (np.minimum(count, cap) / cap).astype(np.float32)


# Keys of the env's end-of-episode info used for the damage label (ADR 0007 ruling 2).
INFO_LAST_DAMAGE = "last_step_damage_taken"
INFO_DEAD = "dead"


def damage_events(next_self: np.ndarray, ended: np.ndarray, final_self: Sequence[np.ndarray | None],
                  final_fresh: np.ndarray, final_infos: Sequence[dict[str, Any] | None],
                  damage_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Damage taken during each env's transition: (N,) float32 (> 0 means damage) and (N,) validity.

    Continuing env: the next observation's damage slot. Ended env: the final
    `info` (death counts as damage; else `last_step_damage_taken`), which is
    training-only data; if the info has neither key, a fresh final observation's
    damage slot; otherwise invalid. Only the sign is used by the binary label.
    """
    n = ended.shape[0]
    ev = np.zeros(n, np.float32)
    valid = np.zeros(n, bool)
    for i in range(n):
        if not ended[i]:
            ev[i], valid[i] = next_self[i, damage_index], True
            continue
        info = final_infos[i] or {}
        if bool(info.get(INFO_DEAD, False)):
            ev[i], valid[i] = 1.0, True
        elif INFO_LAST_DAMAGE in info:
            ev[i], valid[i] = float(info[INFO_LAST_DAMAGE]), True
        elif final_fresh[i] and final_self[i] is not None:
            ev[i], valid[i] = final_self[i][damage_index], True
    return ev, valid


def damage_window_targets(events: np.ndarray, events_valid: np.ndarray, ended: np.ndarray,
                          k: int) -> tuple[np.ndarray, np.ndarray]:
    """Target for step t: any damage in transitions t..t+k-1 (stopping at episode end).

    Invalid if any event in that window is invalid, or if the window runs past the
    rollout end before the episode ends. Shapes (T, N).
    """
    t_len, n = events.shape
    tgt = np.zeros((t_len, n), np.float32)
    ok = np.zeros((t_len, n), bool)
    for t in range(t_len):
        for i in range(n):
            hit, good, complete = False, True, False
            for j in range(t, min(t + k, t_len)):
                good &= bool(events_valid[j, i])
                hit |= bool(events[j, i] > 0)
                if ended[j, i]:
                    complete = True
                    break
            else:
                complete = (t + k) <= t_len
            tgt[t, i], ok[t, i] = float(hit), good and complete
    return tgt, ok
