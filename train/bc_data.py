"""Teacher dataset loading, map-id split and truncated-BPTT chunk streaming for
behaviour cloning.

Dataset contract (fixed by the coordinator, 2026-11-25):
  data/teacher/<name>/ep_<6 digits>.npz   one episode per file: the env observation
      keys except `screen`, `action` (T, 6) int64 = teacher label, `executed_action`,
      `engine_reward`, `done`. `prev_action` is the action actually executed (may be a
      noise action); it is used exactly as saved.
  data/teacher/<name>/index.jsonl         {"file","map_id","stage","length","success","seed","noise"}

Sequencing (truncated BPTT, carried state): `batch_seqs` slots each stream WHOLE
episodes, chunk by chunk, in a seeded shuffled order. An episode always starts at a
chunk boundary with episode_start=1 (the model zeroes the hidden state there); when
it ends mid-chunk the rest of that chunk is padding (valid=False) and the slot takes
the next episode at the next chunk. The trainer carries the GRU state between
chunks, detached. Padding repeats the episode's last real step (a real observation,
so no all-empty entity set is ever encoded); it is excluded from every loss and
metric by `valid`.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from agent.obs_spec import EGOMAP_KEY, ObsSpec, model_keys
from agent.tensorize import TORCH_DTYPES, torch_dtype

from .bc_spec import LABEL_KEY, DatasetMismatch, validate_arrays
from .goal_head import GOAL_LABEL_KEY, GOAL_UNLABELLED, validate_goal

INDEX_NAME = "index.jsonl"
INDEX_FIELDS = ("file", "map_id", "stage", "length", "success", "seed", "noise")
# Model-side dtype per observation key (agent.tensorize; `egomap` stays uint8 until
# its encoder). Old name kept: train/bc_aux.py and its tests import it.
_TORCH_DTYPES: dict[str, torch.dtype] = TORCH_DTYPES
# Saved dtype of optional model keys, used when train/bc_spec.py does not list them.
_OPTIONAL_NP_DTYPES: dict[str, np.dtype] = {EGOMAP_KEY: np.dtype(np.uint8)}
# 0/1 uint8 keys held BIT-PACKED in RAM (np.packbits over the flattened step; 8x
# smaller) and unpacked per batch. The npz files on disk are untouched.
BIT_PACKED_KEYS: tuple[str, ...] = (EGOMAP_KEY,)


def packed_step_bytes(step_shape: tuple[int, ...]) -> int:
    """Bytes per step of a bit-packed 0/1 array: ceil(prod(shape) / 8)."""
    return (int(np.prod(step_shape)) + 7) // 8


def pack_steps(a: np.ndarray, where: str = "array") -> np.ndarray:
    """(T, *step_shape) uint8 0/1 -> (T, packed_step_bytes) uint8. Refuses values other
    than 0/1 (packbits would silently turn them into 1)."""
    if a.dtype != np.uint8:
        raise DatasetMismatch(f"{where}: bit-packing needs uint8, got {a.dtype}")
    if a.size and int(a.max()) > 1:
        raise DatasetMismatch(f"{where}: bit-packing needs values 0/1, found max {int(a.max())}")
    out = np.packbits(a.reshape(a.shape[0], -1), axis=1)
    assert out.shape == (a.shape[0], packed_step_bytes(a.shape[1:])),         f"packed expected (T, {packed_step_bytes(a.shape[1:])}), got {out.shape}"
    return out


def unpack_steps(p: np.ndarray, step_shape: tuple[int, ...]) -> np.ndarray:
    """(..., packed_step_bytes) uint8 -> (..., *step_shape) uint8 0/1 (inverse of pack_steps)."""
    n = int(np.prod(step_shape))
    assert p.shape[-1] == packed_step_bytes(step_shape),         f"packed last dim expected {packed_step_bytes(step_shape)}, got {p.shape[-1]}"
    return np.unpackbits(p, axis=-1, count=n).reshape(*p.shape[:-1], *step_shape)


@dataclass
class Episode:
    obs: dict[str, np.ndarray]     # the model's keys (agent.obs_spec.model_keys), each (T, ...)
    labels: np.ndarray             # (T, n_heads) int64
    map_id: str
    stage: str
    file: str
    # key -> per-step shape, for keys stored bit-packed in `obs` ((T, packed bytes) uint8)
    packed: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # (T,) int64 sub-goal label (train/goal_head.py; negative = unlabelled), or None
    # when the file has no `goal_mode` array. Never a model input.
    goal: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.labels.shape[0])

    def rows(self, key: str, idx: np.ndarray) -> np.ndarray:
        """obs[key] at step rows `idx`, unpacked if stored bit-packed."""
        a = self.obs[key][idx]
        return unpack_steps(a, self.packed[key]) if key in self.packed else a

    def goal_rows(self, idx: np.ndarray) -> np.ndarray:
        """goal labels at step rows `idx`; GOAL_UNLABELLED everywhere when the episode has none."""
        if self.goal is None:
            return np.full(idx.shape, GOAL_UNLABELLED, np.int64)
        return self.goal[idx]


def read_index(data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, INDEX_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {INDEX_NAME} in {data_dir!r}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [k for k in INDEX_FIELDS if k not in r]
            if missing:
                raise DatasetMismatch(f"{path}:{n}: index row lacks {missing}")
            rows.append(r)
    if not rows:
        raise DatasetMismatch(f"{path}: no episodes")
    return rows


def _step_shapes_and_dtypes(spec: ObsSpec) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Per-step (shape, saved dtype) of every model key and the label: train/bc_spec's
    table, plus any model key it does not list (e.g. `egomap` from the spec itself)."""
    from .bc_spec import LABEL_DTYPE, OBS_DTYPES, expected_step_shapes
    out = {k: (tuple(s), LABEL_DTYPE if k == LABEL_KEY else OBS_DTYPES[k])
           for k, s in expected_step_shapes(spec).items()}
    if spec.egomap is not None and EGOMAP_KEY not in out:
        out[EGOMAP_KEY] = (tuple(spec.egomap), OBS_DTYPES.get(EGOMAP_KEY, _OPTIONAL_NP_DTYPES[EGOMAP_KEY]))
    return out


def step_bytes(spec: ObsSpec, pack: bool = True) -> int:
    """RAM per stored step (the model's observation keys + label), for the dataset-size
    guard. `egomap` counts its bit-packed size (640 bytes at (5, 32, 32)) when `pack`,
    as load_episodes stores it; its full size (5120) otherwise."""
    return sum(packed_step_bytes(s) if (pack and k in BIT_PACKED_KEYS) else int(np.prod(s)) * dt.itemsize
               for k, (s, dt) in _step_shapes_and_dtypes(spec).items())


def _validate_extra(arrays: dict[str, np.ndarray], spec: ObsSpec, t: int, where: str) -> None:
    """Check model keys that train/bc_spec.validate_arrays does not cover."""
    from .bc_spec import expected_step_shapes
    covered = set(expected_step_shapes(spec))
    for k, (shape, dt) in _step_shapes_and_dtypes(spec).items():
        if k in covered:
            continue
        if k not in arrays:
            raise DatasetMismatch(f"{where}: missing key {k!r} required by the model (has {sorted(arrays)})")
        a = arrays[k]
        if a.shape != (t, *shape):
            raise DatasetMismatch(f"{where}: {k} shape {a.shape}, expected {(t, *shape)}")
        if a.dtype != dt:
            raise DatasetMismatch(f"{where}: {k} dtype {a.dtype}, expected {dt}")


def load_episodes(data_dir: str, spec: ObsSpec, max_bytes: int, pack: bool = True) -> list[Episode]:
    """Load and validate every indexed episode. Refuses before loading if the index's
    lengths imply more than `max_bytes` of arrays. `pack`: keep BIT_PACKED_KEYS
    bit-packed in RAM (Episode.packed); batches unpack them, giving identical tensors."""
    rows = read_index(data_dir)
    est = sum(int(r["length"]) for r in rows) * step_bytes(spec, pack)
    if est > max_bytes:
        raise MemoryError(f"dataset {data_dir!r} needs about {est / 2**30:.2f} GiB in RAM, above the "
                          f"{max_bytes / 2**30:.2f} GiB cap (BCConfig.max_dataset_bytes)")
    keys = model_keys(spec)
    eps = []
    for r in rows:
        path = os.path.join(data_dir, r["file"])
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in (*keys, LABEL_KEY) if k in z.files}
            goal_raw = z[GOAL_LABEL_KEY] if GOAL_LABEL_KEY in z.files else None
        t = validate_arrays(arrays, spec, path)
        _validate_extra(arrays, spec, t, path)
        if t != int(r["length"]):
            raise DatasetMismatch(f"{path}: length {t} but index says {r['length']}")
        obs, packed = {k: arrays[k] for k in keys}, {}
        if pack:
            for k in BIT_PACKED_KEYS:
                if k in obs:
                    packed[k] = tuple(obs[k].shape[1:])
                    obs[k] = pack_steps(obs[k], f"{path}: {k}")
        goal = None if goal_raw is None else validate_goal(goal_raw, t, path)
        eps.append(Episode(obs, arrays[LABEL_KEY], str(r["map_id"]), str(r["stage"]), str(r["file"]),
                           packed, goal))
    return eps


def split_map_ids(map_ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """(train ids, val ids), disjoint. With val_frac > 0 and at least 2 distinct ids, at
    least one id is held out and at least one kept for training."""
    ids = sorted(set(map_ids))
    if val_frac <= 0.0 or len(ids) < 2:
        return ids, []
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    n_val = min(max(1, int(round(val_frac * len(ids)))), len(ids) - 1)
    return sorted(order[n_val:]), sorted(order[:n_val])


@dataclass
class Chunk:
    obs: dict[str, torch.Tensor]    # (T, B, ...)
    labels: torch.Tensor            # (T, B, n_heads) long
    valid: torch.Tensor             # (T, B) bool
    episode_start: torch.Tensor     # (T, B) float
    # (T, B) long sub-goal labels (negative = unlabelled) when any episode of the
    # stream carries `goal_mode`; None otherwise.
    goal: torch.Tensor | None = None


class ChunkStream:
    """Iterates one epoch of (T=seq_len, B=batch_seqs) chunks over `episodes`, each
    episode streamed whole by one slot (see module docstring)."""

    def __init__(self, episodes: list[Episode], seq_len: int, batch_seqs: int, generator: torch.Generator,
                 shuffle: bool = True, device: torch.device | str = "cpu") -> None:
        if not episodes:
            raise ValueError("no episodes to stream")
        self.eps, self.t, self.b = episodes, seq_len, batch_seqs
        self.gen, self.shuffle, self.device = generator, shuffle, torch.device(device)
        self.filler = episodes[0]   # source of padding for slots with no episode left
        self.keys = tuple(episodes[0].obs)
        if any(tuple(e.obs) != self.keys for e in episodes):
            raise ValueError(f"episodes carry different observation keys (first: {self.keys})")
        self.has_goal = any(e.goal is not None for e in episodes)

    def __iter__(self) -> Iterator[Chunk]:
        order = (torch.randperm(len(self.eps), generator=self.gen).tolist() if self.shuffle
                 else list(range(len(self.eps))))
        queue = iter(order)
        slot_ep: list[int | None] = [None] * self.b
        slot_off = [0] * self.b
        for s in range(self.b):
            slot_ep[s] = next(queue, None)
        while any(e is not None for e in slot_ep):
            yield self._build(slot_ep, slot_off)
            for s in range(self.b):
                e = slot_ep[s]
                if e is None:
                    continue
                slot_off[s] += self.t
                if slot_off[s] >= self.eps[e].length:
                    slot_ep[s], slot_off[s] = next(queue, None), 0

    def _build(self, slot_ep: list[int | None], slot_off: list[int]) -> Chunk:
        t, b = self.t, self.b
        idx = np.zeros((b, t), np.int64)      # per slot: the episode row read at each step
        valid = np.zeros((b, t), bool)
        start = np.zeros((b, t), np.float32)
        srcs: list[Episode] = []
        for s in range(b):
            e = slot_ep[s]
            ep = self.filler if e is None else self.eps[e]
            srcs.append(ep)
            if e is None:
                continue
            off, n = slot_off[s], ep.length
            real = min(t, n - off)
            idx[s, :real] = np.arange(off, off + real)
            idx[s, real:] = n - 1                     # pad with the last real step
            valid[s, :real] = True
            start[s, 0] = 1.0 if off == 0 else 0.0
        obs = {}
        for k in self.keys:
            stacked = np.stack([srcs[s].rows(k, idx[s]) for s in range(b)], axis=1)   # (T, B, ...)
            obs[k] = torch.as_tensor(stacked).to(device=self.device, dtype=torch_dtype(k, stacked))
        labels = np.stack([srcs[s].labels[idx[s]] for s in range(b)], axis=1)
        goal = None
        if self.has_goal:
            goal = torch.as_tensor(np.stack([srcs[s].goal_rows(idx[s]) for s in range(b)], axis=1),
                                   device=self.device).long()
            assert goal.shape == (t, b), f"goal labels expected (T={t}, B={b}), got {tuple(goal.shape)}"
        out = Chunk(obs, torch.as_tensor(labels, device=self.device).long(),
                    torch.as_tensor(valid.T.copy(), device=self.device),
                    torch.as_tensor(start.T.copy(), device=self.device), goal)
        assert out.labels.shape[:2] == (t, b), f"labels expected (T={t}, B={b}, heads)"
        assert out.valid.shape == (t, b) and out.episode_start.shape == (t, b)
        return out
