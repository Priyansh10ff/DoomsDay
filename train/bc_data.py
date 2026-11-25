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
from dataclasses import dataclass

import numpy as np
import torch

from agent.obs_spec import A0_KEYS, ObsSpec

from .bc_spec import LABEL_KEY, DatasetMismatch, validate_arrays

INDEX_NAME = "index.jsonl"
INDEX_FIELDS = ("file", "map_id", "stage", "length", "success", "seed", "noise")
_TORCH_DTYPES: dict[str, torch.dtype] = {
    "rays": torch.float32, "entities": torch.float32, "entity_type": torch.long,
    "entity_mask": torch.bool, "self": torch.float32, "prev_action": torch.long,
    "modality": torch.float32,
}


@dataclass
class Episode:
    obs: dict[str, np.ndarray]     # A0 keys, each (T, ...)
    labels: np.ndarray             # (T, n_heads) int64
    map_id: str
    stage: str
    file: str

    @property
    def length(self) -> int:
        return int(self.labels.shape[0])


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


def step_bytes(spec: ObsSpec) -> int:
    """RAM per stored step (A0 observation keys + label), for the dataset-size guard."""
    from .bc_spec import OBS_DTYPES, expected_step_shapes
    shapes = expected_step_shapes(spec)
    total = 0
    for k, s in shapes.items():
        item = 8 if k == LABEL_KEY else OBS_DTYPES[k].itemsize
        total += int(np.prod(s)) * item
    return total


def load_episodes(data_dir: str, spec: ObsSpec, max_bytes: int) -> list[Episode]:
    """Load and validate every indexed episode. Refuses before loading if the index's
    lengths imply more than `max_bytes` of arrays."""
    rows = read_index(data_dir)
    est = sum(int(r["length"]) for r in rows) * step_bytes(spec)
    if est > max_bytes:
        raise MemoryError(f"dataset {data_dir!r} needs about {est / 2**30:.2f} GiB in RAM, above the "
                          f"{max_bytes / 2**30:.2f} GiB cap (BCConfig.max_dataset_bytes)")
    eps = []
    for r in rows:
        path = os.path.join(data_dir, r["file"])
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in (*A0_KEYS, LABEL_KEY) if k in z.files}
        t = validate_arrays(arrays, spec, path)
        if t != int(r["length"]):
            raise DatasetMismatch(f"{path}: length {t} but index says {r['length']}")
        eps.append(Episode({k: arrays[k] for k in A0_KEYS}, arrays[LABEL_KEY], str(r["map_id"]),
                           str(r["stage"]), str(r["file"])))
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
        for k in A0_KEYS:
            stacked = np.stack([srcs[s].obs[k][idx[s]] for s in range(b)], axis=1)   # (T, B, ...)
            obs[k] = torch.as_tensor(stacked).to(device=self.device, dtype=_TORCH_DTYPES[k])
        labels = np.stack([srcs[s].labels[idx[s]] for s in range(b)], axis=1)
        out = Chunk(obs, torch.as_tensor(labels, device=self.device).long(),
                    torch.as_tensor(valid.T.copy(), device=self.device),
                    torch.as_tensor(start.T.copy(), device=self.device))
        assert out.labels.shape[:2] == (t, b), f"labels expected (T={t}, B={b}, heads)"
        assert out.valid.shape == (t, b) and out.episode_start.shape == (t, b)
        return out
