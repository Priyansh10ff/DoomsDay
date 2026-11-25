"""Behaviour cloning (BC) of the scripted teacher into the A0 recurrent agent
(CLAUDE.md s7, 2026-11-25 evening plan change; coordinator rulings 1-4).

    python -m train.bc --data data/teacher/<name> [--data <dir2>[,<dir3>...]] --out reports/runs/<run-dir>
        [--init-from-bc <bc_model.pt>] [--epochs N] [--seq-len 64] [--batch-seqs 32] [--lr 3e-4] [--device cpu|cuda|auto]
        [--seed S] [--val-frac 0.1] [--max-seconds S] [--head-weights 1,1,1,0,1,1]
        [--torch-threads 2] [--max-dataset-gb 3] [--use-pos-weight 1.0] [--prev-action-dropout 0.0]

What it does
  * Loads every episode of the dataset (train/bc_data.py), validates shapes and
    dtypes against the A0 obs spec built engine-free (train/bc_spec.py), and splits
    BY MAP ID: `val-frac` of the map ids are held out of cloning, so val accuracy is
    measured on maps the model was not cloned on.
  * Truncated BPTT with a carried, detached GRU state: each of `batch-seqs` slots
    streams whole episodes chunk by chunk (`seq-len` decisions); padded steps are
    masked out of every loss and metric.
  * Loss = sum over the six heads of weight_h x cross-entropy(teacher label). Every
    head's default weight is 1, attack included (configs/bc_config.py: a zero-weight
    attack head stays at 50/50 and breaks closed-loop play); a head given weight 0 is
    logged, and every head whose train label is constant is flagged.
  * The value head is NOT trained here (coordinator ruling 2): use
    `python -m train.run_doom --init-from ... --value-warmup-updates N` to fit it
    under PPO's own return convention before the policy moves.
  * Telemetry dropout (ADR 0006) is not applied: A0 trains with probability 0.
  * `--prev-action-dropout P` (default 0 = off): in training chunks only, each (step,
    sequence) independently has its `prev_action` input replaced with probability P by
    the env's reset value (all zeros, env/doom_env.py reset), so the student cannot
    lean on copying its previous action (the wall-wedge of
    reports/eval/2026-11-26-bc-v3-regression.md). Validation is never dropped.
  * Model built with agent.model.build_agent + configs.model_config.ModelConfig(),
    the same path the PPO trainer uses, so weights are interchangeable.

Outputs in --out: `bc_model.pt` (weights-only checkpoint in the trainer's blob
layout, train/checkpoint.save_weights_checkpoint; loads in evaluate.run and
run_doom --init-from, refused by --resume), `bc_metrics.jsonl` (one line per epoch),
`bc_summary.json`. Below-normal process priority. Starts no engine.

The sensor signature stored in bc_model.pt is ASSUMED (see train/bc_spec.py), not
read from the dataset; the record says so.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from agent.distributions import FactorizedCategorical
from agent.model import A0Agent, build_agent
from agent.obs_spec import ObsSpec
from configs.bc_config import GIB, USE_HEAD_INDEX, USE_PRESS_LABEL, BCConfig
from configs.model_config import ModelConfig
from configs.train_config import HealthMonitorConfig, RunConfig

from .bc_data import INDEX_NAME, ChunkStream, Episode, load_episodes, split_map_ids, step_bytes
from .bc_spec import SIGNATURE_SOURCE
from .checkpoint import save_weights_checkpoint

MODEL_NAME = "bc_model.pt"
METRICS_NAME = "bc_metrics.jsonl"
SUMMARY_NAME = "bc_summary.json"
# `prev_action` value at episode start: env/doom_env.py reset() sets np.zeros(6, int64).
NO_PREV_ACTION_VALUE = 0
PREV_ACTION_KEY = "prev_action"
VALUE_HEAD_NOTE = ("value head untouched by BC (no value loss); fit it with "
                   "train.run_doom --init-from <bc_model.pt> --value-warmup-updates N")
# Key read_init_bc adds to the returned record when it grew the egomap input.
INIT_GROWTH_KEY = "init_egomap_growth"


def head_names(n: int) -> list[str]:
    names = HealthMonitorConfig().head_names
    return [names[i] if i < len(names) else f"head{i}" for i in range(n)]


# ---------------------------------------------------------------- loss
@dataclasses.dataclass
class ChunkLoss:
    total: torch.Tensor                 # scalar, weighted sum of per-head mean CE
    head_loss_sum: list[float]          # per head: CE summed over valid steps
    head_correct: list[int]             # per head: argmax == label, over valid steps
    head_entropy_sum: list[float]       # per head: policy entropy summed over valid steps
    n_valid: int
    # USE head press counts over valid steps: true positives, label presses, predicted presses
    use_tp: int = 0
    use_pos: int = 0
    use_pred_pos: int = 0


def bc_loss(logits: list[torch.Tensor], labels: torch.Tensor, valid: torch.Tensor,
            head_weights: tuple[float, ...], use_pos_weight: float = 1.0) -> ChunkLoss:
    """logits: per head (T, B, n_h); labels (T, B, H) long; valid (T, B) bool.
    Padded steps (valid False) contribute nothing to the loss, its gradient, or the
    metrics: they are removed by boolean indexing before any reduction.
    `use_pos_weight` != 1: in the USE head's cross-entropy, press-label steps count
    that many times (still divided by the valid-step count); 1 = the plain loss."""
    t, b = valid.shape
    assert labels.shape == (t, b, len(logits)), f"labels expected (T={t}, B={b}, {len(logits)})"
    assert len(head_weights) == len(logits), f"{len(head_weights)} head weights for {len(logits)} heads"
    n_valid = int(valid.sum())
    denom = max(n_valid, 1)
    total = logits[0].new_zeros(())
    loss_sums, correct, ent_sums = [], [], []
    ent = FactorizedCategorical(logits).entropy_per_head()            # (T, B, H)
    for h, lg in enumerate(logits):
        assert lg.shape[:2] == (t, b), f"head {h} logits expected (T={t}, B={b}, n)"
        lv, yv = lg[valid], labels[..., h][valid]                     # (n_valid, n_h), (n_valid,)
        ce = F.cross_entropy(lv, yv, reduction="sum")
        loss_ce = ce
        if h == USE_HEAD_INDEX and use_pos_weight != 1.0:
            w = torch.where(yv == USE_PRESS_LABEL, use_pos_weight, 1.0).to(lv.dtype)
            loss_ce = (F.cross_entropy(lv, yv, reduction="none") * w).sum()
        if head_weights[h] != 0.0:
            total = total + head_weights[h] * loss_ce / denom
        loss_sums.append(float(ce.detach()))
        correct.append(int((lv.detach().argmax(-1) == yv).sum()))
        ent_sums.append(float(ent[..., h][valid].detach().sum()))
    out = ChunkLoss(total, loss_sums, correct, ent_sums, n_valid)
    if len(logits) > USE_HEAD_INDEX:
        pred = logits[USE_HEAD_INDEX].detach()[valid].argmax(-1) == USE_PRESS_LABEL
        lab = labels[..., USE_HEAD_INDEX][valid] == USE_PRESS_LABEL
        out.use_tp, out.use_pos, out.use_pred_pos = int((pred & lab).sum()), int(lab.sum()), int(pred.sum())
    return out


# ---------------------------------------------------------------- prev-action dropout
def drop_prev_action(prev: torch.Tensor, p: float, generator: torch.Generator) -> torch.Tensor:
    """prev (T, B, H) long -> same shape, where each (t, b) row is replaced with
    probability p by the reset value (NO_PREV_ACTION_VALUE in every head). The draw is
    made on CPU from `generator` (T x B uniforms per call) and moved to prev's device."""
    assert prev.dim() == 3, f"prev_action expected (T, B, H), got {tuple(prev.shape)}"
    t, b, _ = prev.shape
    drop = (torch.rand((t, b), generator=generator) < p).to(prev.device)
    out = torch.where(drop.unsqueeze(-1), torch.full_like(prev, NO_PREV_ACTION_VALUE), prev)
    assert out.shape == prev.shape
    return out


# ---------------------------------------------------------------- epochs
class Accumulator:
    def __init__(self, n_heads: int) -> None:
        self.loss = [0.0] * n_heads
        self.correct = [0] * n_heads
        self.ent = [0.0] * n_heads
        self.steps = 0
        self.weighted_loss_sum = 0.0
        self.chunks = 0
        self.use_tp = self.use_pos = self.use_pred_pos = 0

    def add(self, c: ChunkLoss) -> None:
        for h in range(len(self.loss)):
            self.loss[h] += c.head_loss_sum[h]
            self.correct[h] += c.head_correct[h]
            self.ent[h] += c.head_entropy_sum[h]
        self.steps += c.n_valid
        self.weighted_loss_sum += float(c.total.detach())
        self.chunks += 1
        self.use_tp += c.use_tp
        self.use_pos += c.use_pos
        self.use_pred_pos += c.use_pred_pos

    def metrics(self, prefix: str, names: list[str]) -> dict[str, float | None]:
        n = max(self.steps, 1)
        out: dict[str, float | None] = {f"{prefix}_steps": float(self.steps),
                                 f"{prefix}_loss": self.weighted_loss_sum / max(self.chunks, 1)}
        for h, name in enumerate(names):
            out[f"{prefix}_loss_{name}"] = self.loss[h] / n
            out[f"{prefix}_acc_{name}"] = self.correct[h] / n
            out[f"{prefix}_entropy_{name}"] = self.ent[h] / n
        # USE press recall / precision (NaN when there is no label / predicted press)
        out[f"{prefix}_use_recall"] = self.use_tp / self.use_pos if self.use_pos else None
        out[f"{prefix}_use_precision"] = self.use_tp / self.use_pred_pos if self.use_pred_pos else None
        out[f"{prefix}_use_label_presses"] = float(self.use_pos)
        return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def run_epoch(model: A0Agent, stream: ChunkStream, head_weights: tuple[float, ...],
              optimizer: torch.optim.Optimizer | None, max_grad_norm: float,
              should_stop: Callable[[], bool] = lambda: False,
              use_pos_weight: float = 1.0, prev_action_dropout: float = 0.0,
              dropout_generator: torch.Generator | None = None) -> tuple[Accumulator, bool]:
    """One pass over `stream`. Trains when `optimizer` is given, else evaluates under
    no_grad. Hidden state carried across chunks, detached. Returns (metrics, stopped).
    `prev_action_dropout` > 0 applies drop_prev_action (with `dropout_generator`) to
    training chunks only; an evaluation pass never drops."""
    acc = Accumulator(model.spec.n_heads)
    device = next(model.parameters()).device
    h = model.initial_state(stream.b, device)
    train = optimizer is not None
    dropping = train and prev_action_dropout > 0.0
    if dropping and dropout_generator is None:
        raise ValueError("prev_action_dropout > 0 needs a dropout_generator")
    model.train(train)
    for chunk in stream:
        if should_stop():
            return acc, True
        obs = chunk.obs
        if dropping:
            assert dropout_generator is not None
            obs = {**obs, PREV_ACTION_KEY: drop_prev_action(obs[PREV_ACTION_KEY], prev_action_dropout,
                                                            dropout_generator)}
        with torch.set_grad_enabled(train):
            out, h_last = model.forward_sequence(obs, h, chunk.episode_start)
            c = bc_loss(out.logits, chunk.labels, chunk.valid, head_weights, use_pos_weight)
        if train and c.n_valid > 0 and c.total.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            c.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        h = h_last.detach()
        acc.add(c)
    return acc, False


# ---------------------------------------------------------------- training
def constant_label_heads(episodes: list[Episode], n_heads: int) -> dict[int, int]:
    """{head: the single label value} for heads whose label never varies."""
    out = {}
    for h in range(n_heads):
        vals: set[int] = set()
        for ep in episodes:
            vals.update(int(v) for v in set(ep.labels[:, h].tolist()))
            if len(vals) > 1:
                break
        if len(vals) == 1:
            out[h] = vals.pop()
    return out


def index_sha256(data_dir: str) -> str:
    with open(os.path.join(data_dir, INDEX_NAME), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_init_bc(path: str, spec: ObsSpec, model_cfg: ModelConfig, signature: tuple
                 ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """(model state, its BC record) of a previous BC model, for --init-from-bc (DAgger
    rounds). Refuses a different sensor signature, model config or obs spec, except an
    obs spec with extra egomap channels appended (and a signature differing only by a
    higher obs_version): the state is then grown with zero input slices
    (train/egomap_growth.py) and the returned record carries INIT_GROWTH_KEY."""
    from .checkpoint import WEIGHTS_ONLY_KEY, CheckpointMismatch, canonical_json, load_checkpoint
    from .egomap_growth import egomap_growth, grow_state, obs_version_change, state_shapes
    blob = load_checkpoint(path)
    run_sig = canonical_json(signature)
    version_change = None
    if blob["sensor_signature_json"] != run_sig:
        version_change = obs_version_change(blob["sensor_signature_json"], run_sig)
        if version_change is None:
            raise CheckpointMismatch(f"--init-from-bc {path!r}: sensor signature differs")
    if blob["model_config"] != model_cfg:
        raise CheckpointMismatch(f"--init-from-bc {path!r}: model config differs")
    rec = json.loads(blob[WEIGHTS_ONLY_KEY]) if WEIGHTS_ONLY_KEY in blob else {}
    state = blob["trainer"]["model"]
    if blob["obs_spec_json"] != canonical_json(dataclasses.asdict(spec)):
        growth = egomap_growth(blob["obs_spec_json"], spec)
        if growth is None:
            raise CheckpointMismatch(f"--init-from-bc {path!r}: observation spec differs")
        state = grow_state(state, growth, state_shapes(build_agent(spec, model_cfg, seed=0)))
        grec = growth.record()
        if version_change is not None:
            grec["obs_version_from"], grec["obs_version_to"] = version_change
        rec[INIT_GROWTH_KEY] = grec
        print(f"--init-from-bc GROWTH: {growth.describe()}; obs_version {version_change or 'unchanged'}")
    elif version_change is not None:
        raise CheckpointMismatch(f"--init-from-bc {path!r}: obs_version differs {version_change} "
                                 f"but the observation spec is the same; refusing")
    return state, rec


def train_bc(train_eps: list[Episode], val_eps: list[Episode], spec: ObsSpec, model_cfg: ModelConfig,
             cfg: BCConfig, device: str, out_dir: str, signature: tuple, record: dict[str, Any],
             log: Callable[[str], None] = print,
             init_state: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    """Train, write bc_model.pt (after every epoch), bc_metrics.jsonl and
    bc_summary.json into `out_dir`; returns the summary. `init_state` (strict) starts
    from earlier weights instead of a fresh initialisation."""
    if not train_eps:
        raise ValueError("no training episodes")
    names = head_names(spec.n_heads)
    weights = tuple(float(w) for w in cfg.head_weights)
    if len(weights) != spec.n_heads:
        raise ValueError(f"--head-weights has {len(weights)} values for {spec.n_heads} heads")
    for h, w in enumerate(weights):
        if w == 0.0:
            log(f"head {names[h]}: loss weight 0 (not cloned; its entropy is left for PPO)")
    const = constant_label_heads(train_eps, spec.n_heads)
    for h, v in const.items():
        note = "weight 0, fine" if weights[h] == 0.0 else "WARNING: weight > 0 will crush its entropy"
        log(f"head {names[h]}: teacher label constant ({v}) in the train split ({note})")
    log(VALUE_HEAD_NOTE)
    log(f"prev_action dropout P = {cfg.prev_action_dropout} (training chunks only"
        f"{'; off' if cfg.prev_action_dropout == 0.0 else ''})")

    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    model = build_agent(spec, model_cfg, seed=cfg.seed)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    drop_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + cfg.prev_action_dropout_seed_offset)
    train_stream = ChunkStream(train_eps, cfg.seq_len, cfg.batch_seqs, gen, shuffle=True, device=dev)
    val_stream = (ChunkStream(val_eps, cfg.seq_len, cfg.batch_seqs, gen, shuffle=False, device=dev)
                  if val_eps else None)
    run_cfg = dataclasses.replace(RunConfig(), seed=cfg.seed, maps="generated")
    t0 = time.monotonic()
    deadline = t0 + cfg.max_seconds if math.isfinite(cfg.max_seconds) else float("inf")
    should_stop = lambda: time.monotonic() >= deadline  # noqa: E731
    history: list[dict[str, Any]] = []
    stopped = False
    model_path = os.path.join(out_dir, MODEL_NAME)
    with open(os.path.join(out_dir, METRICS_NAME), "a", encoding="utf-8") as mf:
        for epoch in range(cfg.epochs):
            tr, stopped = run_epoch(model, train_stream, weights, opt, cfg.max_grad_norm, should_stop,
                                    cfg.use_pos_weight, cfg.prev_action_dropout, drop_gen)
            row: dict[str, Any] = {"epoch": epoch + 1, "partial": stopped, "seconds": time.monotonic() - t0,
                                   "prev_action_dropout": cfg.prev_action_dropout,
                                   **tr.metrics("train", names)}
            if val_stream is not None:
                va, _ = run_epoch(model, val_stream, weights, None, cfg.max_grad_norm,
                                  use_pos_weight=cfg.use_pos_weight)
                row.update(va.metrics("val", names))
            history.append(row)
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            log(f"epoch {epoch + 1}{' (stopped early)' if stopped else ''}: " + " ".join(
                f"{n} acc {row['train_acc_' + n]:.3f}/{row.get('val_acc_' + n, float('nan')):.3f}"
                for n in names) + f"  loss {row['train_loss']:.4f}  (train/val)  use recall/precision "
                f"{_fmt(row.get('train_use_recall'))}/{_fmt(row.get('train_use_precision'))} train, "
                f"{_fmt(row.get('val_use_recall'))}/{_fmt(row.get('val_use_precision'))} val")
            rec = {**record, "epochs_done": epoch + 1, "stopped_early": stopped, "last_epoch_metrics": row}
            save_weights_checkpoint(model_path, model, model_cfg, spec, run_cfg, signature, rec)
            if stopped:
                break
    if not history:   # epochs == 0: still write the (untrained) model so the path exists
        save_weights_checkpoint(model_path, model, model_cfg, spec, run_cfg, signature,
                                {**record, "epochs_done": 0, "stopped_early": False})
    summary = {"model": model_path, "epochs_done": len(history), "stopped_early": stopped,
               "seconds": time.monotonic() - t0, "head_names": names, "head_weights": list(weights),
               "constant_label_heads": {names[h]: v for h, v in const.items()},
               "value_head": VALUE_HEAD_NOTE, "sensor_signature_source": SIGNATURE_SOURCE,
               "use_pos_weight": cfg.use_pos_weight, "prev_action_dropout": cfg.prev_action_dropout,
               "final_use_recall_precision": ({k: history[-1].get(k) for k in (
                   "train_use_recall", "train_use_precision", "val_use_recall", "val_use_precision")}
                   if history else None),
               "final": history[-1] if history else None, **record}
    with open(os.path.join(out_dir, SUMMARY_NAME), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    return summary


# ---------------------------------------------------------------- CLI
def parse_head_weights(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(x) for x in text.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--head-weights must be comma-separated numbers: {text!r}") from e


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = BCConfig()
    a = argparse.ArgumentParser(prog="python -m train.bc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--data", required=True, action="append",
                   help="dataset dir with index.jsonl and ep_*.npz; repeat (or comma-separate) to "
                        "train on the union of several dirs (DAgger rounds)")
    a.add_argument("--init-from-bc", default=None, metavar="PATH",
                   help="start from this BC model's weights; its BC-val map ids are kept as the "
                        "val split (so maps held out of cloning stay held out)")
    a.add_argument("--out", required=True, help="run dir for bc_model.pt and logs")
    a.add_argument("--epochs", type=int, default=d.epochs)
    a.add_argument("--seq-len", type=int, default=d.seq_len)
    a.add_argument("--batch-seqs", type=int, default=d.batch_seqs)
    a.add_argument("--lr", type=float, default=d.lr)
    a.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    a.add_argument("--seed", type=int, default=d.seed)
    a.add_argument("--val-frac", type=float, default=d.val_frac)
    a.add_argument("--max-seconds", type=float, default=d.max_seconds)
    a.add_argument("--head-weights", type=parse_head_weights, default=d.head_weights,
                   help="per-head loss weights, order move,strafe,turn,attack,use,speed; default "
                        + ",".join(str(w) for w in d.head_weights))
    a.add_argument("--torch-threads", type=int, default=d.torch_threads)
    a.add_argument("--max-dataset-gb", type=float, default=d.max_dataset_bytes / GIB)
    a.add_argument("--use-pos-weight", type=float, default=d.use_pos_weight,
                   help="USE head: weight of press-label steps in its cross-entropy; default 1 (unweighted)")
    a.add_argument("--prev-action-dropout", type=float, default=d.prev_action_dropout, metavar="P",
                   help="training only: per step, with probability P replace the prev_action input by "
                        "the episode-start value (no previous action); default 0 (off)")
    return a.parse_args(argv)


def config_from_args(ns: argparse.Namespace) -> BCConfig:
    return BCConfig(epochs=ns.epochs, seq_len=ns.seq_len, batch_seqs=ns.batch_seqs, lr=ns.lr, seed=ns.seed,
                    val_frac=ns.val_frac, head_weights=tuple(ns.head_weights), max_seconds=ns.max_seconds,
                    torch_threads=ns.torch_threads, max_dataset_bytes=int(ns.max_dataset_gb * GIB),
                    use_pos_weight=ns.use_pos_weight, prev_action_dropout=ns.prev_action_dropout)


def data_dirs_from_args(values: list[str] | str) -> list[str]:
    """Repeatable and/or comma-separated --data values -> list of dirs (order kept, no repeats)."""
    vals = [values] if isinstance(values, str) else list(values)
    out: list[str] = []
    for v in vals:
        for d in v.split(","):
            d = d.strip()
            if d and d not in out:
                out.append(d)
    if not out:
        raise ValueError("no --data dir given")
    return out


def prepare(data_dir: str | list[str], spec: ObsSpec, cfg: BCConfig, log: Callable[[str], None] = print,
            val_map_ids: list[str] | None = None) -> tuple[list[Episode], list[Episode], dict[str, Any]]:
    """Load, validate and split; returns (train, val, record for the checkpoint).
    `data_dir` may be several dirs (their union is used). `val_map_ids` given (from
    --init-from-bc): exactly those map ids are val, whatever maps the data holds."""
    dirs = data_dirs_from_args(data_dir)
    eps: list[Episode] = []
    budget = cfg.max_dataset_bytes
    for d in dirs:
        part = load_episodes(d, spec, budget)
        budget -= sum(e.length for e in part) * step_bytes(spec)
        eps.extend(part)
    if val_map_ids is None:
        train_ids, val_ids = split_map_ids([e.map_id for e in eps], cfg.val_frac, cfg.seed)
    else:
        val_ids = sorted(set(val_map_ids))
        train_ids = sorted({e.map_id for e in eps} - set(val_ids))
    vs = set(val_ids)
    train = [e for e in eps if e.map_id not in vs]
    val = [e for e in eps if e.map_id in vs]
    log(f"dataset {dirs}: {len(eps)} episodes, {sum(e.length for e in eps)} steps; train {len(train)} "
        f"episodes on {len(train_ids)} maps, val {len(val)} episodes on {len(val_ids)} held-out maps {val_ids}")
    record = {"kind": "behaviour_cloning", "data_dir": os.path.abspath(dirs[0]),
              "data_dirs": [os.path.abspath(d) for d in dirs],
              "index_sha256": index_sha256(dirs[0]),
              "index_sha256_per_dir": [index_sha256(d) for d in dirs],
              "train_map_ids": train_ids, "val_map_ids": val_ids,
              "n_train_episodes": len(train), "n_val_episodes": len(val),
              "n_train_steps": sum(e.length for e in train), "n_val_steps": sum(e.length for e in val),
              "bc_config": {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                            for k, v in dataclasses.asdict(cfg).items()}, "value_head": VALUE_HEAD_NOTE,
              "sensor_signature_assumed": True, "sensor_signature_source": SIGNATURE_SOURCE}
    return train, val, record


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    cfg = config_from_args(ns)
    if cfg.below_normal_priority:
        from . import process_control as PC
        PC.set_below_normal()
    torch.set_num_threads(cfg.torch_threads)
    from .device import DeviceUnavailable, resolve_device
    try:
        device = resolve_device(ns.device)
    except DeviceUnavailable as e:
        print(str(e), file=sys.stderr)
        return 6
    from .bc_spec import a0_training_spec_and_signature
    spec, signature = a0_training_spec_and_signature()
    init_state, val_ids = None, None
    if ns.init_from_bc:
        init_state, init_rec = read_init_bc(ns.init_from_bc, spec, ModelConfig(), signature)
        val_ids = init_rec.get("val_map_ids")
        if val_ids is None:
            print(f"--init-from-bc {ns.init_from_bc}: no val_map_ids in its record; splitting afresh",
                  file=sys.stderr)
    train, val, record = prepare(ns.data, spec, cfg, val_map_ids=val_ids)
    record["argv"] = list(sys.argv[1:] if argv is None else argv)
    record["device"] = device
    if ns.init_from_bc:
        from .init_from import sha256_file
        record["init_from_bc"] = os.path.abspath(ns.init_from_bc)
        record["init_from_bc_sha256"] = sha256_file(ns.init_from_bc)
        if INIT_GROWTH_KEY in init_rec:
            record["init_from_bc_growth"] = init_rec[INIT_GROWTH_KEY]
    summary = train_bc(train, val, spec, ModelConfig(), cfg, device, ns.out, signature, record,
                       init_state=init_state)
    print(f"wrote {summary['model']} after {summary['epochs_done']} epochs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
