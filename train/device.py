"""Torch device selection for `--device cpu|cuda|auto|hybrid`.

  cpu     today's behaviour, exactly.
  cuda    CUDA or nothing: refused with a clear message when torch has no usable CUDA;
          never a silent fallback to CPU.
  auto    CUDA when available, otherwise CPU (the user asked for either).
  hybrid  rollout inference on CPU, PPO update on CUDA (refused without CUDA, like
          `cuda`). The trainer keeps a CPU actor copy of the CUDA learner and copies
          the learner's weights into it after every update and every load
          (train/trainer.py). Measured 2026-11-25 at 8x6: CPU sampling is faster at
          batch 48, CUDA updates 2x faster.

With cpu / cuda one model on that device does both rollout inference and the PPO
update; the samplers' torch.Generators stay on CPU (agent.distributions moves the
probabilities to the generator's device).
"""
from __future__ import annotations

from collections.abc import Callable

import torch

DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "auto", "hybrid")
HYBRID = "hybrid"
# resolved RunConfig.device -> (rollout inference device, PPO update device)
DEVICE_ROLES: dict[str, tuple[str, str]] = {"cpu": ("cpu", "cpu"), "cuda": ("cuda", "cuda"),
                                            HYBRID: ("cpu", "cuda")}


def device_roles(resolved: str) -> tuple[str, str]:
    """(act device, learn device) of a resolved device string."""
    if resolved not in DEVICE_ROLES:
        raise ValueError(f"unresolved or unknown device {resolved!r}; expected one of {sorted(DEVICE_ROLES)}")
    return DEVICE_ROLES[resolved]


class DeviceUnavailable(RuntimeError):
    pass


def resolve_device(choice: str, cuda_available: Callable[[], bool] = torch.cuda.is_available) -> str:
    if choice not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {choice!r}; choices: {DEVICE_CHOICES}")
    if choice == "cpu":
        return "cpu"
    has_cuda = bool(cuda_available())
    if choice in ("cuda", HYBRID) and not has_cuda:
        raise DeviceUnavailable(
            f"--device {choice} was requested but torch.cuda.is_available() is False (torch "
            f"{torch.__version__}, built with CUDA {torch.version.cuda}). Install a CUDA build of "
            f"torch or use --device cpu; there is no automatic fallback.")
    if choice == HYBRID:
        return HYBRID
    return "cuda" if has_cuda else "cpu"
