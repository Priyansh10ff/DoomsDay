"""Running return scale for value-target normalization (ADR 0004).

Targets are divided by the running standard deviation of the GAE returns seen so
far (no mean shift), floored at `min_scale` (`PPOConfig.return_scale_min`, 0.1 per
ADR 0008 point 3, which replaced ADR 0007 ruling 2's 1.0).
"""
from __future__ import annotations

import torch


class ReturnScale:
    def __init__(self, eps: float, min_scale: float) -> None:
        self.eps, self.min_scale = eps, min_scale
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().double().flatten()
        n = float(x.numel())
        if n == 0:
            return
        b_mean = float(x.mean())
        b_m2 = float(((x - b_mean) ** 2).sum())
        tot = self.count + n
        delta = b_mean - self.mean
        self.mean += delta * n / tot
        self.m2 += b_m2 + delta * delta * self.count * n / tot
        self.count = tot

    @property
    def scale(self) -> float:
        if self.count < 2:
            return 1.0
        var = self.m2 / self.count
        return max(float(var) ** 0.5 + self.eps, self.min_scale)

    def state_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, d: dict[str, float]) -> None:
        self.count, self.mean, self.m2 = d["count"], d["mean"], d["m2"]
