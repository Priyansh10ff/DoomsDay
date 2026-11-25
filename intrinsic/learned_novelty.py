"""E2: learned episodic novelty in embedding space. PLACEHOLDER, NOT BUILT.

ADR 0005 schedules this module for "once Stage A1 learns"; ADR 0010 does not build
it. This file marks where it goes and what it must respect, so nobody mistakes the
coverage bonus (E1, intrinsic/coverage.py) for the submitted curiosity module.

What the later module must do (ADR 0005):
  * our own code: an elliptical or nearest-neighbour bonus over features trained
    with an inverse-dynamics loss; memory emptied at every episode start;
  * features must exclude anything the agent changes just by firing (ammo count,
    own projectiles), to avoid the ICM firing farm;
  * its own value head (`value_int` in agent/model.py) and discount; coefficient
    annealed; a vanilla-reward-only evaluation always run;
  * NOT ICM.

Where it plugs in: unlike E1 it needs torch and the policy's features, so it cannot
live in the env-worker wrapper (workers never import torch). It belongs on the
trainer side, producing a (T, N) intrinsic reward per rollout, reusing
`intrinsic.schedule.LinearAnneal` and the bounded-per-episode convention of E1 so
the two are comparable in the ADR 0005 ablation. That integration is `worker`'s.
"""
from __future__ import annotations


class LearnedEpisodicNovelty:
    """Not implemented. See the module docstring and ADR 0005."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("E2 (learned episodic novelty) is not built yet: ADR 0005 "
                                  "schedules it after Stage A1; use intrinsic.coverage (E1)")
