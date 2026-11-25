"""Exploration signals (ADR 0005, ADR 0010). TRAINING ONLY.

  coverage.py        E1: episodic coverage bonus from privileged position (built).
  schedule.py        the annealed coefficient.
  learned_novelty.py E2: learned episodic novelty in embedding space. NOT BUILT; a
                     marked placeholder until Stage A1 learns (ADR 0005).
ICM is not used (ADR 0005). Nothing here is imported by `agent/` or `env/`, runs at
inference, or writes to an observation.
"""
