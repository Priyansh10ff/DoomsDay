"""Exploration signals (ADR 0005, ADR 0010). TRAINING ONLY.

  coverage.py        E1: episodic coverage bonus from privileged position (built).
  schedule.py        the annealed coefficient.
  learned_novelty.py E2: learned episodic novelty in embedding space. NOT BUILT; a
                     marked placeholder until Stage A1 learns (ADR 0005).
The Random Network Distillation bonus (Burda et al. 2018) lives in train/rnd.py, not
here: it imports torch, and this package must stay torch-free (env workers import it).
ICM's curiosity reward is not used (ADR 0005); only its inverse-dynamics idea is used,
as an auxiliary loss, in train/inverse_dynamics.py. Nothing here is imported by `agent/`
or `env/`, runs at inference, or writes to an observation.
"""
