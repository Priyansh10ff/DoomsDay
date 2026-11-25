"""Extrinsic reward shaping (ADR 0010). TRAINING ONLY.

Nothing here is imported by `agent/` or `env/`, nothing here runs at inference, and
nothing here writes to an observation. Rewards read the privileged `info` dict; the
policy never sees it. No torch and no vizdoom import (env workers import this).
"""
