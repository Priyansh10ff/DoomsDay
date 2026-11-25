"""Evaluation / inference entry point (`python -m evaluate.run`).

Plays episodes on the EVAL configuration (vanilla rules, no shaping, no curriculum,
no cheats) with one engine instance at a time. Every action is the network's output
through `agent.inference.PolicyRunner` (or, for the baseline only, a uniform random
draw). Map ids, stages and seeds go to the log only, never to the model.
"""
