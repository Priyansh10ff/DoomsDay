"""Scripted teachers for imitation learning (TRAINING-TIME ONLY).

Team decision 2026-11-25 (CLAUDE.md section 7, "Plan change"): a scripted teacher is
behaviour-cloned, then the student is fine-tuned with our own PPO. Everything in this
package may use privileged data (absolute pose from the env `info` dict, the map's own
WAD geometry and generator record) and classical planning (grid search, trigonometry,
if/else state machines). That is allowed ONLY because none of it is in the decision
path of the final agent:

  * nothing under agent/, env/, evaluate/ (or any inference code) imports `teachers`
    (tests/test_teachers_isolation.py checks this with an AST scan);
  * the dataset stores the env observation dict exactly as the env returned it; the
    privileged data the teacher reads is never written into an observation.

This __init__ imports nothing, so importing a submodule does not start the engine.
"""
