# Hard Constraints Checklist

Source: `docs/TRACK_BRIEF.md` (section 5, Ground Rules, plus sections 2 and 3). Breaking any of these can disqualify the entry. Check every item before proposing or merging agent, environment, or inference code. If a task would break one, stop and say so.

## Decision path (inference)

- [ ] The final agent decides **only via trained neural parameters**: every action comes from the network's forward pass.
- [ ] **No hardcoded if/else decision trees** choosing actions.
- [ ] **No A\*, Dijkstra, or any other classical pathfinding.**
- [ ] **No atan2 / trigonometry aimbots** in the inference path.

## Engine and evaluation

- [ ] Engine, ACS map-script, and game-variable modifications are used **ONLY in training** (pretraining, curriculum).
- [ ] **Evaluation runs on the unmodified engine with vanilla rules**: nothing the agent needs at inference depends on a mod.

## Original work

- [ ] The **core agent architecture and training pipeline are written during the hackathon**. Pre-existing libraries are allowed; pre-existing agent or training code is not.

## Observations

- [ ] Observations **must not let the agent memorize absolute map coordinates**.
- [ ] **No god-view radar**: partial observability only (what is in the player's field of view and local sensory range).

## Judging

- [ ] Judging is **zero-shot on unseen maps**: never tune or validate against a single map only.
