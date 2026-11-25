# OSC: Curiosity and exploration implementations (ICM, RND, episodic curiosity)

Scouted 2026-11-25 by osc-scout. Facts come from web reads (GitHub API, README pages, raw files) unless tagged "general knowledge, not verified in this run". That tag covers algorithm descriptions taken from the papers, not from the repos. **No clones.** "Last push" is the GitHub API `pushed_at`. Unconfirmed values are marked "not verified".

## Recommendation (for the exploration-mechanism ADR)

- **Nothing here is an IMPORT_AS_DEPENDENCY.** Every curiosity module is part of the agent and training pipeline, which the brief treats as an innovation area, so we must write our own. All finds are READ_FOR_REFERENCE, or AVOID where they are dead or unlicensed.
- **Most useful reading, in order:**
  1. CleanRL `ppo_rnd_envpool.py`: PyTorch, single file. Gives the exact mechanics of RND with a two-headed critic.
  2. E3B (episodic elliptical bonus). Its README lists **ViZDoom experiments**. The licence is CC-BY-NC, so read the paper only.
  3. Sample Factory's landmark-novelty wrapper: the simplest working Doom episodic bonus, and a good baseline to compare against.
  4. noreward-rl (ICM), which has Doom sparse-reward experiments.
  5. episodic-curiosity (reachability network): DMLab only, TF.
- **Design notes the ADR should weigh:**
  - Global novelty (RND/ICM) fades over training. Episodic novelty (E3B, landmarks, reachability) resets each episode and fits "explore unfamiliar rooms in an unseen map" better. NovelD-style hybrids combine both (general knowledge, not verified in this run).
  - An intrinsic reward may use privileged training-time information such as absolute position, because it is not in the inference decision path. The policy *observation* must not.
  - Rule 4.2: intrinsic rewards are training-only, so they are naturally switchable off at inference.

---

## 1. CleanRL: `ppo_rnd_envpool.py` (RND + PPO)

- **URL:** https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
- **Stars:** 10716 (repo)
- **Last commit:** 2026-04-20 (repo)
- **License:** MIT (repo LICENSE file). This file has no separate notice listed in the LICENSE.
- **What is reusable (technique, as read from the file):**
  - A frozen random target network vs a trained predictor. The intrinsic reward is half the squared feature error.
  - The predictor trains on a random 25% of each minibatch (`update_proportion`).
  - Observation normalisation with `RunningMeanStd`, warmed up with random-action rollouts over 50 iterations and clipped to [-5, 5].
  - Intrinsic reward normalised by the std of a discounted running sum (`RewardForwardFilter`), with no mean subtraction.
  - **Separate extrinsic and intrinsic value heads** and separate GAE. `gamma=0.999` (extrinsic), `int_gamma=0.99`. Advantages are combined as `ext_coef(2.0)*ext + int_coef(1.0)*int`. The extrinsic value loss is clipped; the intrinsic one is not.
  - The argparse help strings for `int_coef`/`ext_coef` are swapped relative to how the code uses them. Trust the code.
  - Whether the intrinsic return ignores episode boundaries (non-episodic, as in the RND paper) was not confirmed in this run.
- **Rule risk:** It is a complete PPO pipeline, so copying it violates the originality rule. Reading the normalisation details and re-implementing them is fine.
- **Label: READ_FOR_REFERENCE.**

## 2. OpenAI `random-network-distillation` (original RND)

- **URL:** https://github.com/openai/random-network-distillation
- **Stars:** 930
- **Last push:** 2020-10-01. The repo is **archived**.
- **License:** none declared (GitHub reports `license: null`). Treat it as all rights reserved.
- **What is reusable:** It is the paper companion (TensorFlow, per general knowledge, not verified in this run). The CleanRL file covers the same algorithm in PyTorch with a clear licence.
- **Rule risk:** No licence, so do not copy. It is stale.
- **Label: AVOID** (superseded by the CleanRL reference; licence unclear). Read the paper instead.

## 3. jcwleo `random-network-distillation-pytorch`

- **URL:** https://github.com/jcwleo/random-network-distillation-pytorch
- **Stars:** 264
- **Last push:** 2019-03-04
- **License:** MIT (GitHub API)
- **What is reusable:** An alternative PyTorch RND. Not inspected.
- **Rule risk:** It is a pipeline; do not copy it. It is stale.
- **Label: AVOID.** It is unmaintained and adds nothing over CleanRL.

## 4. pathak22 `noreward-rl` (ICM, "Curiosity-driven Exploration by Self-supervised Prediction")

- **URL:** https://github.com/pathak22/noreward-rl
- **Stars:** 1490
- **Last push:** 2022-12-07
- **License:** The LICENSE file holds two permissive licences: BSD-3-Clause (Copyright 2017 Deepak Pathak) and MIT (Copyright 2016 OpenAI). No copyleft.
- **What is reusable (technique):**
  - **Intrinsic Curiosity Module (general knowledge, not verified in this run):**
    - An inverse model predicts the action from (φ(s), φ(s')). This trains features that ignore what the agent cannot control.
    - A forward model predicts φ(s') from (φ(s), a), and its error is the curiosity reward.
  - **Repo facts:**
    - The README says the code is TensorFlow-based.
    - Doom runs use `--env-id doom`, `doomSparse` and `doomVerySparse`, with a customised Doom setup in `doomFiles/`. Which ViZDoom map they use (the paper's MyWayHome) is not verified in the repo.
    - The file containing the ICM model is **not verified**. The README lists `src/train.py`, `demo.py`, `inference.py`, and `constants.py` with `PREDICTION_BETA`/`ENTROPY_BETA`.
- **Rule risk:** Old TensorFlow stack, so there is nothing to import. Copying would breach originality anyway.
- **Label: READ_FOR_REFERENCE** (paper companion: loss weighting and Doom setup).

## 5. google-research `episodic-curiosity` (reachability-based episodic novelty)

- **URL:** https://github.com/google-research/episodic-curiosity
- **Stars:** 205
- **Last push:** 2020-10-02. The repo is **archived**.
- **License:** Apache-2.0
- **What is reusable (technique):**
  - A reachability (R) network trained to tell whether two observations are within k steps of each other. The repo has `episodic_curiosity/train_r.py` and `generate_r_training_data.py`.
  - An episodic memory of embeddings. The bonus is given when the current observation is not reachable from memory. This is the paper's description (general knowledge, not verified in this run); the memory-buffer file is not verified.
  - It maps well onto "remember rooms already visited" without coordinates.
- **Rule risk:** TF/Keras plus OpenAI baselines PPO2. Experiments use patched DeepMind Lab and optional MuJoCo; **no ViZDoom**. Nothing is importable.
- **Label: READ_FOR_REFERENCE.**

## 6. facebookresearch `e3b` (Exploration via Elliptical Episodic Bonuses)

- **URL:** https://github.com/facebookresearch/e3b
- **Stars:** 88
- **Last push:** 2024-03-22. The repo is **archived**.
- **License:** the README says "The majority of this project is licensed under CC-BY-NC". NovelD parts are Apache-2.0 and the Sample Factory implementation of E3B is MIT. **CC-BY-NC is non-commercial and not designed for code reuse. Do not copy anything.**
- **What is reusable (technique):**
  - **Repo facts:** environments are MiniHack, **VizDoom** and Habitat. VizDoom and MiniHack runs live in `minihack/torchbeast` (IMPALA). There are also `minihack/sample_factory` (APPO) and `habitat-lab` (DD-PPO). The file implementing the bonus is not verified.
  - **Method (general knowledge, not verified in this run):**
    - The bonus is b = φ(s)ᵀ C⁻¹ φ(s), with C the per-episode covariance of embeddings.
    - φ is trained with an inverse-dynamics loss.
    - C⁻¹ is updated cheaply each step.
  - This makes E3B a strong concept for our exploration ADR: episodic, coordinate-free, and tested on ViZDoom.
- **Rule risk:** Licence (CC-BY-NC) and originality both block copying. Implementing the idea from the paper is fine.
- **Label: READ_FOR_REFERENCE** (paper and results only; do not copy code).

## 7. tianjunz `NovelD`

- **URL:** https://github.com/tianjunz/NovelD
- **Stars:** 40
- **Last push:** 2021-11-23
- **License:** GitHub reports `NOASSERTION`. The E3B README says NovelD code is Apache-2.0; this was **not verified** in the NovelD repo.
- **What is reusable (general knowledge, not verified in this run):**
  - Bonus = max(novelty(s') − α·novelty(s), 0), gated to the first visit of a state in the episode.
  - Novelty is the RND error, so the bonus combines global and episodic novelty.
  - The repo itself was not inspected.
- **Rule risk:** Stale. The licence is unclear, so do not copy.
- **Label: READ_FOR_REFERENCE** (paper idea only).

## 8. Sample Factory `doom/wrappers/exploration.py` (landmark novelty)

- **URL:** https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/wrappers/exploration.py
- **Stars / date / license:** 1030; last commit 2026-10-19; MIT.
- **What is reusable (technique, as read from the file):**
  - An episodic landmark set (max 200) of (x, y, angle).
  - A +0.1 intrinsic reward when the player is farther than 75 units (Euclidean plus half the wrapped angle difference) from every landmark. It is written to `info["intrinsic_reward"]`.
  - The set is cleared on reset; when full, a random landmark is dropped.
  - It is a good **baseline** to compare our curiosity module against. It uses absolute coordinates, which is fine for a training-time reward but never as an observation.
- **Rule risk:** Copying breaches originality; re-implementing the idea is trivial.
- **Label: READ_FOR_REFERENCE.**

---

## Sources
- https://github.com/vwxyzjn/cleanrl, https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_rnd_envpool.py
- https://api.github.com/repos/openai/random-network-distillation
- https://api.github.com/repos/jcwleo/random-network-distillation-pytorch
- https://github.com/pathak22/noreward-rl, https://raw.githubusercontent.com/pathak22/noreward-rl/master/LICENSE
- https://github.com/google-research/episodic-curiosity
- https://github.com/facebookresearch/e3b
- https://api.github.com/repos/tianjunz/NovelD
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/wrappers/exploration.py
