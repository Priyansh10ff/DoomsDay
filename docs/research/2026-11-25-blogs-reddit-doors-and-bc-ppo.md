# Doors, BC-to-PPO handoff, wall-pinning, exploration: practitioner evidence

Date: 2026-11-25. Author: `research` subagent. Brief came directly from the coordinator (the ADR/advisor step is paused). The "Top 5" list at the end replaces the usual three recommendations, as the brief asked.

## How sources were read, and limits

- Pages were read through a small extraction model that returned quotes and numbers. I did not read any page end to end myself. Papers were read on ar5iv; VPT, ADVISOR and the forgetting paper are over 100,000 characters and only VPT's tail was read.
- **Reddit could not be read at all.** Two searches restricted to reddit.com / old.reddit.com were refused ("domains are not accessible to our user agent") and one direct fetch of old.reddit.com was refused. I stopped after three attempts, as instructed. No Reddit thread is cited anywhere below. General searches surfaced no Reddit content either.
- Blocked or missing: the Kieliger Medium post (HTTP 403), the MazeExplorer PDF (HTTP 404). Both are marked UNVERIFIED.
- Anything seen only in a search-result summary is marked **UNVERIFIED**.
- No quote is longer than 15 words. No fetched page contained instructions aimed at the agent.
- Local file used as evidence: `docs/reference/engine-probes.md` (our own probes R2.2, R5.1, R5.3).
- Not repeated here: `docs/research/exploration.md` (ICM, RND, EC, NGU, Go-Explore) and `docs/research/vizdoom-agents.md` (Arnold, DFP, Sample Factory). Read those for the paper-level detail.

## 0. The facts that order everything below

From the brief: doors sit near 5% under pure PPO, under BC + DAgger, and (after erosion from 10%) under BC then PPO. The exception is the once-per-door reward, where doors were passed in about 38% and then about 10% of door episodes.

How I read that (my reasoning, not a measured result):

- Three different optimisers landing near the same low number points at something they share: what the student sees, what the labels are, or how the door responds to the action. That is why door mechanics, labels and start-state curriculum are ranked above optimiser fixes.
- The 38% figure shows the student can pass doors at a real rate when the reward points at them, and that the gain did not hold. So the ceiling is not purely mechanical, and the erosion is also evidence for stabilising training (item 4 of the Top 5).
- A better BC-to-PPO handoff alone can only preserve a policy that passes doors 5 to 10% of the time.

## 1. Summary

- **Nobody I could find has published a neural Doom agent that generalises through doors and keys to unseen levels.** The two practitioner projects that finish real levels (doom-player, vizdoom_ppo_rnd) each train and test on the same level. The one transfer test found scored 0 of 15.
- **What made full levels work in those projects was dense progress shaping toward the exit, computed from privileged map data at training time**, not a door reward. Exit reward alone scored 0 of 5, below random.
- **Door mechanics are a concrete, checkable suspect.** The original engine source and our own probe agree: USE acts on the press edge only, and a second press on a moving or open manual door reverses it. A sampled USE head that presses, releases and presses again shuts the door it just opened.
- **For BC then PPO, the published recipe with numbers is: freeze the policy and train only the critic first, then low learning rate with warm-up, and a decaying KL to the frozen BC policy in place of the entropy bonus.** Our "entropy rises, doors erode" symptom matches what these recipes exist to prevent.
- **Wall-pinning and "aim point the student cannot see" both have names in the literature**: the inertia problem (causal confusion) and the imitation gap (teacher uses information the student lacks). The strongest number: imitation of shortest-path demonstrations gave 6.4% success where human demonstrations gave 52 to 64%.
- **Exploration bonuses are not the bottleneck the evidence points at.** The cheapest change with support is a reverse curriculum by spawn placement in our own map generator.

## 2. Evidence table

| Claim | Evidence (numbers + setup) | Source link | Venue / peer-reviewed or not | Relevance to our design | Confidence |
|---|---|---|---|---|---|
| USE fires only on the press edge in the original engine | `P_PlayerThink`: `P_UseLines` is called only `if (!player->usedown)`, then `usedown = true`; cleared only when the button is released. | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_user.c | Original engine source (id Software) | A policy that holds USE while walking up to a door never fires inside range. ViZDoom is ZDoom-based; our probe section 6 ("Held USE fires once") agrees. | high for vanilla source; medium for ZDoom being identical in every case |
| A second player press on a manual door reverses it, also when open and waiting | `EV_VerticalDoor`, specials 1, 26, 27, 28, 117 with an active thinker: closing door goes "back up"; otherwise a player press sets direction -1, "start going down immediately". | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_doors.c | Original engine source | Press, release, press closes the door. Matches our probe R2.2 rows 2 and 3 (gap 32, 64, 32, 0). | high |
| Doors take several decisions to open, and the speed differs between maps | Our probes: generated maps, gap rises 8 units per decision at frame skip 4, to 68-92, so about 9 to 12 decisions (R5.1). freedoom2 MAP01 sector 27: 32, 64, 92, so about 3 decisions (R2.2). DR doors close again after a wait. | `docs/reference/engine-probes.md` R5.1, R2.2 | Own measurement | Any re-press while the door opens, or while walking through, reverses it. Door speed on unseen maps will not match our generated maps; the two probes differ by 4x and the reason was not established here. | high for the readings |
| Sample Factory agents could open doors, but only in self-play deathmatch | Action space extended with "object interaction, which allows the agent to open doors and call elevators"; Fig. 9: "Agent opening a secret door to get a more powerful weapon." 2.5x10^9 frames per agent (from `vizdoom-agents.md`). No door reward or door observation described. | https://ar5iv.labs.arxiv.org/html/2006.11751 | ICML 2020 | Doors emerged from pixels at roughly 170x the frames of our first run (3.71M decisions x skip 4 = 14.8M frames; my arithmetic). No recipe to borrow. | high |
| Full E1M1 clears came from dense route-progress shaping; exit reward alone failed | E1M1, screen pixels, 11 button combinations, SB3 PPO. Shaped (5M steps, about 3 h, 3 seeds): 15/15 clears. Exit reward only (1M steps): 0/5, progress 8.5%; random 0/5, 20%. Shaping "pays for new ground toward the exit and charges for dying" (100 route, 25 death). | https://github.com/lucas-moont/doom-player/pull/31 | One person's GitHub pull request; not peer-reviewed; single map; unequal budgets (5M vs 1M) | Dense distance-to-exit progress, from a privileged distance field at training time (allowed by rule 4.2), is what worked. Trained and tested on the same map. | medium |
| That E1M1 agent did not transfer | E1M1-trained agents on E1M2: 0/15 clears, progress 0.6%, 2.0%, 4.2% (random 2.3%). "No Contender picked up the red key." Policies pushed into walls near the spawn. | https://github.com/lucas-moont/doom-player/pull/32 | Same; not peer-reviewed | Single-map shaping memorises a route. Also shows wall-pushing as the default failure on an unseen map. | medium |
| Progress shaping must route through keys, not through locked doors | Same PR: a doors-open distance rule paid up to 20.1% for ground toward a door that will not open, and nothing for the key detour. Keyed route on E1M2: 9,177 units (2,222 to red key, then 6,955) vs 4,959 doors-open. | https://github.com/lucas-moont/doom-player/pull/32 | Same | For S2 maps: the progress potential must be distance to the next required objective (key, then door, then exit). | medium |
| PPO + RND finished two real levels, one level per training run | Doom 2 MAP01 and Doom E1M2, about 6M steps and about 8 h each, 20 envs, RTX 3080; RGB + labels 80x60 plus a HUD key crop; intrinsic coefficient 0.01; extrinsic advantages weighted double. "More testing is required to get the agent to generalize across further levels." | https://github.com/callumhay/vizdoom_ppo_rnd | One person's GitHub README; not peer-reviewed | Same-level only. Its observation includes x, y position, which rule 4.4 forbids for us. | medium |
| Naive RL from a BC policy dips early; a critic-first handoff ends higher | ObjectNav (Habitat, HM3D val, 20k demos): BC 52.0% success; BC then RL without the critic phase 53.6%; full PIRLNav recipe 61.9%. Random critic "leads to a rapid drop in performance"; appendix: performance "drops significantly early on" and then recovers. With 77k demos: 64.1% to 70.4%. | https://ar5iv.labs.arxiv.org/html/2301.07302 | arXiv 2301.07302; venue not shown on the page (I recall CVPR 2023, UNVERIFIED) | Direct match to our erosion. Note the naive handoff recovered and still ended slightly above BC: our 20-minute erosion may be that early dip, not a final result. Recipe in section 4. | high for numbers; size of the early dip not quantified |
| KL-to-pretrained anchor, replacing the entropy bonus, prevents forgetting | VPT: KL coefficient 0.2, decay 0.9995 per iteration; "this KL divergence loss replaces the common entropy maximization loss"; learning rate 2e-5; value head "initialized with zero weights". Without KL, progress "stalls after 100,000 episodes". | https://ar5iv.labs.arxiv.org/html/2206.11795 | arXiv 2206.11795; venue not checked (I recall NeurIPS 2022, UNVERIFIED) | Simplest fix with published coefficients. Our entropy bonus (0.01) pushes a sharp BC policy toward uniform, which fits "entropy rises" (my inference). | high |
| Same anchor reused by others at smaller scale | DECKARD Table 5: initial KL coefficient 0.1, decay 0.999 per iteration, learning rate 1e-4; they "replace the traditional entropy loss in the PPO algorithm with a KL loss". | https://ar5iv.labs.arxiv.org/html/2301.12050 | arXiv 2301.12050; venue not checked | Second set of coefficients; 1e-4 is nearer our learning rate. | high |
| KL anchor was slightly worse than critic-first in one comparison | PIRLNav, HM3D val, 20k demos: VPT-style KL finetuning 59.7% vs PIRLNav 61.9%. | https://ar5iv.labs.arxiv.org/html/2301.07302 | as above | Both beat the naive handoff; the gap between them is small. Use whichever is quicker to build. | medium (one comparison) |
| Forgetting hits states the finetuned policy does not visit early; KL on visited states cannot protect them | Wolczyk et al.: forgetting when the model "deteriorates on the state subspace of the downstream task not visited" early. NetHack (APPO): kickstarting best (KL 0.5, decay 0.99998 per step), BC loss second (coefficient 2.0, no decay). RoboticSequence (SAC): BC on expert states best (80% all four stages); kickstarting failed there. | https://ar5iv.labs.arxiv.org/html/2402.02868 | arXiv 2402.02868; venue not shown (I recall ICML 2024, UNVERIFIED) | Our student rarely gets past a door, so post-door states are "not visited early". A BC loss on teacher-visited states covers them; a KL on the student's own states does not. | high for the paper's results; the mapping to doors is my inference |
| Imitating a teacher that uses information the student cannot see fails badly | PIRLNav: BC on 240k shortest-path demos gave 6.4% success, attributed to privileged map access; BC on 20k human demos gave 52.0%. | https://ar5iv.labs.arxiv.org/html/2301.07302 | as above | Closest published match to "turn labels quantise an aim point the student cannot see". Our teacher is a pathfinder. | high |
| The imitation gap: a fixed imitation + RL mix did not beat RL alone; an adaptive mix did | ADVISOR: privileged information is "marginalized out" in the student. PointNav success: ADVISOR 88.2, static IL+RL mix 78.2, RL only 79.0. MiniGrid Lava Crossing (once-switch): ADVISOR 0.80, all others at or below 0.04. Weight w = exp(-alpha x KL(teacher, auxiliary imitation-only head)). The authors also argue (not measured in what I read) that IL warm-start then RL can be worse than RL alone. | https://ar5iv.labs.arxiv.org/html/2007.12173 | arXiv 2007.12173; venue not shown (I recall NeurIPS 2021, UNVERIFIED) | Says to imitate only where the student can predict the teacher from its own observation, and use RL elsewhere. | high for numbers (first 100k characters read) |
| Agents that sit still or press into nothing are a known BC artefact (inertia problem) | Driving BC: failure defined as stopped with no throttle for 8 s; cause "a spurious correlation between low speed and no acceleration"; grows with more data. A speed-prediction head helped but is "not a final solution to this problem". Seed variance: success "can change by up to 42%". | https://ar5iv.labs.arxiv.org/html/1904.08980 | arXiv 1904.08980; venue not shown (I recall ICCV 2019, UNVERIFIED) | Wall-pinning analogue (my inference): if `prev_action` is an input, "forward was pressed" predicts "press forward". Also a warning to use more than one BC seed. | medium (figures had no numbers in the text) |
| Noise in the teacher's executed actions, with clean labels, matches DAgger at lower cost | DART: grasping in clutter 79% vs 49% for BC; MuJoCo: matches DAgger, about 3x faster on Humanoid. Loss compares the learner to the supervisor's policy at the visited state; executed control is noisy. Appendix derives an epsilon-greedy form for discrete actions. No fixed noise level given (it is optimised). | https://ar5iv.labs.arxiv.org/html/1703.09327 and https://bair.berkeley.edu/blog/2017/10/26/dart | CoRL 2017 (stated on the page) | Teacher-driven recovery data: the teacher gets knocked into walls and door-misalignments and shows the way out. Robotics, not navigation. | high for numbers; "clean label" is implied by the objective, not stated outright |
| DAgger can underperform BC in practice | BAIR post: in a 10-person study BC beat DAgger; DAgger "may add systematic noise" to labels; the aggregate dataset can go stale. | https://bair.berkeley.edu/blog/2017/10/26/dart | Lab blog (secondary; the paper is the row above) | Human-labelling result; a scripted teacher does not have that problem. Our DAgger plateau is more likely a label-quality (imitation gap) issue (my inference). | medium |
| Reverse curriculum from the goal turns exponential exploration into roughly quadratic | Salimans and Chen: start episodes near the end of a demonstration, move the start back when 20% of rollouts match the demo's return. Montezuma: 74,500. Cost: 1,024 workers, about 50 billion frames. Could not tune it for Gravitar or Pitfall. | https://ar5iv.labs.arxiv.org/html/1812.03381 | arXiv 1812.03381; venue not checked | The idea is cheap for us: our generator can place the player start, so no state restore is needed. Their scale is not ours. | high for the paper; transfer to us untested |
| Practitioner pitfalls with RND in Doom | "normalizing both rewards keeps the agent from learning properly"; tying extrinsic reward to exploration "results in all sorts of poor agent behaviour" (circling, standing still); LSTM struggles with a key far from its door. | https://github.com/callumhay/vizdoom_ppo_rnd | GitHub README; not peer-reviewed | Do not normalise the extrinsic stream; keep coverage in the intrinsic stream. | medium |
| Random critic at handoff as the cause of erasure | GitHub issue: BC 97% validation accuracy and 100% pre-RL success lost after PPO updates; value head was random. Fix proposed (pretrain value head on demo returns) was **not tested** in the issue. | https://github.com/amadou-6e/theseo-anysearch/issues/203 | Open GitHub issue; untested proposal | Anecdote that agrees with PIRLNav. Not evidence that the fix works. | low |
| MimicBot, Cycle-of-Learning, Stanford CS224R report, Ray forum thread on BC then PPO | Seen in search summaries only: keep a BC loss during RL; entropy bonus only delays collapse; load weights after trainer creation. | https://arxiv.org/pdf/2108.09478 , https://arxiv.org/pdf/1910.04281 , https://discuss.ray.io/t/how-to-pretrain-a-model-with-behavior-cloning/278 | not opened | Consistent with the rows above; do not cite. | UNVERIFIED |
| Kieliger blog: movement bonus +0.00005 per step, -0.0025 when not moved; agents "often get stuck against walls" | Search summary only; the page returned HTTP 403. | https://lkieliger.medium.com/playing-doom-with-deep-reinforcement-learning-part-3-boosting-performance-with-reward-shaping-b1de246bda1d | Blog, not opened | Would need the page to cite. | UNVERIFIED |
| LevDoom and MazeExplorer have no doors | Search summaries only: LevDoom (CoG 2022) has four scenarios, none with doors found; MazeExplorer (IEEE CoG 2019) collects keys in generated mazes, doors not mentioned, texture randomisation helped on unseen maps. | https://github.com/TTomilin/LevDoom , https://github.com/microsoft/MazeExplorer | not opened (MazeExplorer PDF returned 404) | No door recipe from either, as far as I could see. | UNVERIFIED |

## 3. Findings per question

### Q1. How did people get Doom agents through doors and multi-room levels?

1. **By dense progress-to-exit shaping from privileged map data, on one level.** doom-player: 15/15 on E1M1 with shaping, 0/5 with exit reward only. https://github.com/lucas-moont/doom-player/pull/31
   - *For us:* a potential based on walking distance to the next required objective is the reward term with outside evidence. I found no outside source for a once-per-door bonus; our own run showed it helps for a while (38%) and then fades (10%). Our `maps/` solvability search may already hold the distances.
2. **Key-aware routing matters.** Paying for progress toward a locked door misleads training. https://github.com/lucas-moont/doom-player/pull/32
   - *For us:* on S2 the potential must go key, then door, then exit.
3. **By scale, with no door-specific design.** Sample Factory self-play agents opened a secret door after 2.5x10^9 frames. https://ar5iv.labs.arxiv.org/html/2006.11751
   - *For us:* not reachable; no recipe.
4. **By PPO + RND with a HUD key crop, one level per run.** https://github.com/callumhay/vizdoom_ppo_rnd
   - *For us:* key bits in the observation (being added tonight) have a practitioner precedent; the author still reports key-far-from-door failures with an LSTM.
5. **Door mechanics nobody writes about but the engine source settles.** USE is edge-triggered and a second press reverses a manual door. https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_user.c , https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_doors.c , and our probe R2.2.
   - *For us (a check for `env-engineer`, not a diagnosis):* with a sampled binary USE head, any press / no-press / press pattern while the door opens (9 to 12 decisions on our generated maps), or while walking through, shuts the door. A policy that holds USE from outside range never triggers. Measure before changing anything: per door episode, count USE press edges inside range, and count steps where the door gap falls right after a press. Also compare door success with sampled versus argmax USE.
   - Door speed is not constant across maps (our own probes: 8 units per decision on generated maps, 32 on freedoom2 MAP01), so a policy that learns "wait N decisions" on our maps may mistime vanilla doors.
6. **No source found** for a door-specific observation channel or door reward that was shown to help. Our line-of-sight door channel is untested by anyone else. I found nothing on LevDoom or MazeExplorer doors (UNVERIFIED rows).

### Q2. Stopping PPO from destroying the BC initialisation

1. **Critic-first handoff (PIRLNav).** Freeze the policy, roll out by sampling, train only the critic until its loss plateaus; critic's last layer initialised near zero. Then unfreeze with policy learning rate warming up from 0 while the critic's decays. BC 52.0%; naive handoff dips early, then recovers to 53.6%; full recipe 61.9%. https://ar5iv.labs.arxiv.org/html/2301.07302
   - *For us:* simplest to build (a flag that zeroes the policy loss for the first N updates). Needs no teacher at handoff. Our 20-minute erosion may be the early dip the paper describes; the paper does not say how long theirs lasted.
2. **KL to the frozen BC policy, replacing the entropy bonus, with decay.** VPT 0.2 x 0.9995 per iteration; DECKARD 0.1 x 0.999; learning rates 2e-5 and 1e-4. https://ar5iv.labs.arxiv.org/html/2206.11795 , https://ar5iv.labs.arxiv.org/html/2301.12050
   - *For us:* our entropy term at 0.01 works against a sharp BC policy. Half-lives: 0.9995 is about 1,390 iterations, 0.999 about 690 (my arithmetic). One "iteration" in those systems may not equal one of our updates, so set the decay by the fraction of the run it should cover, not by copying the number. Our first run was 1,814 updates.
3. **BC loss on teacher states versus KL on student states.** KL on visited states failed where later stages were not visited early; BC on expert states worked there. https://ar5iv.labs.arxiv.org/html/2402.02868
   - *For us:* keep a BC term on the teacher's demonstration buffer (which contains post-door states) during PPO, coefficient order 1 to 2 as in that paper, in addition to or instead of the KL.
   - Our teacher can also be queried live, so its label on the student's own states (a DAgger loss inside PPO) is available. Caution: that is ADVISOR's static IL+RL mix, which scored 78.2 on PointNav against 79.0 for RL alone and 88.2 for the adaptive mix. A fixed live-teacher loss is not supported as an improvement when the teacher is privileged.
4. **Lower learning rate at the start.** Every recipe above finetunes at 1.5e-5 to 1e-4, against our 3e-4.
5. **Which is simplest with numbers:** critic-only warm-up plus swapping the entropy bonus for a decaying KL. Both are a few lines in our own trainer. AWAC/IQL-style methods are off-policy and PIRLNav reports off-policy RL is hard to adapt to recurrent policies; I found no reason to go there tonight. DAPG itself was not opened (PIRLNav only says it was infeasible at their scale).

### Q3. Compounding error and wall-pinning

1. **The teacher's labels may not be learnable from the student's observation (imitation gap).** 6.4% for shortest-path imitation; ADVISOR's analysis. https://ar5iv.labs.arxiv.org/html/2301.07302 , https://ar5iv.labs.arxiv.org/html/2007.12173
   - *For us:* the cleanest fix is on the teacher side: make it steer only toward things the student's rays or entity set currently show (a visible opening, a lit door surface), and explore the way the student would have to when nothing is visible. DAgger cannot fix labels that depend on hidden state; more rounds will plateau (my inference from these two sources).
   - A cheap diagnostic from ADVISOR: train an imitation-only head and look at where its loss stays high. Those are the states where the teacher uses hidden information.
2. **Inertia / causal confusion.** https://ar5iv.labs.arxiv.org/html/1904.08980
   - *For us:* check whether the student copies `prev_action` (forward stays forward). Tests: drop or noise `prev_action` during BC; add an auxiliary head predicting own displacement next step (the paper's speed head analogue; the paper says it helps but does not cure).
3. **Noise injection in the teacher's rollouts, clean labels (DART).** https://ar5iv.labs.arxiv.org/html/1703.09327
   - *For us:* with some probability execute a random action (or the student's action) but store the teacher's label. This fills the dataset with "pressed into a wall, here is the way out" and "misaligned at the door" states. Our DAgger already does a version of this; DART's version does not depend on the student reaching useful states. DART gives no noise level to copy (it optimises it); any rate we pick is our own.
4. **Use several BC seeds.** Up to 42% success swing across seeds in the driving study.
5. **Not found:** evidence for action chunking, longer frame skip or continuous turn curing wall-pinning in navigation. A 2025 preprint on action chunking plus noise injection appeared in a search summary only (UNVERIFIED, not opened). The ViZDoom paper's frame-skip result is already in `vizdoom-agents.md`.

### Q4. Exploration in Doom-like mazes, with effort

See `docs/research/exploration.md` for ICM, RND, EC, NGU and Go-Explore. Only what is new:

1. **Reverse curriculum by spawn placement.** https://ar5iv.labs.arxiv.org/html/1812.03381
   - *For us:* generate S1/S2 variants whose player start is just before the exit, then just past the door, then facing the door at USE range, then further back; advance when success passes a threshold (they used 20%). Our generator writes the start position, so no save-state is needed. Effort 2 to 3 hours (my estimate). The brief also explicitly allows auto-opening doors in training, which can be mixed in and annealed out. Evidence is at 50 billion frames on Atari; treat as promising, not proven.
2. **Keep coverage intrinsic and un-mixed.** The RND project found extrinsic rewards tied to exploration caused circling. https://github.com/callumhay/vizdoom_ppo_rnd
   - *For us:* already our design (episodic coverage bonus, separate stream). Do not normalise the extrinsic reward.
3. **Keyed progress potential** (Q1 items 1 and 2) acts as directed exploration and has the only full-level Doom numbers found. Effort 2 to 4 hours if distances exist (my estimate). Training-only.
4. Go-Explore style return-to-frontier and NGU: nothing new found; `exploration.md` already rules them out as systems.

### Q5. What did not work (reported honestly by the sources)

| What failed or underperformed | Numbers | Source |
|---|---|---|
| PPO with exit reward only on a full level | 0/5 clears, 8.5% progress, below random's 20% (1M steps) | https://github.com/lucas-moont/doom-player/pull/31 |
| Zero-shot transfer of a single-level shaped agent | 0/15, no key picked up | https://github.com/lucas-moont/doom-player/pull/32 |
| Progress shaping that ignores locks | paid up to 20.1% for a dead end | same |
| Normalising both reward streams; exploration-tied extrinsic reward | qualitative | https://github.com/callumhay/vizdoom_ppo_rnd |
| LSTM remembering a key far from its door | qualitative | same |
| Imitating a privileged shortest-path teacher | 6.4% vs 52.0% | https://ar5iv.labs.arxiv.org/html/2301.07302 |
| Naive BC then PPO with a random critic | early drop, then recovery to 53.6% (BC 52.0%); 61.9% with the recipe. Underperformed; did not end below BC. | same |
| Kickstarting (KL on visited states) when later stages are unvisited | failed on RoboticSequence | https://ar5iv.labs.arxiv.org/html/2402.02868 |
| EWC as the anti-forgetting method | deteriorated or saturated lower | same |
| Fixed (static) imitation + RL mix under an imitation gap | 78.2 vs RL alone 79.0 vs adaptive 88.2 (PointNav). That IL warm-start then RL can be worse than RL alone is argued by the authors, not measured in what I read. | https://ar5iv.labs.arxiv.org/html/2007.12173 |
| Backward curriculum on other games | "unable to find hyperparameters that worked" (Gravitar, Pitfall) | https://ar5iv.labs.arxiv.org/html/1812.03381 |
| DAgger with noisy human labels | BC beat it in a 10-person study | https://bair.berkeley.edu/blog/2017/10/26/dart |
| ICM with a fire button; curiosity on a keys-and-doors task | see `exploration.md` | `docs/research/exploration.md` |

My reading of where not to spend tonight (judgement, resting on the rows above): a new curiosity module, more DAgger rounds with the same teacher, EWC, or longer PPO finetuning with the entropy bonus still on.

## 4. Reusable hyperparameters, architecture details, compute

- **PIRLNav handoff** (https://ar5iv.labs.arxiv.org/html/2301.07302): phase 1, policy and shared RNN frozen, sampled actions, critic only, 8M steps there; critic's final layer near zero. Phase 2: critic learning rate 2.5e-4 decaying to 1.5e-5 while policy learning rate warms from 0 to 1.5e-5 (over their steps 8M to 12M), then both constant. Visual encoder stays frozen. Compute: only the 8M and 12M step marks were extracted; total budget and hardware not found. The phases scale down; any shorter length is our own choice.
- **VPT finetuning** (https://ar5iv.labs.arxiv.org/html/2206.11795): KL coefficient 0.2, decay 0.9995 per iteration, replaces entropy bonus (random-init baselines used entropy 0.01), learning rate 2e-5, value head zero-initialised. About 1.3 million 10-minute episodes; not reachable; only the loss form transfers.
- **DECKARD** (https://ar5iv.labs.arxiv.org/html/2301.12050): KL 0.1, decay 0.999 per iteration, learning rate 1e-4. Compute not extracted.
- **Wolczyk et al.** (https://ar5iv.labs.arxiv.org/html/2402.02868): NetHack APPO: BC loss coefficient 2.0 without decay; kickstarting 0.5 with decay 0.99998 per train step. SAC RoboticSequence: BC coefficient 1, 10,000-sample expert buffer. Compute not extracted.
- **ADVISOR** (https://ar5iv.labs.arxiv.org/html/2007.12173): loss w x CE(teacher, student) + (1 - w) x RL loss, w = exp(-alpha x KL(teacher, auxiliary head)); auxiliary head trained by imitation only on the shared trunk. Alpha value not extracted.
- **Salimans and Chen** (https://ar5iv.labs.arxiv.org/html/1812.03381): move the start back when 20% of rollouts succeed; replay a short segment to warm the RNN state and mask it from training. 1,024 workers, 50 billion frames: not reachable; the schedule rule is.
- **doom-player** (https://github.com/lucas-moont/doom-player/pull/31): route progress 100, death 25; 5M steps in about 3 h with SB3 PPO on pixels, one map. Reachable in size; it is an off-the-shelf SB3 setup, which our brief says scores poorly, so take the reward lesson only.
- **vizdoom_ppo_rnd** (https://github.com/callumhay/vizdoom_ppo_rnd): intrinsic coefficient 0.01, RND loss coefficient 0.005 (range 0.001 to 0.05), rollout 256, extrinsic advantage weight 2, intrinsic normalised and extrinsic not; LSTM hidden at least a quarter of its input size; 6M steps, about 8 h, 20 envs, RTX 3080. Reachable in size on our laptop.
- **Door timing, ours** (`docs/reference/engine-probes.md`): generated maps 8 units of gap per decision, open at 68 to 92; freedoom2 MAP01 door 32 per decision, open at 92; face blocks the middle ray band up to gap 40.

## 5. Top 5 changes to try in the next 12 hours

Ranked by expected effect on door success per hour. The ranking is my reasoning from section 0, not something a source measured. Effort figures are my estimates. Each item has a rules note.

1. **Measure, then fix, how USE presses interact with the door.**
   - Do: log per door episode, as three separate counts, door opened (gap rose), door passed, exit reached; plus the number of USE press edges inside range, door-gap reversals right after a press, and decisions spent facing a lit door. Compare sampled against argmax USE at evaluation. Check what press pattern the teacher's labels contain (one press then release, or held).
   - Expected effect: if many doors are opened and then reversed, or never triggered because USE was already held, that is a mechanical loss on top of everything else and could be a large part of the gap between "38% for a while" and "5%". If reversals are rare, it is ruled out in an hour.
   - Effort: 1 hour to measure; 1 to 2 hours to act.
   - Evidence: mechanism strong (engine source plus our probe R2.2); that it is a main cause for us is a hypothesis.
   - Rules: logging is training-side. Any change to how the wrapper turns the USE head into button presses must be the same at inference and must not encode strategy; ask `rules-auditor` before touching it.

2. **Make the teacher's labels learnable from the student's observation, and add teacher-driven recovery data.**
   - Do: restrict the teacher's turn targets to what the rays, door channel and entity set show this step; collect new demonstrations in which a fraction of executed actions are random or the student's, while storing the teacher's clean label (the fraction is ours to choose; DART gives none); drop or noise `prev_action` during BC; train 2 to 3 BC seeds.
   - Expected effect: attacks wall-pinning, the stall at the first door and the unseen-aim-point labels together; this is where BC success should move most.
   - Effort: 3 to 5 hours.
   - Evidence: strong that privileged-teacher imitation fails (6.4% vs 52.0%, PIRLNav; ADVISOR); medium for noise injection (robotics, CoRL 2017); the inertia analogy is my inference.
   - Rules: teacher stays in `teachers/`, training only.

3. **Reverse curriculum by spawn placement on S1 and S2, optionally with auto-opening doors annealed out.**
   - Do: generated variants starting past the door, at the door within USE range, then progressively further back; advance on a success threshold (20% in the source).
   - Expected effect: gives PPO door-passing returns from the first update, which a 5% base rate cannot.
   - Effort: 2 to 3 hours (generator already places starts).
   - Evidence: medium-weak for us (one paper at 50 billion frames; authors report tuning trouble on other games). The brief itself names auto-opening doors as allowed in training.
   - Rules: training-only map changes; evaluation maps unchanged.

4. **Fix the BC-to-PPO handoff: critic-only warm-up, then a decaying KL to the frozen BC policy in place of the entropy bonus, at a lower learning rate, plus a BC term on teacher states.**
   - Do: policy loss off for an initial stretch until the value loss flattens (length is ours to choose; the source used 8M steps at its scale); then KL coefficient 0.1 to 0.2 with a decay set to cover most of the run (sources: 0.999 to 0.9995 per iteration), entropy coefficient 0, learning rate 1e-4 or lower with warm-up; BC loss on the demonstration buffer at coefficient about 1 to 2.
   - Expected effect: stops the 10% to 6% erosion and the entropy rise, and may also be what the once-per-door run needed to hold its 38%. It preserves behaviour; it does not by itself teach doors, which is why it is fourth.
   - Effort: 2 to 3 hours.
   - Evidence: strong (PIRLNav 53.6% to 61.9%; VPT; DECKARD; Wolczyk et al.), all at larger scale and none in Doom.
   - Rules: the frozen BC policy is our own network; training-only loss terms.

5. **Add a keyed progress potential (distance to the next required objective: key, door, exit) alongside or in place of the once-per-door bonus.**
   - Do: per-step reward proportional to the decrease in walking distance along the solvable route, from the generator's own map data; no payment for approaching a locked door without its key.
   - Expected effect: dense signal through and beyond doors; the only shaping with full-level Doom numbers from outside (15/15 vs 0/5). Our once-per-door bonus has only our own evidence (38% then 10%).
   - Effort: 2 to 4 hours if the solvability search exposes distances.
   - Evidence: medium-weak (one practitioner, one map, unequal budgets), and the same source shows 0/15 transfer when trained on a single map; ours trains across 80 maps, which is the mitigation.
   - Rules: privileged distances in the reward only, never in the observation; must be off at inference.

## 6. Gaps

- No Reddit content could be read.
- No source shows door or keycard generalisation to unseen Doom maps by any method.
- No source tests a door-surface ray channel or a telemetry-only Doom policy.
- Whether ZDoom's USE handling differs from the 1997 source in edge cases (for example a press while out of range) was not checked beyond our own probes.
- Why door opening speed differs between our generated maps and freedoom2 MAP01 in our own probes was not looked into.
- DAPG, kickstarting (original paper), AWAC and IQL were not opened.
