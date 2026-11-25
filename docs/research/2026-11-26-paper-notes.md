# Paper notes for DoomsDay (five papers)

**STATUS: DRAFT, INCOMPLETE.** Work was paused on 2026-11-26 on the user's instruction, before the notes were finished and before any review. What is here is the evidence gathered so far; the "what it means for DoomsDay" parts are short first passes, and the standard research layout (claims table, three recommendations for advisor) has not been written yet.

## How these were read (affects confidence)

- The arXiv PDFs could not be opened directly (no PDF renderer on this machine). Each paper was read through a full-text HTML or text rendering of the arXiv paper, with the extraction done by a small summarising model, not by reading the pages myself.
  - Papers 1, 3, 4, 5: ar5iv rendering (`ar5iv.labs.arxiv.org/html/<id>`).
  - Paper 2 (PPO): ar5iv had no body text; read through a text conversion of the arXiv PDF (`r.jina.ai`, a third-party converter).
  - Paper 3 (ICM): read through both; they disagree on one number (see below).
- Every number below came from one of those readings. Nothing is filled in from memory. Items the text did not give are listed as "not stated" or **UNVERIFIED**.
- Not yet done: checking `docs/research/` and `docs/decisions/` for overlap with earlier notes (directory listing was not possible with the tools available; individual files were not opened).

---

## 1. Mnih et al. 2013, Playing Atari with Deep Reinforcement Learning (DQN)

**Problem.** Learn control policies straight from raw pixels with RL, one agent and one set of settings across seven Atari 2600 games, no game-specific information.

**Method in plain words.**
- A convolutional network takes the stacked frames and outputs one Q-value per action in a single forward pass.
- Loss: squared difference between the network's Q(s, a) and a target "reward plus discounted best Q-value of the next state". The background section holds the previous iteration's parameters fixed for the target; Algorithm 1 writes the target with the current parameters. **This 2013 paper does not describe a separate, periodically copied target network** (that belongs to the 2015 Nature paper, which was not read in this session: UNVERIFIED here).
- Experience replay: transitions (s, a, r, s') are stored in a memory and minibatches are sampled uniformly at random, which breaks correlation between consecutive samples and reuses data.
- Behaviour policy: epsilon-greedy, epsilon annealed linearly from 1 to 0.1 over the first million frames, then fixed.

**Setup and hyperparameters.**
- Preprocessing: 210x160 RGB to grayscale, downsample to 110x84, crop 84x84; last 4 frames stacked.
- Network: conv 16 filters 8x8 stride 4, conv 32 filters 4x4 stride 2, fully connected 256, linear output (4 to 18 actions).
- RMSProp, minibatch 32, replay memory of one million frames, 10 million training frames, frame skip 4 (3 for Space Invaders), rewards clipped to +1 / -1 / 0 during training, evaluation epsilon 0.05.
- Not stated in the text: learning rate, discount, hardware.

**Results (Table 1, average total reward).**

| Game | Random | Sarsa | Contingency | DQN | Human |
|---|---|---|---|---|---|
| Beam Rider | 354 | 996 | 1743 | 4092 | 7456 |
| Breakout | 1.2 | 5.2 | 6 | 168 | 31 |
| Enduro | 0 | 129 | 159 | 470 | 368 |
| Pong | -20.4 | -19 | -17 | 20 | -3 |
| Q*bert | 157 | 614 | 960 | 1952 | 18900 |
| Seaquest | 110 | 665 | 723 | 1705 | 28010 |
| Space Invaders | 179 | 271 | 268 | 581 | 3690 |

- Abstract claim: it "outperforms all previous approaches on six of the games" and beats a human expert on three (Breakout, Enduro, Pong).

**Failure modes / limits stated.** No convergence guarantee; far from human on games needing long-horizon strategy (Q*bert, Seaquest, Space Invaders); reward clipping hides reward magnitude; uniform replay treats all transitions alike; only seven games.

**What it means for DoomsDay (first pass).**
- Already do: frame skip, a shared encoder feeding action outputs, reward scaling.
- Not applicable: replay memory and bootstrapped Q targets are off-policy machinery. Our trainer is on-policy recurrent PPO; replaying old transitions would break the assumption that data comes from the current policy, and stored GRU states go stale.
- Worth noting: DQN's weak games are the long-horizon ones, which is the same weakness our key/door tasks probe.
- Write-up sentence (draft): "Deep RL from high-dimensional input dates from DQN (Mnih et al., 2013); we use an on-policy method instead, so its replay memory does not apply."

---

## 2. Schulman et al. 2017, Proximal Policy Optimization Algorithms (PPO)

**Problem.** Policy-gradient methods use each batch of data for one update; doing several updates on the same data without a constraint gives "destructively large policy updates". TRPO constrains the update but is complicated. PPO aims for the same stability with a first-order method.

**Method in plain words.**
- Probability ratio r = new policy's probability of the taken action divided by the old policy's.
- **Clipped surrogate:** for each step take the smaller of (ratio x advantage) and (ratio clipped to [1 - eps, 1 + eps] x advantage), then average. Taking the minimum makes it a "pessimistic bound": the policy gains nothing from pushing the ratio outside the clip range in the direction that helps.
- **KL-penalty variant:** ratio x advantage minus beta x KL(old, new); beta is halved if measured KL is below target / 1.5 and doubled if above target x 1.5. The paper found this worse than clipping.
- **Combined loss:** clipped surrogate minus c1 x squared value error plus c2 x entropy bonus (needed when policy and value share parameters).
- **Truncated GAE:** advantage is the sum of TD errors (r + gamma V(s') - V(s)) discounted by gamma x lambda, over the T-step rollout.
- **Algorithm 1:** N actors each collect T steps; compute advantages; optimise for K epochs with minibatches of size M <= N x T; repeat.

**Results.**
- Table 1 (7 MuJoCo tasks, 3 seeds each, 1M timesteps, normalised score): no clipping or penalty -0.39; clipping eps 0.1 / 0.2 / 0.3 = 0.76 / 0.82 / 0.70; adaptive KL (targets 0.003 / 0.01 / 0.03) = 0.68 / 0.74 / 0.71; fixed KL beta 0.3 / 1 / 3 / 10 = 0.62 / 0.71 / 0.72 / 0.69.
- Atari, 49 games, 40M frames, games won (A2C / ACER / PPO / tie): average reward over all of training 1 / 18 / 30 / 0; average over last 100 episodes 1 / 28 / 19 / 1. So PPO learns faster; ACER is ahead on final score.

**Hyperparameters.**

| | MuJoCo (Table 3) | Atari (Table 5) |
|---|---|---|
| Horizon T | 2048 | 128 |
| Adam step size | 3e-4 | 2.5e-4 x alpha |
| Epochs K | 10 | 3 |
| Minibatch | 64 | 32 x 8 = 256 |
| gamma / lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| Actors | (not extracted) | 8 |
| Clip eps | 0.2 in Table 1 best | 0.1 x alpha |
| c1 (value) / c2 (entropy) | no sharing, no entropy bonus | 1 / 0.01 |

alpha is annealed linearly from 1 to 0 over training.

**Things the paper does NOT contain (matters for the audit).** Advantage normalisation, value-function clipping and gradient-norm clipping are not mentioned. Recurrent policies are mentioned in passing but not evaluated. Anything we do in those areas comes from later implementation practice, not from this paper, and should be justified separately.

**Failure modes.** Unclipped multi-epoch updates collapse (the -0.39 row). No dedicated limitations section.

**What it means for DoomsDay (first pass).**
- Already do: our own recurrent PPO with GAE, clipped surrogate, entropy bonus.
- Adding: an implementation audit against this paper. Checklist drafted from the above: ratio computed from stored old log-probs; minimum of clipped and unclipped terms; sign of value and entropy terms; GAE truncated at rollout end with bootstrap value; K, M, T, eps recorded per run. With six categorical heads the ratio should use the joint log-probability (sum over heads): this is our design consequence, not a statement from the paper.
- The first run's mid-run collapse (entropy minimum 0.122) is the kind of event the clip range and entropy coefficient control; the paper gives no direct evidence on recurrent or multi-head cases.
- Write-up sentence (draft): "The trainer is our own implementation of the clipped-surrogate PPO objective with GAE (Schulman et al., 2017), extended with a recurrent core and six action heads."

---

## 3. Pathak et al. 2017, Curiosity-driven Exploration by Self-supervised Prediction (ICM), ICML 2017

**Problem.** Extrinsic reward is sparse or absent; predicting raw pixels as a curiosity signal gets distracted by things the agent cannot affect.

**Method in plain words.**
- A feature encoder phi turns each state into a feature vector (288-dimensional).
- **Inverse model:** given features of s_t and s_t+1, predict the action taken. Loss is cross-entropy over actions. This trains phi to keep only what the agent's actions influence.
- **Forward model:** given phi(s_t) and the action, predict phi(s_t+1). Loss is half the squared error.
- **Intrinsic reward:** eta / 2 x that same squared forward-prediction error.
- **Overall objective:** policy loss weighted by lambda, plus (1 - beta) x inverse loss plus beta x forward loss; beta = 0.2, lambda = 0.1. **eta's value is not stated in the paper text** (both renderings agree).

**Setup.** A3C base, 20 workers, ADAM, learning rate 1e-3. Input 42x42 grayscale, 4 stacked frames. Action repeat 4 in VizDoom (6 in Mario) during training. Policy: four conv layers (32 filters, 3x3, stride 2, ELU), LSTM 256, value and action heads. Inverse model: concatenated features, FC 256, FC 4. Forward model: features plus action, FC 256, FC 288.

**VizDoom setup: this is our TRAIN scenario (`DoomMyWayHome-v0` / `my_way_home`).**
- 4 discrete actions: forward, left, right, no-action.
- Reward +1 on reaching the vest, 0 otherwise; episode ends at the goal or after 2100 steps.
- Dense: spawn uniformly among 17 locations. Sparse: fixed spawn in Room-13, about 270 steps from the goal under an optimal policy. Very sparse: Room-17, about 350 steps.
- Number of rooms: the PDF-text reading says "9 rooms"; the ar5iv rendering says "99 rooms". 9 is taken as correct (it is the PDF text, and room labels 13 and 17 are spawn labels, not a room count), but this was not checked by eye: **medium confidence**.

**VizDoom results.**

| Setting | ICM + A3C | ICM-pixels + A3C | A3C (epsilon-greedy) | Notes |
|---|---|---|---|---|
| Dense | learns, faster than A3C | learns | learns | curve only, no numbers in text |
| Sparse | succeeds | succeeds, similar convergence | fails | curve only |
| Very sparse | perfect score in 66% of runs | never succeeds | never succeeds | |

Sparse setting at convergence, mean (median) score: TRPO 26.0% (0.0%), A3C 0.0% (0.0%), VIME + TRPO 46.1% (27.1%), ICM + A3C 100.0% (100.0%).

- Figure 5 x-axis runs 0 to 9 million training steps; exact convergence steps are **not given in the text** (UNVERIFIED from the curves).
- Noise test: with a fixed white-noise region covering 40% of the image, ICM still solves sparse; ICM-pixels suffers.
- No-reward exploration: the agent reaches rooms over 250 steps away; no room counts given.
- Generalisation: pretraining with curiosity only on a map with different textures, then fine-tuning on very sparse, learns faster than ICM from scratch (no numbers in text). Mario table numbers were extracted but the column mapping was scrambled, so they are not reported here.

**Failure modes.**
- Unpredictable but irrelevant input: the intro uses a white-noise television example; the phrase "noisy TV" itself does not appear in this paper (it is from later work, including RND). ICM's inverse-model features are its defence, and it only covers noise the agent cannot affect.
- Mario: progress capped near 30% of Level-1 by a pit at 38% needing 15 to 20 specific key presses.
- "Curiosity blockade": fine-tuning on Level-3 degraded the policy because curiosity reward dried up.
- Authors say it does not directly extend to settings where interaction opportunities are rare.

**What it means for DoomsDay (first pass).**
- Comparability warning for our own `my_way_home` number (success 0.726 over the last 500 episodes, first run): the paper's observation (42x42 pixels), action set (4 actions), spawn setting and algorithm all differ from ours, and which spawn setting the bundled scenario uses was not checked here. Do not put our number beside the paper's table as a like-for-like comparison.
- Adding: the inverse-dynamics loss as an **auxiliary loss** on our encoder (predict the action from consecutive encoded observations), not as a reward. The paper supports the representation argument (features that track what actions change); it does not test the inverse loss alone without the curiosity reward, so the benefit of the auxiliary-only use is **not evidenced by this paper**.
- With six action heads the inverse model needs six classification outputs; the paper only covers a single 4-way output.
- Write-up sentence (draft): "Following ICM (Pathak et al., 2017) we train an inverse-dynamics head so the encoder keeps action-relevant features; we use it as an auxiliary loss, not a reward."

---

## 4. Burda et al. 2018, Exploration by Random Network Distillation (RND)

Venue: the arXiv page lists none. ICLR 2019 is recalled but **UNVERIFIED** in this session.

**Problem.** Sparse-reward exploration (Montezuma's Revenge) with a bonus that scales to many parallel environments.

**Method in plain words.**
- A **target** network with fixed random weights maps an observation to a feature vector. A **predictor** network is trained on the agent's own experience to output the same vector.
- **Intrinsic reward** = squared error between predictor and target on the next observation. States seen often are predicted well (low reward); new states are predicted badly (high reward).
- The paper lists four sources of prediction error: little training data (the useful one), stochasticity, model misspecification, learning dynamics. Because the target is deterministic and in the predictor's model class, the second and third are avoided.
- **Two value heads:** V_E for extrinsic return (episodic, gamma_E = 0.999) and V_I for intrinsic return (**non-episodic**: not cut at game over, gamma_I = 0.99). The advantage combines them with coefficients 2 (extrinsic) and 1 (intrinsic).
- **Normalisation:** intrinsic reward divided by a running estimate of the standard deviation of intrinsic returns. Observations for predictor and target (not the policy) are whitened with running statistics and clipped to [-5, 5]; statistics are initialised from a short random-agent run (length not stated).

**Hyperparameters (Tables 2 to 5).** Rollout 128; 128 parallel envs (default); 4 minibatches; 4 epochs; Adam, learning rate 1e-4; lambda 0.95; entropy 0.001; clip range [0.9, 1.1]; predictor trained on a 0.25 share of experience; frame skip 4; sticky actions 0.25; extrinsic reward clipped to [-1, 1], intrinsic not clipped; 30K rollouts per environment, about 1.97 billion frames for most experiments. Feature dimension and layer sizes are not in the paper text.

**Results (Table 1, final mean score).**

| Game | RND | PPO | Dynamics | Prior SOTA | Avg. human |
|---|---|---|---|---|---|
| Gravitar | 3,906 | 3,426 | 3,371 | 2,209 | 3,351 |
| Montezuma's Revenge | 8,152 | 2,497 | 400 | 3,700 | 4,753 |
| Pitfall! | -3 | 0 | 0 | 0 | 6,464 |
| Private Eye | 8,666 | 105 | 33 | 15,806 | 69,571 |
| Solaris | 3,282 | 3,387 | 3,246 | 12,380 | 12,327 |
| Venture | 1,859 | 0 | 1,712 | 1,813 | 1,188 |

- Montezuma's Revenge: best agent often finds 22 of 24 first-level rooms; one RNN run with 32 envs visited all 24 (mean return 7,570 after 1.6B frames, best 17,500); RNN with 1,024 envs reached mean 10,070.
- Raising gamma_I to 0.999 hurt; a higher gamma_E helped. More parallel envs gave higher returns.

**Failure modes stated.**
- Dynamics-based (forward-prediction) bonus plus sticky actions made the agent oscillate between two rooms: the noisy-TV effect.
- "Dancing with skulls": the agent keeps seeking dangerous, rarely seen states.
- Only local exploration; long-horizon coordinated exploration is "beyond the reach of our method".
- Keys and doors: the incentive to save a key for a later door is too weak. Directly relevant to our S2 key and locked-door stage.
- High variance across seeds.

**Compute.** About 2 billion frames with 128 or more parallel envs. **Not reachable on our laptop** (the first run did 3.71M decisions). We can borrow the mechanism, not the scale; whether RND helps at a few million steps is not shown by this paper.

**What it means for DoomsDay (first pass).**
- Already do: an episodic coverage bonus (count-style novelty within an episode).
- Adding: RND intrinsic reward with a separate `value_int` head. The paper's recipe to copy: separate heads and discounts, non-episodic intrinsic return, reward normalisation by return standard deviation, observation whitening and clipping for the RND networks only, predictor trained on a subsample.
- Open points for advisor: (a) our input is rays plus an entity set, not images, so the RND networks need their own input design and the paper gives no evidence for that; (b) RND novelty is over the whole of training, so on procedurally varied maps it may fade differently than on one fixed Atari game; (c) ADR 0011 says verdicts use `engine_return`, which fits the paper's separation of extrinsic and intrinsic streams.
- Write-up sentence (draft): "Exploration combines an episodic coverage bonus with a Random Network Distillation bonus (Burda et al., 2018), using a separate intrinsic value head and non-episodic intrinsic returns."

---

## 5. Mnih et al. 2014, Recurrent Models of Visual Attention (RAM)

Venue: none on the arXiv page. NIPS 2014 is recalled but **UNVERIFIED** in this session.

**Problem.** Convolutional networks cost grows with image size. RAM processes only a few small regions chosen by the model itself.

**Method in plain words.**
- **Glimpse sensor:** at a location l, cut several square patches centred on l, each twice as wide as the last, all resized to the same small size (sharp centre, blurry surround).
- **Glimpse network:** patch through a 128-unit layer, location through a 128-unit layer, then both combined by a linear layer each and summed into a 256-unit glimpse vector.
- **Core network:** recurrent, 256 units; new state from previous state plus glimpse vector (256 LSTM units for the game task).
- **Location network:** linear layer on the hidden state gives the mean of a fixed-variance Gaussian; the next location is sampled from it.
- **Action network:** linear softmax on the hidden state (the class label, or the game action).
- **Training:** the location choice is a sample, so it is not differentiable. REINFORCE: gradient of log-probability of the chosen locations times (return minus a learned baseline). The classifier is trained with ordinary cross-entropy and backpropagation through the core and glimpse networks.
- SGD with momentum 0.9; learning rate and location variance by random search (values not stated).

**Results (test error).**
- MNIST 28x28, 8x8 glimpses: 2 / 3 / 4 / 5 / 6 / 7 glimpses = 6.27% / 2.7% / 1.73% / 1.55% / 1.29% / 1.47%; fully connected 2x256 baseline 1.35%.
- Translated MNIST 60x60: conv 2.31%; RAM 4 / 6 / 8 glimpses = 2.29% / 1.86% / 1.84%.
- Cluttered translated 60x60: conv 7.83%; RAM 4 / 6 / 8 glimpses = 7.1% / 5.88% / 5.23%.
- Cluttered translated 100x100: conv 16.51%; RAM 4 / 6 / 8 glimpses = 14.95% / 11.58% / 10.83%.
- "Catch" game (24x24 screen, 6x6 retina at three scales, reward only at the end, 20 million frames): best agent catches about 85% of the time and learns to track the ball unprompted.

**Failure modes / limits.** Hard attention is non-differentiable, so REINFORCE's variance applies; hyperparameters needed random search; only MNIST variants and one toy game; the rendering shows an inconsistency in the conv baseline description (8 vs 16 filters).

**What it means for DoomsDay (first pass).**
- Already do: attention over the visible-entity set, then a GRU 256.
- Adding: an attention readout whose **query comes from the GRU state**, so what the agent attends to depends on what it remembers. This borrows RAM's idea that the recurrent state decides where to look next.
- Honest difference: RAM uses hard, sampled attention trained by REINFORCE; ours would be soft (differentiable) attention over entities trained by backpropagation. The paper gives no evidence about soft, state-queried attention over sets, so the citation is for inspiration only, not for a performance claim.
- Write-up sentence (draft): "Inspired by the Recurrent Attention Model (Mnih et al., 2014), the entity readout is queried by the recurrent state, so memory steers attention; ours is soft attention trained end to end."

---

## Still to do when work resumes

1. Check existing `docs/research/` and `docs/decisions/` notes for overlap and contradictions.
2. Resolve the ICM "9 rooms" reading and the convergence step counts by looking at the PDF directly.
3. Confirm venues for RND and RAM; read the DQN Nature 2015 paper only if the write-up needs the target-network citation.
4. Confirm which spawn setting our bundled `my_way_home` config uses before any comparison with ICM's numbers.
5. Add the standard claims table and the three recommendations for advisor.

## References

1. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. *Playing Atari with Deep Reinforcement Learning.* 2013. NIPS Deep Learning Workshop 2013 (arXiv comments field). https://arxiv.org/abs/1312.5602
2. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. *Proximal Policy Optimization Algorithms.* 2017. arXiv preprint, no venue listed. https://arxiv.org/abs/1707.06347
3. Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell. *Curiosity-driven Exploration by Self-supervised Prediction.* 2017. ICML 2017 (arXiv comments field and project page). https://arxiv.org/abs/1705.05363 ; project page https://pathak22.github.io/noreward-rl/
4. Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. *Exploration by Random Network Distillation.* 2018. Venue UNVERIFIED. https://arxiv.org/abs/1810.12894
5. Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu. *Recurrent Models of Visual Attention.* 2014. Venue UNVERIFIED. https://arxiv.org/abs/1406.6247

Full-text renderings used: https://ar5iv.labs.arxiv.org/html/1312.5602 , https://ar5iv.labs.arxiv.org/html/1705.05363 , https://ar5iv.labs.arxiv.org/html/1810.12894 , https://ar5iv.labs.arxiv.org/html/1406.6247 , and a text conversion of https://arxiv.org/pdf/1707.06347 and https://arxiv.org/pdf/1705.05363 .
