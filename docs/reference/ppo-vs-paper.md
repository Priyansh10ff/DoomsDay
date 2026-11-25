# Our PPO vs the PPO paper (and DQN)

Date: 2026-11-26. Code audited: `train/ppo.py`, `train/gae.py`, `train/trainer.py`,
`train/running_stats.py`, `agent/distributions.py`, `configs/train_config.py` (PPOConfig).
Paper facts come from `docs/research/2026-11-26-paper-notes.md` (a DRAFT; its PPO section
was read through a text conversion of the arXiv PDF). Where the paper is silent, the
row says so.

## 1. Items the paper defines

| Item | Paper (Schulman et al. 2017) | Ours | Same? |
|---|---|---|---|
| Ratio | pi_new(a\|s) / pi_old(a\|s) | `exp(logp - old_logp)`, old log-prob stored at collection. With six independent categorical heads, logp is the **sum over heads** (joint probability; `FactorizedCategorical.log_prob`). This is our design consequence, not stated in the paper. | yes |
| Clipped surrogate | mean of min(r A, clip(r, 1-eps, 1+eps) A) | `-min(ratio*a, ratio.clamp(1-clip, 1+clip)*a)`, masked mean over valid steps; `clip` 0.2 (`--clip`) | yes (eps 0.2 = paper's best MuJoCo value; paper's Atari used 0.1 x alpha, annealed) |
| KL-penalty variant | adaptive beta, found worse than clipping | not used | n/a |
| Combined loss | L_clip - c1 L_VF + c2 S | `pg_loss - entropy_coef*entropy + value_coef*v_loss + aux terms` (minimised, so signs match) | yes |
| Value loss | squared error (V - V_target)^2 | `0.5*(V - target)^2`, value_coef 0.5, so effective weight 0.25 on the squared error; no value clipping | equivalent up to the constant |
| Entropy bonus | c2 = 0.01 (Atari) | 0.01 on the **summed** entropy of the six heads (`--entropy-coef`) | same value; summing over heads is ours |
| Advantage | truncated GAE, gamma, lambda over T steps | `compute_gae`, gamma 0.997, lambda 0.95, T = 64 (`rollout_len`), bootstrapped with V(next obs) at the rollout end | yes; differences below |
| Epochs / minibatches | K epochs, minibatch M <= NT (Atari K=3, 32x8) | 3 epochs (`--ppo-epochs`), 4 minibatches of **whole env sequences** (recurrent: each recomputed from its stored h0) | yes in spirit; whole sequences are ours (recurrence) |
| Learning rate | Adam 2.5e-4 x alpha, annealed to 0 (Atari) | Adam 3e-4 constant (ADR 0008: decay only for a fixed-length final run, floored at 10%); optional linear warm-up | deliberate difference |
| Parallel actors | N actors x T steps | N engine instances (layout approved per run), synchronous | yes |

### GAE differences (ours, deliberate)
- Termination and truncation are kept apart: death/exit -> next value 0; time-out -> V(true
  final observation) (`bootstrap_value`). The paper's truncated estimator does not discuss
  time-outs.
- A time-out whose final observation is stale is dropped (`valid` False, advantage 0).
- Values are trained on returns divided by a running return std (`ReturnScale`, floor 0.1).
  Not in the paper.

## 2. Later practice we use (NOT in the paper; justify on its own)

The paper notes say advantage normalisation, value clipping and gradient-norm clipping are
not mentioned in the paper.

| Practice | Ours | Note |
|---|---|---|
| Advantage normalisation | per minibatch, masked mean/std over valid steps, eps 1e-8 | per minibatch, not per batch; with RND on it normalises the combined ext+int advantage |
| Gradient clipping | global norm 1.0 over model parameters | the inverse-dynamics head has its own clip (1.0) and optimizer, so the model's clip is unchanged by it |
| KL early stop | `--kl-stop K`: a minibatch whose approx KL (k3 estimator `(r-1) - log r`, measured before its step) exceeds K is not stepped and the update ends; the first minibatch always steps | off by default |
| Value clipping | not used | |
| Value warm-up / LR warm-up | ours (for `--init-from`) | |
| Recurrent core | GRU, hidden reset where `episode_start` is 1, same convention in collection and update | the paper mentions recurrence in passing, does not evaluate it |

## 3. Additions on 2026-11-26 (this task)
- **RND** (`train/rnd.py`, `--rnd-int-coef`, off by default): second, non-episodic reward
  stream; `value_int` head trained with the same value_coef; advantage
  `ext_coef*A_ext + int_coef*A_int`. Follows Burda et al. 2018's recipe (two value heads,
  gamma_int 0.99, reward / running std of intrinsic return, whitened + clipped RND input,
  predictor on 25% of the data), at laptop scale.
- **Inverse dynamics** (`train/inverse_dynamics.py`, `--inv-dyn-coef`, off by default):
  auxiliary cross-entropy predicting a_t from encoder features of (o_t, o_{t+1}), as in
  ICM's inverse model; not used as a reward. `prev_action` is zeroed for this loss
  (o_{t+1}.prev_action equals a_t, which would leak the label).

## 4. DQN (Mnih et al. 2013) in one paragraph
DQN learns action values off-policy: transitions go to a replay memory and minibatches are
sampled uniformly from it, and the bootstrapped target uses (held-fixed) earlier parameters;
the separate periodically copied target network is from the 2015 Nature paper, not the 2013
one (paper notes). Both mechanisms exist to make **off-policy value learning** stable. PPO is
on-policy: each update uses only data from the current policy (plus the clip), so neither a
replay memory nor a target network applies, and stored GRU states would go stale in replay.
Our only "replay" is the teacher data in the behaviour-cloning auxiliary loss
(`train/bc_aux.py`), a supervised term on fixed demonstrations, not bootstrapped value
learning.

## 5. Audit result
No clear bug found, so no code was changed for the audit. Checked: ratio from stored old
log-probs with the joint (summed) log-probability; min of clipped and unclipped terms;
signs of the value and entropy terms; GAE bootstraps at the rollout end and at time-outs and
cuts at every episode end; the approx-KL estimator; the KL stop measured before the step;
the clip over all model parameters. Points worth a decision (not bugs): (a) advantage
normalisation per minibatch rather than per batch; (b) the value targets use the return
scale updated with the current rollout, while the stored values used the previous scale
(small, standard drift); (c) no learning-rate or clip annealing, unlike the paper's Atari setup.
