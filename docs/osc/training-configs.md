# OSC: Reference training configurations (numbers only)

Scouted 2026-11-25 by osc-scout. **Read-only web fetches; nothing cloned, no weights downloaded, nothing run.**

The repos are classified in `vizdoom-and-rl-references.md` and `curiosity.md`, and this file does not repeat that.

Every value below was read from the file named in its section. "not verified" means the source was not opened, or the opened source did not show the value. "Derived" marks arithmetic done here, not a value read from a source.

**Use rule.** These are reference values for the team's own recurrent PPO.
- Do not copy code from any of these repos.
- Do not load, fine-tune or distil from the Hugging Face checkpoints. They are recorded for reading only.

## Recommendation (headline)

**Recurrent PPO starting point.** The Sample Factory Doom battle runs and CleanRL's Atari LSTM PPO agree on:
- `gamma` 0.99
- GAE `lambda` 0.95
- PPO clip 0.1
- frame skip 4

**Exceptions:**
- Sample Factory duel/deathmatch runs use gamma 0.995 and frame skip 2.
- CleanRL RND uses gamma 0.999 extrinsic.
- CleanRL plain `ppo.py` uses clip 0.2.
- Arnold's frame skip varies from 2 to 5 by scenario.
- CleanRL is not a Doom source.

**Where they differ:**
- **Recurrent width:** Sample Factory 512 (GRU by default; one published run uses LSTM); CleanRL LSTM 128.
- **Sample Factory Doom:**
  - `lr` 1e-4, constant
  - rollout 32 = recurrence 32 (BPTT window)
  - batch 2048, 1 epoch
  - input normalisation on, `obs_scale` 255
- **CleanRL:**
  - `lr` 2.5e-4, annealed
  - `num_steps` 128, 4 epochs x 4 minibatches
  - `max_grad_norm` 0.5

**Entropy.** Sample Factory Doom does **not** use a plain entropy bonus. It uses a `symmetric_kl` exploration loss at 0.001. CleanRL uses an entropy coefficient of 0.01, or 0.001 in RND.

**RND.** Use CleanRL's split as the reference:
- `ext_coef` 2.0, `int_coef` 1.0
- `gamma` 0.999 extrinsic, 0.99 intrinsic
- intrinsic returns non-episodic
- observation normalisation clipped to [-5, 5]
- predictor trained on 25% of samples

**Compute.** Sample Factory's Doom runs (4e9 env steps, 72 workers x 8 envs, 4 GPUs) are far beyond the laptop; see the last section. A laptop-scale run is closer to CleanRL LSTM's shape: 8 to 16 envs and 1e7 to 5e7 steps.

---

## 1. Sample Factory, ViZDoom setup

**Repo:** https://github.com/alex-petrenko/sample-factory (master; files fetched 2026-11-25)

**Three layers of configuration, kept apart below:**
- **(a) Core defaults.** From `sample_factory/cfg/cfg.py`: https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sample_factory/cfg/cfg.py
- **(b) Doom overrides.** From `doom_override_defaults` in `sf_examples/vizdoom/doom/doom_params.py`: https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/doom_params.py
- **(c) Per-run CLI flags.** From the documented training commands: https://www.samplefactory.dev/09-environment-integrations/vizdoom/

### 1.1 Hyperparameters

Columns (c1) and (c2) are both from source (c):
- **(c1) battle / battle2:** the battle and battle2 commands.
- **(c2) duel / deathmatch:** the `doom_duel_bots`, `doom_deathmatch_bots`, `doom_duel` and `doom_deathmatch_full` commands.

A dash means "not overridden at this layer", so the value is inherited from the layer to its left.

| Parameter | (a) cfg.py default | (b) Doom override | (c1) battle / battle2 | (c2) duel / deathmatch |
|---|---|---|---|---|
| algo | APPO | - | APPO | APPO |
| learning_rate | 1e-4 | - | - | - |
| lr_schedule | constant | - | - | - |
| gamma | 0.99 | - | - | **0.995** |
| gae_lambda | 0.95 | - | - | - |
| rollout | 32 | - | - | - |
| recurrence | -1 (= rollout if RNN, else 1) | - | - | - |
| batch_size | 1024 | - | **2048** | 2048 |
| num_batches_per_epoch | 1 | - | - | - |
| num_epochs | 1 | - | - | - |
| shuffle_minibatches | False | - | - | - |
| num_workers | cpu_count() | - | **72** | 72 or 80 |
| num_envs_per_worker | 2 | - | **8** | 16, 24 or 32 |
| worker_num_splits | 2 | - | - | - |
| num_policies | 1 | - | 1 | 8 (PBT) |
| max_policy_lag | 1000 | - | - | - |
| exploration_loss | entropy | **symmetric_kl** | - | - |
| exploration_loss_coeff | 0.003 | **0.001** | - | - |
| value_loss_coeff | 0.5 | - | - | - |
| ppo_clip_ratio | 0.1 | - | - | - |
| ppo_clip_value | 1.0 | **0.2** | - | - |
| max_grad_norm | 4.0 | - | **0.0** | - |
| with_vtrace | False | - | - | - |
| vtrace_rho / vtrace_c | 1.0 / 1.0 | - | - | - |
| kl_loss_coeff | 0.0 | - | - | - |
| optimizer | adam | - | - | - |
| adam_eps / beta1 / beta2 | 1e-6 / 0.9 / 0.999 | - | - | - |
| reward_scale | 1.0 | - | - | **0.5** (`doom_duel_bots` only) |
| reward_clip | 1000.0 | - | - | - |
| value_bootstrap | False | - | - | - |
| normalize_returns | True | True | - | - |
| normalize_input | True | True | - | - |
| obs_subtract_mean | 0.0 | 0.0 | - | - |
| obs_scale | 1.0 | **255.0** | - | - |
| env_frameskip | 1 | **4** | 4 | **2** |
| eval_env_frameskip | None | **1** | - | - |
| fps | 0 | 35 | - | - |
| env_framestack | 1 | - | - | - |
| pixel_format | CHW | - | - | - |
| use_rnn | True | - | True | True |
| rnn_type | gru | - | - | - |
| rnn_size | 512 | - | - | - |
| rnn_num_layers | 1 | - | - | - |
| encoder_conv_architecture | convnet_simple | - | - | - |
| encoder_conv_mlp_layers | [512] | - | - | - |
| encoder_mlp_layers | [512, 512] | - | - | - |
| decoder_mlp_layers | [] | - | - | - |
| nonlinearity | elu | - | - | - |
| policy_initialization | orthogonal, gain 1.0 | - | - | - |
| actor_critic_share_weights | True | - | - | - |
| res_w x res_h | - | 128 x 72 (`add_doom_env_args` default) | - | 128 x 72 (passed explicitly) |
| wide_aspect_ratio | - | False | False | False |
| train_for_env_steps | 1e10 | - | **4e9** (basic envs: 5e8) | train_for_seconds 360,000 or 3,600,000 |
| PBT | off | - | off | on; `pbt_period_env_steps` 5e6; replace-gap values vary per run |

**Notes on the table:**
- **`max_grad_norm = 0.0`.** Every battle run sets this. Whether 0.0 disables clipping in the learner is **not verified**: `learner.py` was not opened.
- **Unit of `train_for_env_steps`.** Whether it counts engine frames (after frame skip) or agent decisions is **not verified**. `summaries_use_frameskip = True` in cfg.py hints at frames, but that is not confirmed.
- **`doom_benchmark` commands (docs page, not training runs):**
  - 10-core i9-7900X: batch 4096 with 20 workers x 32 envs, reported at 50K+ FPS.
  - 36-core machine: batch 8192 with 72 x 24, reported at 100K+ FPS.
- **Hardware:** the docs page describes the battle runs as using a "36-core server, 4 GPUs".

### 1.2 Observation pipeline and resolution

Sources: `doom_utils.py` (https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/doom_utils.py) and `doom_params.py`.

**Pixels:**
- The engine renders at 160x120, or 256x144 when `wide_aspect_ratio` is set.
- A `ResizeWrapper` scales the frame to `res_w x res_h`, default 128x72, in colour (`grayscale=False`).
- Pixels are divided by 255 (`obs_scale`). A running input normaliser is also on (`normalize_input=True`).
- Returns are normalised (`normalize_returns=True`).

**Wrapper order:**
1. VizdoomEnv
2. (recording)
3. MultiplayerStatsWrapper
4. SetResolution
5. Resize
6. TimeLimit (when timeout > 0)
7. PixelFormatChw (when CHW)
8. the spec's extra wrappers, in order
9. RewardScalingWrapper (when scaling != 1.0)

**Measurements vector (`DoomAdditionalInput`, used by battle and deathmatch).** Source: https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/wrappers/additional_input.py

The vector has 7 + 2N entries, with N = `NUM_WEAPONS` = 8 (from `reward_shaping.py`), so it has **23 floats**. That size is derived.

| Entry | Variable | Scaling |
|---|---|---|
| 1 | SELECTED_WEAPON | clamped >= 0, rounded |
| 2 | SELECTED_WEAPON_AMMO | / 15, capped at 5 |
| 3 | HEALTH | clamped >= 0, / 30 |
| 4 | ARMOR | / 30 |
| 5 | USER2 (kills) | / 10 |
| 6 | ATTACK_READY | none |
| 7 | PLAYER_COUNT | / 5 |
| 8 to 15 | WEAPON0..7 | clamped >= 0 |
| 16 to 23 | AMMO0..7 | / 15, capped at 5 |

### 1.3 Encoder and RNN sizes

Sources:
- conv stacks: `sample_factory/model/encoder.py` (https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sample_factory/model/encoder.py)
- Doom encoder: `doom_model.py` (https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/doom_model.py)
- MLP and RNN sizes: cfg.py

| Component | Value |
|---|---|
| convnet_simple (out_ch, kernel, stride) | (32, 8, 4), (64, 4, 2), (128, 3, 2); nonlinearity after each |
| Alternatives | convnet_atari: (32,8,4), (64,4,2), (64,3,1). convnet_impala: (16,8,4), (32,4,2). resnet_impala: three stages of 16/32/32 channels, each with a 3x3 conv, a stride-2 max-pool and 2 residual blocks |
| Conv MLP | `encoder_conv_mlp_layers` [512] (cfg.py default) |
| Measurement branch (Doom) | Linear(23 -> 128), act, Linear(128 -> 128), act |
| Fusion | concat(image features, 128-d measurement features) |
| Core | GRU 512, 1 layer (default). The edbeeching run used LSTM 512 |
| Heads | `decoder_mlp_layers` [] (none); actor and critic share weights |

### 1.4 Action spaces

Source: `action_space.py` (https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/action_space.py), with the env-to-space mapping in `doom_utils.py`.

| Env | Action space |
|---|---|
| doom_basic | Discrete(4) |
| doom_my_way_home_flat_actions, doom_health_gathering(_supreme) | Discrete(5) |
| doom_defend_the_center_flat_actions | Discrete(4) |
| doom_benchmark | Discrete(9) |
| doom_my_way_home, doom_two_colors_easy/hard | `basic`: Tuple(turn 3, move fwd/back 3) |
| doom_deadly_corridor | `extended`: Tuple(turn 3, fwd/back 3, strafe 3, attack 2) |
| doom_defend_the_center, doom_defend_the_line | `turn_and_attack_only`: Tuple(turn 3, attack 2) |
| **doom_battle, doom_battle2** | `discretized_no_weap`: Tuple(fwd/back 3, strafe 3, attack 2, sprint 2, turn = Discretized(11 bins over [-10, 10])) |
| doom_dm, doom_dwango5 | `doom_action_space`: Tuple(fwd/back 3, strafe 3, prev/next weapon 3, attack 2, sprint 2, turn Box[-1, 1]) |
| doom_deathmatch_bots | `full_discretized`: Tuple(fwd/back 3, strafe 3, weapon select 8, attack 2, sprint 2, turn = Discretized(21 bins over [-12.5, 12.5])) |
| doom_duel_bots, doom_duel, doom_deathmatch_full | `full_discretized(with_use=True)`: as above plus use 2 before turn (7 heads) |

**Episode timeouts (`default_timeout`):**
- battle and battle2: 2100
- basic: 300
- dm, duel and deathmatch: 1e9

**`reward_scaling` per spec:**
- basic and deadly_corridor: 0.01
- all others: 1.0

### 1.5 Reward shaping: battle and deathmatch

Source: `wrappers/reward_shaping.py` (https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/wrappers/reward_shaping.py)

Rewards are applied to the per-step change in a game variable, as (reward per unit gained, reward per unit lost).

| Variable | DEATHMATCH_V0 = BATTLE | DEATHMATCH_V1 (self-play) |
|---|---|---|
| FRAGCOUNT | +1 / -1.5 | +1 / -0.001 |
| DEATHCOUNT | -0.75 / +0.75 | -1 / +1 |
| HITCOUNT | +0.01 / -0.01 | 0 / 0 |
| DAMAGECOUNT | +0.003 / -0.003 | +0.01 / -0.01 |
| HEALTH | +0.005 / -0.003 | +0.01 / -0.01 |
| ARMOR | +0.005 / -0.001 | same as V0 |
| WEAPON0-2 / 3-5 / 6-7 | +0.02/-0.01, +0.10/-0.05, +0.20/-0.10 | same as V0 |
| AMMO0-2 / 3-5 / 6-7 | +0.0002/-0.0001, +0.001/-0.0005, +0.002/-0.001 | same as V0 |
| Selected-weapon bonus (per step) | SELECTED0-2: 0.0002; 3-5: 0.001; 6-7: 0.002 | same as V0 |

**Other constants:**
- **Weapon preference multipliers:** pistol 1; shotguns, machinegun and rocket launcher 5; plasmagun and BFG 10.
- **Per-step delta caps:** DAMAGECOUNT 200, HITCOUNT 5.
- **Selected-weapon bonus timing:** paid only when a 5-step history window is full and all 5 entries are the same weapon.
- **True objectives:**
  - battle: `None`. What is used instead was not read.
  - deathmatch_bots and duel_bots: frag count.
  - duel and deathmatch_full: 1.0 for finishing first, otherwise 0.

---

## 2. Published Sample Factory Doom checkpoints on Hugging Face (configs only)

These are **reading only. Do not download or use the weights.** Only `cfg.json`, the model card and the API file listing were read.

### 2a. andrewzhang505/sample-factory-2-doom-battle

**Config:** https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle/raw/main/cfg.json

**Older schema.** This config uses keys that the current cfg.py no longer has: `encoder_type`, `encoder_subtype`, `encoder_custom`, `encoder_extra_fc_layers` and `hidden_size`. There is no `rnn_size` key. Do not assume `hidden_size` maps to `rnn_size`.

| Key | Value |
|---|---|
| env | doom_battle |
| algo | APPO, async_rl true |
| num_workers x num_envs_per_worker | 16 x 20 (320 envs, derived) |
| batch_size / num_epochs / num_batches_per_epoch | 2048 / 1 / 1 |
| rollout / recurrence | 32 / 32 |
| shuffle_minibatches | true |
| gamma / gae_lambda | 0.99 / 0.95 |
| learning_rate / lr_schedule | 0.0001 / constant |
| exploration_loss / coeff | symmetric_kl / 0.001 |
| value_loss_coeff | 0.5 |
| ppo_clip_ratio / ppo_clip_value | 0.1 / 0.2 |
| with_vtrace | **true** (rho 1.0, c 1.0) |
| max_policy_lag | 10000 |
| max_grad_norm | 0.0 |
| adam eps / beta1 / beta2 | 1e-6 / 0.9 / 0.999 |
| reward_scale / reward_clip | 1.0 / 1000.0 |
| normalize_input / normalize_returns | true / **false** |
| obs_scale / obs_subtract_mean | 255.0 / 0.0 |
| encoder | conv, convnet_simple, custom "vizdoom", 1 extra FC layer |
| hidden_size | 512 |
| rnn_type / rnn_num_layers | gru / 1 |
| nonlinearity | elu |
| policy_initialization | orthogonal, gain 1.0 |
| env_frameskip / env_framestack | 4 / **4** |
| res_w x res_h | 128 x 72 |
| force_envs_single_thread | true |
| train_for_env_steps (target) | 4,000,000,000 |
| seed | 8433121 |

**Remaining keys in `cfg.json`:**

- **System and sampling:**
  - device gpu
  - num_policies 1
  - serial_mode false, batched_sampling false
  - num_batches_to_accumulate 2
  - worker_num_splits 2
  - policy_workers_per_policy 1
  - decorrelate_experience_max_seconds 0
  - decorrelate_envs_on_one_worker true
  - set_workers_cpu_affinity true
  - actor_worker_gpus []
  - default_niceness 0
  - env_gpu_actions false
  - pixel_format CHW
  - fps 35
- **Loss and optimisation:**
  - kl_loss_coeff 0.0
  - value_bootstrap false
  - lr_schedule_kl_threshold 0.008
  - normalize_input_keys null
- **Policy:**
  - actor_critic_share_weights true
  - adaptive_stddev true
  - initial_stddev 1.0
- **Checkpointing and logging:**
  - train_for_seconds 1e10
  - save_every_sec 120
  - keep_checkpoints 3
  - load_checkpoint_kind latest
  - save_milestones_sec -1
  - save_best_every_sec 5
  - save_best_metric reward
  - save_best_after 100000
  - experiment_summaries_interval 10
  - stats_avg 100
  - benchmark false
  - with_wandb true
- **PBT (off):**
  - period 5e6, start_mutation 2e7
  - replace_fraction 0.3, mutation_rate 0.15
  - replace_reward_gap 0.1, absolute gap 1e-6
  - optimize_batch_size false, optimize_gamma false
  - target true_objective
  - perturb 1.05 to 1.5
  - pbt_mix_policies_in_one_env true
- **Doom env:**
  - num_agents -1, num_humans 0, num_bots -1
  - start_bot_difficulty null
  - timelimit null
  - wide_aspect_ratio false

**Scores reported (they differ by source):**
- **Model card:** 56.20 +/- 6.72 mean reward, self-reported. https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle
- **Docs page:** 59.37 +/- 3.93, described as "evaluation over 10 runs". https://www.samplefactory.dev/09-environment-integrations/vizdoom/

**Training frames reached:** not stated on the card.
- The API file listing (https://huggingface.co/api/models/andrewzhang505/sample-factory-2-doom-battle) shows a final checkpoint named `checkpoint_000488283_4000014336`.
- The second number matches the 4e9 target. Reading it as env steps is an inference and **not verified**.
- The best checkpoint is named `..._3550044160_reward_57.570`.

### 2b. edbeeching/doom_battle2_2222

**Config:** https://huggingface.co/edbeeching/doom_battle2_2222/raw/main/cfg.json

This config uses the current schema. It was produced at sample-factory git hash `9da68b57`.

| Key | Value |
|---|---|
| env | doom_battle2 |
| algo | APPO, async_rl true |
| num_workers x num_envs_per_worker | 20 x 12 (240 envs, derived) |
| batch_size / num_epochs / num_batches_per_epoch | 2048 / 1 / 1 |
| rollout / recurrence | 32 / 32 |
| shuffle_minibatches | false |
| gamma / gae_lambda | 0.99 / 0.95 |
| learning_rate / lr_schedule | 0.0001 / constant |
| exploration_loss / coeff | symmetric_kl / 0.001 |
| value_loss_coeff | 0.5 |
| ppo_clip_ratio / ppo_clip_value | 0.1 / 0.2 |
| with_vtrace | false |
| max_policy_lag | 1000 |
| max_grad_norm | 0.0 |
| adam eps / beta1 / beta2 | 1e-6 / 0.9 / 0.999 |
| reward_scale / reward_clip | 1.0 / 1000.0 |
| normalize_input / normalize_returns | true / true |
| obs_scale / obs_subtract_mean | 255.0 / 0.0 |
| encoder_conv_architecture | convnet_simple |
| encoder_conv_mlp_layers / encoder_mlp_layers | [512] / [512, 512] |
| decoder_mlp_layers | [] |
| rnn_type / rnn_size / rnn_num_layers | **lstm** / 512 / 1 |
| nonlinearity | **relu** |
| env_frameskip / env_framestack | 4 / 1 |
| eval_env_frameskip | 1 |
| res_w x res_h | 128 x 72 |
| decorrelate_experience_max_seconds | 1 |
| train_for_seconds / train_for_env_steps (limits) | 3,600,000 / 1e10 |
| seed | 2222 |

**Remaining keys in `cfg.json`:**

- **System and sampling:**
  - device gpu
  - num_policies 1
  - serial_mode false, batched_sampling false
  - num_batches_to_accumulate 2
  - worker_num_splits 2
  - policy_workers_per_policy 1
  - decorrelate_envs_on_one_worker true
  - set_workers_cpu_affinity true
  - force_envs_single_thread false
  - actor_worker_gpus []
  - default_niceness 0
  - env_gpu_actions false
  - use_env_info_cache false
  - use_record_episode_statistics false
  - pixel_format CHW
  - fps 35
- **Loss and optimisation:**
  - kl_loss_coeff 0.0
  - value_bootstrap false
  - lr_schedule_kl_threshold 0.008
  - normalize_input_keys null
- **Policy:**
  - policy_initialization orthogonal, gain 1.0
  - actor_critic_share_weights true
  - adaptive_stddev true
  - continuous_tanh_scale 0.0
  - initial_stddev 1.0
- **Checkpointing and logging:**
  - log_to_file true
  - experiment_summaries_interval 10
  - flush_summaries_interval 30
  - stats_avg 100
  - summaries_use_frameskip true
  - heartbeat_interval 10
  - heartbeat_reporting_interval 300
  - save_every_sec 120
  - keep_checkpoints 2
  - load_checkpoint_kind latest
  - save_milestones_sec -1
  - save_best_every_sec 5
  - save_best_metric reward
  - save_best_after 100000
  - benchmark false
  - with_wandb false
- **PBT (off):**
  - period 5e6, start_mutation 2e7
  - replace_fraction 0.3, mutation_rate 0.15
  - replace_reward_gap 0.1, absolute gap 1e-6
  - optimize_gamma false
  - target true_objective
  - perturb 1.1 to 1.5
  - pbt_mix_policies_in_one_env true
- **Doom env:**
  - num_agents -1, num_humans 0, num_bots -1
  - start_bot_difficulty null
  - timelimit null
  - wide_aspect_ratio false

**Score:** 30.93 +/- 0.00 mean reward on doom_battle2, self-reported on the model card. https://huggingface.co/edbeeching/doom_battle2_2222

The docs page reports 36.40 +/- 4.20 for a *different* battle2 model (andrewzhang505/sample-factory-2-doom-battle2).

**Training frames reached:** not stated.
- The API listing (https://huggingface.co/api/models/edbeeching/doom_battle2_2222) shows `checkpoint_000396822_3250765824`, and best `..._3143630848_reward_35.888`.
- Reading the large number as env steps is **not verified**.

### 2c. Where the published runs differ from the current defaults

- **RNN type:** GRU in the default and in 2a; LSTM in 2b.
- **Nonlinearity:** ELU in the default and in 2a; ReLU in 2b.
- **Frame stack:** 4 in 2a, 1 in 2b.
- **V-trace:** on in 2a, off in 2b.
- **Return normalisation:** off in 2a, on in 2b.
- **Envs:** both runs used about 240 to 320 envs and a 4e9 to 1e10 step budget.

---

## 3. CleanRL single-file PPO variants

**Repo:** https://github.com/vwxyzjn/cleanrl (master; files fetched 2026-11-25)

The sentences on reset and minibatching below are written in our own words.

| Parameter | `ppo.py` | `ppo_atari_lstm.py` | `ppo_rnd_envpool.py` |
|---|---|---|---|
| env_id | CartPole-v1 | BreakoutNoFrameskip-v4 | MontezumaRevenge-v5 |
| total_timesteps | 500,000 | 10,000,000 | 2,000,000,000 |
| learning_rate | 2.5e-4 | 2.5e-4 | 1e-4 |
| anneal_lr | True | True | True |
| num_envs | 4 | 8 | 128 |
| num_steps (rollout) | 128 | 128 | 128 |
| batch / minibatch (derived from formulas) | 512 / 128 | 1024 / 256 | 16,384 / 4,096 |
| num_minibatches | 4 | 4 | 4 |
| update_epochs | 4 | 4 | 4 |
| gamma | 0.99 | 0.99 | 0.999 (extrinsic) |
| int_gamma | - | - | 0.99 |
| gae_lambda | 0.95 | 0.95 | 0.95 |
| norm_adv | True | True | True |
| clip_coef | 0.2 | 0.1 | 0.1 |
| clip_vloss | True | True | True (extrinsic only; intrinsic value loss unclipped) |
| ent_coef | 0.01 | 0.01 | 0.001 |
| vf_coef | 0.5 | 0.5 | 0.5 |
| max_grad_norm | 0.5 | 0.5 | 0.5 |
| target_kl | None | None | None |
| int_coef / ext_coef | - | - | 1.0 / 2.0 |
| update_proportion | - | - | 0.25 |
| num_iterations_obs_norm_init | - | - | 50 |

**Sources:**
- https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo.py
- https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py
- https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_rnd_envpool.py

### 3.1 Network sizes

**`ppo.py`:**
- Separate actor and critic MLPs, each with 2 x 64 hidden units and Tanh.
- Orthogonal init: hidden gain sqrt(2), actor output 0.01, critic output 1.0; biases 0.

**`ppo_atari_lstm.py`:**
- Input: one 84x84 grayscale frame (frame stack 1).
- Conv: (32, 8, 4), (64, 4, 2), (64, 3, 1), then Linear(3136 -> 512), all ReLU.
- **LSTM: 512 -> 128, 1 layer.** Weights orthogonal with gain 1.0, biases 0.
- Actor: Linear(128 -> actions), gain 0.01. Critic: Linear(128 -> 1), gain 1.
- Env wrappers: noop-reset 30, MaxAndSkip 4, episodic life, fire-reset, reward clip, resize 84x84, grayscale.

**`ppo_rnd_envpool.py` policy:**
- Input: 4 stacked 84x84 frames, divided by 255.
- Conv: the same Atari stack as above.
- Trunk: Linear(3136 -> 256), ReLU, Linear(256 -> 448), ReLU.
- Extra layer: Linear(448 -> 448).
- Actor: Linear(448 -> 448), then Linear(448 -> actions).
- Two value heads (extrinsic and intrinsic), each Linear(448 -> 1).

**`ppo_rnd_envpool.py` RND networks** (input: a single 84x84 frame, the last frame of the stack):
- **Target (frozen):** 3 LeakyReLU convs (same shapes as the Atari stack), then Linear(3136 -> 512).
- **Predictor:** the same convs, then Linear(3136 -> 512), ReLU, Linear(512 -> 512), ReLU, Linear(512 -> 512).

**`ppo_rnd_envpool.py` env settings:**
- envpool with `episodic_life=True`, `reward_clip=True`, sticky actions `repeat_action_probability=0.25`.
- Frame skip is not set in the script (envpool default, **not verified**).

### 3.2 RND normalisation details (`ppo_rnd_envpool.py`)

- **Observation normalisation:** running mean and std over the single RND input frame.
  - It is seeded by 50 x 128 steps of random actions before training, then updated every iteration.
  - The input is normalised as (x - mean) / std and clipped to [-5, 5].
- **Intrinsic reward:** half the squared prediction error, summed over features.
  - It is divided by the running std of a per-env discounted sum of intrinsic rewards (discount `int_gamma`).
  - It is not mean-centred.
- **Intrinsic GAE ignores episode ends** (non-episodic). Extrinsic GAE respects them.
- **Combined advantage:** 1.0 x intrinsic + 2.0 x extrinsic. The combined advantage is then normalised per minibatch.
- **Predictor loss:** computed on a random 25% mask of samples (`update_proportion`).
- **Total loss:** pg_loss - ent_coef x entropy + vf_coef x (ext_v_loss + int_v_loss) + forward_loss.

### 3.3 LSTM reset and minibatching

**`ppo.py`:**
- There is no recurrent state, so there is nothing to reset.
- Terminations and truncations are OR-ed into one done flag, so GAE never bootstraps across a time-limit cut.
- Minibatches are random shuffles of flattened single transitions.

**`ppo_atari_lstm.py`:**
- **Reset:** at every timestep, the hidden and cell states are multiplied by (1 - done) before the LSTM step. Memory is wiped at the step where a new episode starts, both while collecting and when replaying the rollout during the update.
- **Minibatching:** minibatches are built from whole environments, not single transitions. Each minibatch takes `num_envs / num_minibatches` = 2 environment columns with all 128 steps in time order.
- **Initial state:** the LSTM state saved at the start of the rollout is sliced to match, so every sequence is replayed from its true starting memory.

**`ppo_rnd_envpool.py`:**
- There is no LSTM; the policy is feed-forward. There is nothing to reset.
- Minibatches are random shuffles of the flattened 16,384 transitions.

---

## 4. Older ViZDoom agents (quick look)

### 4a. Arnold: DRQN, 2017 ViZDoom competition

**Source:** README at https://github.com/glample/Arnold. Repo facts are already recorded in `vizdoom-and-rl-references.md`: no licence, so treat the code as AVOID.

| Parameter | Value (README defaults) |
|---|---|
| network_type | dqn_rnn (DRQN) |
| recurrence / n_rec_layers / hidden_dim | lstm / 1 / 512 |
| n_rec_updates | 5 |
| input | 60 x 108 screen (height x width), RGB (`gray` false), hist_size 4; depth off |
| frame_skip | 4 by default. Example commands: 2 (defend_the_center), 5 (health_gathering), 4 (deathmatch) |
| batch_size | 32 |
| replay_memory_size | 1,000,000 |
| epsilon decay | 0 -> 1,000,000 steps, final 0.1 |
| gamma | 0.99 |
| update_frequency | 4 |
| optimizer | rmsprop, lr 0.0002 |
| clip_delta / dropout | 1.0 / 0.5 |
| game_features (auxiliary prediction targets) | "target,enemy" by default; "enemy" in the deathmatch example |
| action_combinations | default "attack+move_lr;turn_lr;move_fb"; deathmatch "move_fb;move_lr;turn_lr;attack" |
| speed / crouch | on / off |

**Training length:** the README states about 4M steps for defend_the_center (to 56 frags) and about 1.5M steps for health_gathering supreme. It gives none for deathmatch.

### 4b. DFP, "Learning to Act by Predicting the Future"

**Repo:** https://github.com/isl-org/DirectFuturePrediction
- 150 stars
- archived; last push 2024-09-02
- license "Other" (NOASSERTION), terms not read
- Source of the table: `examples/D3_battle/run_exp.py` (https://raw.githubusercontent.com/isl-org/DirectFuturePrediction/master/examples/D3_battle/run_exp.py) and `maps/D3_battle.cfg`

| Parameter | Value |
|---|---|
| Learning type | supervised prediction of future measurements (not PPO or DQN) |
| resolution / colour | 84 x 84, GRAY (cfg renders at 160x120, GRAY8) |
| frame_skip | 4 |
| num_simulators | 8 |
| conv | (32, 8, 4), (64, 4, 2), (64, 3, 1) |
| image FC / measurement FC / joint FC | 512 / 3 x 128 / 512 then output |
| measurements | AMMO2, HEALTH, USER2 (kills); normalised x / 100 - 0.5 |
| prediction offsets | [1, 2, 4, 8, 16, 32] steps; targets scaled by [7.5, 30, 1] |
| objective weights | temporal [0, 0, 0, 0.5, 0.5, 1]; measurement [0.5, 0.5, 1] |
| exploration | epsilon = 0.02 + 145000 / (step + 150000) |
| memory_capacity | 20,000 |
| batch_size | 64 |
| optimizer | Adam, lr 1e-4, beta1 0.95, eps 1e-4; lr x 0.3 every 250,000 iterations |
| new_memories_per_batch | 8 |
| num_train_iterations | 820,000 |
| buttons | MOVE_FORWARD, MOVE_BACKWARD, MOVE_RIGHT, MOVE_LEFT, TURN_LEFT, TURN_RIGHT, ATTACK, SPEED (`action_format` "enumerate") |
| episode_timeout | 2100 tics |

**Not verified:** the number of discrete actions after enumeration.

**Derived:** 820,000 iterations x 8 new memories gives about 6.6M environment steps. That assumes one memory is one env step, which is not verified.

---

## 5. Side-by-side: where sources agree and differ

Key: SF = Sample Factory; RND = CleanRL RND; n/a = the value has no equivalent in that method.

| Value | SF Doom (battle) | SF duel / deathmatch | CleanRL ppo | CleanRL LSTM | CleanRL RND | Arnold (DRQN) | DFP |
|---|---|---|---|---|---|---|---|
| Learning rate | 1e-4 const | 1e-4 const | 2.5e-4 annealed | 2.5e-4 annealed | 1e-4 annealed | 2e-4 (RMSprop) | 1e-4, step decay |
| Gamma | 0.99 | 0.995 | 0.99 | 0.99 | 0.999 ext / 0.99 int | 0.99 | n/a (fixed horizons) |
| GAE lambda | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | n/a | n/a |
| Rollout length | 32 | 32 | 128 | 128 | 128 | n/a (replay) | n/a |
| BPTT / recurrence | 32 | 32 | n/a | 128 (full rollout) | n/a | n_rec_updates 5 | n/a |
| Entropy-type coefficient | symmetric_kl 0.001 (not plain entropy) | same | ent 0.01 | ent 0.01 | ent 0.001 | n/a (epsilon-greedy) | n/a |
| Clip ratio | 0.1 | 0.1 | 0.2 | 0.1 | 0.1 | n/a | n/a |
| Max grad norm | 0.0 in runs (4.0 default) | 4.0 default | 0.5 | 0.5 | 0.5 | clip_delta 1.0 | n/a |
| Frame skip | 4 | 2 | n/a | 4 | not verified | 4 (2 to 5 per scenario) | 4 |
| RNN size | GRU 512 (LSTM 512 in 2b) | GRU 512 | none | LSTM 128 | none | LSTM 512 | none |
| Resolution | 128x72 RGB | 128x72 | n/a | 84x84 gray | 84x84 gray x4 | 60x108 RGB | 84x84 gray |
| Envs in parallel | 576 (72x8) | 1,152 to 2,304 | 4 | 8 | 128 | not verified | 8 |

**Agreement:**
- **GAE lambda:** 0.95 in every PPO source.
- **Gamma:** 0.99 in SF battle, all CleanRL PPO files (extrinsic, except RND's 0.999) and Arnold. The exception is SF duel/deathmatch at 0.995.
- **Clip ratio:** 0.1 in SF and CleanRL LSTM/RND. The exception is plain `ppo.py` at 0.2.
- **Frame skip:** 4 in SF battle, CleanRL Atari, DFP and Arnold's default. The exceptions are SF duel/deathmatch at 2 and Arnold's per-scenario 2 to 5.
- **Conv stack:** an Atari-style 3-layer conv in all pixel sources. SF's `convnet_simple` widens the last layer to 128 channels with stride 2.

**Disagreement:**
- **Learning rate:** constant 1e-4 (SF) vs annealed 2.5e-4 (CleanRL).
- **Rollout and BPTT:** 32 vs 128.
- **Entropy regulariser:** symmetric_kl 0.001 vs entropy 0.01 or 0.001.
- **Gradient clipping:** 0.0 in SF runs (meaning not verified) vs 0.5.
- **Recurrent width and type:** GRU 512, LSTM 512, LSTM 128.
- **Epochs per batch:** SF 1 vs CleanRL 4.

## 6. Compute assumptions vs the team laptop

The laptop has 24 logical CPUs, 15.4 GB RAM, Windows, and CPU-only PyTorch.

**Measured baseline** (`reports/runs/2026-11-25-env-single-bench.md`): one env gives about 370 decisions/s conservatively (up to 703) at frame skip 4. That is about 1,474 tics/s. Peak working set is about 137 MB per env (Python plus `vizdoom.exe`). Vectorised throughput was **not measured**.

**Far beyond the laptop:**
- **SF battle/battle2:** 4e9 env steps on a 36-core, 4-GPU server with 72 x 8 envs.
  - Derived: if the 4e9 counts engine frames (unit **not verified**, see §1.1), one env at 1,474 tics/s needs about 31 days. If it counts decisions, about 4x longer.
  - Even with perfect 16-env scaling, which is not measured, that is days of engine time alone, before any CPU-only learner cost.
  - 576 envs at about 137 MB each is about 79 GB (derived), roughly 5x the laptop's RAM.
- **SF duel/deathmatch:** 360,000 s (about 100 h) to 3,600,000 s (about 42 days) with PBT over 8 policies and 1,000+ envs.
- **CleanRL RND:** 2e9 steps with 128 envs, sized for GPU and envpool. 128 Doom envs would need about 17.5 GB RAM (derived), which is over budget.
- **Batch 2048 x conv + 512 GRU/LSTM:** fine on a GPU. On a CPU-only learner, expect the update to compete with the env workers for the 24 threads. The cost has not been measured.

**Laptop-scale reference points:**
- CleanRL LSTM's shape: 8 envs, 128-step rollouts, a 1,024-sample batch, LSTM 128, 1e7 steps.
  - Derived: about 7.5 h of single-env engine time at 370 decisions/s. Less with parallel envs, if the scaling holds.
- DFP: 8 simulators and about 6.6M steps (derived).
- Arnold: 1.5M to 4M steps for single scenarios.
- These are the closest sourced shapes to what the laptop can run.
- *Our own interpolation, not a sourced value:* 8 to 16 envs, an RNN between CleanRL's 128 and SF's 512, rollout and BPTT of 32 to 128, and 1e7 to 5e7 steps.

## Prompt-injection check

No fetched page or file contained instructions aimed at an agent.

## Sources

- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sample_factory/cfg/cfg.py
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sample_factory/model/encoder.py
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/doom_params.py
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/doom_utils.py
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/doom_model.py
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/action_space.py
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/wrappers/reward_shaping.py
- https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/sf_examples/vizdoom/doom/wrappers/additional_input.py
- https://www.samplefactory.dev/09-environment-integrations/vizdoom/
- https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle (card, `raw/main/cfg.json`, `/api/models/...`)
- https://huggingface.co/edbeeching/doom_battle2_2222 (card, `raw/main/cfg.json`, `/api/models/...`)
- https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo.py
- https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_atari_lstm.py
- https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_rnd_envpool.py
- https://github.com/glample/Arnold
- https://github.com/isl-org/DirectFuturePrediction, https://api.github.com/repos/isl-org/DirectFuturePrediction
- https://raw.githubusercontent.com/isl-org/DirectFuturePrediction/master/examples/D3_battle/run_exp.py
- https://raw.githubusercontent.com/isl-org/DirectFuturePrediction/master/maps/D3_battle.cfg
