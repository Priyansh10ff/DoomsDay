# Lesson 05: Memory and the action heads (GRU, hidden state, six heads, inference)

Files taught: `agent/model.py` (the core, the heads and the sequence pass), `agent/distributions.py`, `agent/inference.py`, and the tests `tests/test_a0_model.py:54-134`.

Design records: ADR 0003 (Stage A0: "GRU (256), hidden state reset on episode end", six policy heads, two value heads, auxiliary heads) and ADR 0002 (factorized heads, argmax or sample at inference). **All ADRs are "Proposed", not ratified.** Nothing has been trained on DOOM yet.

Builds on lesson 03 (the six heads and their sizes) and lesson 04 (the 192-number fused vector).

---

## 1. What problem this piece solves

The observation only shows what is in front of the agent right now. But DOOM needs memory:

- the agent saw a blue key in the left corridor 30 seconds ago and now stands at a blue door;
- an imp went out of view behind a pillar two steps ago;
- the agent just turned away from a wall it was stuck against.

None of that is in the current observation. The network needs a **memory that it updates every decision** and carries forward. Then, from that memory, it must produce the six choices of lesson 03, plus a guess of how well things are going (used in training).

There is also a hard rule here: **every action must come from the network** (rule 4.1). So the step from "network output" to "buttons" must contain nothing but picking from the network's own probabilities.

## 2. The idea in plain words

Think of a **ship's logbook kept by a navigator who can only see out of one porthole**. Every few seconds the navigator looks out (the observation), reads the last log entry (the memory), and writes a new entry that blends the two: some of the old entry is kept, some is replaced by what is new. The captain reads only the newest entry to give orders.

- The **log entry** is the **hidden state**: 256 numbers the network carries from one decision to the next.
- **Writing the new entry** is the **GRU** step.
- **The captain's orders** are the six action heads.
- When the voyage ends, the logbook is closed and a fresh one starts (the hidden state is **reset to zeros** at the start of every episode).

Where the analogy stops being accurate:

- The navigator writes in a language nobody chose. The 256 numbers have no assigned meanings; training shapes them.
- A real logbook keeps every page. Ours keeps **only the newest entry**, of a fixed size. Anything not carried forward in those 256 numbers is lost. That is why ADR 0003 plans a larger memory later (Stage B) and calls the GRU memory "only a recurrent vector".
- The captain's orders are not certain. Each head gives **probabilities**, and the action is drawn from them (or the most likely one is taken).

## 3. A walk through our actual code

### 3.1 Recurrent memory: the GRU core, `model.py:46` and `:84-93`

A **recurrent** network is one that feeds part of its own output back in as input at the next step. That fed-back part is the **hidden state**.

A **GRU** (gated recurrent unit) is a standard recurrent layer. At each step it takes the new input `x` (our 192 numbers) and the old hidden state `h` (256 numbers) and computes the new hidden state with two learned **gates**. A gate is a set of numbers between 0 and 1, computed from `x` and `h`, used to mix things:

- the **reset gate** decides how much of the old state to use when proposing new content;
- a **candidate** new state is computed from `x` and the (reset-gated) old state;
- the **update gate** decides, number by number, how much to keep the old state versus take the candidate.

So each of the 256 numbers can choose to hold its value for a long time (update gate near "keep") or be overwritten quickly. That is what lets a GRU carry "I saw a blue key" across many steps. The code uses PyTorch's built-in `nn.GRUCell(192, 256)` (`model.py:46`); the gate equations are PyTorch's, our code decides how it is fed and reset.

`core_step(x, h, episode_start)` (`model.py:84-93`):

- `:89-90`: **reset**. `episode_start` is 1 for an environment whose observation is the first of a new episode. The old hidden state is multiplied by `1 - episode_start`, so it becomes zeros there and is unchanged everywhere else. Only the environments that just started are reset; the others keep their memory (`tests/test_a0_model.py:74-88`).
- `:91-92`: **the memoryless switch.** If `cfg.recurrent` is false, the state is zeroed at *every* step. That turns the same network into one with no memory. It exists only for the test in lesson 07, which compares the two.
- `:93`: one GRU step.

The convention is written at the top of the file (`model.py:3-5`): "`episode_start[t]` is 1 when obs[t] is the first observation of an episode; the hidden state is zeroed BEFORE obs[t] is processed". The same rule is used when collecting experience and when training (lessons 06 and 07). If the two disagreed, training would be computed on a different memory from the one that chose the actions.

### 3.2 One step and a whole sequence: `model.py:106-123`

- `step(obs, h, episode_start)` (`:106-109`): encode (lesson 04), one GRU step, then the heads. Used when playing.
- `forward_sequence(obs, h0, episode_start)` (`:111-123`): the same thing for a whole stretch of T decisions. It encodes all T x B observations at once (faster), then runs the GRU step by step from a stored starting state `h0`, applying the resets. Training uses this to recompute what the network would output over a stored sequence (lesson 07).

`tests/test_a0_model.py:54-71` checks that running `forward_sequence` over 9 steps, with resets in the middle for some environments, gives exactly the same outputs as calling `step` 9 times.

### 3.3 The output heads: `model.py:48-64` and `:95-103`

All outputs read the same 256-number hidden state. Each is a single linear layer.

**Six policy heads** (`:48`): `Linear(256, n)` for n = 3, 3, 7, 2, 2, 2. Each produces **logits**: raw scores, one per option, that are not yet probabilities. A **softmax** turns a list of logits into probabilities that are positive and sum to 1; a higher logit means a higher probability.

`:54-56`: these layers start with very small random weights (`policy_head_init_gain = 0.01`, marked PROVISIONAL in `model_config.py:39-40`). Tiny weights mean all logits start near 0, so every option starts with nearly equal probability. An untrained agent then tries everything, instead of starting with a strong arbitrary habit.

**Two value heads** (`:49-50`). A **value** is the network's guess of how much total future reward will follow from the current situation. It is not used to choose actions; it is used in training to judge whether an action turned out better or worse than expected (lesson 06).

- `value_ext`: value for the game's ("extrinsic") reward.
- `value_int`: reserved for a future exploration ("intrinsic") reward (ADR 0005). **No loss trains it yet**; it is computed and ignored.

**Three auxiliary heads** (`:51-53`), training-only:

- `aux_keys`: 3 logits, "do I hold the blue / yellow / red key?";
- `aux_damage`: 1 logit, "will I take damage soon?";
- `aux_monsters`: 1 number, "how many monsters can I see?".

Their labels come from `info` and from the true observation during training (lesson 07). The model's docstring is explicit: "Auxiliary heads are training-only loss terms; nothing reads them to act" (`model.py:7`). Their purpose is to push the hidden state to *contain* this information, which the policy heads can then use. ADR 0003 cites the MERLIN result: memory trained only by the reward signal fails over long delays, and a predictive extra loss helps.

`heads(h)` (`:95-103`) returns all of these in an `AgentOutput` (`:25-32`).

### 3.4 Six choices as one action: `agent/distributions.py`

A **probability distribution** over options says how likely each option is. A **categorical distribution** is one over a fixed list of options. We have six of them, one per head, treated as independent. `FactorizedCategorical` (`distributions.py:12-50`) bundles them. "Factorized" means the joint probability of an action is the product of the six heads' probabilities.

- `:19`: `log_softmax` turns each head's logits into **log-probabilities** (the natural log of the probabilities). Working in logs turns products into sums and avoids tiny numbers.
- `log_prob(actions)` (`:22-29`): for each head, pick the log-probability of the chosen option, and **add the six**. The sum of logs is the log of the product, so this is the log-probability of the whole action.
- `entropy_per_head()` and `entropy()` (`:31-37`): **entropy** measures how spread out a distribution is: 0 when one option is certain, and largest (the log of the number of options) when all are equally likely. It is computed per head and summed. Lesson 07 uses it to encourage exploration, and per-head entropy is logged.
- `sample(generator)` (`:39-47`): draw one option per head at random according to the probabilities. It takes an explicit random generator, so runs are repeatable from a seed (`tests/test_a0_model.py:112-121`).
- `mode()` (`:49-50`): the most likely option per head (**argmax**: the index of the largest value).

### 3.5 The inference path: `agent/inference.py`

**Inference** means using the trained network to play, with no learning. This is the code path the judges' "every action from the network" rule applies to most directly.

`PolicyRunner` (`inference.py:18-34`):

- `:19-26`: it is given the model, the number of environments, `argmax_heads` (one true/false per head: take the most likely option, or sample), and a seed. It holds one hidden state per environment.
- `:28-34`, `act(obs, episode_start)`: under `torch.no_grad()` (no learning bookkeeping), call `model.step` with the stored hidden state, store the new state, build the distribution, and per head take `mode()` or `sample()`. Return six integers per environment.

That is the whole path: **forward pass, then sample or argmax per head**. The module docstring says it: "Nothing else ... no code here inspects the observation to change an action" (`inference.py:1-4`). There is no "if the network is unsure, do X" and no override. Which heads are argmax is fixed before play starts; it does not depend on what the agent sees. ADR 0002 says this choice should live in an inference config and be evaluated on held-out maps. **At present it is only a constructor argument; no inference config file, and no script that loads a checkpoint and plays DOOM, exist yet.** The caller must also pass `episode_start = 1` at the start of each episode so the memory is reset.

`tests/test_a0_model.py:124-134` checks that argmax is deterministic regardless of seed, and that sampling with the same seed repeats exactly.

### 3.6 Building the model reproducibly: `model.py:126-134`

`build_agent(spec, cfg, seed)` creates the model inside `torch.random.fork_rng`, so the starting weights depend only on `seed` and building a model does not disturb any other random numbers (`tests/test_a0_model.py:101-109`). `count_parameters` counts learned numbers.

## 4. A tiny worked example with real numbers

Same demo as lesson 04: the A0 model built with seed 0, **untrained**, given one real observation (A0 settings, step 7 of a `DoomEnv` run), 4 torch threads. Real output:

```
  GRU h (1, 256)  logits [(1, 3), (1, 3), (1, 7), (1, 2), (1, 2), (1, 2)]  value_ext [0.01724710687994957]  aux_keys (1, 3)
  head 0 probs [0.3337000012397766, 0.33309999108314514, 0.33320000767707825]
  head 1 probs [0.3336000144481659, 0.33309999108314514, 0.33329999446868896]
  head 2 probs [0.1428000032901764, 0.1429000049829483, 0.1429000049829483, 0.14270000159740448, 0.1428000032901764, 0.1428000032901764, 0.14300000667572021]
  head 3 probs [0.5002999901771545, 0.49970000982284546]
  head 4 probs [0.5, 0.5]
  head 5 probs [0.49970000982284546, 0.5002999901771545]
  entropy per head [1.0986120700836182, 1.0986120700836182, 1.9459099769592285, 0.6931470036506653, 0.6931471824645996, 0.6931470036506653]  max ln(n): [1.0986, 1.0986, 1.9459, 0.6931, 0.6931, 0.6931]
  sampled [[2, 1, 5, 1, 1, 0]]  logp [-6.224734306335449]  sum ln(1/n) -6.222576268071369  mode [[0, 0, 6, 0, 1, 1]]
```

Reading it:

- **Shapes.** Hidden state 256; six logit lists of sizes 3, 3, 7, 2, 2, 2; one value.
- **Near-uniform start.** Every head gives each option almost exactly 1/n (1/3 ≈ 0.333, 1/7 ≈ 0.143, 1/2). That is the small initial weights at work.
- **Entropy.** Each head's entropy equals its maximum, ln(n): ln 3 = 1.0986, ln 7 = 1.9459, ln 2 = 0.6931. The untrained agent is as undecided as possible.
- **Log-probability of the sampled action.** By hand: if every head were exactly uniform, the log-probability would be ln(1/3) + ln(1/3) + ln(1/7) + 3 ln(1/2) = -6.2226. The real value is -6.2247, very close.
- **Argmax of an untrained network.** `mode` is `[0, 0, 6, 0, 1, 1]`: "turn hard right, hold USE, run". It comes from differences in the fourth decimal place. An argmax policy would repeat that choice every step regardless of what it sees. That is why training samples, and why ADR 0002 wants argmax-versus-sample decided per head on held-out maps.

**The memory and its reset.** Real output:

```
  same obs, keep h vs reset h: value 0.03364763781428337 0.01724710687994957  reset equals fresh: True
  PolicyRunner argmax act: [[0, 0, 6, 0, 1, 1]]
```

The same observation was fed twice more: once keeping the hidden state from the first step, once with `episode_start = 1`. The value differs (0.0336 against 0.0172), because the memory changed what the network computes. With the reset, the new hidden state is exactly the one computed from a fresh start ("reset equals fresh: True"). `PolicyRunner` with argmax on all heads returns the same `[0, 0, 6, 0, 1, 1]` as `mode`.

**Where the parameters are.** From lesson 04's count, the GRU (`core`) has 345,600 of the 474,474 parameters. By hand: a GRU cell has three sets (reset gate, update gate, candidate), each with input weights 192 x 256, state weights 256 x 256 and two bias vectors of 256: 3 x (49,152 + 65,536 + 512) = 3 x 115,200 = **345,600**.

## 5. Why we chose this design over the alternatives

From ADR 0003 and ADR 0002 (**Proposed, not ratified**):

- **GRU memory first, a map later.** ADR 0003 builds in stages: Stage A0 is the GRU (this lesson); Stage B adds an ego-centric neural map (a grid of learned features addressed by the agent's own motion), only if gates are met; ADR 0006 further makes Stage B wait for the organizers' answer on whether telemetry is available. The research found grid memory beat a plain LSTM (a recurrent layer similar to the GRU) on unseen DOOM mazes, but in a form that used absolute coordinates, which we may not use. So the GRU is the safe first step, honestly weaker as "spatial memory".
- **A transformer (stacked attention layers, lesson 04) over history (option D) was rejected**: its cost grows with the length of history, and a keycard detour is thousands of decisions.
- **Six independent heads** (ADR 0002, lesson 03): joint log-probability and entropy are sums over heads, which keeps the training loss simple.
- **Auxiliary heads** (ADR 0003): keys held, damage soon, visible monsters, as training-only losses to shape the memory. ADR 0003 states the risk plainly: "If any auxiliary output were fed back into action selection outside the network, it would breach rule 4.1. They are loss terms only."
- **Inference = forward pass + sample/argmax** (ADR 0002): "no code inspects the observation to override a head".

## 6. How it connects

- **Lesson 04** produced the 192-number input to the GRU.
- **Lesson 06 (next)** shows how the trainer runs `step` across many environments, stores everything, sets `episode_start` after each episode end, and uses `value_ext` to compute advantages.
- **Lesson 07** shows how `forward_sequence` recomputes the stored stretch from the stored starting state, and how the policy, value, entropy and auxiliary losses train all of this.

## 7. Common mistakes, and how we would notice them

| Mistake | How it would show up |
|---|---|
| Not resetting the hidden state at a new episode. | Memory leaks between episodes: the agent may "remember" a key from the previous level. Hard to see in reward; checked by `tests/test_a0_model.py:74-88`. In inference, it happens if the caller forgets `episode_start = 1`. |
| Resetting (or zeroing) the state every step. | That is the memoryless ablation. On the toy memory task it gives chance-level success (0.505 in the worker's run, lesson 07). |
| Using different reset rules in collection and in training. | The recomputed log-probabilities no longer match the stored ones. The trainer logs `first_minibatch_ratio_max_dev`, which should be about 0 (lesson 07). |
| Using argmax on all heads from the start, or in training. | The agent repeats one action (in the demo, "turn hard right, hold USE, run") and never explores. In logs: per-head entropy stays high but behaviour is fixed; episode returns do not move. |
| Reading an auxiliary output to choose actions ("if aux_keys says blue, go to the blue door"). | A rule 4.1 breach (a hand-written rule using a network output). Only `PolicyRunner.act` should turn outputs into actions, and it uses only the six policy heads. |
| Thinking `value_int` is trained. | It is not; no loss uses it yet. Its output is meaningless until ADR 0005's exploration reward is built. |

## 8. Check your understanding

1. Four environments run together. After a step, environment 2 ended its episode and was reset; the others continue. What value of `episode_start` does each get on the next step, and what happens inside `core_step` for each?
2. Why does the untrained network give almost exactly 1/3, 1/3, 1/3 for the `move` head, and why is that useful?
3. A teammate wants the inference code to "press USE automatically whenever the network's `use` probability is between 0.4 and 0.6, to avoid hesitation". What is wrong with that?

---

## Answers

1. `episode_start = [0, 0, 1, 0]`. In `core_step`, the stored hidden state is multiplied by `1 - episode_start` (`model.py:89-90`): environments 0, 1 and 3 keep their memory, and environment 2's memory becomes all zeros before its first observation is processed. Then the GRU step runs for all four.
2. The policy layers start with very small weights (gain 0.01, `model.py:54-56`), so all logits are near 0, and the softmax of equal logits is uniform. It is useful because an untrained agent should try every option; a strong arbitrary preference at the start would hide most actions from learning.
3. It is a hand-written rule that turns network output into an action outside the network's own sampling or argmax: a "safety override" in the decision path, which rule 4.1 forbids. The inference path must be only the forward pass plus a fixed per-head sample-or-argmax choice (`inference.py:1-4`). If hesitation is a problem, it has to be fixed by training (for example rewards or entropy settings), not by code.
