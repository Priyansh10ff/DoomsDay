# Lesson 04: The encoders (from observation to one feature vector)

Files taught: `agent/obs_spec.py`, `agent/tensorize.py`, `agent/encoders.py`, `agent/model.py:35-82` (building the encoders and fusing them), `configs/model_config.py`, and the tests `tests/test_a0_encoders.py` and `tests/test_a0_model.py:25-51`.

Design records: ADR 0003 (model architecture, Stage A0) and ADR 0006 (the model must size itself from the observation space). **Both are marked "Proposed, pending ratification by the user."** Nothing has been trained on DOOM yet. The code in this lesson has only been run in tests, on a toy task (lesson 07), and in the small demo below.

Builds on lesson 02 (what each observation key means) and lesson 03 (`prev_action` and the head sizes).

---

## 1. What problem this piece solves

The observation is a dictionary of very different things: 240 distances, a table of up to 24 objects of 14 kinds, 28 body numbers, 6 previous button choices, 3 flags. The part of the network that decides what to do needs **one list of numbers** as input. Something must turn each kind of input into numbers that capture what matters about it, and then join them.

The entity table is the hard part. In one frame the agent may see no monsters; in the next, five imps and a barrel. The rows arrive in whatever order the engine listed them. If the network treated "imp in row 1, barrel in row 2" differently from "barrel in row 1, imp in row 2", it would see two different worlds where there is one. The brief asks us to show how the agent "handles a variable number of enemies". This lesson is our answer.

## 2. The idea in plain words

Think of a **briefing room**. Several specialists each read one kind of report and write a short summary:

- the **range officer** reads the 240 distance readings and writes 64 numbers,
- the **lookout** reads the list of visible things and writes 64 numbers,
- the **medic** reads the body numbers and writes 32 numbers,
- the **clerk** notes what was ordered last time in 16 numbers.

The commander staples the summaries together with the three "which instruments work" flags, and one more layer mixes them into a single 192-number briefing. That briefing goes to the memory (lesson 05).

The lookout works in a particular way. They first write a small card for each visible thing. Then the cards are laid on a table and each card is updated after "looking at" the others ("this imp is next to a barrel"). Finally the lookout asks one fixed question of the whole pile ("what here matters?") and writes the answer as one summary. Because cards on a table have no order, shuffling them cannot change the summary.

Where the analogy stops being accurate:

- The specialists are not given rules. Every summary is computed by learned weights, which start random and are shaped by training.
- The "fixed question" is not written by us. It is a learned vector, the same for every frame.
- The briefing is not readable by humans. Its 192 numbers have no assigned meanings.

## 3. A walk through our actual code

### 3.1 Sizes come from the environment: `agent/obs_spec.py`

`spec_from_spaces(observation_space, action_space)` (`obs_spec.py:35-61`) reads every size the model needs from the environment's own description:

- `:38-40`: the keys the A0 model uses (`A0_KEYS`, `:14-16`) must all exist. `screen` is not among them; A0 does not use pixels.
- `:42-47`: ray bands and columns, and entity count and features, from the array shapes.
- `:48`, `:57`: the number of entity types is the `entity_type` Box's upper bound plus 1. With 14 types (lesson 02) that is 13 + 1 = 14.
- `:49`, `:60`: the head sizes come from `action_space.nvec`, the `[3, 3, 7, 2, 2, 2]` of lesson 03, not from the looser `prev_action` Box.

The result is a frozen `ObsSpec` (`:19-32`). Nothing in the model hard-codes "3 x 80" or "14 types". ADR 0006 required this: when `hazard` was added, or when the screen is switched off, the model follows without code changes. `tests/test_a0_model.py:25-44` builds the model from three different space layouts (13 or 14 types, screen on or off, 3 or 4 ray bands) and checks every output shape.

`agent/tensorize.py:18-19` converts a batch of numpy observations into torch tensors of the right types, **for the A0 keys only**. If a `screen` key is present, it is dropped here.

### 3.2 Model sizes: `configs/model_config.py`

`ModelConfig` (`model_config.py:8-61`; since ADR 0008 it also holds the initialisation switches taught in lesson 09) holds the widths ADR 0003 sets: ray code 64, entity tokens 64 with a 16-number type embedding and 4 attention heads, self code 32, previous-action code 16, fusion 192, memory 256. Some values were marked **PROVISIONAL** by the worker, because no ADR set them; ADR 0007 (`0007-a0-trainer-rulings-and-first-run-prerequisites.md`, ruling 2) has since **accepted** both:

- `prev_action_dim` (`:26-28`): ADR 0003 says "six small embeddings to 16". The code reads this as one table per head, each 16 wide, **added together** into one 16-number vector. Another reading (six 16-wide pieces side by side, 96 numbers) was possible; the worker raised it as a question for `advisor`.
- `policy_head_init_gain` and `learned_token_init_std` (`:38-40`): how large the starting random weights are. No ADR sets them.

### 3.3 The ray encoder: `encoders.py:10-33`

A **1D convolution** is a layer that slides a small window along a sequence and applies the same learned weights at every position. Here the sequence is the 80 ray columns, and the 3 bands are the "channels" (like the red, green and blue of an image). Because the same weights are used everywhere along the row, a pattern such as "a gap between two near walls" is recognised wherever it appears in the view.

- `:16-22`: a stack of `Conv1d` layers, each followed by `ReLU` (a function that keeps positive numbers and turns negatives into 0). The defaults (`model_config.py:11-14`) are two layers with 16 and 32 channels, window 5, step 2, and padding 2. "Step 2" (stride) means the window moves 2 columns at a time, so each layer halves the length: 80, then 40, then 20.
- `:23-24`: run a dummy input once to find how many numbers come out (32 x 20 = 640).
- `:25`: a linear layer and ReLU turn those 640 numbers into the 64-number **ray code**.
- `:28-33`: the forward pass, with shape checks.

### 3.4 The entity set encoder: `encoders.py:36-81`

This is the custom module ADR 0003 asks for. First, some new terms.

- **Embedding**: a learned lookup table that turns a whole number (here, an entity type id) into a short list of learned numbers. Type 1 (monster) and type 13 (hazard) each get their own 16 numbers, which training adjusts.
- **Token**: a fixed-size vector that stands for one item; here, one 64-number vector per visible entity.
- **Attention**: a layer in which each item computes a weighted average over a set of items, with the weights computed from the items' content. In detail: each item is turned (by learned linear maps) into a *query* ("what am I looking for?"), a *key* ("what do I offer?") and a *value* ("what do I pass on?"). The weight item A gives item B comes from how well A's query matches B's key, made to sum to 1 with a softmax (a function that turns scores into positive weights that sum to 1; lesson 05 uses it again). A's output is the weighted average of everyone's values.
- **Self-attention**: attention where the items look at each other (queries, keys and values all come from the same set).
- **Multi-head attention**: several attention calculations side by side, each on a slice of the numbers, so different heads can look for different relationships. Here 4 heads, each with 64 / 4 = 16 numbers. These attention heads are unrelated to the six *action* heads of lesson 03.
- **Mask** (in attention): a list of positions that must receive no weight. Padded rows are masked so nothing can attend to them.
- **Residual connection**: adding a layer's input back to its output (`x + layer(x)`), so the layer only has to learn a change. It makes deeper networks easier to train.
- **Layer normalization**: rescaling each token's numbers to average 0 and spread 1 (then applying a learned scale and shift). It keeps the numbers in a stable range.
- **Permutation invariance**: the output does not change when the input rows are put in a different order.

Now the code, step by step (`forward`, `:63-81`):

1. `:68-70`: **wipe the padded rows.** Wherever the mask is false, features become 0 and type becomes 0 (`pad`). Whatever garbage a padded row held, it can no longer affect anything. `tests/test_a0_encoders.py:70-81` fills padded rows with `NaN` and huge numbers and checks the output is unchanged and finite.
2. `:71`: **one token per row.** Look up the 16-number type embedding, put it next to the 8 features (24 numbers), and pass them through a shared two-layer network (`token_mlp`, `:51-54`) to get a 64-number token. "Shared" means every row uses the same weights.
3. `:72-73`: **add the "no entity" token.** A learned token (`empty_token`, `:55`) is put in front of the list and is always marked valid. Without it, a frame with no visible entities would ask attention to average over nothing, which is undefined (it gives NaN). With it, the empty set has a well-defined code. `tests/test_a0_encoders.py:58-67` checks that with an all-false mask the output is finite and the same whatever the rows contain.
4. `:74-76`: **self-attention with a residual and layer norm.** Each token looks at every valid token (`key_padding_mask=pad` blocks the padded ones) and is updated: `tokens = norm(tokens + attention(tokens))`.
5. `:77-79`: **pooling by a learned query.** A single learned vector (`pool_query`, `:58`) is the query; the tokens are the keys and values. The result is one weighted average of the valid tokens: the 64-number **entity code**. Padded tokens are masked here too.

**Why the order cannot matter.** Nothing in steps 1 to 5 uses a row's position. There is no "position number" added to the tokens (no positional encoding, as the docstring says, `:42`). Self-attention gives each token the same update whichever row it sits in, and the final pooling is a weighted sum, and a sum does not care about order. So shuffling the rows only shuffles the tokens, and the pooled code is the same. The tests check this for 1, 5 and 24 entities (`tests/test_a0_encoders.py:39-55`). They also check that the same 3 entities placed in rows 0-2 or in rows 20, 2 and 11 give the same code (`:84-93`).

**Why any count works.** The mask decides which tokens take part, from 0 (only the "no entity" token) to 24. The output is always 64 numbers.

One detail differs from ADR 0003's wording: the ADR describes "self-attention block (4 heads) with residual". The code also applies **layer normalization** after the residual (`:57`, `:76`). ADR 0003 does not mention it. ADR 0007 ruling 3 later **kept** it and recorded it as an addition to the Stage A0 design: it is applied to each token separately, so it does not affect order-independence or the empty-set case (the tests cover both).

The type table size comes from the spec (`:50`). A type id the table does not have raises an error instead of being silently wrong (`tests/test_a0_encoders.py:96-101`).

### 3.5 The self and previous-action encoders: `encoders.py:84-111`

- `SelfEncoder` (`:84-94`): a two-layer network, 28 → 64 → 32, with ReLU. Output: the 32-number **self code**.
- `PrevActionEncoder` (`:97-111`): six embedding tables, one per action head, sized 3, 3, 7, 2, 2, 2 rows by 16 numbers. Each head's last choice picks one row from its table, and the six rows are **added** (`:109`). Output: the 16-number **previous-action code**. (See the provisional note in 3.2.)

### 3.6 Fusion: `model.py:39-45` and `:61-73`

`A0Agent.__init__` builds the four encoders from the spec (`model.py:39-42`). `encode()` (`:61-73`) runs them and **concatenates** (joins side by side) their outputs with the raw `modality` flags:

```
ray code 64 + entity code 64 + self code 32 + previous-action code 16 + modality 3 = 179 numbers
```

`fusion` (`:42-44`) is one linear layer and ReLU from 179 to **192** numbers. This 192-number vector is the input to the memory, the subject of lesson 05.

The `modality` flags are passed in unchanged, so the network can learn to rely on other inputs when the flags say telemetry is missing (lesson 02).

## 4. A tiny worked example with real numbers

The demo below builds the A0 model and passes one **real observation** through it. The observation is from step 7 of a `DoomEnv` run with A0's settings (`a0_eval_env_config()`, screen off; one engine instance, closed before the model ran). For the observation and action *spaces* I used the test helper `tests/a0_helpers.py:doom_like_spaces`, which builds the same shapes by hand (14 types, screen off), so the model could be built without keeping the engine open. Torch was limited to 4 threads. Real output:

```
spec: ObsSpec(ray_channels=3, ray_width=80, max_entities=24, entity_feat_dim=8, n_entity_types=14, self_dim=28, modality_dim=3, head_sizes=(3, 3, 7, 2, 2, 2))
input shapes: {'rays': (1, 3, 80), 'entities': (1, 24, 8), 'entity_type': (1, 24), 'entity_mask': (1, 24), 'self': (1, 28), 'prev_action': (1, 6), 'modality': (1, 3)}
real entity rows: 1 types [11]
  ray conv0: (1, 16, 40)
  ray conv2: (1, 32, 20)
  ray flatten: 640 -> ray code (1, 64)
  type embed (1, 24, 16) + feats (1, 24, 8) -> tokens (1, 24, 64) + empty token -> (1, 25, 64)
  entity code (1, 64)
  self code (1, 32)  prev_action code (1, 16)  modality (1, 3) [[1.0, 0.0, 1.0]]
  concat width 179 -> fusion (1, 192)
```

Reading it:

- The batch size is 1 (one environment). Every shape starts with that 1.
- Rays: 3 bands x 80 columns → 16 channels x 40 → 32 channels x 20 (each convolution halves the length) → 640 numbers → 64.
- Entities: 24 rows, of which 1 is real (the bullet puff from lesson 02, type 11 = decoration). Each row becomes 16 + 8 = 24 numbers, then a 64-number token. The "no entity" token makes 25 tokens. Attention and pooling give 64 numbers.
- `modality` is `[1, 0, 1]`: telemetry valid, **screen absent** (A0 settings), ego-motion valid.
- 64 + 64 + 32 + 16 + 3 = 179 → 192.

**Order does not matter, shown with numbers.** I gave the entity encoder a random set of 5 entities, then the same 5 in a different row order. For comparison, I did the same with a plain linear layer applied to all 24 rows flattened into one list (an untrained layer created for the demo only). Real output:

```
  set code max |diff| after reordering rows: 1.1920928955078125e-07
  plain Linear on flattened rows, max |diff| after reordering: 0.9689280986785889
```

The set encoder's difference is about 0.0000001, which is floating-point rounding. The flattened layer changes by almost 1, so it treats reordered rows as a different scene. That is the difference between ADR 0003's option A (flatten the padded table) and the chosen option B.

**Parameter counts** (a **parameter** is one learned number in the network). Real output:

```
params total: 474474
  rays          43872
  entities      39520
  self_enc      3936
  prev_action   304
  fusion        34560
  core          345600
```

Checking two by hand: `prev_action` is 16 x (3 + 3 + 7 + 2 + 2 + 2) = 16 x 19 = **304**. `fusion` is 179 x 192 weights + 192 biases = 34368 + 192 = **34560**. The memory (`core`, lesson 05) holds most of the parameters. The whole model is under ADR 0003's limit of 1 million.

## 5. Why we chose this design over the alternatives

From ADR 0003 (**Proposed, not ratified**):

- **A. An MLP (a plain stack of fully connected layers) and a GRU (the memory layer of lesson 05) over the flattened, padded observation.** Half a day to build and the fastest, but **no permutation invariance** over entities (shown in section 4), and the weakest story for the judges.
- **B. A set-attention entity encoder, a ray encoder and a GRU core, with auxiliary heads (chosen for Stage A0).** About a day, under 1 million parameters, trainable on a CPU. ADR 0003's evidence: entity sets with a shared embedding, masked attention and pooling are well supported outside DOOM (hide-and-seek work), but **there is no DOOM evidence for them**.
- **C. B plus an ego-centric neural map** (Stage B, later and gated).
- **D. A transformer (a network built from stacked attention layers) over a long history of observations.** Memory and compute grow with the history length, and a keycard detour is thousands of decisions. Not chosen.

ADR 0003 also chose 1D convolutions for rays, so the same "gap in the wall" pattern is recognised anywhere across the view, and it asked for three unit checks on the set encoder: unchanged under row permutation, finite with zero entities, unchanged with garbage in padded rows. All three are in `tests/test_a0_encoders.py`.

ADR 0006 added the rule that the model reads its sizes from the spaces, after the entity table grew to 14 types and the screen became optional.

## 6. How it connects

- **Lesson 02** defined every input. Note what the encoder does *not* do: it does not compute bearings or threat scores; it only learns from `u`, `v`, `d` and the type, as rule 4.1 requires.
- **Lesson 05 (next)** takes the 192-number fused vector into the GRU memory and out through the six action heads.
- **Lesson 06** shows that during training, the trainer can blank telemetry before the encoders see it (masking); A0 trains with that off.
- **Later (Stage A1)**: a small image encoder for `screen` will join the fusion. It is not built.

## 7. Common mistakes, and how we would notice them

| Mistake | How it would show up |
|---|---|
| Adding a positional encoding (or any use of the row index) to the entity tokens. | The permutation tests in `tests/test_a0_encoders.py:39-55` and `:84-93` fail. In training, behaviour would depend on the engine's label order. |
| Forgetting to mask padded rows in attention or pooling. | The empty-set and garbage tests fail (`:58-81`). In training, the entity code would change with the number of padded rows, and a scene with 1 monster would look different from the same scene with a padded table of another size. |
| Hard-coding sizes (14 types, 3 x 80 rays) in the model. | Changing the environment (adding a type, turning the screen on) would crash with a shape error or, worse, an index error at run time. `spec_from_spaces` prevents this; keep using it. |
| Loading a checkpoint trained with 13 types into a 14-type environment. | An index error on the first `hazard` entity (`tests/test_a0_encoders.py:96-101` shows the error). Lesson 06 shows checkpoints carry the sensor signature so this can be caught before running. |
| Expecting A0 to use the screen because `screen_enabled` is true. | `tensorize.py` drops `screen`; A0 never sees pixels. That is by design until Stage A1. |
| Reading meaning into individual numbers of the codes. | They have no assigned meaning. To ask "does the memory know X?", ADR 0003 proposes a *linear probe*: a separate small model trained to read X from the frozen vector. |

## 8. Check your understanding

1. A frame shows no entities at all. Walk through what `EntitySetEncoder.forward` does, and explain why the output is still a valid 64-number vector.
2. The same three monsters are listed in rows 0, 1, 2 in one frame and in rows 5, 2, 9 in another (all other rows padded). Which parts of the encoder guarantee the entity code is the same, and which test checks it?
3. Someone adds a 15th entity type. Which files must change for the model to handle it, and what happens if an old checkpoint (14 types) is loaded?

---

## Answers

1. Step 1 wipes every row to zeros and type `pad`, because the mask is all false. Step 2 still builds 24 tokens, but step 3 puts the learned "no entity" token in front, and it is the only one marked valid. In steps 4 and 5, attention and pooling are masked to valid tokens only, so they use only the "no entity" token. The output is therefore a well-defined 64-number vector that depends only on learned weights. Without that token, attention would average over an empty set, which gives NaN.
2. The rows' positions are never used: no positional encoding, and each row goes through the same shared token network. Self-attention updates each token the same way whichever row it sits in, and the masked pooling is a weighted sum over valid tokens, which does not depend on order. Padded rows are wiped and masked, so their positions do not matter either. `tests/test_a0_encoders.py::test_set_encoder_any_count_and_row_position` checks exactly this case; the permutation tests check general reordering.
3. In the agent and environment code, only `env/observation.py` (append the new name at the end of `ENTITY_TYPES`, and map it in the type function). Some tests and helpers pin 14 types and would need updating too (`tests/a0_helpers.py:25`, `tests/test_a0_checkpoint_and_env_config.py:65`, and the toy's `train/toy_memory.py:29`). The model reads the type count from the `entity_type` Box (`obs_spec.py:48`, `:57`), so a new model gets a 15-row embedding automatically. An old checkpoint has a 14-row embedding table, so its weights do not fit the new model's shape (the load fails), or, if forced to run, an index error occurs the first time type 14 appears. Lesson 06 shows that checkpoints store the observation spec and sensor signature, so this mismatch can be caught before running.
