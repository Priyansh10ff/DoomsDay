# Gymnasium API reference (condensed for the DoomsDay env wrapper)

Fetched 2026-11-25 by web (doc fetcher). Installed: **gymnasium 1.4.0**.

## Sources and version

**Gymnasium docs** (unversioned "latest" site):
- https://gymnasium.farama.org/api/env/
- https://gymnasium.farama.org/api/spaces/fundamental/
- https://gymnasium.farama.org/api/spaces/composite/
- https://gymnasium.farama.org/api/vector/
- https://gymnasium.farama.org/api/vector/sync_vector_env/
- https://gymnasium.farama.org/api/vector/async_vector_env/
- https://gymnasium.farama.org/api/vector/wrappers/
- https://gymnasium.farama.org/introduction/migration_guide/

**Release notes:** https://gymnasium.farama.org/gymnasium_release_notes/. The newest release is **v1.4.0 (2026-10-05)**.

**Farama blog, "Deep Dive: VectorEnv Autoreset"** (M. Towers, **2025-02-20**, written for v1.0/v1.1; linked from the API docs): https://farama.org/Vector-Autoreset-Mode

**Repository source at tag v1.4.0.** This is outside the "official docs" scope of this run and is quarantined in **section 5**. Delete section 5 and the [repo] tags if the user rejects it.
- https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v1.4.0/gymnasium/vector/sync_vector_env.py
- https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v1.4.0/gymnasium/vector/vector_env.py

**Version check.** The docs pages show no version number. They do document 1.4.0-only features: `Dict(sort_keys=...)` and `AsyncVectorEnv(max_concurrency=...)` both appear in the v1.4.0 release notes. So the docs match the installed version.

---

## 1. Env contract

```python
Env.reset(*, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]
Env.step(action) -> tuple[ObsType, SupportsFloat, bool, bool, dict]  # obs, reward, terminated, truncated, info
Env.render() -> str | ndarray | list | None   # mode fixed at construction (render_mode)
Env.close() -> None                            # idempotent: closing twice is no error
```

**Termination flags:**
- **terminated** means a terminal state of the MDP was reached (goal, death).
- **truncated** means the episode was ended by something outside the MDP, typically a time limit.
- Either flag means the caller must `reset()`. `step()` after an end is undefined.
- `done` and the info key `"TimeLimit.truncated"` are gone; they were removed in v0.26.

**Seeding:**
- Custom envs must begin `reset` with `super().reset(seed=seed)`. That sets `self.np_random`, a `numpy.random.Generator`, and `np_random_seed`.
- `seed=None` keeps the existing RNG. An int reseeds it.
- Docs advice: seed once right after construction, then never again.
- Action-space sampling is seeded separately with `env.action_space.seed(n)`.

**Fresh objects:** each `reset` and `step` must return **new** observation and info objects, not a reused, mutated buffer. The env checker enforces this since 1.4.0.

**Attributes:**
- `action_space`, `observation_space`
- `metadata` (default `{"render_modes": []}`), `render_mode`
- `np_random`, `np_random_seed` (-1 if not set via reset)
- `spec` (set by `make`), `unwrapped`

## 2. Spaces

```python
Box(low, high, shape=None, dtype=np.float32, seed=None)
Discrete(n, seed=None, start=0, *, dtype=np.int64)
MultiDiscrete(nvec, dtype=np.int64, seed=None, start=None)
MultiBinary(n, seed=None)
Dict(spaces=None, seed=None, *, sort_keys=True, **spaces_kwargs)
Tuple(spaces, seed=None)
```

**Box:**
- Scalar bounds broadcast to `shape`. Array bounds imply the shape.
- If all of `shape`, `low` and `high` are None, it raises `ValueError`.
- Integer bounds are limited to `[-2**63, 2**63-2]`.
- `sample()` takes no mask or probability.

**Discrete:**
- `{start, ..., start+n-1}`.
- The `sample(mask=...)` mask is int8 of shape `(n,)`. An all-zero mask returns `start`.

**MultiDiscrete:**
- Cartesian product of Discretes; `nvec` can be multi-axis.
- Its mask is a **tuple** of int8 arrays, one per sub-action.
- The docs note that some wrappers assume **0 means NOOP**.

**MultiBinary:** the mask uses 0 (force 0), 1 (force 1) and 2 (random).

**Dict:**
- `spaces` is a dict, a sequence of `(key, space)` pairs, or keyword arguments.
- A plain dict has its **keys sorted by default**. Pass `sort_keys=False` (new in 1.4.0) to keep insertion order. `OrderedDict` and sequence inputs always keep their order.
- Dicts can nest.
- `seed` accepts an int (derived subseeds, not guaranteed unique) or a dict of per-key seeds.

**contains:**
- The fundamental and composite pages do **not** document `contains` semantics, including dtype strictness.
- The release notes say that since 1.4.0, `Box.contains` and `Discrete.contains` return False for out-of-range values instead of raising `OverflowError`.

## 3. Vector API (docs)

```python
VectorEnv.reset(*, seed: int | None = None, options: dict | None = None) -> (obs, infos)
VectorEnv.step(actions) -> (obs, rewards, terminations, truncations, infos)   # batched arrays
VectorEnv.close(**kwargs) -> None   # sub-envs closed in close_extras()

SyncVectorEnv(env_fns, copy=True, observation_mode="same",
              autoreset_mode=AutoresetMode.NEXT_STEP)
AsyncVectorEnv(env_fns, shared_memory=True, copy=True, context=None, daemon=True,
               worker=None, observation_mode="same",
               autoreset_mode=AutoresetMode.NEXT_STEP, max_concurrency=None)
```

**Attributes:**
- `num_envs`
- `action_space` and `observation_space` are batched; `single_action_space` and `single_observation_space` are per env.
- `metadata["autoreset_mode"]` holds the mode in use. The docs say it "should be available" there.

**Seeds:** the SyncVectorEnv page (not the base `VectorEnv` page) says `reset` also accepts a **list** of ints, one per env. An int seed `s` expands to `s, s+1, ...`.

**`step` return:** SyncVectorEnv returns rewards as a float64 array and terminations and truncations as bool arrays.

**AsyncVectorEnv parameters:**
- **`context`:** a multiprocessing start-method name. `None` uses the default context. The Gymnasium docs do not name spawn, fork or forkserver, and say **nothing about Windows**. The PyTorch Windows FAQ states that Windows starts child processes with spawn.
- **`shared_memory=True`:** raises `ValueError` for custom (non-builtin) observation spaces. Built-ins such as Box, Discrete and Dict are fine.
- **`daemon=True`:** a worker cannot spawn its own children. The docs say some envs need `False`. **ViZDoom launches a child `vizdoom.exe` process**, so test whether `daemon=True` blocks that. This is untested and the docs do not say.
- **`max_concurrency`:** new in 1.4.0. Caps how many sub-envs run `reset` or `step` at once, via a semaphore. It does not apply to `call`, `get_attr` or `set_attr`.
- **Other methods:** `call(name, *a, **kw)`, `get_attr(name)` and `set_attr(name, values)` each return a tuple with one entry per env.
- **Not documented:** how a worker exception surfaces to the caller, and `step_async` / `step_wait`.

**Info format (docs):**
- `infos` is one dict whose values are arrays indexed by env.
- Each key `k` has a boolean companion `_k` that marks which envs reported `k`. See the `DictInfoToList` example: `{"k": [0,0,.5,.3], "_k": [F,F,T,T]}` becomes `[{}, {}, {"k":.5}, {"k":.3}]`.
- `RecordEpisodeStatistics` (vector) writes `info["episode"]` (`r`, `l`, `t`) plus `info["_episode"]`.

## 4. Autoreset modes (docs and the 2025-02-20 article)

- Introduced in v1.1.0: `gymnasium.vector.AutoresetMode` with `NEXT_STEP` (the default), `SAME_STEP` and `DISABLED`.
- The mode is stored as that enum in `metadata["autoreset_mode"]`, per the release notes.

**NEXT_STEP:**
- A sub-env that ended is reset on the **next** `step()` call, and **the action passed for it there is ignored**.
- That step's transition is not a valid transition and must be masked out. The article's pattern: `episode_start = terminations | truncations`, and skip stored transitions where `episode_start` was True.
- The article says the episode's last transition is valid. It does **not** specify the exact values returned on the finishing step or on the reset step.

**SAME_STEP:**
- The finishing step already returns the **reset** obs.
- The terminal obs is in `info["final_obs"]`.
- Some observation wrappers do not support this mode.

**DISABLED:**
- No automatic reset.
- The caller must call `envs.reset(options={"reset_mask": mask})` for finished envs.

**Not documented:** neither the docs nor the article document `final_info`, or how the final info of a finished episode is delivered.

**Training code is mode-specific.** Episode lengths in step calls differ between modes, so code cannot be switched between them.

## 5. [repo] Details found only in repository source (v1.4.0 tag), not in the docs

Listed so the worker knows where they come from. Confirm them by running a small test on the installed 1.4.0.

**Enum string values:** `NEXT_STEP` = "NextStep", `SAME_STEP` = "SameStep", `DISABLED` = "Disabled".

**NEXT_STEP:**
- The finishing step returns the env's real terminal obs, reward and flags; no reset happens yet.
- On the reset step, the env returns the reset obs with **reward 0.0** and terminated and truncated both False. Its info is the reset info.
- No `final_obs` or `final_info` keys are added in this mode.

**SAME_STEP:**
- The terminal info is delivered as `info["final_info"]`, alongside `info["final_obs"]`.
- Each key gets its `_key` mask (`_final_obs`, `_final_info`).
- The reset info is merged into the same `infos`.

**DISABLED:**
- Stepping an env that finished and was not reset raises an `AssertionError`.
- `reset_mask` must be an `np.bool_` array of shape `(num_envs,)` with at least one True.
- `reset_mask` is **popped from your `options` dict**, which mutates it.
- Unmasked envs keep their current obs.

**`_add_info` dtypes:**
- `final_obs` uses an object array.
- A new scalar key uses a `np.zeros(num_envs, dtype=type(value))` array.
- An ndarray key uses an array of shape `(num_envs, *shape)`, which assumes the same shape across envs.
- Any other type uses an object array of None.
- Nested dicts recurse.

**`VectorObservationWrapper`:** warns if `metadata["autoreset_mode"]` is missing. It raises `ValueError` for SAME_STEP.

## 6. Gotchas for a custom vectorised env

1. **Declare `metadata["autoreset_mode"]`** on any custom `VectorEnv` subclass. The docs say it should be present. [repo] The vector observation wrappers warn when it is missing.
2. **Pick one autoreset mode and write the rollout and GAE code for it.** With NEXT_STEP, mask the reset step, whose action was ignored (docs/article). [repo] That step's reward is 0 and its flags are False.
3. **ViZDoom's `get_state()` is None at episode end.** The wrapper must build the final observation itself, e.g. from the last valid state or zeros (the ViZDoom FAQ suggests a zero dummy). Never hand None to Gymnasium as an observation.
4. **Bootstrapping:** the standard RL practice is to bootstrap V(s') on **truncated** (ViZDoom `is_episode_timeout_reached()`), not on terminated (death or exit). That needs the true last obs, so with NEXT_STEP keep it from the finishing step. The Gymnasium docs define the two flags but do not prescribe bootstrapping.
5. **Return fresh arrays** from `step` and `reset`; do not reuse buffers. The 1.4.0 checker flags reuse.
6. **`Dict` key order:** keys are sorted by default. If code flattens a Dict by insertion order, pass `sort_keys=False` (1.4.0+) or flatten by sorted key.
7. **Windows and AsyncVectorEnv:**
   - Under spawn, the main script must sit behind `if __name__ == "__main__":` (PyTorch Windows FAQ).
   - PyTorch's `spawn` docs require target functions to be defined at module top level so they can be pickled. Apply the same care to env factories; this is an inference for Gymnasium, which does not document pickling.
   - Each worker also starts a `vizdoom.exe`, which counts toward the laptop's CPU budget. Consider `max_concurrency`.
8. **`shared_memory=True`** only works with built-in spaces.
9. **Masks in `sample()`** are int8; probability arrays are float64 and must sum to 1.
