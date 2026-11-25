# PyTorch API reference (condensed for the DoomsDay model and trainer)

Fetched 2026-11-25 by web (doc fetcher). Installed: **torch 2.9.1, CPU build**, Windows 11, Python 3.12.10.

## Sources and version

Every page is from the **PyTorch 2.9 docs**: the URL path is `/docs/2.9/`, and the source links point to tag `v2.9.1`.

- https://docs.pytorch.org/docs/2.9/generated/torch.nn.GRU.html
- https://docs.pytorch.org/docs/2.9/generated/torch.nn.GRUCell.html
- https://docs.pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html
- https://docs.pytorch.org/docs/2.9/distributions.html
- https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad_norm_.html
- https://docs.pytorch.org/docs/2.9/generated/torch.set_num_threads.html
- https://docs.pytorch.org/docs/2.9/generated/torch.set_num_interop_threads.html
- https://docs.pytorch.org/docs/2.9/notes/cpu_threading_torchscript_inference.html
  - In 2.9 this page shows **only a "TorchScript is deprecated" notice**, with no body. The v2.9.1 `.rst` source in the official repo also held only that notice.
  - So **none of the pages fetched** give guidance on OMP_NUM_THREADS / MKL_NUM_THREADS or default thread counts. None is added here.
- https://docs.pytorch.org/docs/2.9/notes/multiprocessing.html
- https://docs.pytorch.org/docs/2.9/multiprocessing.html
- https://docs.pytorch.org/docs/2.9/notes/windows.html
- https://docs.pytorch.org/docs/2.9/generated/torch.save.html
- https://docs.pytorch.org/docs/2.9/generated/torch.load.html
- https://docs.pytorch.org/docs/2.9/notes/serialization.html

Lines marked *(ours)* are design notes or inferences, not documentation statements.

---

## 1. nn.GRU

```python
nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
       dropout=0.0, bidirectional=False, device=None, dtype=None)
out, h_n = gru(input, h_0=None)
```

D = 2 if bidirectional, else 1.

**Shapes:**

| Tensor | Unbatched | Batched, `batch_first=False` | Batched, `batch_first=True` |
|---|---|---|---|
| input | (L, H_in) | (L, N, H_in) | (N, L, H_in) |
| output | (L, D·H_out) | (L, N, D·H_out) | (N, L, D·H_out) |
| h_0 / h_n | (D·num_layers, H_out) | (D·num_layers, N, H_out) | **(D·num_layers, N, H_out), unchanged** |

- **`batch_first` does NOT apply to the hidden state.**
- `h_0` defaults to zeros.
- Packed sequences are accepted (`pack_padded_sequence`, `pack_sequence`), and the output is then packed too.
- `dropout` applies to each layer's output except the last layer.

**Weights and initialisation:**
- `weight_ih_l{k}` stacks gates in the order **(r | z | n)**, shape (3·H, in). `weight_hh_l{k}` has shape (3·H, H). The biases follow the same order.
- Initialisation is U(-√k, √k) with k = 1/hidden_size.
- The reset gate is applied **after** the W_hn product and its bias. The docs say this differs from the original paper on purpose, for efficiency.

## 2. nn.GRUCell

```python
nn.GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
h_next = cell(input, hx=None)
```

**Shapes:**
- input: (N, H_in) or unbatched (H_in).
- hx and the output: (N, H_out) or (H_out).
- hx defaults to zeros.

**Weights:** `weight_ih` (3·H, H_in), `weight_hh` (3·H, H), biases (3·H). Initialisation is U(-√k, √k), k = 1/hidden_size.

**Gate order:** the equations list r, z, n, but this page **does not explicitly state** the stacking order of the 3·H rows.

*(ours)* GRUCell steps one tick at a time. That suits rollout loops where each env's hidden state is zeroed at episode boundaries, e.g. multiplied by `1 - done`.

## 3. nn.MultiheadAttention

```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                      add_zero_attn=False, kdim=None, vdim=None, batch_first=False,
                      device=None, dtype=None)
attn_out, attn_w = mha(query, key, value, key_padding_mask=None, need_weights=True,
                       attn_mask=None, average_attn_weights=True, is_causal=False)
```

**`batch_first`:**
- **The default is False**, meaning (seq, batch, feature).
- True means (batch, seq, feature).
- It is ignored for unbatched input.

**`key_padding_mask`:**
- Shape (N, S), or (S) for unbatched input.
- **Bool: True means IGNORE that key** (padding).
- Float: the docs say it is "added directly to the corresponding key value".

**`attn_mask`:**
- Shape (L, S), or (N·num_heads, L, S).
- Bool True means "may not attend". A float mask is added to the attention weights.

**Masks together:** if both masks are given, **their types must match** (both bool or both float).

**Outputs:**
- `attn_output`: (L, E), (L, N, E), or (N, L, E) with `batch_first`.
- Weights, only if `need_weights=True`: (N, L, S) averaged over heads. With `average_attn_weights=False` they are (N, num_heads, L, S).
- `need_weights=False` lets the layer use the optimised `scaled_dot_product_attention` path.

**Fast path:** requires all of the following:
- self-attention with q = k = v
- 3-D batched input with `batch_first=True`
- no autograd, and `.eval()` mode
- no `add_bias_kv` or `add_zero_attn`
- `kdim` = `vdim` = `embed_dim`
- autocast off

**Not documented: a fully masked row.** The docs **do not say** what happens when every key in a row is masked. That is our "zero visible enemies" case.

*(ours)* Test it on 2.9.1 for NaN output, which is a common concern. Guard with an always-unmasked "null" token or an explicit check.

## 4. torch.distributions.Categorical

```python
Categorical(probs=None, logits=None, validate_args=None)   # exactly one of probs / logits
```

**Parameters:**
- **The category dimension is the LAST dimension.** The leading dims are the batch. Categories run from 0 to K-1.
- `probs` must be non-negative, finite, with a non-zero sum. They are normalised to sum to 1 along the last dim.
- `logits` are unnormalised log-probabilities and are normalised along the last dim. The docs do not state the method.
- The `.logits` and `.probs` properties return the **normalised** values.

**Methods:**
- `sample(sample_shape=())` returns integer indices of shape `sample_shape + batch_shape`. The base-class docs say this; the Categorical example shows a scalar for 1-D probs.
- `log_prob(value)` returns a Tensor. **The page states no shape rule for Categorical.** *(ours)* Pass `value` shaped `batch_shape` and assert the output shape in a unit test.
- `entropy()` returns a Tensor of shape `batch_shape` (base-class docs).
- `enumerate_support(expand=True)` returns shape `(K,) + batch_shape`.

**Validation:**
- `validate_args` is on by default and off under `python -O`.
- Change it globally with `Distribution.set_default_validate_args(bool)`.
- The docs note validation can be expensive.

**REINFORCE:** the docs' example uses `loss = -m.log_prob(action) * reward`.

*(ours)* For a factored MultiDiscrete action, use one Categorical per head and sum the per-head log_probs and entropies.

## 5. nn.utils.clip_grad_norm_

```python
total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0,
                                            error_if_nonfinite=False, foreach=None)
```

- The norm is computed over **all gradients together**, as one concatenated vector. Gradients are scaled **in place**.
- It returns a Tensor holding the total norm.
- The docs say it is equivalent to `get_total_norm()` followed by `clip_grads_with_norm_()` with that norm. They do **not** explicitly say whether the returned value is the pre-clip or the post-clip norm.
- `error_if_nonfinite=True` raises on a NaN or Inf total norm.
- `norm_type` accepts `'inf'`.

## 6. CPU threading

- `torch.set_num_threads(int)` sets the threads for **intra-op** parallelism on CPU. Warning from the docs: call it **before running any eager, JIT or autograd code**.
- `torch.set_num_interop_threads(int)` sets the **inter-op** threads. It can only be called **once, before any inter-op parallel work** starts.

**Multiprocessing note** ("CPU in multiprocessing"):
- If M processes each use N threads on an N-vCPU machine, you get **oversubscription**: contention and context switching.
- The fix the docs give is `torch.set_num_threads(floor(N / M))` **in each subprocess**. The docs cite about a 30x speedup on their Hogwild example.

*(ours)* The env workers and their `vizdoom.exe` children also take cores, so count them in M.

## 7. Multiprocessing and Windows

- `torch.multiprocessing` is a drop-in wrapper for `multiprocessing`. Tensors sent through its Queue move to shared memory, and only a handle is sent.
- **Windows uses spawn, not fork** (Windows FAQ). Each child re-imports the main module.
  - Wrap entry code in `main()` and call it only under `if __name__ == "__main__":`.
  - Otherwise you get a RuntimeError about starting a process during bootstrapping.
  - Under non-fork start methods, unguarded module-level statements run in every child (multiprocessing notes).
- `torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')`:
  - `fn(i, *args)` must be defined at **module top level** so it can be pickled.
  - If one process fails, the others are killed and the exception is re-raised with its traceback.
- **Sharing strategies:** `file_descriptor` is the default (not supported on macOS); `file_system` is the alternative. The docs **do not say** what Windows uses; `torch.multiprocessing.get_all_sharing_strategies()` reports it.
- **Deadlocks:** the most common cause is a background thread holding a lock at fork time. `SimpleQueue` avoids helper threads.
- **Reuse buffers through queues:** each `put` of a non-shared tensor costs a copy into shared memory.
- **Hogwild:** call `model.share_memory()`, send the `state_dict()` for periodic sync, and prefer `multiprocessing.Queue`.
- **Leaks:** an abrupt parent exit may leave child processes unreaped.
- CUDA IPC is not supported on Windows. That is irrelevant on our CPU build.

## 8. torch.save / torch.load

```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)
torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)
```

**torch.save:**
- Writes a ZIP64 archive by default (since 1.6).
- Storage sharing and views are preserved, so **saving a slice saves the whole underlying storage**. Call `.clone()` first if you only want the slice.

**torch.load and `weights_only`:**
- **`weights_only` defaults to True** since 2.6, when `pickle_module` is not passed (serialization notes).
- It allows only tensors, primitive types, dicts, and allow-listed globals. Allow-list them with `torch.serialization.add_safe_globals([...])` or the `safe_globals(...)` context manager.
- `get_unsafe_globals_in_checkpoint(f)` lists the offending globals.
- An unsupported global raises `_pickle.UnpicklingError` with "WeightsUnpickler error: Unsupported global".
- Environment overrides: `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

**Security:** with `weights_only=False`, loading runs pickle and **can execute arbitrary code**. "Only load data you trust."

**Recommendation from the docs:** save a module's `state_dict`, not the whole module.

**`map_location`:** accepts a device or string (`"cpu"`), a dict that remaps location tags, or a callable `(storage, location)`. Without it, tensors go back to the device they were saved from.

**`mmap=True`:** loads storages lazily.

## 9. Gotchas

1. **GRU hidden state is (layers, N, H) even with `batch_first=True`.**
2. **MultiheadAttention defaults to `batch_first=False`.** Set it explicitly.
3. **`key_padding_mask` True means "ignore".** Do not pass a "valid" mask by mistake.
4. **Bool and float masks cannot be mixed.**
5. **A fully masked attention row is undocumented.** Test it.
6. **Categorical normalises over the last dim.** `entropy()` has shape `batch_shape`.
7. **Call `set_num_threads` before any torch op, in every worker process.** `set_num_interop_threads` can be called only once.
8. **On Windows, spawn re-imports the main module.** Guard the entry point, and keep worker functions top-level and picklable.
9. **Checkpoints:**
   - Save `state_dict()`, and keep any metadata to the types `weights_only` allows: tensors, primitives, dicts.
   - *(ours)* Avoid NumPy objects in metadata. They are not on that documented list and would need allow-listing.
   - Never load untrusted `.pt` files with `weights_only=False`.
10. **`clip_grad_norm_` returns the norm as a Tensor.** Whether it is the pre-clip or post-clip norm is not stated explicitly.
