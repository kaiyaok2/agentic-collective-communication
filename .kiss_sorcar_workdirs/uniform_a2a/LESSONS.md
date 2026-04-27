# Lessons

- The cost model is: `real_local_ops * 29us + num_dispatches * 2 * 100us`. `real_local_ops` excludes both collective ops AND free XLA ops. Each collective dispatch costs 200us (2x factor for fwd+bwd).
- Free XLA ops (0 cost): `view`, `reshape`, `unsqueeze`, `squeeze`, `flatten`, `narrow`, `transpose`, `permute`, `expand`, `contiguous`, `slice` (via `__getitem__` with slice/tuple).
- Non-free local ops (29us each): `index_select`, `cat`, `stack`, `gather`, `scatter`, `scatter_`, `cumsum`, `where`, `clamp`, `repeat_interleave`, `chunk`, `split`.
- `torch.tensor()`, `zeros()`, `ones()`, `empty()`, `full()`, `arange()` do NOT record ops.
- TrackedTensor `__getitem__` with int key does NOT record an op; with slice/tuple it records "slice" (free).
- `xm.all_gather(tensor, dim=0)` works directly on 1D tensors—no need to unsqueeze to 2D first.
- For uniform_a2a, theoretical minimum is 200us: 1 all_gather(200us) + 0 real local ops. Achieved by using view/slice/reshape (all free) instead of index_select (29us).
- Always prefer free XLA ops (view, reshape, slice, permute, transpose, narrow) over non-free ops (index_select, cat, gather, stack) for data extraction after collectives.
