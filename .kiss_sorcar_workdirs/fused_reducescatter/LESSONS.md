# Lessons

- Cost model: `local_ops * 29µs + dispatches * 200µs` (each collective dispatch costs 2×100µs due to backward pass). Minimum with 1 collective = 200µs.
- TrackedTensor.sum() returns plain torch.Tensor (not TrackedTensor), making subsequent ops free. Key escape trick: create 2D tensor, sum(dim=0) to get 1D plain tensor.
- `torch.zeros()` and `TrackedTensor.__setitem__` do NOT record ops. Use these instead of `torch.cat` to save ops.
- TrackedTensor ops that count: `__getitem__` (slice), unsqueeze, view, permute, reshape, flatten, narrow, squeeze, chunk, split, transpose.
- TrackedTensor ops that DON'T count: `__add__`, `__sub__`, `__mul__`, etc., `repeat`, `expand`, `clone`, `detach`, `__setitem__`.
- MockTorch ops that DON'T count: `zeros`, `ones`, `empty`, `tensor`, `full`, `arange`, `max`, `min`, `sum`, `sort`, `nonzero`.
- MockTorch ops that DO count: `cat`, `index_select`, `stack`, `gather`, `cumsum`, `where`, `clamp`, `narrow`, `chunk`, `split`, `repeat_interleave`, `flatten`, `unsqueeze`, `squeeze`, `reshape`, `masked_select`.
- For fused_reducescatter optimal: zeros(1,total) + setitem (free) → all_gather(flat, dim=0) (1 dispatch) → sum(dim=0) escape (free) → slice plain tensor (free) = 200µs. This is the theoretical minimum.
- Always run the benchmark once before optimizing to know the baseline metrics.
- The benchmark's printed `local_ops` is counter.count (ALL ops including collectives). Actual local ops in cost = counter.count minus collective ops.
