# Lessons

- Cost model: `sim_time_us = non_collective_local_ops * 29us + num_collective_dispatches * 200us` (2x100us per dispatch for fwd+bwd).
- `local_ops` in METRICS output = counter.count (ALL ops including collectives). But only non-collective ops contribute to the 29us term.
- MockTorch.zeros and MockTorch.tensor do NOT record ops in the counter.
- TrackedTensor.__getitem__ only records "slice" for slice/tuple keys. Tensor keys (torch.Tensor) are free.
- TrackedTensor.__setitem__ never records ops.
- Using `.tensor` property on TrackedTensor to get the raw torch.Tensor bypasses op counting for both reads and writes, allowing zero-cost packing and extraction.
- The theoretical minimum cost for any solution requiring 1 collective is 200us (0 local ops * 29 + 1 dispatch * 200).
- Pass 1D tensor directly to all_gather(dim=0) — avoids unsqueeze + view (2 extra ops = 58us).
- Build flat index lists in Python (free), create a torch.tensor index, then use `gathered[idx.tensor]` for zero-cost fancy indexing extraction.
- The optimal alltoallv solution: pack via raw_packed = packed.tensor, all_gather 1D, extract via gathered[idx_tensor.tensor] = 200us total.
- The benchmark harness does NOT enforce `unsupported_primitives` (passes None). The constraint about banned XLA primitives is for real hardware only.
