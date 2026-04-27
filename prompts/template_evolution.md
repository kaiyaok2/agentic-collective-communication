# XLA AllToAllV Algorithm Evolution

You are optimizing an AllToAllV communication algorithm for AWS Trainium using XLA (torch_xla).

## Hardware: trn1.32xlarge cluster ({num_nodes} node(s))
- {num_devices} NeuronDevices in a 2D 4x4 torus per node, {cores_per_device} NeuronCores each ({world_size} ranks total)
- Each device has 4 bidirectional NeuronLinks (~192 GB/s each)
- Intra-device core-to-core is free (shared HBM)
- {ranks_per_node} ranks per node
- **Inter-node (EFA)**: {efa_bandwidth} GB/s per adapter, {num_nodes} node(s), {efa_latency} us latency

## XLA Programming Model

Your function uses XLA operations via `xm` (torch_xla.core.xla_model) and `torch`:

### Collectives (via `xm`)

**`xm.collective_permute(tensor, pairs)`** — Point-to-point exchange
```python
# pairs: list of (src_rank, dst_rank)
received = xm.collective_permute(send_tensor, pairs=[(0,1), (1,0), ...])
# Cost: ~100us dispatch + bandwidth-limited transfer. All ranks must participate.
```

**`xm.all_gather(tensor, dim, groups=None)`** — Gather from all ranks
```python
gathered = xm.all_gather(tensor.unsqueeze(0), dim=0).view(world_size, -1)
# or with topology-aware groups:
gathered = xm.all_gather(tensor.unsqueeze(0), dim=0, groups=[rank_order]).view(-1)
# Cost: ~100us dispatch + ring-based bandwidth transfer.
```

**NOTE**: `xm.all_to_all` is NOT supported on this hardware. Do NOT use it.

### Local Tensor Operations (via `torch`)
```python
packed = torch.zeros(size, device=x.device, dtype=x.dtype)         # Allocate
packed[start:end] = x[a:b]                                          # Slice assign
result = torch.cat([t1, t2, ...], dim=0)                            # Concatenate
result = torch.index_select(gathered, 0, idx_tensor)                # Indexed gather
idx = torch.tensor([...], device=x.device, dtype=torch.long)       # Index tensor
```

### CRITICAL: XLA Cost Model

Each XLA operation has a per-op dispatch cost. The exact costs vary by operation type — some
operations are essentially free (metadata-only), while others create real HLO IR nodes with
significant overhead. Collective operations also have per-dispatch overhead.

Your optimization target: minimize total estimated latency (sum of all op costs).

For small-to-medium data sizes (up to ~8MB), per-op dispatch overhead dominates bandwidth.
Reducing the total number of costly operations matters more than reducing data transferred.

### CRITICAL: XLA Lazy Execution Rules

XLA tensors are lazy — they build a computation graph but DON'T hold concrete values.
You **CANNOT** use device tensors as Python-level values (slice indices, loop bounds, conditions).

**FORBIDDEN patterns — these will crash on real hardware:**
```python
# WRONG: creating device tensors for offsets, then using as Python ints
send_offsets = torch.zeros(world_size + 1, device=input_tensor.device, dtype=torch.long)
for i in range(world_size):
    send_offsets[i + 1] = send_offsets[i] + send_counts[i]  # send_offsets is lazy!
packed[i * max_chunk:i * max_chunk + sc] = input_tensor[send_offsets[i]:send_offsets[i + 1]]  # CRASH

# WRONG: using len() on a device tensor result
total = recv_offsets[world_size]  # lazy tensor, can't use as int
result = torch.zeros(total, ...)  # CRASH — total is a tensor not an int

# WRONG: using device tensor in if/for condition
if recv_counts_tensor[i] > 0:  # CRASH if recv_counts_tensor is on device
```

**CORRECT patterns — use Python lists/ints for control flow:**
```python
# RIGHT: compute offsets as plain Python ints
send_off = 0
for i in range(world_size):
    sc = send_counts[i]        # send_counts is a Python list of ints
    if sc > 0:
        packed[i * max_chunk:i * max_chunk + sc] = input_tensor[send_off:send_off + sc]
    send_off += sc

# RIGHT: build index tensor from Python list, then use single torch.index_select
flat_idx = []
for src in range(world_size):
    count = recv_counts[src]   # Python int
    base = src * pack_size + rank * max_chunk
    flat_idx.extend(range(base, base + count))
idx_tensor = torch.tensor(flat_idx, device=input_tensor.device, dtype=torch.long)
result = torch.index_select(gathered, 0, idx_tensor)
```

**Rule of thumb:** `send_counts`, `recv_counts`, `max_chunk`, `rank`, `world_size` are plain Python ints/lists. Use them directly for control flow, slicing, and building index lists. Only create device tensors for data that goes through collectives or is returned.

### CRITICAL: Training Context (autograd + backward pass)

Your function will be called inside a `torch.autograd.Function` with `xm.mark_step()` barriers
before and after each call. In real MoE training it is called multiple times per layer (dispatch
and combine), across multiple layers, in both forward and backward passes.

This means:
- **Dtype preservation**: The input tensor may be `torch.bfloat16` during training, not just
  `torch.float32`. ALWAYS use `input_tensor.dtype` and `input_tensor.device` when creating
  new tensors (e.g., `torch.zeros(..., dtype=input_tensor.dtype, device=input_tensor.device)`).
  Never hardcode `torch.float32`.
- **Backward pass calls `g.contiguous()`**: The gradient tensor may have non-contiguous strides.
  Your function must work with contiguous inputs — do not assume a specific memory layout beyond
  the documented semantics.
- **Multiple compilations**: XLA compiles a separate graph for forward vs backward. Both must
  produce valid HLO. Operations that work in forward may fail in backward if they create
  shapes the compiler cannot statically resolve.
- **Mark-step barriers**: `xm.mark_step()` is called before and after your function. Lazy
  tensors from a previous mark_step epoch are materialized — your function starts with a
  concrete (but still lazy-API) tensor and must return one.

### Constraints:
- **Do NOT use xm.all_to_all**: This primitive is not supported on the target hardware.
- All index arithmetic MUST be done in Python (plain ints/lists), not device tensors.

## Required function signature

```python
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """
    XLA AllToAllV algorithm.

    Args:
        input_tensor: 1D tensor. Data layout:
            input_tensor[send_offsets[i]:send_offsets[i]+send_counts[i]] is data for rank i.
        send_counts: list[int] of length world_size. Elements to send to each rank.
        recv_counts: list[int] of length world_size. Elements to receive from each rank.
        max_chunk: int. Maximum element count across all send/recv pairs.
        rank: int. This rank's index.
        world_size: int. Total ranks.
        num_devices: int. Number of physical devices.
        cores_per_device: int. Cores per device (typically 2).
        xm: XLA model module (provides collective_permute, all_gather, reduce_scatter).
        torch: Torch module (provides zeros, cat, index_select, tensor, etc.).
        num_nodes: int. Number of nodes (default 1).

    Returns:
        1D tensor with received data from all sources, concatenated in
        source-rank order: [data_from_rank_0, data_from_rank_1, ...].
    """
```

## Current best implementation

Performance: {current_sim_time} us, {current_num_permutes} collective_permute, {current_num_gathers} all_gather, {current_local_ops} local ops.

```python
{current_code}
```

## Reference implementations

### Naive AllGather + Slice:
```python
{builtin_naive_allgather}
```

### Permute Ring:
```python
{builtin_permute_ring}
```

## Evolution history

{history}

## Your task

Write an IMPROVED `evolved_alltoallv` function. It MUST:
- Follow the exact function signature above
- Return correct results for ALL rank pairs and ALL traffic patterns
- Handle send_counts that include zeros
- Use ONLY `xm.collective_permute`, `xm.all_gather`, `xm.reduce_scatter` for communication. **Do NOT use xm.all_to_all** (unsupported on target hardware).
- Use `torch.zeros`, `torch.cat`, `torch.index_select`, `torch.tensor`, `torch.arange`, `torch.stack`, `torch.gather`, `torch.narrow`, `torch.chunk`, `torch.split` for local ops
- **Use Python ints/lists for ALL control flow and index computation** (never device tensors)
- **Preserve input dtype**: use `input_tensor.dtype` for all created tensors (bf16-safe)
- Minimize total real XLA ops (collectives + real local ops). Free ops (view, reshape, permute, transpose, slicing) cost nothing.

**PRIMARY GOAL: Minimize total estimated latency while maintaining correctness.**
The current implementation has {current_local_ops} real local ops. Use the cost model above to estimate total latency. All index math MUST happen in Python (lists, ints), NOT as device tensor operations.

Provide your implementation inside a ```python code block.
