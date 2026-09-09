# XLA {display_name} Algorithm Evolution

You are optimizing a {display_name} communication algorithm for AWS Trainium using XLA (torch_xla).

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

**`xm.all_reduce(reduce_type, tensor)`** — Sum/reduce across all ranks
```python
reduced = xm.all_reduce('sum', tensor)
# Cost: ~100us dispatch + ring-based bandwidth transfer. Returns tensor of same shape.
```

**`xm.reduce_scatter(reduce_type, tensor, scale, scatter_dim, shard_count)`** — Reduce then scatter
```python
shard = xm.reduce_scatter(xm.REDUCE_SUM, tensor, scale=1.0/world_size, scatter_dim=0, shard_count=world_size)
# Cost: ~100us dispatch. Each rank gets 1/world_size of the result.
```

### Local Tensor Operations (via `torch`)
```python
packed = torch.zeros(size, device=x.device, dtype=x.dtype)         # Allocate
packed[start:end] = x[a:b]                                          # Slice assign
result = torch.cat([t1, t2, ...], dim=0)                            # Concatenate
result = torch.index_select(gathered, 0, idx_tensor)                # Indexed gather
idx = torch.tensor([...], device=x.device, dtype=torch.long)       # Index tensor
parts = torch.split(flat, split_sizes)                              # Split into list
```

### CRITICAL: XLA Cost Model

Each XLA operation has a per-op dispatch cost. The exact costs vary by operation type — some
operations are essentially free (metadata-only), while others create real HLO IR nodes with
significant overhead. Collective operations also have per-dispatch overhead.

**Consult the profiled cost table in the optimization hints section below for exact per-op costs.**

Your optimization target: minimize the total estimated latency (sum of all op costs).

For small-to-medium data sizes (up to ~8MB), per-op dispatch overhead dominates bandwidth.
Reducing the total number of costly operations matters more than reducing data transferred.

### CRITICAL: XLA Lazy Execution Rules

XLA tensors are lazy — they build a computation graph but DON'T hold concrete values.
You **CANNOT** use device tensors as Python-level values (slice indices, loop bounds, conditions).

**CORRECT patterns — use Python ints/lists for ALL control flow:**
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

**Rule of thumb:** All control-flow variables (counts, offsets, sizes) must be plain Python ints/lists. Only create device tensors for data that goes through collectives or is returned.

### CRITICAL: Training Context (autograd + backward pass)

Your function will be called inside a `torch.autograd.Function` with `xm.mark_step()` barriers
before and after each call. In real MoE training it is called multiple times per layer (dispatch
and combine), across multiple layers, in both forward and backward passes.

This means:
- **Dtype preservation**: The input tensor may be `torch.bfloat16` during training, not just
  `torch.float32`. ALWAYS use `input_tensor.dtype` and `input_tensor.device` when creating
  new tensors. Never hardcode `torch.float32`.
- **Backward pass calls `g.contiguous()`**: The gradient tensor may have non-contiguous strides.
  Your function must work with contiguous inputs.
- **Multiple compilations**: XLA compiles a separate graph for forward vs backward. Both must
  produce valid HLO. Operations that work in forward may fail in backward if they create
  shapes the compiler cannot statically resolve.
- **Mark-step barriers**: `xm.mark_step()` is called before and after your function. Your
  function starts with a materialized (but still lazy-API) tensor and must return one.

{optimization_hints}

## Required function signature

```python
{signature}
```
{signature_doc}

## Current best implementation

Performance: {current_sim_time} us, {current_num_permutes} collective_permute, {current_num_gathers} all_gather, {current_local_ops} local ops.
{current_op_breakdown}

```python
{current_code}
```

## Reference implementations

{reference_implementations}

## Evolution history

{history}

## Your task

Write an IMPROVED `{evolved_fn_name}` function. It MUST:
- Follow the exact function signature above
- Return correct results for ALL rank pairs and ALL traffic patterns
- Handle edge cases (zero sizes, single rank, etc.)
- Use ONLY `xm.collective_permute`, `xm.all_gather`, `xm.all_reduce`, `xm.reduce_scatter` for communication
- Use `torch.zeros`, `torch.cat`, `torch.index_select`, `torch.tensor`, `torch.arange`, `torch.stack`, `torch.split`, `torch.chunk`, `torch.narrow` for local ops
- **Use Python ints/lists for ALL control flow and index computation** (never device tensors)
- **Preserve input dtype**: use `input_tensor.dtype` for all created tensors (bf16-safe)
- Minimize total XLA ops (collectives + local tensor operations)

**PRIMARY GOAL: Minimize total estimated latency while maintaining correctness.**
The current implementation has {current_local_ops} local ops. Study the op cost breakdown above — each op type has a specific cost. Replace expensive ops with cheaper alternatives where possible (e.g., metadata-only ops cost ~0us vs compute ops at ~29us). Use the profiled cost table in the optimization hints to estimate total latency.

Provide your implementation inside a ```python code block.
