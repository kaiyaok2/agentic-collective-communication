# NKI AllToAllV Kernel Evolution

You are optimizing an AllToAllV communication kernel for AWS Trainium using NKI (Neuron Kernel Interface).

**NOTE**: On current trn1 hardware, NKI collectives have ~7x higher dispatch overhead than XLA collectives (~0.85ms vs ~0.12ms). This template is for research/future hardware where NKI dispatch improves.

## Hardware: trn1.32xlarge cluster ({num_nodes} node(s))
- {num_devices} NeuronDevices in a 2D 4x4 torus per node, {cores_per_device} NeuronCores each ({world_size} ranks total)
- Each device has 4 bidirectional NeuronLinks (~192 GB/s each)
- Intra-device core-to-core is free (shared HBM)
- {ranks_per_node} ranks per node

## NKI Programming Model

NKI compiles kernel functions directly to NeuronDevice instructions, bypassing XLA/HLO.

### NKI Memory Operations (nl module)
```python
buf = nl.ndarray((size,), dtype=nl.float32, buffer=nl.shared_hbm)
data = nl.load(hbm_buffer[start:end])
nl.store(hbm_buffer[start:end], data)
```

### NKI Collectives (nccl module)
```python
nccl.collective_permute(dst=recv_buffer, src=send_buffer, source_target_pairs=pairs)
nccl.all_gather(srcs=[local_buffer], dsts=[gathered_buffer], replica_groups=rank_list, all_gather_dim=0)
```

### What is NOT available on trn1
- `nl.gather_flattened` (Trn2+ only), `nki.isa.sendrecv` (Trn2+ only)
- `nccl.all_to_all`, `nccl.send/recv` (unstable)

## Required function signature

```python
def evolved_alltoallv_kernel(input_hbm, send_counts, recv_counts, max_chunk,
                             rank, world_size, num_devices, cores_per_device,
                             nl, nccl, num_nodes=1):
```

## Current best: {current_sim_time} us, {current_num_permutes} permutes, {current_num_gathers} gathers

```python
{current_code}
```

## Reference implementations

### NKI Naive AllGather:
```python
{builtin_nki_naive_allgather}
```

### NKI Permute Ring:
```python
{builtin_nki_permute_ring}
```

## Evolution history
{history}

## Your task

Write an IMPROVED `evolved_alltoallv_kernel`. Minimize collective dispatches.
Provide your implementation inside a ```python code block.
