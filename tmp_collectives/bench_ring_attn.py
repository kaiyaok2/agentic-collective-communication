#!/usr/bin/env python3
"""
Benchmark #4: Ring Attention (Sequence-Parallel KV rotation)
Baseline: naive ring with collective_permute per step
Optimized: all_gather-based KV distribution (fewer dispatches)
"""
import os, time, json
os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache_ring')

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def baseline_ring_permute(kv_chunk, ws):
    """Naive ring attention: rotate KV chunks ws-1 times via collective_permute.
    Each step is a separate collective dispatch."""
    rank = xr.global_ordinal()
    results = [kv_chunk.clone()]

    current = kv_chunk
    for step in range(1, ws):
        src = (rank - step) % ws
        dst = (rank + step) % ws
        pairs = [[i, (i + step) % ws] for i in range(ws)]
        current = xm.collective_permute(current, pairs)
        results.append(current)

    return torch.cat(results, dim=0)


def optimized_allgather_kv(kv_chunk, ws):
    """All-gather all KV chunks at once — 1 collective dispatch instead of ws-1.
    Trades bandwidth for fewer dispatches."""
    gathered = xm.all_gather(kv_chunk.unsqueeze(0), dim=0)
    return gathered.view(-1, kv_chunk.shape[-1]) if kv_chunk.dim() > 1 else gathered.view(-1)


def optimized_2step_ring(kv_chunk, ws):
    """2-step hierarchical: all_gather within node, then 1 cross-node permute.
    Intra-node: 1 dispatch (NeuronLink). Cross-node: 1 dispatch (EFA).
    Total: 2 dispatches instead of ws-1."""
    rank = xr.global_ordinal()
    rpn = 32  # ranks per node

    intra_groups = [list(range(n * rpn, (n + 1) * rpn)) for n in range(ws // rpn)]
    local_gathered = xm.all_gather(kv_chunk.unsqueeze(0), dim=0, groups=intra_groups)

    inter_groups = [[lr + n * rpn for n in range(ws // rpn)] for lr in range(rpn)]
    full_gathered = xm.all_gather(local_gathered, dim=0, groups=inter_groups)

    return full_gathered.view(-1) if kv_chunk.dim() == 1 else full_gathered.reshape(-1, kv_chunk.shape[-1])


def run():
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()

    warmup = 3
    iters = 10

    for head_dim in [128]:
        for seq_per_rank in [64, 256]:
            kv_size = seq_per_rank * head_dim * 2  # K + V
            kv = torch.randn(kv_size, device=dev, dtype=torch.bfloat16)

            if rank == 0:
                print(f"\n[ring_attn] ws={ws}  seq/rank={seq_per_rank}  hd={head_dim}  kv_elems={kv_size:,}")

            # Only test subset of ring steps to avoid massive compilation
            ring_steps_to_test = min(ws - 1, 4)  # limit ring to 4 steps for compile time

            methods = {}

            # Partial ring baseline (limited steps)
            def partial_ring():
                current = kv
                for step in range(1, ring_steps_to_test + 1):
                    pairs = [[i, (i + step) % ws] for i in range(ws)]
                    current = xm.collective_permute(current, pairs)
                    xm.mark_step()
                return current
            methods[f'ring_{ring_steps_to_test}steps'] = partial_ring

            methods['allgather_1dispatch'] = lambda: optimized_allgather_kv(kv, ws)
            methods['hierarchical_2dispatch'] = lambda: optimized_2step_ring(kv, ws)

            for name, fn in methods.items():
                try:
                    for _ in range(warmup):
                        fn()
                        xm.mark_step()

                    xm.wait_device_ops()
                    t0 = time.time()
                    for _ in range(iters):
                        fn()
                        xm.mark_step()
                    xm.wait_device_ops()
                    avg_ms = (time.time() - t0) / iters * 1000
                    if rank == 0:
                        print(f"  {name:30s}: {avg_ms:.3f} ms")
                except Exception as e:
                    if rank == 0:
                        print(f"  {name:30s}: FAILED ({type(e).__name__}: {e})")

    if rank == 0:
        print("\nDone.")


if __name__ == '__main__':
    run()
