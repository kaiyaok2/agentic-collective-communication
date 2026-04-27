#!/usr/bin/env python3
"""
Benchmark #5: Fused AllReduce (Gradient Bucketing)
Baseline: one all_reduce per parameter tensor (many dispatches)
Optimized: flatten all grads, single all_reduce (1 dispatch)
Also: smart bucketing (few dispatches, less padding)
"""
import os, time, json
os.environ.setdefault('NEURON_NUM_RECENT_MODELS_TO_KEEP', '1')
os.environ.setdefault('NEURON_COMPILE_CACHE_URL', '/tmp/neuron_cache_bucket')

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def baseline_per_param_allreduce(grads, ws):
    """One all_reduce per gradient tensor — N dispatches for N params."""
    for g in grads:
        xm.all_reduce('sum', g)
        g.data /= ws


def optimized_single_allreduce(grads, ws):
    """Flatten all grads into one buffer, single all_reduce, unflatten."""
    flat = torch.cat([g.view(-1) for g in grads])
    reduced = xm.all_reduce('sum', flat)
    reduced /= ws
    off = 0
    for g in grads:
        numel = g.numel()
        g.data.copy_(reduced[off:off + numel].view_as(g))
        off += numel


def optimized_bucket_allreduce(grads, ws, bucket_mb=8):
    """Bucket grads into ~8MB chunks, one all_reduce per bucket."""
    bucket_bytes = bucket_mb * 1024 * 1024
    elem_bytes = 2  # bf16

    buckets = []
    current_bucket = []
    current_size = 0

    for g in grads:
        sz = g.numel() * elem_bytes
        if current_size + sz > bucket_bytes and current_bucket:
            buckets.append(current_bucket)
            current_bucket = []
            current_size = 0
        current_bucket.append(g)
        current_size += sz
    if current_bucket:
        buckets.append(current_bucket)

    for bucket in buckets:
        flat = torch.cat([g.view(-1) for g in bucket])
        reduced = xm.all_reduce('sum', flat)
        reduced /= ws
        off = 0
        for g in bucket:
            numel = g.numel()
            g.data.copy_(reduced[off:off + numel].view_as(g))
            off += numel


def optimized_list_allreduce(grads, ws):
    """Pass list of tensors to xm.all_reduce — let XLA bucket internally."""
    xm.all_reduce('sum', grads)
    for g in grads:
        g.data /= ws


def run():
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    rank = xr.global_ordinal()
    ws = xr.world_size()
    dev = xm.xla_device()

    warmup = 5
    iters = 20

    # Simulate realistic model gradient shapes (DeepSeek-MoE-Lite-like)
    shapes = [
        (32768, 2048),  # embedding
        (2048,),        # norm
    ]
    for _ in range(12):  # 12 layers
        shapes.extend([
            (2048, 6144),   # QKV
            (2048, 2048),   # O proj
            (2048,),        # norm1
            (2048, 64),     # gate
            (2048, 1408),   # expert up
            (1408, 2048),   # expert down
            (2048,),        # norm2
        ])
    shapes.append((2048, 32768))  # LM head

    grads_template = [torch.randn(s, device=dev, dtype=torch.bfloat16) for s in shapes]
    n_params = len(shapes)
    total_elems = sum(g.numel() for g in grads_template)

    if rank == 0:
        print(f"[grad_bucket] ws={ws}  n_params={n_params}  total_elems={total_elems:,}")
        print(f"  warmup={warmup}  iters={iters}")

    methods = {
        'per_param_allreduce': lambda gs: baseline_per_param_allreduce(gs, ws),
        'single_flat_allreduce': lambda gs: optimized_single_allreduce(gs, ws),
        'bucket_8mb_allreduce': lambda gs: optimized_bucket_allreduce(gs, ws, 8),
        'list_allreduce_xla': lambda gs: optimized_list_allreduce(gs, ws),
    }

    results = {}
    for name, fn in methods.items():
        for _ in range(warmup):
            gs = [g.clone() for g in grads_template]
            fn(gs)
            xm.mark_step()

        xm.wait_device_ops()
        t0 = time.time()
        for _ in range(iters):
            gs = [g.clone() for g in grads_template]
            fn(gs)
            xm.mark_step()
        xm.wait_device_ops()
        avg_ms = (time.time() - t0) / iters * 1000
        results[name] = round(avg_ms, 3)
        if rank == 0:
            print(f"  {name:30s}: {avg_ms:.3f} ms")

    if rank == 0:
        path = '/home/ubuntu/trainium-llm-search/tmp_collectives/results_bucket.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {path}")


if __name__ == '__main__':
    run()
