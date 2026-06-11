#!/usr/bin/env python3
"""Auto-generated AllToAllV (evolved_ag). Run: torchrun --nproc_per_node=32 <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
import random as _rng_mod

def make_moe_send_counts(world, shard_size):
    """MoE traffic: Zipf(s=1.2) expert popularity, shuffled assignment."""
    rng = _rng_mod.Random(42)
    raw = [1.0 / (i + 1) ** 1.2 for i in range(world)]
    perm = list(range(world))
    rng.shuffle(perm)
    probs = [0.0] * world
    for i, p in enumerate(perm):
        probs[p] = raw[i]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    cdf, acc = [], 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    matrix = [[0] * world for _ in range(world)]
    for s in range(world):
        counts = [0] * world
        for _ in range(shard_size):
            r = rng.random()
            for d in range(world):
                if r <= cdf[d]:
                    counts[d] += 1
                    break
        matrix[s] = counts
    return matrix

SHARD_SIZE = 1024

def alltoallv(x, send_counts, matrix):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype

    # Compute uniform padded buffer size
    max_total = max(sum(matrix[r]) for r in range(world))
    max_total = max(max_total, 1)

    # Build flat extraction index
    flat_idx_list = []
    for src in range(world):
        src_sc = matrix[src]
        src_off = [0]
        for c in src_sc[:-1]: src_off.append(src_off[-1] + c)
        count = src_sc[rank]
        base = src * max_total + src_off[rank]
        flat_idx_list.extend(range(base, base + count))
    ag_flat_idx = torch.tensor(flat_idx_list, device=device, dtype=torch.long)

    # Pad, all_gather, index_select (~3 XLA ops)
    total_send = sum(send_counts)
    if total_send < max_total:
        x_padded = torch.cat([x, torch.zeros(max_total - total_send, device=device, dtype=dtype)])
    else:
        x_padded = x
    gathered = xm.all_gather(x_padded.unsqueeze(0), dim=0).view(-1)
    return torch.index_select(gathered, 0, ag_flat_idx)

def main():
    device = xla.device()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    world, rank = xr.world_size(), xr.global_ordinal()
    matrix = make_moe_send_counts(world, SHARD_SIZE)
    send_counts = matrix[rank]
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32) + rank*100000
    alltoallv(x, send_counts, matrix); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters): alltoallv(x, send_counts, matrix); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"evolved_ag latency: {(end-start)/iters*1000:.3f} ms")
if __name__ == "__main__": main()
