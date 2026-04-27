#!/usr/bin/env python3
"""Auto-generated AllToAllV (fused_alltoall). Run: torchrun --nproc_per_node=32 <file>.py"""
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

def alltoallv(x, send_counts, recv_counts, max_chunk):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
    send_off, recv_off = [0], [0]
    for c in send_counts[:-1]: send_off.append(send_off[-1] + c)
    for c in recv_counts[:-1]: recv_off.append(recv_off[-1] + c)
    # Pack: slot i = data destined for rank i, padded to max_chunk
    packed = torch.zeros(world * max_chunk, device=device, dtype=dtype)
    for i in range(world):
        packed[i * max_chunk:i * max_chunk + send_counts[i]] = \
            x[send_off[i]:send_off[i] + send_counts[i]]
    # Single all_to_all: chunk i goes to rank i
    received = xm.all_to_all(packed, split_dimension=0,
                             concat_dimension=0, split_count=world)
    # Unpack
    output = torch.empty(sum(recv_counts), device=device, dtype=dtype)
    for i in range(world):
        rc = recv_counts[i]
        output[recv_off[i]:recv_off[i] + rc] = \
            received[i * max_chunk:i * max_chunk + rc]
    return output

def main():
    device = xla.device()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    world, rank = xr.world_size(), xr.global_ordinal()
    matrix = make_moe_send_counts(world, SHARD_SIZE)
    send_counts = matrix[rank]
    recv_counts = [matrix[src][rank] for src in range(world)]
    mc = max(matrix[s][d] for s in range(world) for d in range(world))
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
    out = alltoallv(x, send_counts, recv_counts, mc); out.sum().backward(); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters):
        x_i = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
        out = alltoallv(x_i, send_counts, recv_counts, mc); out.sum().backward(); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"fused_alltoall latency: {(end-start)/iters*1000:.3f} ms")
if __name__ == "__main__": main()
