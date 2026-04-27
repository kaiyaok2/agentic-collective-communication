#!/usr/bin/env python3
"""Auto-generated AllToAllV (evolved). Run: torchrun --nproc_per_node=32 <file>.py"""
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
NUM_DEVICES = 32 // 2
CORES_PER_DEVICE = 2

def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """Optimized AllGather with single index_select for unpacking."""
    
    pack_size = world_size * max_chunk
    
    # Pack: single buffer fill using Python-computed offsets
    packed = torch.zeros(pack_size, device=input_tensor.device,
                         dtype=input_tensor.dtype)
    send_off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = \
                input_tensor[send_off:send_off + sc]
        send_off += sc
    
    # Single all_gather: collect all packed buffers
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0).view(-1)
    
    # Build flat index list in Python for single index_select
    # gathered layout: [rank0_pack | rank1_pack | ... | rank(N-1)_pack]
    # where rankX_pack has data for all destinations at [dest*max_chunk:dest*max_chunk+count]
    flat_idx = []
    for src in range(world_size):
        count = recv_counts[src]
        if count > 0:
            base = src * pack_size + rank * max_chunk
            flat_idx.extend(range(base, base + count))
    
    # Single index_select to extract all received data
    if len(flat_idx) == 0:
        return torch.zeros(0, device=input_tensor.device, dtype=input_tensor.dtype)
    
    idx_tensor = torch.tensor(flat_idx, device=input_tensor.device, dtype=torch.long)
    result = torch.index_select(gathered, 0, idx_tensor)
    
    return result

def main():
    device = xla.device()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    world, rank = xr.world_size(), xr.global_ordinal()
    matrix = make_moe_send_counts(world, SHARD_SIZE)
    send_counts = matrix[rank]
    recv_counts = [matrix[src][rank] for src in range(world)]
    mc = max(matrix[s][d] for s in range(world) for d in range(world))
    mc = max(mc, 1)
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
    out = evolved_alltoallv(x, send_counts, recv_counts, mc, rank, world,
                      NUM_DEVICES, CORES_PER_DEVICE, xm, torch)
    out.sum().backward(); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters):
        x_i = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
        out = evolved_alltoallv(x_i, send_counts, recv_counts, mc, rank, world,
                          NUM_DEVICES, CORES_PER_DEVICE, xm, torch)
        out.sum().backward(); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"evolved latency: {(end-start)/iters*1000:.3f} ms")
if __name__ == "__main__": main()
