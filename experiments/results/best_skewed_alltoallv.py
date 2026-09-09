#!/usr/bin/env python3
"""Auto-generated AllToAllV with optimized schedule. Run with: torchrun --nproc_per_node=32 <this_file>.py"""
import time
import torch
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist

SCHEDULE = [1, 2, 7, 31, 30, 9, 8, 6, 5, 25, 26, 3, 24, 15, 10, 4, 14, 23, 29, 18, 27, 28, 16, 17, 11, 13, 22, 21, 19, 20, 12]
SHARD_SIZE = 1024

def compute_recv_counts(send_counts):
    rank = xr.global_ordinal()
    world = xr.world_size()
    device = xm.xla_device()
    send_counts_tensor = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(send_counts_tensor)
    gathered = gathered.view(world, world)
    recv_counts = gathered[:, rank].tolist()
    return recv_counts

def alltoallv_optimized(x, send_counts, recv_counts, max_chunk):
    rank = xr.global_ordinal()
    world = xr.world_size()
    device = x.device
    dtype = x.dtype

    send_offsets = [0]
    for c in send_counts[:-1]:
        send_offsets.append(send_offsets[-1] + c)
    recv_offsets = [0]
    for c in recv_counts[:-1]:
        recv_offsets.append(recv_offsets[-1] + c)

    total_recv = sum(recv_counts)
    output = torch.empty(total_recv, device=device, dtype=dtype)

    # Prebuild padded shards
    shards = []
    for i in range(world):
        start = send_offsets[i]
        end = start + send_counts[i]
        chunk = x[start:end]
        if send_counts[i] < max_chunk:
            pad = torch.zeros(max_chunk - send_counts[i], device=device, dtype=dtype)
            chunk = torch.cat([chunk, pad], dim=0)
        shards.append(chunk)

    # Precompute permute pairs
    permute_pairs = [
        [(r, (r + d) % world) for r in range(world)]
        for d in SCHEDULE
    ]

    # Self copy
    self_chunk = shards[rank][:recv_counts[rank]]
    output[recv_offsets[rank]:recv_offsets[rank] + recv_counts[rank]] = self_chunk

    # Communication with optimized schedule
    for i, d in enumerate(SCHEDULE):
        send_to = (rank + d) % world
        recv_from = (rank - d) % world
        send_tensor = shards[send_to]
        recv_tensor = xm.collective_permute(send_tensor, pairs=permute_pairs[i])
        recv_tensor = recv_tensor[:recv_counts[recv_from]]
        output[recv_offsets[recv_from]:recv_offsets[recv_from] + recv_counts[recv_from]] = recv_tensor

    return output

def main():
    device = xla.device()
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")

    world = xr.world_size()
    rank = xr.global_ordinal()

    # Generate send counts based on pattern
    send_counts = [1024] * world  # uniform
    recv_counts = compute_recv_counts(send_counts)
    total_send = sum(send_counts)
    max_chunk = max(max(send_counts), max(recv_counts))

    x = torch.arange(total_send, device=device, dtype=torch.float32) + rank * 100000

    # Warmup
    alltoallv_optimized(x, send_counts, recv_counts, max_chunk)
    xla.step()

    # Benchmark
    iters = 10
    start = time.time()
    for _ in range(iters):
        alltoallv_optimized(x, send_counts, recv_counts, max_chunk)
        xla.step()
    xm.wait_device_ops()
    end = time.time()

    if rank == 0:
        print(f"Optimized AllToAllV latency: {(end-start)/iters*1000:.3f} ms")
        print(f"Schedule: {SCHEDULE}")

if __name__ == "__main__":
    main()
