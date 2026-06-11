#!/usr/bin/env python3
"""Auto-generated AllToAllV (hierarchical). Run: torchrun --nproc_per_node=32 <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist

INTER_SCHEDULE = [4, 3, 15, 1, 12, 13, 5, 2, 7, 11, 9, 14, 8, 6, 10]
CPD = 2  # cores per device

def compute_recv_counts(send_counts):
    rank, world, device = xr.global_ordinal(), xr.world_size(), xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    g = xm.all_gather(t).view(world, world)
    return g[:, rank].tolist()

def alltoallv(x, send_counts, recv_counts, max_chunk):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
    my_dev = rank // CPD
    peer = rank ^ 1

    send_off, recv_off = [0], [0]
    for c in send_counts[:-1]: send_off.append(send_off[-1] + c)
    for c in recv_counts[:-1]: recv_off.append(recv_off[-1] + c)
    output = torch.empty(sum(recv_counts), device=device, dtype=dtype)

    shards = []
    for i in range(world):
        chunk = x[send_off[i]:send_off[i]+send_counts[i]]
        if send_counts[i] < max_chunk:
            chunk = torch.cat([chunk, torch.zeros(max_chunk-send_counts[i], device=device, dtype=dtype)])
        shards.append(chunk)

    # Intra-device (free)
    output[recv_off[rank]:recv_off[rank]+recv_counts[rank]] = shards[rank][:recv_counts[rank]]
    output[recv_off[peer]:recv_off[peer]+recv_counts[peer]] = shards[peer][:recv_counts[peer]]

    # Inter-device
    num_devices = world // CPD
    for d in INTER_SCHEDULE:
        for core_off in range(CPD):
            rank_d = d * CPD + core_off
            if rank_d == 0 or rank_d >= world: continue
            send_to = (rank + rank_d) % world
            recv_from = (rank - rank_d) % world
            if rank // CPD == send_to // CPD: continue
            pairs = [(r, (r + rank_d) % world) for r in range(world)]
            rt = xm.collective_permute(shards[send_to], pairs=pairs)
            output[recv_off[recv_from]:recv_off[recv_from]+recv_counts[recv_from]] = rt[:recv_counts[recv_from]]
    return output

def main():
    device = xla.device()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    world, rank = xr.world_size(), xr.global_ordinal()
    send_counts = [1024]*world
    recv_counts = compute_recv_counts(send_counts)
    mc = max(max(send_counts), max(recv_counts))
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32) + rank*100000
    alltoallv(x, send_counts, recv_counts, mc); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters): alltoallv(x, send_counts, recv_counts, mc); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"hierarchical latency: {(end-start)/iters*1000:.3f} ms")
if __name__ == "__main__": main()
