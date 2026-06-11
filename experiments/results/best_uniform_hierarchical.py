#!/usr/bin/env python3
"""Auto-generated AllToAllV (hierarchical). Run: torchrun --nproc_per_node=32 <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist

INTER_SCHEDULE = [4, 1, 13, 3, 12, 5, 2, 7, 9, 15, 11, 10, 8, 14, 6]
CPD = 2  # cores per device
NUM_DEVICES = 16

def compute_recv_counts(send_counts):
    rank, world, device = xr.global_ordinal(), xr.world_size(), xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    g = xm.all_gather(t).view(world, world)
    return g[:, rank].tolist()

def _precompute_pairs(world):
    """Precompute device-level permute pairs for each inter-device distance."""
    pairs_list = []
    for d in INTER_SCHEDULE:
        pairs = []
        for r in range(world):
            r_dst = ((r // CPD + d) % NUM_DEVICES) * CPD + (r % CPD)
            pairs.append((r, r_dst))
        pairs_list.append(pairs)
    return pairs_list

def alltoallv(x, send_counts, recv_counts, max_chunk, precomputed_pairs):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
    my_dev = rank // CPD

    send_off, recv_off = [0], [0]
    for c in send_counts[:-1]: send_off.append(send_off[-1] + c)
    for c in recv_counts[:-1]: recv_off.append(recv_off[-1] + c)
    output = torch.empty(sum(recv_counts), device=device, dtype=dtype)

    # Level 1: Intra-device (free, shared HBM)
    output[recv_off[rank]:recv_off[rank]+recv_counts[rank]] = x[send_off[rank]:send_off[rank]+send_counts[rank]]
    peer = rank ^ 1
    if peer < world:
        output[recv_off[peer]:recv_off[peer]+recv_counts[peer]] = x[send_off[peer]:send_off[peer]+send_counts[peer]]

    # Level 2: Inter-device with aggregation (one collective_permute per step)
    for step_idx, d in enumerate(INTER_SCHEDULE):
        dst_dev = (my_dev + d) % NUM_DEVICES
        src_dev = (my_dev - d + NUM_DEVICES) % NUM_DEVICES
        # Aggregate data for all cores on dst_device
        chunks = []
        for c in range(CPD):
            dst_rank = dst_dev * CPD + c
            chunk = x[send_off[dst_rank]:send_off[dst_rank]+send_counts[dst_rank]]
            if send_counts[dst_rank] < max_chunk:
                chunk = torch.cat([chunk, torch.zeros(max_chunk-send_counts[dst_rank], device=device, dtype=dtype)])
            chunks.append(chunk)
        send_tensor = torch.cat(chunks, dim=0)
        recv_tensor = xm.collective_permute(send_tensor, pairs=precomputed_pairs[step_idx])
        # Unpack
        for c in range(CPD):
            from_rank = src_dev * CPD + c
            rc = recv_counts[from_rank]
            output[recv_off[from_rank]:recv_off[from_rank]+rc] = recv_tensor[c*max_chunk:c*max_chunk+rc]
    return output

def main():
    device = xla.device()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    world, rank = xr.world_size(), xr.global_ordinal()
    send_counts = [1024]*world
    recv_counts = compute_recv_counts(send_counts)
    mc = max(max(send_counts), max(recv_counts))
    precomputed_pairs = _precompute_pairs(world)
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32) + rank*100000
    alltoallv(x, send_counts, recv_counts, mc, precomputed_pairs); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters): alltoallv(x, send_counts, recv_counts, mc, precomputed_pairs); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"hierarchical latency: {(end-start)/iters*1000:.3f} ms")
if __name__ == "__main__": main()
