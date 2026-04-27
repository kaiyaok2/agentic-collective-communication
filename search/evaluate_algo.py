"""
Evaluate AllToAllV algorithms across templates: simulation + hardware validation + C++ codegen.
"""

import os
import json
import re
import subprocess
import tempfile
from pathlib import Path

from simulator.topology import TrainiumTopology
from simulator.alltoallv import AllToAllVSimulator
from simulator.cost_model import CostModel
from search.profiling import profile_schedule, format_profiling_report


def evaluate_template(template_name, params, send_counts_matrix,
                      topology=None, element_bytes=4):
    """
    Evaluate a template + params using simulation.

    Returns:
        dict with simulation metrics including per-step profiling data
    """
    if topology is None:
        topology = TrainiumTopology()

    cost = CostModel(topology, send_counts_matrix, element_bytes)
    sim = AllToAllVSimulator(topology, send_counts_matrix, element_bytes)
    lb = sim.lower_bound()

    score, breakdown = cost.evaluate_template(template_name, params)

    result = {
        "template": template_name,
        "params": {k: v for k, v in params.items() if not k.startswith("_")},
        "lower_bound_us": lb * 1e6,
        "efficiency": lb / max(breakdown["sim_time_us"] * 1e-6, 1e-15),
        "cost_score": score,
        **breakdown,
    }

    # Attach per-step profiling breakdown
    try:
        prof = profile_schedule(template_name, params, send_counts_matrix,
                                topology, element_bytes)
        result["step_times_us"] = [d.get("time_us", 0) for d in prof.step_details]
        result["bottleneck_steps"] = prof.bottleneck_steps(3)
    except Exception:
        pass

    return result


# Keep backward compat
def evaluate_schedule_sim(schedule, send_counts_matrix, topology=None, element_bytes=4):
    result = evaluate_template("permute_ring", {"schedule": schedule},
                               send_counts_matrix, topology, element_bytes)
    result["schedule"] = schedule
    return result


def generate_trainium_code(template_name, params, shard_size=1024, world=32, num_nodes=1):
    """
    Generate runnable Python code for any template.

    Generated code uses MoE traffic (Zipf-distributed expert popularity)
    with non-uniform per-rank send_counts.

    Returns:
        str: Python source code for torchrun
    """
    if template_name == "permute_ring":
        return _gen_permute_ring(params["schedule"], shard_size, world)
    elif template_name == "allgather_slice":
        return _gen_allgather_slice(shard_size, world,
                                     params.get("chunk_factor", 1))
    elif template_name == "hierarchical":
        return _gen_hierarchical(params["inter_schedule"], shard_size, world)
    elif template_name == "pairwise":
        return _gen_pairwise(params["_matchings"], params["round_order"],
                              shard_size, world)
    elif template_name == "hybrid_ag_perm":
        return _gen_hybrid(params, shard_size, world)
    elif template_name == "fused_alltoall":
        return _gen_fused_alltoall(shard_size, world)
    elif template_name == "multinode_hierarchical":
        return _gen_multinode_hierarchical(params, shard_size, world, num_nodes)
    elif template_name == "allgather_reduce_scatter":
        return _gen_allgather_reduce_scatter(shard_size, world)
    elif template_name.startswith("evolved") or template_name.startswith("sorcar_"):
        return _gen_evolved(params, shard_size, world)
    elif params.get("evolved_code"):
        return _gen_evolved(params, shard_size, world)
    else:
        raise ValueError(f"Unknown template: {template_name}")


# MoE traffic generation snippet embedded in generated benchmark code
_MOE_TRAFFIC_SNIPPET = '''\
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
'''


def _gen_permute_ring(schedule, shard_size, world):
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (permute_ring). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
SCHEDULE = {repr(schedule)}
SHARD_SIZE = {shard_size}

def alltoallv(x, send_counts, recv_counts, max_chunk):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
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
    pairs = [[(r, (r+d)%world) for r in range(world)] for d in SCHEDULE]
    output[recv_off[rank]:recv_off[rank]+recv_counts[rank]] = shards[rank][:recv_counts[rank]]
    for i, d in enumerate(SCHEDULE):
        sf, rf = (rank+d)%world, (rank-d)%world
        rt = xm.collective_permute(shards[sf], pairs=pairs[i])
        output[recv_off[rf]:recv_off[rf]+recv_counts[rf]] = rt[:recv_counts[rf]]
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
    if rank == 0: print(f"permute_ring latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_allgather_slice(shard_size, world, chunk_factor=1):
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (allgather_slice). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
SHARD_SIZE = {shard_size}

def alltoallv(x, send_counts, matrix):
    rank, world = xr.global_ordinal(), xr.world_size()
    # Pad to max total so all_gather gets uniform-sized inputs
    max_total = max(sum(matrix[r]) for r in range(world))
    total_send = sum(send_counts)
    if total_send < max_total:
        x_padded = torch.cat([x, torch.zeros(max_total - total_send, device=x.device, dtype=x.dtype)])
    else:
        x_padded = x
    gathered = xm.all_gather(x_padded.unsqueeze(0), dim=0).view(world, max_total)
    chunks = []
    for src in range(world):
        src_sc = matrix[src]
        src_off = [0]
        for c in src_sc[:-1]: src_off.append(src_off[-1] + c)
        chunks.append(gathered[src, src_off[rank]:src_off[rank]+src_sc[rank]])
    return torch.cat(chunks, dim=0)

def main():
    device = xla.device()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    world, rank = xr.world_size(), xr.global_ordinal()
    matrix = make_moe_send_counts(world, SHARD_SIZE)
    send_counts = matrix[rank]
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
    out = alltoallv(x, send_counts, matrix); out.sum().backward(); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters):
        x_i = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
        out = alltoallv(x_i, send_counts, matrix); out.sum().backward(); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"allgather_slice latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_hierarchical(inter_schedule, shard_size, world):
    num_devices = world // 2
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (hierarchical). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
INTER_SCHEDULE = {repr(inter_schedule)}
CPD = 2  # cores per device
NUM_DEVICES = {num_devices}
SHARD_SIZE = {shard_size}

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
    matrix = make_moe_send_counts(world, SHARD_SIZE)
    send_counts = matrix[rank]
    recv_counts = [matrix[src][rank] for src in range(world)]
    mc = max(matrix[s][d] for s in range(world) for d in range(world))
    precomputed_pairs = _precompute_pairs(world)
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
    out = alltoallv(x, send_counts, recv_counts, mc, precomputed_pairs); out.sum().backward(); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters):
        x_i = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
        out = alltoallv(x_i, send_counts, recv_counts, mc, precomputed_pairs); out.sum().backward(); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"hierarchical latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_pairwise(matchings, round_order, shard_size, world):
    # Emit matchings inline
    ordered_matchings = [matchings[i] for i in round_order]
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (pairwise). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
MATCHINGS = {repr(ordered_matchings)}
SHARD_SIZE = {shard_size}

def alltoallv(x, send_counts, recv_counts, max_chunk):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
    send_off, recv_off = [0], [0]
    for c in send_counts[:-1]: send_off.append(send_off[-1] + c)
    for c in recv_counts[:-1]: recv_off.append(recv_off[-1] + c)
    output = torch.empty(sum(recv_counts), device=device, dtype=dtype)
    output[recv_off[rank]:recv_off[rank]+recv_counts[rank]] = x[send_off[rank]:send_off[rank]+send_counts[rank]]

    shards = []
    for i in range(world):
        chunk = x[send_off[i]:send_off[i]+send_counts[i]]
        if send_counts[i] < max_chunk:
            chunk = torch.cat([chunk, torch.zeros(max_chunk-send_counts[i], device=device, dtype=dtype)])
        shards.append(chunk)

    for pairs in MATCHINGS:
        partner = None
        for a, b in pairs:
            if a == rank: partner = b; break
            if b == rank: partner = a; break
        if partner is None: continue
        permute_pairs = []
        for a, b in pairs:
            permute_pairs.extend([(a, b), (b, a)])
        rt = xm.collective_permute(shards[partner], pairs=permute_pairs)
        output[recv_off[partner]:recv_off[partner]+recv_counts[partner]] = rt[:recv_counts[partner]]
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
    if rank == 0: print(f"pairwise latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_hybrid(params, shard_size, world):
    near = params["near_distances"]
    far_schedule = params["permute_schedule"]
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (hybrid). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
NEAR_DISTANCES = {repr(near)}
FAR_SCHEDULE = {repr(far_schedule)}
SHARD_SIZE = {shard_size}

def alltoallv(x, send_counts, recv_counts, max_chunk, matrix):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
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

    # Self copy
    output[recv_off[rank]:recv_off[rank]+recv_counts[rank]] = shards[rank][:recv_counts[rank]]

    # Near: allgather approach (pad to max_total for uniform all_gather)
    if NEAR_DISTANCES:
        max_total = max(sum(matrix[r]) for r in range(world))
        total_send = sum(send_counts)
        if total_send < max_total:
            x_padded = torch.cat([x, torch.zeros(max_total - total_send, device=device, dtype=dtype)])
        else:
            x_padded = x
        gathered = xm.all_gather(x_padded.unsqueeze(0), dim=0).view(world, max_total)
        for d in NEAR_DISTANCES:
            recv_from = (rank - d) % world
            src_sc = matrix[recv_from]
            src_off = [0]
            for c in src_sc[:-1]: src_off.append(src_off[-1] + c)
            output[recv_off[recv_from]:recv_off[recv_from]+recv_counts[recv_from]] = \\
                gathered[recv_from, src_off[rank]:src_off[rank]+src_sc[rank]][:recv_counts[recv_from]]

    # Far: permute approach
    pairs = [[(r, (r+d)%world) for r in range(world)] for d in FAR_SCHEDULE]
    for i, d in enumerate(FAR_SCHEDULE):
        sf, rf = (rank+d)%world, (rank-d)%world
        rt = xm.collective_permute(shards[sf], pairs=pairs[i])
        output[recv_off[rf]:recv_off[rf]+recv_counts[rf]] = rt[:recv_counts[rf]]
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
    out = alltoallv(x, send_counts, recv_counts, mc, matrix); out.sum().backward(); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters):
        x_i = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
        out = alltoallv(x_i, send_counts, recv_counts, mc, matrix); out.sum().backward(); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"hybrid latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_fused_alltoall(shard_size, world):
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (fused_alltoall). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
SHARD_SIZE = {shard_size}

def alltoallv(x, send_counts, recv_counts, max_chunk):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
    send_off, recv_off = [0], [0]
    for c in send_counts[:-1]: send_off.append(send_off[-1] + c)
    for c in recv_counts[:-1]: recv_off.append(recv_off[-1] + c)
    # Pack: slot i = data destined for rank i, padded to max_chunk
    packed = torch.zeros(world * max_chunk, device=device, dtype=dtype)
    for i in range(world):
        packed[i * max_chunk:i * max_chunk + send_counts[i]] = \\
            x[send_off[i]:send_off[i] + send_counts[i]]
    # Single all_to_all: chunk i goes to rank i
    received = xm.all_to_all(packed, split_dimension=0,
                             concat_dimension=0, split_count=world)
    # Unpack
    output = torch.empty(sum(recv_counts), device=device, dtype=dtype)
    for i in range(world):
        rc = recv_counts[i]
        output[recv_off[i]:recv_off[i] + rc] = \\
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
    if rank == 0: print(f"fused_alltoall latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_evolved(params, shard_size, world):
    evolved_code = params.get("evolved_code", "")
    if not evolved_code:
        raise ValueError("evolved template has no evolved_code in params")
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (evolved). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
SHARD_SIZE = {shard_size}
NUM_DEVICES = {world} // 2
CORES_PER_DEVICE = 2

{evolved_code.strip()}

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
    if rank == 0: print(f"evolved latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_allgather_reduce_scatter(shard_size, world):
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (allgather+reduce_scatter). Run: torchrun --nproc_per_node={world} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
SHARD_SIZE = {shard_size}

def alltoallv(x, send_counts, matrix):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype

    max_chunk = max(matrix[s][d] for s in range(world) for d in range(world))
    max_chunk = max(max_chunk, 1)
    recv_counts = [matrix[s][rank] for s in range(world)]
    pack_size = world * max_chunk

    packed = torch.zeros(pack_size, device=device, dtype=dtype)
    send_off = 0
    for i in range(world):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = x[send_off:send_off + sc]
        send_off += sc

    gathered = xm.all_gather(packed.unsqueeze(0), dim=0)
    reshaped = gathered.view(world, world, max_chunk)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)

    my_shard = xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0/world,
                                  scatter_dim=0, shard_count=world)

    flat_idx = []
    for src in range(world):
        count = recv_counts[src]
        base = src * max_chunk
        flat_idx.extend(range(base, base + count))
    idx_t = torch.tensor(flat_idx, device=device, dtype=torch.long)
    return torch.index_select(my_shard, 0, idx_t)

def main():
    device = xla.device()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    world, rank = xr.world_size(), xr.global_ordinal()
    matrix = make_moe_send_counts(world, SHARD_SIZE)
    send_counts = matrix[rank]
    x = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
    out = alltoallv(x, send_counts, matrix); out.sum().backward(); xla.step()
    iters = 10; start = time.time()
    for _ in range(iters):
        x_i = torch.arange(sum(send_counts), device=device, dtype=torch.float32, requires_grad=True) + rank*100000
        out = alltoallv(x_i, send_counts, matrix); out.sum().backward(); xla.step()
    xm.wait_device_ops(); end = time.time()
    if rank == 0: print(f"allgather_reduce_scatter latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


def _gen_multinode_hierarchical(params, shard_size, world, num_nodes):
    rpn = world // max(num_nodes, 1)
    return f'''\
#!/usr/bin/env python3
"""Auto-generated AllToAllV (multinode_hierarchical).
Run: torchrun --nnodes={num_nodes} --nproc_per_node={rpn} <file>.py"""
import time, torch, torch_xla as xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
import torch.distributed as dist
{_MOE_TRAFFIC_SNIPPET}
SHARD_SIZE = {shard_size}
NUM_NODES = {num_nodes}
RANKS_PER_NODE = {rpn}

def alltoallv(x, send_counts, recv_counts, max_chunk):
    rank, world = xr.global_ordinal(), xr.world_size()
    device, dtype = x.device, x.dtype
    rpn = RANKS_PER_NODE
    send_off = [0]
    for c in send_counts[:-1]: send_off.append(send_off[-1] + c)
    # Pack into canonical layout
    pack_size = world * max_chunk
    packed = torch.zeros(pack_size, device=device, dtype=dtype)
    for i in range(world):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = x[send_off[i]:send_off[i] + sc]
    # Phase 1: AllGather within node
    node_groups = [list(range(n * rpn, (n + 1) * rpn)) for n in range(NUM_NODES)]
    p1 = xm.all_gather(packed.unsqueeze(0), dim=0, groups=node_groups)
    p1_flat = p1.view(-1)
    # Phase 2: AllGather across nodes
    cross_groups = [[n * rpn + lr for n in range(NUM_NODES)] for lr in range(rpn)]
    p2 = xm.all_gather(p1_flat.unsqueeze(0), dim=0, groups=cross_groups)
    gf = p2.view(-1)
    # Extract
    idx_list = []
    for src in range(world):
        sn, sl = src // rpn, src % rpn
        base = sn * rpn * pack_size + sl * pack_size + rank * max_chunk
        idx_list.extend(range(base, base + recv_counts[src]))
    idx = torch.tensor(idx_list, device=device, dtype=torch.long)
    return torch.index_select(gf, 0, idx)

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
    if rank == 0: print(f"multinode_hierarchical latency: {{(end-start)/iters*1000:.3f}} ms")
if __name__ == "__main__": main()
'''


NEURON_VENV = os.environ.get(
    "NEURON_VENV", "/opt/aws_neuronx_venv_pytorch_2_9")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")


def run_on_hardware(template_name, params, send_counts_matrix,
                    shard_size=1024, nproc=32, timeout=120,
                    num_nodes=1, master_addr='localhost',
                    worker_addrs=None):
    """Generate code and run on actual Trainium hardware.

    For multi-node (num_nodes > 1 and worker_addrs), copies the script
    to worker nodes and launches torchrun via SSH in parallel.
    """
    code = generate_trainium_code(template_name, params, shard_size, nproc, num_nodes)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/home/ubuntu",
                                     delete=False, prefix="bench_") as f:
        f.write(code)
        script_path = f.name

    try:
        torchrun_bin = os.path.join(NEURON_VENV, "bin", "torchrun")
        if num_nodes > 1:
            cmd = [
                torchrun_bin,
                f"--nnodes={num_nodes}",
                f"--nproc_per_node={nproc}",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={master_addr}:{MASTER_PORT}",
                script_path,
            ]
        else:
            cmd = [torchrun_bin, f"--nproc_per_node={nproc}", script_path]

        if num_nodes > 1 and worker_addrs:
            output = _run_multinode_hw(
                cmd, script_path, worker_addrs, master_addr, timeout)
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd="/home/ubuntu")
            output = result.stdout + result.stderr

        match = re.search(r"latency:\s*([\d.]+)\s*ms", output)
        if match:
            return {
                "template": template_name,
                "hw_latency_ms": float(match.group(1)),
                "output": output[-500:],
            }
        return {"template": template_name, "hw_latency_ms": None,
                "error": "Could not parse latency", "output": output[-1000:]}
    except subprocess.TimeoutExpired:
        return {"template": template_name, "hw_latency_ms": None, "error": "timeout"}
    except Exception as e:
        return {"template": template_name, "hw_latency_ms": None, "error": str(e)}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
        if num_nodes > 1 and worker_addrs:
            for addr in worker_addrs:
                subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     f"ubuntu@{addr}", f"rm -f {script_path}"],
                    capture_output=True, timeout=10)


def _run_multinode_hw(cmd, script_path, worker_addrs, master_addr, timeout):
    """Launch torchrun on master + workers for hardware eval."""
    cmd_str = " ".join(cmd)
    env_setup = (
        f"export PATH={NEURON_VENV}/bin:$PATH && "
        f"export NEURON_RT_NUM_CORES=32 && "
        f"export FI_PROVIDER=efa && "
        f"export FI_EFA_USE_DEVICE_RDMA=1 && "
        f"export MASTER_ADDR={master_addr} && "
        f"export MASTER_PORT={MASTER_PORT}"
    )

    # Copy script to workers
    for addr in worker_addrs:
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no",
             script_path, f"ubuntu@{addr}:{script_path}"],
            capture_output=True, timeout=30)

    # Launch workers
    worker_procs = []
    for addr in worker_addrs:
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{addr}",
            f"{env_setup} && cd /home/ubuntu && {cmd_str}",
        ]
        proc = subprocess.Popen(
            ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        worker_procs.append(proc)

    # Run master locally
    master_env = os.environ.copy()
    master_env["MASTER_ADDR"] = master_addr
    master_env["MASTER_PORT"] = MASTER_PORT
    master_env["NEURON_RT_NUM_CORES"] = "32"
    master_env["FI_PROVIDER"] = "efa"
    master_env["FI_EFA_USE_DEVICE_RDMA"] = "1"
    venv_bin = os.path.join(NEURON_VENV, "bin")
    master_env["PATH"] = venv_bin + ":" + master_env.get("PATH", "")
    try:
        master_result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd="/home/ubuntu", env=master_env)
        output = master_result.stdout + master_result.stderr
    except subprocess.TimeoutExpired:
        output = ""
        for proc in worker_procs:
            proc.kill()

    for proc in worker_procs:
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

    return output
