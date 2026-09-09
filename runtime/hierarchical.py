"""
hierarchical: Topology-aware hierarchical AllToAllV for AWS Trainium trn1.32xlarge

Comparison variant. Exploits the 2-core-per-device structure: intra-device
exchanges are free (shared HBM), reducing 31 inter-rank steps to 15 inter-device
steps + 1 intra-device step. Each inter-device step aggregates both cores' data
into one tensor, performs a single collective_permute, then unpacks.

16 collective dispatches total (~2.3 ms on trn1.32xlarge). Dispatch overhead
(~0.1 ms per collective_permute) dominates; the agent's evolved algorithm avoids
this by using a single all_gather dispatch.

Usage:
    from runtime.hierarchical import all_to_allv, init_alltoallv
    init_alltoallv()
    output = all_to_allv(input_tensor, send_counts)
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_WORLD = None
_NUM_DEVICES = None
# Optimized schedule for 16-device (single-node) 4x4 torus.
_SCHEDULE_16DEV = [4, 3, 13, 12, 2, 15, 1, 7, 5, 11, 9, 14, 8, 6, 10]
_SCHEDULE = None

_rank = None
_world_size = None
_permute_pairs = None


def init_alltoallv():
    """Initialize rank/world info and precompute permute pairs.

    Call once after dist.init_process_group.
    """
    global _rank, _world_size, _permute_pairs, _WORLD, _NUM_DEVICES, _SCHEDULE
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()
    _WORLD = _world_size
    _NUM_DEVICES = _world_size // 2

    # Use optimized schedule for 16 devices, default sequential otherwise
    if _NUM_DEVICES == 16:
        _SCHEDULE = _SCHEDULE_16DEV
    else:
        _SCHEDULE = list(range(1, _NUM_DEVICES))

    num_devices = _NUM_DEVICES
    cpd = _world_size // num_devices

    # Intra-device swap pairs (core 0 <-> core 1 on each device)
    intra_pairs = [(r, r ^ 1) for r in range(_world_size)]
    _permute_pairs = [intra_pairs]

    # Inter-device pairs: each rank sends to same core position on target device
    for d in _SCHEDULE:
        pairs = []
        for r in range(_world_size):
            r_dst = ((r // cpd + d) % num_devices) * cpd + (r % cpd)
            pairs.append((r, r_dst))
        _permute_pairs.append(pairs)


def compute_recv_counts(send_counts):
    """Exchange send_counts to derive recv_counts via all_gather."""
    device = xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(t)
    gathered = gathered.view(_world_size, _world_size)
    return gathered[:, _rank].tolist()


def all_to_allv(x, send_counts, recv_counts=None, max_chunk=None):
    """
    Perform AllToAllV using topology-aware hierarchical algorithm.

    Two levels:
      Level 1: Intra-device exchange (core 0 <-> core 1, via collective_permute)
      Level 2: Inter-device permute with device-level aggregation (15 steps)

    16 collective dispatches total.

    Args:
        x: 1D input tensor (concatenated send data for all ranks)
        send_counts: list[int] of length world_size
        recv_counts: list[int] or None (computed via all_gather if None)
        max_chunk: int or None (auto-computed if None)

    Returns:
        1D tensor with received data concatenated
    """
    if _rank is None:
        raise RuntimeError("Call init_alltoallv() first")

    if recv_counts is None:
        recv_counts = compute_recv_counts(send_counts)

    if max_chunk is None:
        max_chunk = max(max(send_counts), max(recv_counts))

    device = x.device
    dtype = x.dtype
    num_devices = _NUM_DEVICES
    cpd = _world_size // num_devices
    my_device = _rank // cpd

    send_offsets = _cumulative_offsets(send_counts)
    recv_offsets = _cumulative_offsets(recv_counts)

    total_recv = sum(recv_counts)
    output = torch.empty(total_recv, device=device, dtype=dtype)

    # Level 1: Self copy
    output[recv_offsets[_rank]:recv_offsets[_rank] + recv_counts[_rank]] = \
        x[send_offsets[_rank]:send_offsets[_rank] + send_counts[_rank]]

    # Level 1: Intra-device peer (collective_permute for cross-process exchange)
    peer = _rank ^ 1
    if peer < _world_size:
        peer_chunk = x[send_offsets[peer]:send_offsets[peer] + send_counts[peer]]
        if send_counts[peer] < max_chunk:
            peer_chunk = torch.cat([peer_chunk, torch.zeros(
                max_chunk - send_counts[peer], device=device, dtype=dtype)])
        recv_peer = xm.collective_permute(peer_chunk, pairs=_permute_pairs[0])
        output[recv_offsets[peer]:recv_offsets[peer] + recv_counts[peer]] = \
            recv_peer[:recv_counts[peer]]

    # Level 2: Inter-device permute with aggregation
    for i, d in enumerate(_SCHEDULE):
        dst_device = (my_device + d) % num_devices
        src_device = (my_device - d + num_devices) % num_devices

        # Aggregate data for all cores on dst_device into one tensor
        chunks = []
        for c in range(cpd):
            dst_rank = dst_device * cpd + c
            start = send_offsets[dst_rank]
            chunk = x[start:start + send_counts[dst_rank]]
            if send_counts[dst_rank] < max_chunk:
                pad = torch.zeros(max_chunk - send_counts[dst_rank],
                                  device=device, dtype=dtype)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
        send_tensor = torch.cat(chunks, dim=0)

        # One collective_permute per device distance
        recv_tensor = xm.collective_permute(send_tensor,
                                            pairs=_permute_pairs[i + 1])

        # Unpack received data for each core on src_device
        for c in range(cpd):
            from_rank = src_device * cpd + c
            recv_count = recv_counts[from_rank]
            data = recv_tensor[c * max_chunk:c * max_chunk + recv_count]
            output[recv_offsets[from_rank]:recv_offsets[from_rank] + recv_count] = data

    return output


def _cumulative_offsets(counts):
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + c)
    return offsets
