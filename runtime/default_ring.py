"""
default_ring: Topology-unaware ring AllToAllV for AWS Trainium trn1.32xlarge

Comparison variant. Sequential collective_permute with default distance
ordering [1, 2, ..., 31]. Each step sends one shard and receives one shard.

31 collective dispatches total (~2.8 ms on trn1.32xlarge). Dispatch overhead
(~0.1 ms per collective_permute) dominates; the agent's evolved algorithm avoids
this by using a single all_gather dispatch.

Usage:
    from runtime.default_ring import all_to_allv, init_alltoallv
    init_alltoallv()
    output = all_to_allv(input_tensor, send_counts)
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_WORLD = None
_NUM_DEVICES = None
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
    _SCHEDULE = list(range(1, _world_size))

    _permute_pairs = [
        [(r, (r + d) % _world_size) for r in range(_world_size)]
        for d in _SCHEDULE
    ]


def compute_recv_counts(send_counts):
    """Exchange send_counts to derive recv_counts via all_gather."""
    device = xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(t)
    gathered = gathered.view(_world_size, _world_size)
    return gathered[:, _rank].tolist()


def all_to_allv(x, send_counts, recv_counts=None, max_chunk=None):
    """
    Perform AllToAllV using default ring schedule.

    Self-copy first, then 31 collective_permute steps with distance
    ordering [1, 2, ..., 31]. Topology-unaware baseline.

    31 collective dispatches total.

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

    send_offsets = _cumulative_offsets(send_counts)
    recv_offsets = _cumulative_offsets(recv_counts)

    total_recv = sum(recv_counts)
    output = torch.empty(total_recv, device=device, dtype=dtype)

    shards = _build_padded_shards(x, send_counts, send_offsets, max_chunk)

    # Self copy
    output[recv_offsets[_rank]:recv_offsets[_rank] + recv_counts[_rank]] = \
        shards[_rank][:recv_counts[_rank]]

    # 31 collective_permute steps
    for i, d in enumerate(_SCHEDULE):
        send_to = (_rank + d) % _world_size
        recv_from = (_rank - d) % _world_size
        recv_tensor = xm.collective_permute(shards[send_to],
                                            pairs=_permute_pairs[i])
        output[recv_offsets[recv_from]:recv_offsets[recv_from] + recv_counts[recv_from]] = \
            recv_tensor[:recv_counts[recv_from]]

    return output


def _cumulative_offsets(counts):
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + c)
    return offsets


def _build_padded_shards(x, send_counts, send_offsets, max_chunk):
    device = x.device
    dtype = x.dtype
    shards = []
    for i in range(_world_size):
        start = send_offsets[i]
        chunk = x[start:start + send_counts[i]]
        if send_counts[i] < max_chunk:
            pad = torch.zeros(max_chunk - send_counts[i], device=device, dtype=dtype)
            chunk = torch.cat([chunk, pad], dim=0)
        shards.append(chunk)
    return shards
