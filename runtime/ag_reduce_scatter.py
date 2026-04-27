"""
ag_reduce_scatter: AllToAllV via all_gather + transpose + reduce_scatter.

Baseline comparison variant using 2 collective dispatches.

Usage:
    from runtime.ag_reduce_scatter import alltoallv, init_alltoallv
    init_alltoallv()
    output = alltoallv(x, world_size, max_chunk)
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_rank = None
_world_size = None


def init_alltoallv():
    """Initialize rank/world info. Call once after dist.init_process_group."""
    global _rank, _world_size
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()


def alltoallv(x, ws, mc):
    """AllToAllV via all_gather + transpose + reduce_scatter (2 collective dispatches).

    Assumes x is pre-packed: data for rank i is at x[i*mc : (i+1)*mc].
    Returns received data concatenated in source-rank order.
    """
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    reshaped = gathered.view(ws, ws, mc)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                             scale=1.0 / ws, scatter_dim=0, shard_count=ws)


def compute_recv_counts(send_counts):
    """Exchange send_counts via all_gather to derive recv_counts."""
    if _rank is None:
        raise RuntimeError("Call init_alltoallv() first")
    device = xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(t)
    gathered = gathered.view(_world_size, _world_size)
    return gathered[:, _rank].tolist()


def all_to_allv(x, send_counts, recv_counts=None, max_chunk=None):
    """Variable-length AllToAllV with automatic packing.

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
        max_chunk = max(max(send_counts), max(recv_counts), 1)

    pack_size = _world_size * max_chunk
    packed = torch.zeros(pack_size, device=x.device, dtype=x.dtype)
    send_off = 0
    for i in range(_world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = x[send_off:send_off + sc]
        send_off += sc

    shard = alltoallv(packed, _world_size, max_chunk)

    flat_idx = []
    for src in range(_world_size):
        count = recv_counts[src]
        base = src * max_chunk
        flat_idx.extend(range(base, base + count))
    idx_tensor = torch.tensor(flat_idx, device=x.device, dtype=torch.long)
    return torch.index_select(shard, 0, idx_tensor)
