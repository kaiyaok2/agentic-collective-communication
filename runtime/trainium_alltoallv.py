"""
trainium_alltoallv: AllToAllV for AWS Trainium.

Algorithm: all_gather + per-source slice + cat (1 collective dispatch).
Pre-packed input layout: data for rank i lives at x[i*mc : (i+1)*mc].

Usage:
    from runtime.trainium_alltoallv import alltoallv, init_alltoallv
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
    """AllToAllV via all_gather + per-source slice + cat (1 collective dispatch).

    Assumes x is pre-packed: data for rank i is at x[i*mc : (i+1)*mc].
    Returns received data concatenated in source-rank order.
    """
    rank = xr.global_ordinal()
    gathered = xm.all_gather(x.unsqueeze(0), dim=0).view(ws, -1)
    chunks = []
    for src in range(ws):
        chunks.append(gathered[src, rank * mc:rank * mc + mc])
    return torch.cat(chunks, dim=0)


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
        1D tensor with received data concatenated in source-rank order
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

    rank = xr.global_ordinal()
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0).view(_world_size, -1)
    chunks = []
    for src in range(_world_size):
        count = recv_counts[src]
        if count > 0:
            chunks.append(gathered[src, rank * max_chunk:rank * max_chunk + count])
    if not chunks:
        return torch.zeros(0, device=x.device, dtype=x.dtype)
    return torch.cat(chunks, dim=0)
