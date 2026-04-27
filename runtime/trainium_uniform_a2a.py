"""
trainium_uniform_a2a: Uniform AllToAll for AWS Trainium.

Algorithm: all_gather + per-source slice + cat (1 collective dispatch).

Usage:
    from runtime.trainium_uniform_a2a import uniform_a2a, init_uniform_a2a
    init_uniform_a2a()
    output = uniform_a2a(x, chunk_size)
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_rank = None
_world_size = None


def init_uniform_a2a():
    """Initialize rank/world info. Call once after dist.init_process_group."""
    global _rank, _world_size
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()


def uniform_a2a(x, chunk_size):
    """Uniform AllToAll via all_gather + per-source slice + cat (1 collective dispatch).

    Args:
        x: 1D input tensor of size world_size * chunk_size, pre-packed so that
           data for rank i is at x[i*chunk_size : (i+1)*chunk_size].

    Returns:
        1D tensor of size world_size * chunk_size with received data in
        source-rank order.
    """
    rank = xr.global_ordinal()
    ws = xr.world_size()
    gathered = xm.all_gather(x.unsqueeze(0), dim=0).view(ws, -1)
    chunks = []
    for src in range(ws):
        chunks.append(gathered[src, rank * chunk_size:(rank + 1) * chunk_size])
    return torch.cat(chunks, dim=0)
