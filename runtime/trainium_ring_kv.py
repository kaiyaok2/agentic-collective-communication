"""
trainium_ring_kv: Ring Attention KV distribution for AWS Trainium.

Algorithm: all_gather + view (1 collective dispatch).

Usage:
    from runtime.trainium_ring_kv import ring_kv_gather, init_ring_kv
    init_ring_kv()
    all_kv = ring_kv_gather(my_kv_chunk)
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_rank = None
_world_size = None


def init_ring_kv():
    """Initialize rank/world info. Call once after dist.init_process_group."""
    global _rank, _world_size
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()


def ring_kv_gather(kv_chunk):
    """Gather all ranks' KV chunks via all_gather + view (1 collective dispatch).

    Args:
        kv_chunk: 1D tensor containing this rank's local KV data.

    Returns:
        1D tensor of size world_size * kv_chunk.numel() with all ranks' KV
        data concatenated in rank order.
    """
    ws = xr.world_size()
    if ws == 1:
        return kv_chunk
    gathered = xm.all_gather(kv_chunk.unsqueeze(0), dim=0)
    return gathered.view(-1)
