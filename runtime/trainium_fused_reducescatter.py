"""
trainium_fused_reducescatter: Fused ReduceScatter for AWS Trainium.

Algorithm: concatenate tensors + single reduce_scatter (1 collective dispatch).

Usage:
    from runtime.trainium_fused_reducescatter import fused_reducescatter, init_fused_reducescatter
    init_fused_reducescatter()
    shards = fused_reducescatter(gradient_tensors)
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_rank = None
_world_size = None


def init_fused_reducescatter():
    """Initialize rank/world info. Call once after dist.init_process_group."""
    global _rank, _world_size
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()


def fused_reducescatter(tensors):
    """Fused reduce-scatter via single reduce_scatter on concatenated tensor.

    Concatenates all input tensors, performs a single reduce_scatter dispatch,
    then splits the result back. Reduces N dispatches to 1.

    Args:
        tensors: list of 1D tensors. Each tensor's size must be divisible by
                 world_size.

    Returns:
        list of 1D tensors, one shard per input tensor (size = input_size // ws).
    """
    if not tensors:
        return []
    ws = xr.world_size()
    if ws == 1:
        return tensors
    sizes = [t.numel() for t in tensors]
    flat = torch.cat(tensors, dim=0)
    result = xm.reduce_scatter(xm.REDUCE_SUM, flat,
                               scale=1.0, scatter_dim=0, shard_count=ws)
    shard_sizes = [s // ws for s in sizes]
    return list(result.split(shard_sizes))


def fused_reducescatter_flat(x, ws):
    """Single reduce_scatter on a pre-concatenated flat tensor.

    Args:
        x: 1D flat tensor (all gradient tensors already concatenated).
        ws: world_size.

    Returns:
        1D shard tensor of size x.numel() // ws.
    """
    return xm.reduce_scatter(xm.REDUCE_SUM, x,
                             scale=1.0, scatter_dim=0, shard_count=ws)
