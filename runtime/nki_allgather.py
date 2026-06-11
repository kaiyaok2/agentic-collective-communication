"""
nki_allgather: NKI @nki.jit AllGather AllToAllV for AWS Trainium

Baseline comparison variant. The entire AllToAllV (pack -> all_gather -> extract)
runs in a single NKI kernel on the NeuronDevice with zero XLA ops. The kernel
is generated per-rank with fully unrolled pack/extract loops.

~0.87 ms intra-node, ~1.1 ms cross-node on trn1.32xlarge.

NKI collectives have ~7x higher dispatch overhead than XLA on trn1, making
this slower than the XLA-based agent output despite having zero XLA ops.

Usage:
    from runtime.nki_allgather import all_to_allv, init_alltoallv
    init_alltoallv()
    output = all_to_allv(input_tensor, send_counts)

Note: Requires neuronxcc.nki (Neuron SDK). The kernel is compiled on first
call for the given send_counts configuration.
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_rank = None
_world_size = None
_cached_kernel = None
_cached_key = None


def init_alltoallv():
    """Initialize rank/world info. Call once after dist.init_process_group."""
    global _rank, _world_size
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()


def compute_recv_counts(send_counts):
    """Exchange send_counts to derive recv_counts via all_gather."""
    device = xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(t)
    gathered = gathered.view(_world_size, _world_size)
    return gathered[:, _rank].tolist()


def _get_kernel(send_counts_matrix):
    """Get or create the NKI kernel for the given traffic matrix."""
    global _cached_kernel, _cached_key

    key = tuple(tuple(row) for row in send_counts_matrix)
    if _cached_kernel is not None and _cached_key == key:
        return _cached_kernel

    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments"))
    from nki_alltoallv_hw import make_nki_alltoallv_kernel

    kernel_fn, recv_total, pack_size, _ = \
        make_nki_alltoallv_kernel(send_counts_matrix, _world_size, _rank)

    _cached_kernel = (kernel_fn, recv_total)
    _cached_key = key
    return _cached_kernel


def all_to_allv(x, send_counts, recv_counts=None, max_chunk=None):
    """
    Perform AllToAllV using a fully-fused NKI kernel.

    The kernel does pack + all_gather + extract entirely on the NeuronDevice
    with zero XLA IR ops. Requires the full send_counts matrix (exchanged via
    all_gather internally).

    Args:
        x: 1D input tensor (concatenated send data for all ranks)
        send_counts: list[int] of length world_size
        recv_counts: ignored (computed internally by kernel generator)
        max_chunk: ignored (computed internally by kernel generator)

    Returns:
        1D tensor with received data concatenated in source-rank order
    """
    if _rank is None:
        raise RuntimeError("Call init_alltoallv() first")

    # Exchange send_counts to build full matrix
    device = x.device
    sc_tensor = torch.tensor(send_counts, device=device, dtype=torch.int32)
    sc_gathered = xm.all_gather(sc_tensor.unsqueeze(0), dim=0)
    matrix = sc_gathered.view(_world_size, _world_size).tolist()

    kernel_fn, recv_total = _get_kernel(matrix)

    x_2d = x.unsqueeze(0)
    out_2d = kernel_fn(x_2d)
    return out_2d.view(-1)
