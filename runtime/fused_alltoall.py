"""
fused_alltoall: Single xm.all_to_all AllToAllV for AWS Trainium

Baseline comparison variant. Packs data into canonical layout (world * max_chunk),
calls xm.all_to_all which splits and concatenates in one collective, then unpacks.
Uses 1 collective dispatch + ~3 local XLA ops, but XLA decomposes all_to_all
internally into a ring of send/recv pairs.

~1.2 ms intra-node, ~2.3 ms cross-node on trn1.32xlarge.

Usage:
    from runtime.fused_alltoall import all_to_allv, init_alltoallv
    init_alltoallv()
    output = all_to_allv(input_tensor, send_counts)
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


def compute_recv_counts(send_counts):
    """Exchange send_counts to derive recv_counts via all_gather."""
    device = xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(t)
    gathered = gathered.view(_world_size, _world_size)
    return gathered[:, _rank].tolist()


def all_to_allv(x, send_counts, recv_counts=None, max_chunk=None):
    """
    Perform AllToAllV using a single xm.all_to_all call.

    Pack into world * max_chunk canonical layout, call all_to_all (which
    XLA decomposes into an internal ring), then unpack.

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

    send_offsets = _cumulative_offsets(send_counts)
    recv_offsets = _cumulative_offsets(recv_counts)

    packed = torch.zeros(_world_size * max_chunk, device=x.device, dtype=x.dtype)
    for i in range(_world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = \
                x[send_offsets[i]:send_offsets[i] + sc]

    received = xm.all_to_all(packed, split_dimension=0,
                             concat_dimension=0, split_count=_world_size)

    output = torch.empty(sum(recv_counts), device=x.device, dtype=x.dtype)
    for i in range(_world_size):
        rc = recv_counts[i]
        if rc > 0:
            output[recv_offsets[i]:recv_offsets[i] + rc] = \
                received[i * max_chunk:i * max_chunk + rc]

    return output


def _cumulative_offsets(counts):
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + c)
    return offsets
