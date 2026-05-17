"""
allgather_naive: Naive AllGather-based AllToAllV for AWS Trainium trn1.32xlarge

Baseline comparison variant. Each rank pads its send buffer, all ranks gather,
then each rank extracts its data with per-source dynamic slices + concatenation.

This produces ~33 XLA IR ops (32 dynamic-slices + 1 cat), resulting in ~1.05 ms
on trn1.32xlarge. The agent's evolved algorithm reduces this to ~3 ops / ~0.13 ms
by replacing the per-source slicing loop with a single torch.index_select.

Usage:
    from runtime.allgather_naive import all_to_allv, init_alltoallv
    init_alltoallv()
    output = all_to_allv(input_tensor, send_counts)
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

_WORLD = None
_NUM_DEVICES = None

_rank = None
_world_size = None


def init_alltoallv():
    """Initialize rank/world info. Call once after dist.init_process_group."""
    global _rank, _world_size, _WORLD, _NUM_DEVICES
    _rank = xr.global_ordinal()
    _world_size = xr.world_size()
    _WORLD = _world_size
    _NUM_DEVICES = _world_size // 2


def compute_recv_counts(send_counts):
    """Exchange send_counts to derive recv_counts via all_gather."""
    device = xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(t)
    gathered = gathered.view(_world_size, _world_size)
    return gathered[:, _rank].tolist()


def all_to_allv(x, send_counts, recv_counts=None, max_chunk=None):
    """
    Perform AllToAllV using naive AllGather + per-source slicing.

    Pads all inputs to max_total (all_gather requires uniform sizes),
    gathers, then slices per-rank output with a loop.

    ~33 XLA IR ops (32 dynamic-slices + 1 cat).

    Args:
        x: 1D input tensor (concatenated send data for all ranks)
        send_counts: list[int] of length world_size
        recv_counts: list[int] or None (computed via all_gather if None)
        max_chunk: ignored (present for API compatibility)

    Returns:
        1D tensor with received data concatenated
    """
    if _rank is None:
        raise RuntimeError("Call init_alltoallv() first")

    device = x.device

    # Get full send_counts matrix via all_gather
    sc_tensor = torch.tensor(send_counts, device=device, dtype=torch.int32)
    sc_gathered = xm.all_gather(sc_tensor.unsqueeze(0), dim=0)
    matrix = sc_gathered.view(_world_size, _world_size)

    if recv_counts is None:
        recv_counts = matrix[:, _rank].tolist()

    # Pad to max total send across all ranks
    row_sums = matrix.sum(dim=1)
    max_total = int(row_sums.max().item())
    max_total = max(max_total, 1)

    total_send = sum(send_counts)
    if total_send < max_total:
        x_padded = torch.cat([x, torch.zeros(
            max_total - total_send, device=device, dtype=x.dtype)])
    else:
        x_padded = x

    # All_gather: single collective dispatch
    gathered = xm.all_gather(x_padded.unsqueeze(0), dim=0).view(
        _world_size, max_total)

    # Extract per-source slices (32 dynamic-slice XLA ops + 1 cat)
    chunks = []
    for src in range(_world_size):
        src_sc = matrix[src].tolist()
        src_off = _cumulative_offsets(src_sc)
        count = int(src_sc[_rank])
        chunks.append(gathered[src, src_off[_rank]:src_off[_rank] + count])

    return torch.cat(chunks, dim=0)


def _cumulative_offsets(counts):
    offsets = [0]
    for c in counts[:-1]:
        offsets.append(offsets[-1] + int(c))
    return offsets
