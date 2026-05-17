"""
Correctness testing for AllToAllV candidate implementations.

Supports two backend modes:
1. XLA (default): Candidates use torch tensor ops + collective mocks
   (collective_permute, all_gather, index_select, cat, etc.)
2. NKI (optional): Candidates use NKI mock ops (nl.load, nl.store, nccl.*)

The XLA path uses TrackedTensor wrappers to count XLA IR ops, and a
CollectiveSimulator to resolve multi-rank collectives.

The NKI path uses MockNLModule/MockNCCLModule with a NKICollectiveSimulator
to resolve NKI collectives across ranks.

Both paths compare outputs against a gold-standard reference_alltoallv().
"""

import numpy as np
import torch
import math
from copy import deepcopy


# ================================================================
# XLA Collective Simulator (for XLA-based template evolution)
# ================================================================

class CollectiveSimulator:
    """Simulates XLA collective operations across multiple ranks.

    Two-phase execution:
      Phase 1 ("collect"): Run candidates for all ranks. Collective calls
        record what each rank sends and return placeholder zeros.
      Phase 2 ("resolve"): Resolve cross-rank data, then re-run candidates
        with correct received data.
    """

    def __init__(self, world_size):
        self.world_size = world_size
        self.phase = "collect"
        # collective_permute: step -> {rank: (sent_tensor, pairs)}
        self.cp_data = {}
        self.cp_resolved = {}
        # all_gather: step -> {rank: (tensor, dim, groups)}
        self.ag_data = {}
        self.ag_resolved = {}
        # reduce_scatter: step -> {rank: (tensor, reduce_type, scatter_dim, shard_count)}
        self.rs_data = {}
        self.rs_resolved = {}
        # all_reduce: step -> {rank: (tensor, reduce_type)}
        self.ar_data = {}
        self.ar_resolved = {}

    def set_phase(self, phase):
        self.phase = phase

    def clear(self):
        self.cp_data = {}
        self.ag_data = {}
        self.rs_data = {}
        self.ar_data = {}
        self.cp_resolved = {}
        self.ag_resolved = {}
        self.rs_resolved = {}
        self.ar_resolved = {}

    def collective_permute(self, tensor, pairs, rank, step):
        """Mock xm.collective_permute."""
        if self.phase == "collect":
            self.cp_data.setdefault(step, {})[rank] = (
                tensor.clone().detach(), list(pairs))
            resolved = self.cp_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            return torch.zeros_like(tensor)
        else:
            resolved = self.cp_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            return torch.zeros_like(tensor)

    def _group_size_for_rank(self, groups, rank):
        """Find the size of the group containing this rank."""
        if groups and isinstance(groups[0], (list, tuple)):
            for g in groups:
                if rank in g:
                    return len(g)
        if groups:
            return len(groups)
        return self.world_size

    def all_gather(self, tensor, dim, rank, step, groups=None):
        """Mock xm.all_gather."""
        gsz = self._group_size_for_rank(groups, rank)
        if self.phase == "collect":
            self.ag_data.setdefault(step, {})[rank] = (
                tensor.clone().detach(), dim, groups)
            resolved = self.ag_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            rep = [1] * tensor.dim()
            rep[dim] = gsz
            return torch.zeros_like(tensor).repeat(*rep)
        else:
            resolved = self.ag_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            rep = [1] * tensor.dim()
            rep[dim] = gsz
            return torch.zeros_like(tensor).repeat(*rep)

    def all_to_all(self, tensor, split_dim, concat_dim, split_count, rank, step):
        """Mock xm.all_to_all."""
        if self.phase == "collect":
            self.ag_data.setdefault(step, {})[rank] = (
                tensor.clone().detach(), split_dim,
                {"type": "all_to_all", "split_dim": split_dim,
                 "concat_dim": concat_dim, "split_count": split_count})
            resolved = self.ag_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            return torch.zeros_like(tensor)
        else:
            resolved = self.ag_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            return torch.zeros_like(tensor)

    def reduce_scatter(self, tensor, reduce_type, scatter_dim, shard_count, rank, step, scale=1.0):
        """Mock xm.reduce_scatter."""
        if self.phase == "collect":
            self.rs_data.setdefault(step, {})[rank] = (
                tensor.clone().detach(), reduce_type, scatter_dim, shard_count, scale)
            resolved = self.rs_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            shard_size = tensor.shape[scatter_dim] // shard_count
            slices = [slice(None)] * tensor.dim()
            slices[scatter_dim] = slice(rank * shard_size, (rank + 1) * shard_size)
            return torch.zeros_like(tensor[tuple(slices)])
        else:
            resolved = self.rs_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            shard_size = tensor.shape[scatter_dim] // shard_count
            slices = [slice(None)] * tensor.dim()
            slices[scatter_dim] = slice(rank * shard_size, (rank + 1) * shard_size)
            return torch.zeros_like(tensor[tuple(slices)])

    def all_reduce(self, reduce_type, tensor, rank, step, groups=None):
        """Mock xm.all_reduce with optional groups parameter."""
        if self.phase == "collect":
            self.ar_data.setdefault(step, {})[rank] = (
                tensor.clone().detach(), reduce_type, groups)
            resolved = self.ar_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            return torch.zeros_like(tensor)
        else:
            resolved = self.ar_resolved.get(step, {})
            if rank in resolved:
                return resolved[rank].clone()
            return torch.zeros_like(tensor)

    def resolve(self):
        """Resolve all collected collectives using cross-rank data."""
        # Resolve collective_permute
        for step, rank_data in self.cp_data.items():
            any_rank = next(iter(rank_data))
            _, pairs = rank_data[any_rank]
            dst_to_src = {dst: src for src, dst in pairs}
            resolved = {}
            for dst_rank in range(self.world_size):
                src_rank = dst_to_src.get(dst_rank)
                if src_rank is not None and src_rank in rank_data:
                    data, _ = rank_data[src_rank]
                    resolved[dst_rank] = data.clone()
                elif dst_rank in rank_data:
                    data, _ = rank_data[dst_rank]
                    resolved[dst_rank] = torch.zeros_like(data)
            self.cp_resolved[step] = resolved

        # Resolve all_gather and all_to_all
        for step, rank_data in self.ag_data.items():
            any_rank = next(iter(rank_data))
            _, dim_or_split, groups_or_info = rank_data[any_rank]

            if isinstance(groups_or_info, dict) and groups_or_info.get("type") == "all_to_all":
                # all_to_all resolution
                split_count = groups_or_info["split_count"]
                split_dim = groups_or_info["split_dim"]
                concat_dim = groups_or_info["concat_dim"]
                # Each rank's tensor is split into split_count chunks
                resolved = {}
                for r in range(self.world_size):
                    if r not in rank_data:
                        continue
                    chunks_from_all = []
                    for src in range(self.world_size):
                        if src in rank_data:
                            src_tensor = rank_data[src][0]
                            src_chunks = torch.chunk(src_tensor, split_count, dim=split_dim)
                            chunks_from_all.append(src_chunks[r])
                    resolved[r] = torch.cat(chunks_from_all, dim=concat_dim)
                self.ag_resolved[step] = resolved
            else:
                # all_gather resolution
                dim = dim_or_split
                groups = groups_or_info
                resolved = {}
                if groups and isinstance(groups[0], (list, tuple)):
                    for group in groups:
                        gathered_list = []
                        for r in group:
                            if r in rank_data:
                                gathered_list.append(rank_data[r][0])
                            else:
                                ref = rank_data[any_rank][0]
                                gathered_list.append(torch.zeros_like(ref))
                        gathered = torch.cat(gathered_list, dim=dim)
                        for r in group:
                            resolved[r] = gathered.clone()
                elif groups:
                    gathered_list = []
                    for r in groups:
                        if r in rank_data:
                            gathered_list.append(rank_data[r][0])
                        else:
                            ref = rank_data[any_rank][0]
                            gathered_list.append(torch.zeros_like(ref))
                    gathered = torch.cat(gathered_list, dim=dim)
                    for r in range(self.world_size):
                        resolved[r] = gathered.clone()
                else:
                    gathered_list = []
                    for r in range(self.world_size):
                        if r in rank_data:
                            gathered_list.append(rank_data[r][0])
                        else:
                            ref = rank_data[any_rank][0]
                            gathered_list.append(torch.zeros_like(ref))
                    gathered = torch.cat(gathered_list, dim=dim)
                    for r in range(self.world_size):
                        resolved[r] = gathered.clone()
                self.ag_resolved[step] = resolved

        # Resolve reduce_scatter
        for step, rank_data in self.rs_data.items():
            any_rank = next(iter(rank_data))
            _, reduce_type, scatter_dim, shard_count, scale = rank_data[any_rank]
            # Sum all ranks' tensors element-wise
            all_tensors = []
            for r in range(self.world_size):
                if r in rank_data:
                    all_tensors.append(rank_data[r][0])
            summed = torch.stack(all_tensors).sum(dim=0) * scale
            # Scatter: each rank gets shard_count-th portion along scatter_dim
            shard_size = summed.shape[scatter_dim] // shard_count
            resolved = {}
            for r in range(self.world_size):
                slices = [slice(None)] * summed.dim()
                slices[scatter_dim] = slice(r * shard_size, (r + 1) * shard_size)
                resolved[r] = summed[tuple(slices)].clone()
            self.rs_resolved[step] = resolved

        # Resolve all_reduce
        for step, rank_data in self.ar_data.items():
            any_rank = next(iter(rank_data))
            _, reduce_type, groups = rank_data[any_rank]
            resolved = {}

            if groups and isinstance(groups[0], (list, tuple)):
                for group in groups:
                    group_tensors = []
                    for r in group:
                        if r in rank_data:
                            group_tensors.append(rank_data[r][0])
                    if not group_tensors:
                        continue
                    if reduce_type == "sum":
                        result = torch.stack(group_tensors).sum(dim=0)
                    elif reduce_type == "max":
                        result = torch.stack(group_tensors).max(dim=0).values
                    elif reduce_type == "min":
                        result = torch.stack(group_tensors).min(dim=0).values
                    else:
                        result = torch.stack(group_tensors).sum(dim=0)
                    for r in group:
                        resolved[r] = result.clone()
            else:
                all_tensors = []
                for r in range(self.world_size):
                    if r in rank_data:
                        all_tensors.append(rank_data[r][0])
                if reduce_type == "sum":
                    result = torch.stack(all_tensors).sum(dim=0)
                elif reduce_type == "max":
                    result = torch.stack(all_tensors).max(dim=0).values
                elif reduce_type == "min":
                    result = torch.stack(all_tensors).min(dim=0).values
                else:
                    result = torch.stack(all_tensors).sum(dim=0)
                for r in range(self.world_size):
                    resolved[r] = result.clone()

            self.ar_resolved[step] = resolved


# ================================================================
# XLA Op Counter (TrackedTensor)
# ================================================================

# Pure metadata ops in PyTorch: produce a *view* over the source storage,
# never copy. Their per-op cost in the existing measurement table is the
# isolated-mark_step kernel-launch overhead, which is NOT what they cost
# inside a fused HLO graph alongside other ops; treat them as floor-priced.
# (flatten is excluded — it can copy when applied to a non-contiguous
# source, same as reshape; it's handled by the maybe-copy ops list.)
_VIEW_ONLY_OPS = frozenset({
    "view", "unsqueeze", "squeeze",
    "narrow", "transpose", "permute", "expand",
    "slice",
})

# Ops whose cost is contiguity-dependent: cheap (metadata) when input is
# already contiguous and the requested layout is reachable as a view;
# otherwise PyTorch silently inserts a copy of the source storage. The
# simulator detects which case applies via the actual tensor state and
# charges a memory-copy term proportional to bytes touched. This is what
# the previous model missed: contiguous() / reshape() can be free or O(N)
# depending on the chain that produced the input.
#
# Two regimes for the implicit copy, distinguished at trace time by
# inspecting actual tensor strides:
#   *_strided  — source is a sub-region of its underlying storage (e.g.,
#                narrow on a non-leading dim, slice with start>0). Each
#                element of the output reads from a different cache line
#                in a region larger than numel*elem_size. Effective
#                bandwidth = the strided memcpy_bw.
#   *_dense    — source covers full storage but with permuted strides
#                (e.g., result of permute/transpose on a contiguous
#                source). Output reads numel elements from numel-sized
#                storage at predictable strides. Effective bandwidth =
#                the sequential memcpy_bw.
# Distinguishing these two regimes was the difference between picking
# `narrow(non-leading-dim) -> reshape` as ~free (incorrect) and as a real
# strided copy (correct), AND picking a `permute -> reshape` packing
# pattern as expensive-strided (incorrect — Neuron compiler vectorizes
# it well) vs sequential-bandwidth (correct).
# Scalar/elementwise tensor ops. On HLO/Neuron these fuse with neighbouring
# elementwise ops into a single kernel; the agent's isolated-microbench cost
# for one such op is dominated by mark_step kernel-launch overhead, which a
# fused chain pays exactly once. Charge them at the per-op floor (same as
# pure metadata view ops), and treat them as FREE for collective-fusion
# purposes (they don't add a real data dependency that would force two
# back-to-back collectives to serialize on Trainium's NIC).
_FUSED_ELEMENTWISE_OPS = frozenset({
    "mul", "add", "sub", "div", "mod", "neg",
})

_MAYBE_COPY_OPS_STRIDED = frozenset({
    "reshape", "contiguous", "flatten",
})
_MAYBE_COPY_OPS_DENSE = frozenset({
    "reshape_dense", "contiguous_dense", "flatten_dense",
})
_MAYBE_COPY_OPS = _MAYBE_COPY_OPS_STRIDED | _MAYBE_COPY_OPS_DENSE

# Ops whose cost is *always* proportional to the data volume they move.
# The agent's measured isolated cost (~29 us via measure_xla_op_overhead)
# is the floor; for large tensors the actual cost is dominated by the
# implicit memcpy/gather. Charged as max(agent_floor, scaled_bytes / bw).
# index_select: random-access HBM gather, output bytes touched.
# tensor: torch.tensor(python_list, ...) does a host-side O(N) build
#   plus host->device copy; without this charge an algorithm can build
#   a giant Python index list "for free".
_VOLUME_SCALED_OPS = frozenset({
    "index_select", "tensor",
})


def _is_dense_view(t):
    """True if the tensor is a stride-permuted view of contiguous storage
    (e.g., result of permute/transpose on a contiguous source) rather
    than a sub-region of a larger storage (e.g., narrow on a non-leading
    dim). When PyTorch later reshapes/contiguous-es this tensor, the
    induced copy reads/writes numel elements from a numel-sized storage
    at predictable strides, which the hardware vectorizes near
    sequential bandwidth — versus a sub-region copy that gathers
    sub-cache-line elements from a larger region at strided bandwidth.

    Detection: a dense view has the property that every byte of its
    underlying storage is referenced by exactly one tensor element. The
    minimum storage size needed to hold a tensor with given shape and
    strides is sum((dim_size - 1) * stride for each dim) + 1 elements.
    A dense permutation has numel == that minimum; a sub-region has
    numel < that minimum.
    """
    if t.numel() == 0:
        return True
    min_storage_elems = 1
    for dim_size, stride in zip(t.shape, t.stride()):
        if dim_size == 0:
            return True
        if stride < 0:
            return False
        min_storage_elems += (dim_size - 1) * stride
    return t.numel() == min_storage_elems

_FREE_XLA_OPS = _VIEW_ONLY_OPS | _MAYBE_COPY_OPS

_COLLECTIVE_OPS = frozenset({
    "collective_permute", "all_gather", "all_to_all",
    "reduce_scatter", "all_reduce",
})


class TorchOpCounter:
    """Counts XLA IR ops generated by a candidate.

    Each `record(op, copy_bytes=0)` event captures the op name and, for
    ops that may force a memory copy (reshape/contiguous on a
    non-contiguous source), the number of bytes the copy must move. The
    simulator uses copy_bytes to compute a memory-bandwidth term for the
    op so that algorithms that abuse "metadata" ops on non-contiguous
    sources are charged correctly.
    """

    def __init__(self):
        self.ops = []        # list[str], legacy access
        self.events = []     # list[(op_name, copy_bytes)]

    def record(self, op_name, copy_bytes=0):
        self.ops.append(op_name)
        self.events.append((op_name, int(copy_bytes)))

    @property
    def count(self):
        return len(self.ops)

    @property
    def real_local_ops(self):
        """Count only ops that generate real HLO nodes (excludes free metadata ops and collectives)."""
        return sum(1 for op in self.ops
                   if op not in _FREE_XLA_OPS and op not in _COLLECTIVE_OPS)

    def reset(self):
        self.ops = []
        self.events = []


class TrackedTensor:
    """Torch tensor wrapper that records XLA-relevant operations.

    Provides the tensor API needed by AllToAllV candidates while
    counting operations that would become separate XLA IR nodes.
    """

    def __init__(self, data, counter=None):
        if isinstance(data, TrackedTensor):
            self._t = data._t
        elif isinstance(data, torch.Tensor):
            self._t = data
        else:
            self._t = torch.tensor(data, dtype=torch.float32)
        self._counter = counter or TorchOpCounter()

    @property
    def device(self):
        return self._t.device

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def shape(self):
        return self._t.shape

    def __len__(self):
        return len(self._t)

    def __getitem__(self, key):
        if isinstance(key, slice) or isinstance(key, tuple):
            self._counter.record("slice")
        return TrackedTensor(self._t[key], self._counter)

    def __setitem__(self, key, value):
        if isinstance(value, TrackedTensor):
            self._t[key] = value._t
        else:
            self._t[key] = value

    def clone(self):
        return TrackedTensor(self._t.clone(), self._counter)

    def detach(self):
        return TrackedTensor(self._t.detach(), self._counter)

    def unsqueeze(self, dim):
        self._counter.record("unsqueeze")
        return TrackedTensor(self._t.unsqueeze(dim), self._counter)

    def view(self, *shape):
        self._counter.record("view")
        return TrackedTensor(self._t.view(*shape), self._counter)

    def dim(self):
        return self._t.dim()

    def repeat(self, *args):
        return TrackedTensor(self._t.repeat(*args), self._counter)

    def cpu(self):
        return self._t.cpu()

    def numpy(self):
        return self._t.numpy()

    def sum(self, *args, **kwargs):
        return self._t.sum(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self._t.mean(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self._t.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self._t.min(*args, **kwargs)

    def any(self, *args, **kwargs):
        return self._t.any(*args, **kwargs)

    def all(self, *args, **kwargs):
        return self._t.all(*args, **kwargs)

    def element_size(self):
        return self._t.element_size()

    def size(self, *args):
        return self._t.size(*args)

    def numel(self):
        return self._t.numel()

    def item(self):
        return self._t.item()

    def contiguous(self):
        # contiguous() on an already-contiguous tensor is a metadata no-op;
        # on a non-contiguous tensor it forces an O(numel) memory copy.
        # Distinguish dense (full-storage permute) from sub-region (narrow)
        # so the simulator charges the right bandwidth regime.
        if self._t.is_contiguous():
            self._counter.record("contiguous", 0)
        else:
            copy_bytes = self._t.numel() * self._t.element_size()
            if _is_dense_view(self._t):
                self._counter.record("contiguous_dense", copy_bytes)
            else:
                self._counter.record("contiguous", copy_bytes)
        return TrackedTensor(self._t.contiguous(), self._counter)

    def permute(self, *dims):
        self._counter.record("permute")
        return TrackedTensor(self._t.permute(*dims), self._counter)

    def reshape(self, *shape):
        # reshape() returns a view if the requested shape is reachable
        # without changing the storage layout (input contiguous, or the
        # new shape is stride-compatible). Otherwise PyTorch silently
        # invokes a copy of the source storage — the same physics as
        # contiguous() on a non-contiguous source. Detect by trying view()
        # first (PyTorch's reshape uses this internally), and distinguish
        # dense (permute-style) from sub-region (narrow-style) sources so
        # the bandwidth regime is right.
        out = self._t.reshape(*shape)
        try:
            self._t.view(*shape)  # succeeds iff layout is view-compatible
            self._counter.record("reshape", 0)
        except (RuntimeError, TypeError):
            copy_bytes = self._t.numel() * self._t.element_size()
            if _is_dense_view(self._t):
                self._counter.record("reshape_dense", copy_bytes)
            else:
                self._counter.record("reshape", copy_bytes)
        return TrackedTensor(out, self._counter)

    def flatten(self, *args, **kwargs):
        # flatten() is reshape((-1,)) over the flattened dim range; it
        # forces a copy if the source is non-contiguous (and any folded
        # dim range crosses a non-stride-compatible boundary).
        out = self._t.flatten(*args, **kwargs)
        try:
            self._t.view(out.shape)
            self._counter.record("flatten", 0)
        except (RuntimeError, TypeError):
            copy_bytes = self._t.numel() * self._t.element_size()
            if _is_dense_view(self._t):
                self._counter.record("flatten_dense", copy_bytes)
            else:
                self._counter.record("flatten", copy_bytes)
        return TrackedTensor(out, self._counter)

    def narrow(self, dim, start, length):
        self._counter.record("narrow")
        return TrackedTensor(self._t.narrow(dim, start, length), self._counter)

    def squeeze(self, *args):
        self._counter.record("squeeze")
        return TrackedTensor(self._t.squeeze(*args), self._counter)

    def chunk(self, chunks, dim=0):
        self._counter.record("chunk")
        return [TrackedTensor(c, self._counter) for c in self._t.chunk(chunks, dim=dim)]

    def split(self, split_size, dim=0):
        self._counter.record("split")
        return [TrackedTensor(s, self._counter) for s in self._t.split(split_size, dim=dim)]

    def expand(self, *sizes):
        return TrackedTensor(self._t.expand(*sizes), self._counter)

    def transpose(self, dim0, dim1):
        self._counter.record("transpose")
        return TrackedTensor(self._t.transpose(dim0, dim1), self._counter)

    def exp(self):
        self._counter.record("exp")
        return TrackedTensor(self._t.exp(), self._counter)

    def log(self):
        self._counter.record("log")
        return TrackedTensor(self._t.log(), self._counter)

    def gather(self, dim, index):
        self._counter.record("gather")
        idx = _unwrap(index) if isinstance(index, TrackedTensor) else index
        return TrackedTensor(self._t.gather(dim, idx), self._counter)

    def clamp(self, min=None, max=None):
        self._counter.record("clamp")
        return TrackedTensor(self._t.clamp(min=min, max=max), self._counter)

    def abs(self):
        self._counter.record("abs")
        return TrackedTensor(self._t.abs(), self._counter)

    @property
    def tensor(self):
        return self._t

    def __add__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("add")
        return TrackedTensor(self._t + other_t, self._counter)

    def __radd__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("add")
        return TrackedTensor(other_t + self._t, self._counter)

    def __sub__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("sub")
        return TrackedTensor(self._t - other_t, self._counter)

    def __rsub__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("sub")
        return TrackedTensor(other_t - self._t, self._counter)

    def __mul__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("mul")
        return TrackedTensor(self._t * other_t, self._counter)

    def __rmul__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("mul")
        return TrackedTensor(other_t * self._t, self._counter)

    def __truediv__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("div")
        return TrackedTensor(self._t / other_t, self._counter)

    def __floordiv__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("div")
        return TrackedTensor(self._t // other_t, self._counter)

    def __mod__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        self._counter.record("mod")
        return TrackedTensor(self._t % other_t, self._counter)

    def __lt__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return self._t < other_t

    def __le__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return self._t <= other_t

    def __gt__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return self._t > other_t

    def __ge__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return self._t >= other_t

    def __eq__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return self._t == other_t

    def __ne__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return self._t != other_t

    def __neg__(self):
        self._counter.record("neg")
        return TrackedTensor(-self._t, self._counter)

    def __int__(self):
        return int(self._t)

    def __float__(self):
        return float(self._t)

    def __index__(self):
        return int(self._t)

    def __bool__(self):
        return bool(self._t)

    def long(self):
        return TrackedTensor(self._t.long(), self._counter)

    def float(self):
        return TrackedTensor(self._t.float(), self._counter)

    def int(self):
        return TrackedTensor(self._t.int(), self._counter)

    def to(self, *args, **kwargs):
        return TrackedTensor(self._t.to(*args, **kwargs), self._counter)

    def scatter_(self, dim, index, src):
        self._counter.record("scatter_")
        idx = _unwrap(index)
        s = _unwrap(src) if isinstance(src, TrackedTensor) else src
        self._t.scatter_(dim, idx, s)
        return self

    def scatter(self, dim, index, src):
        self._counter.record("scatter")
        idx = _unwrap(index)
        s = _unwrap(src) if isinstance(src, TrackedTensor) else src
        return TrackedTensor(self._t.scatter(dim, idx, s), self._counter)

    def gather(self, dim, index):
        self._counter.record("gather")
        idx = _unwrap(index)
        return TrackedTensor(self._t.gather(dim, idx), self._counter)

    def expand(self, *sizes):
        return TrackedTensor(self._t.expand(*sizes), self._counter)

    def expand_as(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(self._t.expand_as(other_t), self._counter)

    def repeat_interleave(self, repeats, dim=None):
        self._counter.record("repeat_interleave")
        rep = _unwrap(repeats) if isinstance(repeats, TrackedTensor) else repeats
        return TrackedTensor(self._t.repeat_interleave(rep, dim=dim), self._counter)

    def cumsum(self, dim):
        self._counter.record("cumsum")
        return TrackedTensor(self._t.cumsum(dim), self._counter)

    def tolist(self):
        return self._t.tolist()

    def __repr__(self):
        return f"TrackedTensor({self._t})"


def _unwrap(x):
    """Unwrap TrackedTensor to plain torch.Tensor."""
    if isinstance(x, TrackedTensor):
        return x._t
    return x


# ================================================================
# XLA Mock Module (simulates xm.* and torch.* for candidates)
# ================================================================

class MockXM:
    """Mock torch_xla.core.xla_model for sandbox execution."""

    def __init__(self, simulator, rank, counter=None, unsupported_primitives=None):
        self.sim = simulator
        self.rank = rank
        self.counter = counter or TorchOpCounter()
        self._cp_step = 0
        self._ag_step = 0
        self._rs_step = 0
        self._ar_step = 0
        self._unsupported = set(unsupported_primitives or [])

    def collective_permute(self, tensor, pairs):
        self._check_supported("collective_permute")
        step = self._cp_step
        self._cp_step += 1
        self.counter.record("collective_permute")
        t = _unwrap(tensor)
        result = self.sim.collective_permute(t, pairs, self.rank, step)
        return TrackedTensor(result, self.counter)

    def all_gather(self, tensor, dim=0, groups=None):
        self._check_supported("all_gather")
        step = self._ag_step
        self._ag_step += 1
        self.counter.record("all_gather")
        t = _unwrap(tensor)
        result = self.sim.all_gather(t, dim, self.rank, step, groups=groups)
        return TrackedTensor(result, self.counter)

    def _check_supported(self, primitive):
        if primitive in self._unsupported:
            raise RuntimeError(
                f"Primitive '{primitive}' is not supported by the hardware compiler. "
                f"This would fail with a compilation error on real hardware. "
                f"Use a different approach (e.g., all_gather + local extraction).")

    def all_to_all(self, tensor, split_dimension=0, concat_dimension=0,
                   split_count=None):
        self._check_supported("all_to_all")
        step = self._ag_step
        self._ag_step += 1
        self.counter.record("all_to_all")
        t = _unwrap(tensor)
        result = self.sim.all_to_all(
            t, split_dimension, concat_dimension,
            split_count or self.sim.world_size, self.rank, step)
        return TrackedTensor(result, self.counter)

    def reduce_scatter(self, reduce_type, input, scale=1.0, scatter_dim=0,
                       shard_count=None, groups=None, output=None,
                       pin_layout=True, channel_id=None,
                       use_global_device_ids=None):
        self._check_supported("reduce_scatter")
        step = self._rs_step
        self._rs_step += 1
        self.counter.record("reduce_scatter")
        t = _unwrap(input)
        result = self.sim.reduce_scatter(
            t, reduce_type, scatter_dim,
            shard_count or self.sim.world_size, self.rank, step, scale)
        return TrackedTensor(result, self.counter)

    def all_reduce(self, reduce_type, tensor, groups=None):
        self._check_supported("all_reduce")
        step = getattr(self, '_ar_step', 0)
        self._ar_step = step + 1
        self.counter.record("all_reduce")
        t = _unwrap(tensor)
        result = self.sim.all_reduce(reduce_type, t, self.rank, step,
                                     groups=groups)
        return TrackedTensor(result, self.counter)

    REDUCE_SUM = "sum"
    REDUCE_MAX = "max"
    REDUCE_MIN = "min"


class MockTorch:
    """Mock torch module that wraps results in TrackedTensor."""

    def __init__(self, counter=None):
        self.counter = counter or TorchOpCounter()
        self.long = torch.long
        self.float32 = torch.float32

    def zeros(self, *args, device=None, dtype=None, **kwargs):
        return TrackedTensor(
            torch.zeros(*args, dtype=dtype or torch.float32), self.counter)

    def ones(self, *args, device=None, dtype=None, **kwargs):
        return TrackedTensor(
            torch.ones(*args, dtype=dtype or torch.float32), self.counter)

    def empty(self, *args, device=None, dtype=None, **kwargs):
        return TrackedTensor(
            torch.empty(*args, dtype=dtype or torch.float32), self.counter)

    def tensor(self, data, device=None, dtype=None, **kwargs):
        if dtype is None:
            dtype = torch.float32
        out = torch.tensor(data, dtype=dtype)
        # torch.tensor(python_list, device=xla) does an O(N) host-side
        # construction of a CPU tensor and then a host->device copy. When
        # an algorithm builds the list with a Python loop whose length
        # scales with world_size or input size, the total cost (Python-
        # side iteration plus H2D transfer) grows with N and dominates
        # at training scale even though no XLA op fires. Record copy
        # bytes so the simulator charges this against the implicit-copy
        # memory bandwidth term.
        if isinstance(data, (list, tuple)):
            copy_bytes = out.numel() * out.element_size()
        else:
            copy_bytes = 0
        self.counter.record("tensor", copy_bytes)
        return TrackedTensor(out, self.counter)

    def cat(self, tensors, dim=0):
        unwrapped = [_unwrap(t) for t in tensors]
        out = torch.cat(unwrapped, dim=dim)
        copy_bytes = out.numel() * out.element_size()
        self.counter.record("cat", copy_bytes)
        return TrackedTensor(out, self.counter)

    def index_select(self, input, dim, index):
        inp = _unwrap(input)
        idx = _unwrap(index)
        # index_select on Trainium has random-access HBM behavior: the
        # measured isolated-call cost (~29 us) is for tiny indexes and
        # does NOT scale to the index sizes that show up at training
        # scale. Record the gather volume (output bytes the kernel must
        # produce) so the simulator can charge this op proportional to
        # work done, via the same memcpy_bytes_per_us term we use for
        # implicit copies.
        out = torch.index_select(inp, dim, idx)
        copy_bytes = out.numel() * out.element_size()
        self.counter.record("index_select", copy_bytes)
        return TrackedTensor(out, self.counter)

    def full(self, size, fill_value, device=None, dtype=None, **kwargs):
        return TrackedTensor(
            torch.full(size, fill_value, dtype=dtype or torch.float32),
            self.counter)

    def arange(self, *args, device=None, dtype=None, **kwargs):
        unwrapped = [_unwrap(a) for a in args]
        return TrackedTensor(
            torch.arange(*unwrapped, dtype=dtype or torch.float32), self.counter)

    def zeros_like(self, input, dtype=None, **kwargs):
        inp = _unwrap(input) if isinstance(input, TrackedTensor) else input
        return TrackedTensor(
            torch.zeros_like(inp, dtype=dtype) if dtype is not None
            else torch.zeros_like(inp),
            self.counter)

    def ones_like(self, input, dtype=None, **kwargs):
        inp = _unwrap(input) if isinstance(input, TrackedTensor) else input
        return TrackedTensor(
            torch.ones_like(inp, dtype=dtype) if dtype is not None
            else torch.ones_like(inp),
            self.counter)

    def exp(self, input):
        self.counter.record("exp")
        inp = _unwrap(input) if isinstance(input, TrackedTensor) else input
        return TrackedTensor(torch.exp(inp), self.counter)

    def log(self, input):
        self.counter.record("log")
        inp = _unwrap(input) if isinstance(input, TrackedTensor) else input
        return TrackedTensor(torch.log(inp), self.counter)

    def stack(self, tensors, dim=0):
        unwrapped = [_unwrap(t) for t in tensors]
        out = torch.stack(unwrapped, dim=dim)
        copy_bytes = out.numel() * out.element_size()
        self.counter.record("stack", copy_bytes)
        return TrackedTensor(out, self.counter)

    def gather(self, input, dim, index):
        self.counter.record("gather")
        inp = _unwrap(input)
        idx = _unwrap(index)
        return TrackedTensor(torch.gather(inp, dim, idx), self.counter)

    def cumsum(self, input, dim):
        self.counter.record("cumsum")
        inp = _unwrap(input)
        return TrackedTensor(torch.cumsum(inp, dim), self.counter)

    def where(self, condition, x, y):
        self.counter.record("where")
        cond = _unwrap(condition) if isinstance(condition, TrackedTensor) else condition
        xv = _unwrap(x) if isinstance(x, TrackedTensor) else x
        yv = _unwrap(y) if isinstance(y, TrackedTensor) else y
        return TrackedTensor(torch.where(cond, xv, yv), self.counter)

    def clamp(self, input, min=None, max=None):
        self.counter.record("clamp")
        inp = _unwrap(input)
        return TrackedTensor(torch.clamp(inp, min=min, max=max), self.counter)

    def narrow(self, input, dim, start, length):
        self.counter.record("narrow")
        inp = _unwrap(input)
        return TrackedTensor(torch.narrow(inp, dim, start, length), self.counter)

    def chunk(self, input, chunks, dim=0):
        self.counter.record("chunk")
        inp = _unwrap(input)
        result = torch.chunk(inp, chunks, dim=dim)
        return [TrackedTensor(t, self.counter) for t in result]

    def split(self, tensor, split_size_or_sections, dim=0):
        self.counter.record("split")
        inp = _unwrap(tensor)
        result = torch.split(inp, split_size_or_sections, dim=dim)
        return [TrackedTensor(t, self.counter) for t in result]

    def max(self, input, dim=None, keepdim=False):
        inp = _unwrap(input)
        if dim is None:
            return TrackedTensor(torch.max(inp), self.counter)
        result = torch.max(inp, dim=dim, keepdim=keepdim)
        return TrackedTensor(result.values, self.counter), TrackedTensor(result.indices, self.counter)

    def min(self, input, dim=None, keepdim=False):
        inp = _unwrap(input)
        if dim is None:
            return TrackedTensor(torch.min(inp), self.counter)
        result = torch.min(inp, dim=dim, keepdim=keepdim)
        return TrackedTensor(result.values, self.counter), TrackedTensor(result.indices, self.counter)

    def sum(self, input, dim=None, keepdim=False):
        inp = _unwrap(input)
        if dim is None:
            return TrackedTensor(torch.sum(inp), self.counter)
        return TrackedTensor(torch.sum(inp, dim=dim, keepdim=keepdim), self.counter)

    def any(self, input, dim=None, keepdim=False):
        inp = _unwrap(input)
        if dim is None:
            return torch.any(inp)
        return torch.any(inp, dim=dim, keepdim=keepdim)

    def repeat_interleave(self, input, repeats, dim=None):
        self.counter.record("repeat_interleave")
        inp = _unwrap(input)
        rep = _unwrap(repeats) if isinstance(repeats, TrackedTensor) else repeats
        return TrackedTensor(torch.repeat_interleave(inp, rep, dim=dim), self.counter)

    def flatten(self, input, start_dim=0, end_dim=-1):
        self.counter.record("flatten")
        inp = _unwrap(input)
        return TrackedTensor(torch.flatten(inp, start_dim, end_dim), self.counter)

    def unsqueeze(self, input, dim):
        self.counter.record("unsqueeze")
        inp = _unwrap(input)
        return TrackedTensor(torch.unsqueeze(inp, dim), self.counter)

    def squeeze(self, input, dim=None):
        self.counter.record("squeeze")
        inp = _unwrap(input)
        if dim is None:
            return TrackedTensor(torch.squeeze(inp), self.counter)
        return TrackedTensor(torch.squeeze(inp, dim), self.counter)

    def reshape(self, input, shape):
        inp = _unwrap(input)
        out = torch.reshape(inp, shape)
        try:
            inp.view(shape)
            self.counter.record("reshape", 0)
        except (RuntimeError, TypeError):
            copy_bytes = inp.numel() * inp.element_size()
            if _is_dense_view(inp):
                self.counter.record("reshape_dense", copy_bytes)
            else:
                self.counter.record("reshape", copy_bytes)
        return TrackedTensor(out, self.counter)

    def nonzero(self, input, as_tuple=False):
        inp = _unwrap(input)
        result = torch.nonzero(inp, as_tuple=as_tuple)
        if as_tuple:
            return tuple(TrackedTensor(t, self.counter) for t in result)
        return TrackedTensor(result, self.counter)

    def masked_select(self, input, mask):
        self.counter.record("masked_select")
        inp = _unwrap(input)
        m = _unwrap(mask) if isinstance(mask, TrackedTensor) else mask
        return TrackedTensor(torch.masked_select(inp, m), self.counter)

    def sort(self, input, dim=-1, descending=False):
        inp = _unwrap(input)
        values, indices = torch.sort(inp, dim=dim, descending=descending)
        return TrackedTensor(values, self.counter), TrackedTensor(indices, self.counter)


# ================================================================
# XLA Collective Profiler
# ================================================================

class CollectiveProfiler:
    """Records XLA collective operations for simulator benchmarking.

    Counts collective dispatches + local XLA ops and estimates latency.
    """

    def __init__(self, world_size):
        self.world_size = world_size
        self.steps = []

    def reset(self):
        self.steps = []

    def make_xm(self, rank, counter=None, unsupported_primitives=None):
        """Create a MockXM that records profiling info."""
        return _ProfilerXM(self, rank, counter,
                           unsupported_primitives=unsupported_primitives)

    def estimate_latency(self, topology, local_op_overhead_s=29e-6,
                         dispatch_overhead_s=100e-6,
                         dispatch_amortized_s=10e-6,
                         memcpy_bw_GBps=200.0,
                         local_ops=0, events=None):
        """Estimate latency from collective dispatches + bandwidth + local ops.

        Two physics-grounded refinements over the original count-based model
        (used when ``events`` is supplied; otherwise this function falls back
        to the legacy ``local_ops`` count form for backward compatibility):

          1. **Memory bandwidth for copy ops**: ops that record a non-zero
             ``copy_bytes`` (e.g. ``cat``, dense ``reshape``, ``contiguous`` on
             non-contiguous source) are charged
             ``local_op_overhead_s + copy_bytes / memcpy_bw``
             instead of just the flat per-op overhead.

          2. **Back-to-back collective pipelining**: when two or more
             collective dispatches are issued consecutively with no
             intervening non-collective op consuming the first's output,
             only the first pays the full ``dispatch_overhead_s``; subsequent
             back-to-back collectives pay the smaller ``dispatch_amortized_s``.
             This reflects EFA NIC pipelining: once the first collective is
             in flight, subsequent independent issues queue behind it
             rather than each paying a fresh round-trip.

        The 2x training-context factor (forward + implicit backward
        collective) is preserved on the collective term.

        Args:
            topology: TrainiumTopology.
            local_op_overhead_s: Fixed per-op XLA dispatch overhead.
            dispatch_overhead_s: First-collective-in-run dispatch overhead.
            dispatch_amortized_s: Per-collective cost when pipelined behind
                a preceding back-to-back collective in the same run.
            memcpy_bw_GBps: Sequential memcpy throughput (GB/s).
            local_ops: Count of local ops (used only in the legacy fallback
                when ``events`` is not provided).
            events: Optional list of ``(op_name, copy_bytes)`` tuples in
                chronological record order. When provided, enables the
                event-aware cost model.
        """
        # Lazy import to avoid circular reference; these sets are defined
        # in the same module.
        try:
            _coll = _COLLECTIVE_OPS
            _free = _FREE_XLA_OPS
        except NameError:  # pragma: no cover
            _coll = {"all_reduce", "all_gather", "reduce_scatter",
                     "all_to_all", "collective_permute"}
            _free = {"view", "unsqueeze", "squeeze", "reshape",
                     "flatten", "narrow", "transpose", "permute",
                     "expand", "contiguous"}

        if events is None:
            # Legacy fallback: caller didn't supply event order.
            total_time = local_ops * local_op_overhead_s
            for _ in self.steps:
                total_time += 2 * dispatch_overhead_s
            return total_time

        total_time = 0.0
        memcpy_bw_Bps = memcpy_bw_GBps * 1e9
        # Pointer into self.steps for ordered byte info on collectives.
        step_idx = 0
        in_collective_run = False
        for (op, copy_bytes) in events:
            if op in _coll:
                step_bytes = 0
                if step_idx < len(self.steps):
                    step_bytes = self.steps[step_idx].get("tensor_bytes", 0)
                    step_idx += 1
                bw_term = (step_bytes / memcpy_bw_Bps) if step_bytes else 0.0
                setup = (dispatch_amortized_s if in_collective_run
                         else dispatch_overhead_s)
                # 2x for forward + implicit backward collective
                total_time += 2 * (setup + bw_term)
                in_collective_run = True
            else:
                if op in _free and copy_bytes == 0:
                    continue
                bw_term = (copy_bytes / memcpy_bw_Bps) if copy_bytes else 0.0
                total_time += local_op_overhead_s + bw_term
                in_collective_run = False

        return total_time


class _ProfilerXM:
    """Mock XM that records collective patterns for profiling."""

    def __init__(self, profiler, rank, counter=None, unsupported_primitives=None):
        self.profiler = profiler
        self.rank = rank
        self.counter = counter or TorchOpCounter()
        self._step = 0
        self._unsupported = set(unsupported_primitives or [])

    def _check_supported(self, primitive):
        if primitive in self._unsupported:
            raise RuntimeError(
                f"Primitive '{primitive}' is not supported by the hardware compiler. "
                f"This would fail with a compilation error on real hardware.")

    def collective_permute(self, tensor, pairs):
        self._check_supported("collective_permute")
        step = self._step
        self._step += 1
        t = _unwrap(tensor)
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "collective_permute",
                "step": step,
                "pairs": list(pairs),
                "tensor_bytes": t.numel() * t.element_size(),
            })
        self.counter.record("collective_permute")
        return TrackedTensor(torch.zeros_like(t), self.counter)

    def all_gather(self, tensor, dim=0, groups=None):
        self._check_supported("all_gather")
        step = self._step
        self._step += 1
        t = _unwrap(tensor)
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "all_gather",
                "step": step,
                "tensor_bytes": t.numel() * t.element_size(),
                "groups": list(groups) if groups else None,
            })
        self.counter.record("all_gather")
        rep = [1] * t.dim()
        if groups and isinstance(groups[0], (list, tuple)):
            n = next(len(g) for g in groups if self.rank in g)
        elif groups:
            n = len(groups)
        else:
            n = self.profiler.world_size
        rep[dim] = n
        return TrackedTensor(torch.zeros_like(t).repeat(*rep), self.counter)

    def reduce_scatter(self, reduce_type, input, scale=1.0, scatter_dim=0,
                       shard_count=None, groups=None, output=None,
                       pin_layout=True, channel_id=None,
                       use_global_device_ids=None):
        self._check_supported("reduce_scatter")
        step = self._step
        self._step += 1
        t = _unwrap(input)
        sc = shard_count or self.profiler.world_size
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "reduce_scatter",
                "step": step,
                "tensor_bytes": t.numel() * t.element_size(),
                "shard_count": sc,
            })
        self.counter.record("reduce_scatter")
        shard_size = t.shape[scatter_dim] // sc
        slices = [slice(None)] * t.dim()
        slices[scatter_dim] = slice(0, shard_size)
        result = t[tuple(slices)] * scale
        return TrackedTensor(result, self.counter)

    def all_reduce(self, reduce_type, tensor, groups=None):
        self._check_supported("all_reduce")
        step = self._step
        self._step += 1
        t = _unwrap(tensor)
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "all_reduce",
                "step": step,
                "tensor_bytes": t.numel() * t.element_size(),
                "grouped": groups is not None,
            })
        self.counter.record("all_reduce")
        return TrackedTensor(torch.zeros_like(t), self.counter)

    REDUCE_SUM = "sum"
    REDUCE_MAX = "max"
    REDUCE_MIN = "min"

    def all_to_all(self, tensor, split_dimension=0, concat_dimension=0,
                   split_count=None):
        self._check_supported("all_to_all")
        step = self._step
        self._step += 1
        t = _unwrap(tensor)
        sc = split_count or self.profiler.world_size
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "all_to_all",
                "step": step,
                "tensor_bytes": t.numel() * t.element_size(),
                "split_count": sc,
            })
        self.counter.record("all_to_all")
        return TrackedTensor(torch.zeros_like(t), self.counter)


# ================================================================
# Mock NKI Language Module (nl) — for NKI template path
# ================================================================

class MockNLModule:
    """Mock neuronxcc.nki.language for CPU testing with numpy arrays."""

    float32 = np.float32
    float16 = np.float16
    bfloat16 = np.float32
    int32 = np.int32
    int16 = np.int16
    int8 = np.int8
    uint8 = np.uint8

    shared_hbm = "shared_hbm"
    private_hbm = "private_hbm"
    sbuf = "sbuf"
    psum = "psum"

    class tile_size:
        pmax = 128
        gemm_stationary_fmax = 128
        gemm_moving_fmax = 512

    @staticmethod
    def _resolve_dtype(dtype):
        if dtype is None or isinstance(dtype, type) and issubclass(dtype, np.generic):
            return dtype or np.float32
        if isinstance(dtype, np.dtype):
            return dtype
        return np.float32

    @staticmethod
    def ndarray(shape, dtype=np.float32, buffer=None, name='', **kwargs):
        return np.zeros(shape, dtype=MockNLModule._resolve_dtype(dtype))

    @staticmethod
    def zeros(shape, dtype=np.float32, buffer=None, name='', **kwargs):
        return np.zeros(shape, dtype=MockNLModule._resolve_dtype(dtype))

    @staticmethod
    def full(shape, fill_value, dtype=np.float32, buffer=None, name='', **kwargs):
        return np.full(shape, fill_value, dtype=MockNLModule._resolve_dtype(dtype))

    @staticmethod
    def load(src, mask=None, dtype=None, **kwargs):
        result = np.array(src, copy=True)
        if dtype is not None:
            resolved = MockNLModule._resolve_dtype(dtype)
            if resolved is not None:
                result = result.astype(resolved)
        return result

    @staticmethod
    def store(dst, value, mask=None, **kwargs):
        if isinstance(value, (int, float)):
            dst.flat[:] = value
        else:
            val = np.asarray(value)
            if dst.shape == val.shape:
                dst[:] = val
            else:
                n = min(dst.size, val.size)
                dst.flat[:n] = val.flat[:n]

    @staticmethod
    def arange(*args, **kwargs):
        return np.arange(*args)

    @staticmethod
    def copy(src, mask=None, dtype=None, **kwargs):
        result = np.array(src, copy=True)
        if dtype is not None:
            resolved = MockNLModule._resolve_dtype(dtype)
            if resolved is not None:
                result = result.astype(resolved)
        return result

    @staticmethod
    def add(x, y, dtype=None, mask=None, **kwargs):
        return np.add(x, y)

    @staticmethod
    def multiply(x, y, dtype=None, mask=None, **kwargs):
        return np.multiply(x, y)

    @staticmethod
    def subtract(x, y, dtype=None, mask=None, **kwargs):
        return np.subtract(x, y)

    @staticmethod
    def where(condition, x, y, dtype=None, mask=None, **kwargs):
        return np.where(condition, x, y)

    @staticmethod
    def maximum(x, y, dtype=None, mask=None, **kwargs):
        return np.maximum(x, y)

    @staticmethod
    def minimum(x, y, dtype=None, mask=None, **kwargs):
        return np.minimum(x, y)

    @staticmethod
    def program_id(axis=0):
        return 0

    @staticmethod
    def num_programs(axis=0):
        return 1

    @staticmethod
    def par_dim(value):
        return value

    @staticmethod
    def affine_range(*args, **kwargs):
        return range(*[int(a) for a in args])

    @staticmethod
    def sequential_range(*args, **kwargs):
        return range(*[int(a) for a in args])

    @staticmethod
    def shared_constant(constant, dtype=None, **kwargs):
        return np.array(constant)

    @staticmethod
    def device_print(prefix, x, **kwargs):
        print(f"[NKI] {prefix}: {x}")


# ================================================================
# Mock NCCL Collectives Module — for NKI template path
# ================================================================

class MockNCCLModule:
    """Mock neuronxcc.nki.nccl.collectives for multi-rank NKI simulation."""

    def __init__(self, simulator, rank):
        self.sim = simulator
        self.rank = rank
        self._cp_step = 0
        self._ag_step = 0

    def reset_steps(self):
        self._cp_step = 0
        self._ag_step = 0

    def collective_permute(self, *, dst, src, source_target_pairs,
                           mask=None, dtype=None, **kwargs):
        step = self._cp_step
        self._cp_step += 1
        if self.sim.phase == "collect":
            self.sim.cp_data.setdefault(step, {})[self.rank] = (
                np.array(src, copy=True).ravel(), list(source_target_pairs))
            resolved = self.sim.cp_resolved.get(step, {})
            if self.rank in resolved:
                _write_to_view(dst, resolved[self.rank])
            else:
                dst.flat[:] = 0
        else:
            resolved = self.sim.cp_resolved.get(step, {})
            if self.rank in resolved:
                _write_to_view(dst, resolved[self.rank])
            else:
                dst.flat[:] = 0

    def collective_permute_implicit(self, *, dst, src, replica_groups,
                                    channel_id=0, num_channels=1,
                                    mask=None, dtype=None, **kwargs):
        n = len(replica_groups)
        pairs = [(replica_groups[i], replica_groups[(i + 1) % n])
                 for i in range(n)]
        self.collective_permute(dst=dst, src=src, source_target_pairs=pairs)

    def all_gather(self, *args, srcs=None, dsts=None, replica_groups=None,
                   all_gather_dim=0, dtype=None, **kwargs):
        step = self._ag_step
        self._ag_step += 1
        if replica_groups is None:
            replica_groups = list(range(self.sim.world_size))
        if self.sim.phase == "collect":
            src_copies = [np.array(s, copy=True) for s in srcs]
            self.sim.ag_data.setdefault(step, {})[self.rank] = (
                src_copies, list(replica_groups), all_gather_dim)
            resolved = self.sim.ag_resolved.get(step, {})
            if self.rank in resolved:
                for d, r in zip(dsts, resolved[self.rank]):
                    _write_to_view(d, r)
            else:
                for d in dsts:
                    d.flat[:] = 0
        else:
            resolved = self.sim.ag_resolved.get(step, {})
            if self.rank in resolved:
                for d, r in zip(dsts, resolved[self.rank]):
                    _write_to_view(d, r)
            else:
                for d in dsts:
                    d.flat[:] = 0


def _write_to_view(dst, src_data):
    """Write src_data into dst numpy view, handling shape mismatches."""
    d_flat = dst.ravel()
    s_flat = np.asarray(src_data).ravel()
    n = min(len(d_flat), len(s_flat))
    d_flat[:n] = s_flat[:n]


# ================================================================
# NKI Collective Simulator — for NKI template path
# ================================================================

class NKICollectiveSimulator:
    """Orchestrates multi-rank NKI kernel execution with mocked collectives."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.cp_data = {}
        self.ag_data = {}
        self.cp_resolved = {}
        self.ag_resolved = {}
        self.phase = "collect"

    def set_phase(self, phase):
        self.phase = phase

    def clear(self):
        self.cp_data = {}
        self.ag_data = {}
        self.cp_resolved = {}
        self.ag_resolved = {}

    def make_nccl_module(self, rank):
        return MockNCCLModule(self, rank)

    def resolve(self):
        """Resolve all collected NKI collectives using cross-rank data."""
        for step, rank_data in self.cp_data.items():
            any_rank = next(iter(rank_data))
            _, pairs = rank_data[any_rank]
            dst_to_src = {dst: src for src, dst in pairs}
            resolved = {}
            for dst_rank in range(self.world_size):
                src_rank = dst_to_src.get(dst_rank)
                if src_rank is not None and src_rank in rank_data:
                    data, _ = rank_data[src_rank]
                    resolved[dst_rank] = data.copy()
                elif dst_rank in rank_data:
                    data, _ = rank_data[dst_rank]
                    resolved[dst_rank] = data.copy()
            self.cp_resolved[step] = resolved

        for step, rank_data in self.ag_data.items():
            any_rank = next(iter(rank_data))
            _, groups, ag_dim = rank_data[any_rank]
            num_tensors = len(rank_data[any_rank][0])
            gathered_tensors = []
            for t_idx in range(num_tensors):
                ordered = []
                for r in groups:
                    if r in rank_data:
                        ordered.append(rank_data[r][0][t_idx])
                    else:
                        ref = rank_data[any_rank][0][t_idx]
                        ordered.append(np.zeros_like(ref))
                gathered = np.concatenate(ordered, axis=ag_dim)
                gathered_tensors.append(gathered)
            resolved = {}
            for r in groups:
                resolved[r] = [g.copy() for g in gathered_tensors]
            self.ag_resolved[step] = resolved


# ================================================================
# NKI Collective Profiler — for NKI template path
# ================================================================

class NKICollectiveProfiler:
    """Records NKI collective operations for simulator benchmarking."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.steps = []

    def reset(self):
        self.steps = []

    def make_nccl_module(self, rank):
        return _NKIProfilerNCCL(self, rank)

    def estimate_latency(self, topology, dispatch_overhead_s=100e-6):
        topology.reset()
        total_time = 0.0
        for step_info in self.steps:
            total_time += dispatch_overhead_s
            if step_info["type"] in ("collective_permute",
                                     "collective_permute_implicit"):
                step_finish = 0.0
                for src, dst in step_info["pairs"]:
                    if src == dst:
                        continue
                    finish = topology.send(src, dst, step_info["tensor_bytes"])
                    step_finish = max(step_finish, finish)
                total_time += step_finish
            elif step_info["type"] == "all_gather":
                chunk_bytes = step_info["tensor_bytes"]
                groups = step_info.get("groups")
                if groups is not None:
                    group = groups
                    for _ in range(len(group) - 1):
                        step_finish = 0.0
                        for i, r in enumerate(group):
                            dst = group[(i + 1) % len(group)]
                            finish = topology.send(r, dst, chunk_bytes)
                            step_finish = max(step_finish, finish)
                        total_time += step_finish
                else:
                    for _ in range(topology.num_cores - 1):
                        step_finish = 0.0
                        for r in range(topology.num_cores):
                            dst = (r + 1) % topology.num_cores
                            finish = topology.send(r, dst, chunk_bytes)
                            step_finish = max(step_finish, finish)
                        total_time += step_finish
        return total_time


class _NKIProfilerNCCL:
    """NCCL mock that records NKI collective patterns for profiling."""

    def __init__(self, profiler, rank):
        self.profiler = profiler
        self.rank = rank
        self._step = 0

    def reset_steps(self):
        self._step = 0

    def collective_permute(self, *, dst, src, source_target_pairs, **kwargs):
        step = self._step
        self._step += 1
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "collective_permute",
                "step": step,
                "pairs": list(source_target_pairs),
                "tensor_bytes": src.size * src.itemsize,
            })
        dst.flat[:] = 0

    def collective_permute_implicit(self, *, dst, src, replica_groups,
                                    channel_id=0, num_channels=1, **kwargs):
        step = self._step
        self._step += 1
        n = len(replica_groups)
        pairs = [(replica_groups[i], replica_groups[(i + 1) % n])
                 for i in range(n)]
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "collective_permute_implicit",
                "step": step,
                "pairs": pairs,
                "tensor_bytes": src.size * src.itemsize,
                "num_channels": num_channels,
            })
        dst.flat[:] = 0

    def all_gather(self, *args, srcs=None, dsts=None, replica_groups=None,
                   all_gather_dim=0, **kwargs):
        step = self._step
        self._step += 1
        if replica_groups is None:
            replica_groups = list(range(self.profiler.world_size))
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "all_gather",
                "step": step,
                "tensor_bytes": sum(s.size * s.itemsize for s in srcs),
                "group_size": len(replica_groups),
                "groups": list(replica_groups),
            })
        for d in dsts:
            d.flat[:] = 0


# ================================================================
# Mock NKI Module (@nki.jit decorator)
# ================================================================

class MockNKIModule:
    """Mock for the top-level nki module."""

    @staticmethod
    def jit(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f


# ================================================================
# Reference implementation and test utilities
# ================================================================

def reference_alltoallv(inputs, send_counts_matrix, world_size):
    """Gold standard AllToAllV reference implementation."""
    outputs = []
    for dst_rank in range(world_size):
        parts = []
        for src_rank in range(world_size):
            offset = sum(send_counts_matrix[src_rank][:dst_rank])
            count = send_counts_matrix[src_rank][dst_rank]
            parts.append(inputs[src_rank][offset:offset + count])
        outputs.append(torch.cat(parts, dim=0))
    return outputs


def generate_test_inputs(send_counts_matrix, world_size, seed=0):
    """Generate deterministic test inputs with unique values per rank."""
    torch.manual_seed(seed)
    inputs = []
    for rank in range(world_size):
        total_send = sum(send_counts_matrix[rank])
        base = rank * 10000
        inputs.append(torch.arange(base, base + total_send, dtype=torch.float32))
    return inputs


def make_test_traffic(world_size, pattern="moe", shard_size=64):
    """Generate send_counts_matrix for testing."""
    import random as _rng_mod
    matrix = [[0] * world_size for _ in range(world_size)]

    if pattern == "moe":
        rng = _rng_mod.Random(42)
        raw = [1.0 / (i + 1) ** 1.2 for i in range(world_size)]
        perm = list(range(world_size))
        rng.shuffle(perm)
        probs = [0.0] * world_size
        for i, p in enumerate(perm):
            probs[p] = raw[i]
        total_p = sum(probs)
        probs = [p / total_p for p in probs]
        cdf = []
        acc = 0.0
        for p in probs:
            acc += p
            cdf.append(acc)
        for s in range(world_size):
            counts = [0] * world_size
            for _ in range(shard_size):
                r = rng.random()
                for d in range(world_size):
                    if r <= cdf[d]:
                        counts[d] += 1
                        break
            matrix[s] = counts
    elif pattern == "uniform":
        for s in range(world_size):
            for d in range(world_size):
                matrix[s][d] = shard_size
    elif pattern == "skewed":
        for s in range(world_size):
            for d in range(world_size):
                matrix[s][d] = shard_size * 4 if d == 0 else shard_size // 4
    elif pattern == "zero_some":
        for s in range(world_size):
            for d in range(world_size):
                matrix[s][d] = shard_size if (s + d) % 3 != 0 else 0
    elif pattern == "identity":
        for s in range(world_size):
            matrix[s][s] = shard_size
    elif pattern == "variable":
        torch.manual_seed(42)
        for s in range(world_size):
            for d in range(world_size):
                matrix[s][d] = int(torch.randint(1, shard_size * 2, (1,)).item())

    return matrix


# ================================================================
# XLA candidate calling helper
# ================================================================

def _call_xla_candidate(candidate_fn, input_tensor, send_counts, recv_counts,
                        max_chunk, rank, world_size, num_devices,
                        cores_per_device, xm_mock, torch_mock,
                        num_nodes=1):
    """Call an XLA-based candidate function."""
    try:
        return candidate_fn(
            input_tensor, send_counts, recv_counts, max_chunk,
            rank, world_size, num_devices, cores_per_device,
            xm_mock, torch_mock, num_nodes=num_nodes)
    except TypeError as e:
        if 'num_nodes' in str(e):
            return candidate_fn(
                input_tensor, send_counts, recv_counts, max_chunk,
                rank, world_size, num_devices, cores_per_device,
                xm_mock, torch_mock)
        raise


def _call_nki_candidate(candidate_fn, *args, num_nodes=1):
    """Call an NKI-based candidate function."""
    try:
        return candidate_fn(*args, num_nodes=num_nodes)
    except TypeError as e:
        if 'num_nodes' in str(e):
            return candidate_fn(*args)
        raise


# Keep backward compat alias
_call_candidate = _call_nki_candidate


# ================================================================
# XLA Correctness Testing
# ================================================================

def test_xla_candidate(candidate_fn, world_sizes=None, patterns=None,
                       verbose=False, resolve_passes=2, num_nodes=1,
                       unsupported_primitives=None):
    """
    Test an XLA-based AllToAllV candidate for correctness.

    The candidate function signature:
        (input_tensor, send_counts, recv_counts, max_chunk, rank, world_size,
         num_devices, cores_per_device, xm, torch_mock, num_nodes=1)
        -> TrackedTensor or torch.Tensor with received data

    Returns:
        (passed: bool, details: str)
    """
    if world_sizes is None:
        world_sizes = [4, 8]
        if num_nodes > 1:
            world_sizes.append(num_nodes * 32)
    if patterns is None:
        patterns = ["moe", "uniform", "skewed", "zero_some", "variable"]

    for ws in world_sizes:
        num_devices = ws // 2
        cpd = 2

        for pattern in patterns:
            shard_size = 16 if ws > 32 else 32
            matrix = make_test_traffic(ws, pattern, shard_size=shard_size)
            inputs = generate_test_inputs(matrix, ws)
            expected = reference_alltoallv(inputs, matrix, ws)

            max_chunk = max(
                (matrix[s][d] for s in range(ws) for d in range(ws)),
                default=1,
            )

            sim = CollectiveSimulator(ws)

            for _pass in range(resolve_passes):
                sim.set_phase("collect")
                for rank in range(ws):
                    send_counts = matrix[rank]
                    recv_counts = [matrix[src][rank] for src in range(ws)]
                    counter = TorchOpCounter()
                    xm_mock = MockXM(sim, rank, counter,
                                     unsupported_primitives=unsupported_primitives)
                    torch_mock = MockTorch(counter)

                    input_t = TrackedTensor(inputs[rank].clone(), counter)

                    try:
                        _call_xla_candidate(
                            candidate_fn, input_t, send_counts, recv_counts,
                            max_chunk, rank, ws, num_devices, cpd,
                            xm_mock, torch_mock, num_nodes=num_nodes)
                    except Exception as e:
                        return False, (
                            f"CRASH in collect pass {_pass}: world={ws} "
                            f"pattern={pattern} rank={rank}: "
                            f"{type(e).__name__}: {e}")
                sim.resolve()

            # Final resolve run
            sim.set_phase("resolve")
            outputs = []
            for rank in range(ws):
                send_counts = matrix[rank]
                recv_counts = [matrix[src][rank] for src in range(ws)]
                counter = TorchOpCounter()
                xm_mock = MockXM(sim, rank, counter,
                                 unsupported_primitives=unsupported_primitives)
                torch_mock = MockTorch(counter)

                input_t = TrackedTensor(inputs[rank].clone(), counter)

                try:
                    out = _call_xla_candidate(
                        candidate_fn, input_t, send_counts, recv_counts,
                        max_chunk, rank, ws, num_devices, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)
                    out_t = _unwrap(out)
                    outputs.append(out_t.float())
                except Exception as e:
                    return False, (
                        f"CRASH in resolve phase: world={ws} pattern={pattern} "
                        f"rank={rank}: {type(e).__name__}: {e}")

            for rank in range(ws):
                if outputs[rank].shape != expected[rank].shape:
                    return False, (
                        f"SHAPE MISMATCH: world={ws} pattern={pattern} "
                        f"rank={rank}: got {outputs[rank].shape}, "
                        f"expected {expected[rank].shape}")
                if not torch.allclose(outputs[rank], expected[rank], atol=1e-5):
                    diff = (outputs[rank] - expected[rank]).abs()
                    max_diff_idx = diff.argmax().item()
                    return False, (
                        f"VALUE MISMATCH: world={ws} pattern={pattern} "
                        f"rank={rank}: max_diff={diff.max():.6f} "
                        f"at index {max_diff_idx}")

            if verbose:
                print(f"  PASS: world={ws} pattern={pattern}")

    return True, "All correctness tests passed"


# ================================================================
# NKI Correctness Testing
# ================================================================

def test_nki_candidate(candidate_fn, world_sizes=None, patterns=None,
                       verbose=False, resolve_passes=2, num_nodes=1):
    """Test an NKI-based AllToAllV candidate for correctness."""
    if world_sizes is None:
        world_sizes = [4, 8]
        if num_nodes > 1:
            world_sizes.append(num_nodes * 32)
    if patterns is None:
        patterns = ["moe", "uniform", "skewed", "zero_some", "variable"]

    nl = MockNLModule()

    for ws in world_sizes:
        num_devices = ws // 2
        cpd = 2

        for pattern in patterns:
            shard_size = 16 if ws > 32 else 32
            matrix = make_test_traffic(ws, pattern, shard_size=shard_size)
            inputs = generate_test_inputs(matrix, ws)
            expected = reference_alltoallv(inputs, matrix, ws)

            max_chunk = max(
                (matrix[s][d] for s in range(ws) for d in range(ws)),
                default=1,
            )

            sim = NKICollectiveSimulator(ws)

            for _pass in range(resolve_passes):
                sim.set_phase("collect")
                for rank in range(ws):
                    send_counts = matrix[rank]
                    recv_counts = [matrix[src][rank] for src in range(ws)]
                    nccl_mock = sim.make_nccl_module(rank)
                    input_np = inputs[rank].clone().numpy()
                    try:
                        _call_nki_candidate(
                            candidate_fn,
                            input_np, send_counts, recv_counts,
                            max_chunk, rank, ws, num_devices, cpd,
                            nl, nccl_mock, num_nodes=num_nodes)
                    except Exception as e:
                        return False, (
                            f"CRASH in collect pass {_pass}: world={ws} "
                            f"pattern={pattern} rank={rank}: "
                            f"{type(e).__name__}: {e}")
                sim.resolve()

            sim.set_phase("resolve")
            outputs = []
            for rank in range(ws):
                send_counts = matrix[rank]
                recv_counts = [matrix[src][rank] for src in range(ws)]
                nccl_mock = sim.make_nccl_module(rank)
                input_np = inputs[rank].clone().numpy()
                try:
                    out_np = _call_nki_candidate(
                        candidate_fn,
                        input_np, send_counts, recv_counts,
                        max_chunk, rank, ws, num_devices, cpd,
                        nl, nccl_mock, num_nodes=num_nodes)
                    if isinstance(out_np, np.ndarray):
                        outputs.append(torch.from_numpy(out_np.copy()).float())
                    else:
                        outputs.append(torch.tensor(out_np, dtype=torch.float32))
                except Exception as e:
                    return False, (
                        f"CRASH in resolve phase: world={ws} pattern={pattern} "
                        f"rank={rank}: {type(e).__name__}: {e}")

            for rank in range(ws):
                if outputs[rank].shape != expected[rank].shape:
                    return False, (
                        f"SHAPE MISMATCH: world={ws} pattern={pattern} "
                        f"rank={rank}: got {outputs[rank].shape}, "
                        f"expected {expected[rank].shape}")
                if not torch.allclose(outputs[rank], expected[rank], atol=1e-5):
                    diff = (outputs[rank] - expected[rank]).abs()
                    max_diff_idx = diff.argmax().item()
                    return False, (
                        f"VALUE MISMATCH: world={ws} pattern={pattern} "
                        f"rank={rank}: max_diff={diff.max():.6f} "
                        f"at index {max_diff_idx}")

            if verbose:
                print(f"  PASS: world={ws} pattern={pattern}")

    return True, "All correctness tests passed"


# Backward-compat aliases
test_candidate = test_nki_candidate


# ================================================================
# XLA Benchmarking
# ================================================================

def benchmark_xla_candidate(candidate_fn, topology, send_counts_matrix,
                            world_size=32, element_bytes=4, num_nodes=1,
                            unsupported_primitives=None):
    """Benchmark an XLA candidate by profiling collective operations."""
    num_devices = topology.num_devices
    cpd = topology.cores_per_device
    inputs = generate_test_inputs(send_counts_matrix, world_size, seed=99)

    profiler = CollectiveProfiler(world_size)
    counter = TorchOpCounter()
    rank = 0
    send_counts = send_counts_matrix[rank]
    recv_counts = [send_counts_matrix[src][rank] for src in range(world_size)]
    max_chunk = max(
        (send_counts_matrix[s][d]
         for s in range(world_size) for d in range(world_size)),
        default=1,
    )

    xm_prof = profiler.make_xm(rank, counter,
                               unsupported_primitives=unsupported_primitives)
    torch_mock = MockTorch(counter)
    input_t = TrackedTensor(inputs[rank], counter)

    try:
        _call_xla_candidate(
            candidate_fn, input_t, send_counts, recv_counts, max_chunk,
            rank, world_size, num_devices, cpd,
            xm_prof, torch_mock, num_nodes=num_nodes)
    except Exception as e:
        return {"error": str(e)}

    local_ops = counter.real_local_ops
    latency = profiler.estimate_latency(topology, local_ops=local_ops,
                                         events=counter.events)
    total_bytes = sum(s["tensor_bytes"] for s in profiler.steps)
    num_cp = sum(1 for s in profiler.steps
                 if s["type"] == "collective_permute")
    num_ag = sum(1 for s in profiler.steps
                 if s["type"] == "all_gather")
    num_a2a = sum(1 for s in profiler.steps
                  if s["type"] == "all_to_all")

    return {
        "sim_time_us": latency * 1e6,
        "num_collective_permute": num_cp,
        "num_all_gather": num_ag,
        "num_all_to_all": num_a2a,
        "local_ops": local_ops,
        "total_bytes": total_bytes,
        "steps": len(profiler.steps),
    }


# ================================================================
# NKI Benchmarking
# ================================================================

def benchmark_nki_candidate(candidate_fn, topology, send_counts_matrix,
                            world_size=32, element_bytes=4, num_nodes=1):
    """Benchmark an NKI candidate by profiling its collective operations."""
    num_devices = topology.num_devices
    cpd = topology.cores_per_device
    inputs = generate_test_inputs(send_counts_matrix, world_size, seed=99)

    profiler = NKICollectiveProfiler(world_size)
    nl = MockNLModule()
    rank = 0
    send_counts = send_counts_matrix[rank]
    recv_counts = [send_counts_matrix[src][rank] for src in range(world_size)]
    max_chunk = max(
        (send_counts_matrix[s][d]
         for s in range(world_size) for d in range(world_size)),
        default=1,
    )

    nccl_prof = profiler.make_nccl_module(rank)
    input_np = inputs[rank].numpy()

    try:
        _call_nki_candidate(
            candidate_fn,
            input_np, send_counts, recv_counts, max_chunk,
            rank, world_size, num_devices, cpd,
            nl, nccl_prof, num_nodes=num_nodes)
    except Exception as e:
        return {"error": str(e)}

    latency = profiler.estimate_latency(topology)
    total_bytes = sum(s["tensor_bytes"] for s in profiler.steps)
    num_cp = sum(1 for s in profiler.steps
                 if s["type"] in ("collective_permute",
                                  "collective_permute_implicit"))
    num_ag = sum(1 for s in profiler.steps if s["type"] == "all_gather")

    return {
        "sim_time_us": latency * 1e6,
        "num_collective_permute": num_cp,
        "num_all_gather": num_ag,
        "total_bytes": total_bytes,
        "steps": len(profiler.steps),
    }


# Backward-compat aliases
benchmark_candidate = benchmark_nki_candidate


# ================================================================
# Generic (problem-driven) test and benchmark functions
# ================================================================

def test_xla_candidate_generic(problem, candidate_fn, world_sizes=None,
                                patterns=None, verbose=False,
                                resolve_passes=8, num_nodes=1,
                                unsupported_primitives=None):
    """
    Test any XLA collective candidate for correctness using a problem definition.

    The problem provides: generate_test_case, call_candidate, and reference outputs.
    """
    if world_sizes is None:
        world_sizes = [4, 8]
        if num_nodes > 1:
            world_sizes.append(min(num_nodes * 4, 16))
    if patterns is None:
        patterns = _get_patterns_for_problem(problem)

    for ws in world_sizes:
        num_devices = max(ws // 2, 1)
        cpd = 2

        for pattern in patterns:
            shard_size = 16 if ws > 32 else 32
            test_case = problem.generate_test_case(ws, pattern, shard_size, seed=0)
            expected = test_case["expected"]

            sim = CollectiveSimulator(ws)

            for _pass in range(resolve_passes):
                sim.set_phase("collect")
                for rank in range(ws):
                    counter = TorchOpCounter()
                    xm_mock = MockXM(sim, rank, counter,
                                     unsupported_primitives=unsupported_primitives)
                    torch_mock = MockTorch(counter)

                    rank_args = _wrap_rank_args(test_case["per_rank_args"][rank], counter)

                    try:
                        problem.call_candidate(
                            candidate_fn, rank_args,
                            test_case["shared_args"],
                            rank, ws, num_devices, cpd,
                            xm_mock, torch_mock, num_nodes=num_nodes)
                    except Exception as e:
                        return False, (
                            f"CRASH in collect pass {_pass}: world={ws} "
                            f"pattern={pattern} rank={rank}: "
                            f"{type(e).__name__}: {e}")
                sim.resolve()

            sim.set_phase("resolve")
            outputs = []
            for rank in range(ws):
                counter = TorchOpCounter()
                xm_mock = MockXM(sim, rank, counter,
                                 unsupported_primitives=unsupported_primitives)
                torch_mock = MockTorch(counter)

                rank_args = _wrap_rank_args(test_case["per_rank_args"][rank], counter)

                try:
                    out = problem.call_candidate(
                        candidate_fn, rank_args,
                        test_case["shared_args"],
                        rank, ws, num_devices, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)
                    out_t = _unwrap_generic(out)
                    outputs.append(out_t)
                except Exception as e:
                    return False, (
                        f"CRASH in resolve phase: world={ws} pattern={pattern} "
                        f"rank={rank}: {type(e).__name__}: {e}")

            ok, err = _compare_outputs_generic(outputs, expected, ws, pattern)
            if not ok:
                return False, err

    return True, "All tests passed"


def _interp_compilation_cost(samples, tensor_bytes):
    """Estimate NEFF compilation/load cost for an arbitrary tensor size.

    Within the measured range: piecewise-linear interpolation between
    adjacent samples (matches the agent-facing measure_compilation_cost
    tool).

    Above the measured range: log-linear extrapolation using the slope
    of the last two samples in log-log space. The previous formulation
    clamped to the largest sample, which made the cost model unable to
    distinguish a 200 MB single-collective from a 100 MB one even when
    the underlying NEFF physics says compilation cost grows
    super-linearly past a hardware-specific threshold.

    Below the measured range: clamp to the smallest sample (cost
    floor; sub-floor compilation cost is not interesting for ranking).
    """
    import math
    if not samples:
        return 0.0
    pts = sorted(samples, key=lambda s: s["tensor_bytes"])
    if tensor_bytes <= pts[0]["tensor_bytes"]:
        return float(pts[0]["neff_load_us"])
    if tensor_bytes <= pts[-1]["tensor_bytes"]:
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a["tensor_bytes"] <= tensor_bytes <= b["tensor_bytes"]:
                span = max(b["tensor_bytes"] - a["tensor_bytes"], 1)
                frac = (tensor_bytes - a["tensor_bytes"]) / span
                return float(a["neff_load_us"] + frac * (
                    b["neff_load_us"] - a["neff_load_us"]))
        return float(pts[-1]["neff_load_us"])
    # Above the largest measured sample: log-linear extrapolation using
    # the trend of the last two samples.
    a, b = pts[-2], pts[-1]
    log_x_a = math.log(max(a["tensor_bytes"], 1))
    log_x_b = math.log(max(b["tensor_bytes"], 1))
    log_y_a = math.log(max(a["neff_load_us"], 1.0))
    log_y_b = math.log(max(b["neff_load_us"], 1.0))
    if log_x_b == log_x_a:
        return float(b["neff_load_us"])
    slope = (log_y_b - log_y_a) / (log_x_b - log_x_a)
    log_x_q = math.log(max(tensor_bytes, 1))
    log_y_q = log_y_b + slope * (log_x_q - log_x_b)
    return float(math.exp(log_y_q))


def benchmark_xla_candidate_generic(problem, candidate_fn, topology,
                                     send_counts_matrix, world_size,
                                     num_nodes=1,
                                     unsupported_primitives=None,
                                     op_costs=None,
                                     dispatch_overhead_us=100.0,
                                     graph_launch_overhead_us=0.0,
                                     compilation_cost_samples=None,
                                     compilation_load_events_per_run=2,
                                     compilation_amortize_steps=5000,
                                     training_scale_bytes_multiplier=1.0,
                                     min_local_op_us=1.0,
                                     view_op_us=None,
                                     memcpy_bytes_per_us=0.0,
                                     memcpy_seq_bytes_per_us=0.0):
    """Benchmark any XLA collective candidate using the problem definition.

    Args:
        op_costs: dict mapping op_name -> cost_in_us (from agent profiling).
            If provided, per-op costs are used instead of a flat per-op overhead.
        dispatch_overhead_us: per-collective dispatch overhead in microseconds.
        graph_launch_overhead_us: per-mark_step graph launch overhead beyond
            isolated dispatch overhead. Each mark_step boundary in the
            algorithm pays this once.
        compilation_cost_samples: list of {tensor_bytes, neff_load_us} points
            characterizing per-NEFF compilation/load cost vs largest-tensor
            size. Used to penalize algorithms that pack a single very large
            tensor into one collective. If None or empty, the term is skipped.
        compilation_load_events_per_run: number of NEFF load events per
            training run (initial compile + cache evictions).
        compilation_amortize_steps: number of training steps over which the
            compilation cost is amortized.
        training_scale_bytes_multiplier: factor to scale the largest observed
            collective tensor up to its expected training-time size when
            evaluating compilation cost. The correctness test uses small
            shard sizes, but real training tensors are typically much larger.
        view_op_us: per-op cost for true metadata view ops (view, narrow,
            transpose, permute, expand, squeeze, unsqueeze, flatten, slice).
            These ops produce a non-owning view in PyTorch and never copy,
            so their isolated-mark_step microbenchmark cost (typically
            ~28 us) does NOT reflect their cost when fused inside a single
            HLO graph alongside other ops. If None, falls back to op_costs.
        memcpy_bytes_per_us: device memory-copy throughput (bytes per us)
            for the *strided / sub-region* regime. Used to charge the
            implicit O(N) copy that PyTorch silently inserts when
            reshape() or contiguous() is applied to a sub-region view
            (e.g., narrow on a non-leading dim). Set to 0 to disable.
        memcpy_seq_bytes_per_us: device memory-copy throughput (bytes
            per us) for the *sequential / dense* regime. Used to charge
            the implicit copy when reshape() or contiguous() is applied
            to a stride-permuted full-storage view (e.g., result of
            permute/transpose on a contiguous source). Trainium's
            compiler vectorizes these copies so they run at near-HBM
            sequential bandwidth, much faster than the strided regime.
            Defaults to memcpy_bytes_per_us if 0.
    """
    num_devices = topology.num_devices
    cpd = topology.cores_per_device

    test_case = problem.generate_test_case(world_size, "uniform", 64, seed=99)

    profiler = CollectiveProfiler(world_size)
    rank = 0
    counter = TorchOpCounter()
    xm_mock = profiler.make_xm(rank, counter,
                                unsupported_primitives=unsupported_primitives)
    torch_mock = MockTorch(counter)

    rank_args = _wrap_rank_args(test_case["per_rank_args"][rank], counter)

    try:
        problem.call_candidate(
            candidate_fn, rank_args,
            test_case["shared_args"],
            rank, world_size, num_devices, cpd,
            xm_mock, torch_mock, num_nodes=num_nodes)
    except Exception as e:
        return {"error": str(e)}

    # Filter the recorded events to local ops (drop collective entries).
    # Each event is (op_name, copy_bytes); copy_bytes > 0 only for ops that
    # PyTorch detected would force an implicit memory copy on this input.
    local_events = [(op, b) for (op, b) in counter.events
                    if op not in _COLLECTIVE_OPS]
    local_ops_list = [op for (op, _b) in local_events]

    # Per-op cost calculation:
    #   1. View-only ops (view, narrow, transpose, permute, expand, squeeze,
    #      unsqueeze, flatten, slice) are pure metadata in PyTorch — they
    #      never copy. Their isolated-mark_step microbench cost (~28 us)
    #      reflects kernel-launch overhead, not their cost when fused with
    #      neighbors in a single HLO graph. Charge them at view_op_us
    #      (defaulting to min_local_op_us) instead of the agent's measured
    #      isolated cost so multi-view chains don't get spuriously
    #      penalized.
    #   2. Maybe-copy ops (reshape, contiguous): cheap when the input is
    #      already contiguous and the new shape is view-reachable; otherwise
    #      PyTorch silently inserts a copy of bytes_touched. The simulator
    #      detected this at trace time and recorded copy_bytes per event.
    #      Charge max(op_floor, scaled_bytes / memcpy_bytes_per_us).
    #   3. All other ops (cat, stack, index_select, sum, ...) keep their
    #      agent-measured isolated cost as before, with the min_local_op_us
    #      floor.
    base_view_us = view_op_us if view_op_us is not None else min_local_op_us
    seq_bw = (memcpy_seq_bytes_per_us if memcpy_seq_bytes_per_us > 0
              else memcpy_bytes_per_us)

    def _op_cost(op_name, copy_bytes):
        # cat / stack: torch.cat always allocates a new buffer and copies
        # all input bytes through HBM sequentially. Charge the floor plus
        # bytes / sequential-memcpy-bandwidth so a 32 MB cat is not the
        # same flat cost as a 32 KB cat. The agent's measured isolated
        # per-call overhead (~29 us) is the floor.
        if op_name in ("cat", "stack"):
            base = (max(op_costs.get(op_name, 29.0), min_local_op_us)
                    if op_costs else 29.0)
            if copy_bytes <= 0 or seq_bw <= 0:
                return base
            scaled = copy_bytes * float(training_scale_bytes_multiplier)
            return max(base, scaled / float(seq_bw))
        # Pure metadata view ops: floor only, regardless of agent's
        # isolated-microbench measurement (which mostly reports kernel
        # launch overhead at a mark_step boundary).
        if op_name in _VIEW_ONLY_OPS or op_name in _FUSED_ELEMENTWISE_OPS:
            return base_view_us
        # Reshape / contiguous on a sub-region source: gather-style
        # copy at the strided memcpy bandwidth.
        if op_name in _MAYBE_COPY_OPS_STRIDED:
            if copy_bytes <= 0 or memcpy_bytes_per_us <= 0:
                return base_view_us
            scaled = copy_bytes * float(training_scale_bytes_multiplier)
            return max(scaled / float(memcpy_bytes_per_us), base_view_us)
        # Reshape / contiguous on a dense (full-storage permute) source:
        # predictable strided access that the compiler vectorizes,
        # charged at sequential memcpy bandwidth.
        if op_name in _MAYBE_COPY_OPS_DENSE:
            if copy_bytes <= 0 or seq_bw <= 0:
                return base_view_us
            scaled = copy_bytes * float(training_scale_bytes_multiplier)
            return max(scaled / float(seq_bw), base_view_us)
        # Volume-scaled ops (index_select, tensor-from-python-list): the
        # agent's isolated microbench cost is the floor, but actual
        # cost is dominated by data volume the op moves. Without this
        # term, an algorithm can score arbitrarily low by replacing a
        # cat+narrow chain with index_select over a Python-built index
        # tensor, even though the index_select touches the entire output
        # at random-access bandwidth and the index tensor itself was
        # built by an O(N) host-side loop.
        if op_name in _VOLUME_SCALED_OPS:
            base = (max(op_costs.get(op_name, 29.0), min_local_op_us)
                    if op_costs else 29.0)
            if copy_bytes <= 0 or memcpy_bytes_per_us <= 0:
                return base
            scaled = copy_bytes * float(training_scale_bytes_multiplier)
            return max(base, scaled / float(memcpy_bytes_per_us))
        # Compute / copy ops: agent-measured cost with the floor.
        if op_costs:
            return max(op_costs.get(op_name, 29.0), min_local_op_us)
        return 29.0

    # Walk the recorded event stream in chronological order. Charge each
    # local op via _op_cost and each collective dispatch via either the full
    # per-issue overhead (first in a back-to-back run, or after any
    # non-collective op) or an amortized per-issue overhead (subsequent
    # back-to-back issue with no intervening data-consume — captures EFA
    # pipelining). Run length resets on any non-collective, non-FREE op.
    dispatch_amortized_us = max(0.0, dispatch_overhead_us * 0.10)
    total_us = 0.0
    in_collective_run = False
    for (op, b) in counter.events:
        if op in _COLLECTIVE_OPS:
            per_issue = (dispatch_amortized_us if in_collective_run
                         else dispatch_overhead_us)
            total_us += 2 * per_issue
            in_collective_run = True
        else:
            total_us += _op_cost(op, b)
            # Break the run only on a non-free op (free metadata ops between
            # collectives don't create a real data dependency).
            if op not in _FREE_XLA_OPS:
                in_collective_run = False
    # Preserve the legacy `local_cost_us` reporting term (only local ops).
    local_cost_us = sum(_op_cost(op, b) for (op, b) in local_events)
    latency = total_us * 1e-6

    # Per-mark_step framework overhead. The candidate algorithm runs inside
    # one autograd.Function (one forward mark_step pair, one backward), so
    # all of its collective dispatches share a single graph-launch cost on
    # each pass. The previous formulation charged graph_launch_overhead per
    # collective step, which double-counted the framework overhead for
    # algorithms with many in-graph dispatches (e.g. per-tensor loops that
    # XLA fuses into one HLO graph).
    n_steps = len(profiler.steps)
    if n_steps == 0:
        graph_launch_total_us = 0.0
    else:
        graph_launch_total_us = 2 * graph_launch_overhead_us
    latency += graph_launch_total_us * 1e-6

    # NEFF compilation/load cost amortized over a training run. Driven by
    # the LARGEST single-collective tensor in the algorithm (graph size
    # tracks largest tensor). Scaled by training_scale_bytes_multiplier
    # because correctness-test inputs are smaller than real training tensors.
    compilation_amortized_us = 0.0
    if compilation_cost_samples:
        per_step_max_bytes = max(
            (s.get("tensor_bytes", 0) for s in profiler.steps), default=0)
        scaled_max_bytes = per_step_max_bytes * float(training_scale_bytes_multiplier)
        if scaled_max_bytes > 0:
            load_us = _interp_compilation_cost(
                compilation_cost_samples, scaled_max_bytes)
            compilation_amortized_us = (
                load_us * compilation_load_events_per_run /
                max(int(compilation_amortize_steps), 1))
            latency += compilation_amortized_us * 1e-6

    total_bytes = sum(s.get("tensor_bytes", 0) for s in profiler.steps)
    num_cp = sum(1 for s in profiler.steps
                 if s["type"] in ("collective_permute",
                                  "collective_permute_implicit"))
    num_ag = sum(1 for s in profiler.steps if s["type"] == "all_gather")
    num_rs = sum(1 for s in profiler.steps if s["type"] == "reduce_scatter")
    num_ar = sum(1 for s in profiler.steps if s["type"] == "all_reduce")

    op_breakdown = {}
    for op, copy_bytes in local_events:
        cost = _op_cost(op, copy_bytes)
        if op not in op_breakdown:
            op_breakdown[op] = {"count": 0, "per_op_us": cost,
                                "total_us": 0.0, "copy_bytes": 0}
        op_breakdown[op]["count"] += 1
        op_breakdown[op]["total_us"] += cost
        op_breakdown[op]["copy_bytes"] += copy_bytes

    return {
        "sim_time_us": latency * 1e6,
        "num_collective_permute": num_cp,
        "num_all_gather": num_ag,
        "num_reduce_scatter": num_rs,
        "num_all_reduce": num_ar,
        "total_bytes": total_bytes,
        "local_ops": counter.count,
        "local_cost_us": local_cost_us,
        "op_breakdown": op_breakdown,
        "steps": len(profiler.steps),
        "graph_launch_overhead_us": graph_launch_total_us,
        "compilation_amortized_us": compilation_amortized_us,
    }


def _get_patterns_for_problem(problem):
    """Return appropriate test patterns for each problem type."""
    pattern_map = {
        "alltoallv": ["moe", "uniform", "skewed", "zero_some", "variable"],
        "uniform_a2a": ["uniform", "large", "small", "moe_capacity"],
        "ring_kv": ["uniform", "large", "small", "head_dim"],
    }
    return pattern_map.get(problem.name, ["uniform"])


def _wrap_rank_args(rank_args, counter):
    """Wrap tensor values in rank_args with TrackedTensor for op counting."""
    wrapped = {}
    for k, v in rank_args.items():
        if isinstance(v, torch.Tensor):
            wrapped[k] = TrackedTensor(v.clone(), counter)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            wrapped[k] = [TrackedTensor(t.clone(), counter) for t in v]
        else:
            wrapped[k] = v
    return wrapped


def _unwrap_generic(out):
    """Unwrap output that may be a tensor, TrackedTensor, or list thereof."""
    if isinstance(out, list):
        return [_unwrap(t).float() if hasattr(t, 'float') else t for t in out]
    return _unwrap(out).float()


def _compare_outputs_generic(outputs, expected, world_size, pattern):
    """Compare outputs vs expected, handling both tensor and list-of-tensor cases."""
    for rank in range(world_size):
        out = outputs[rank]
        exp = expected[rank]

        if isinstance(exp, list):
            if not isinstance(out, list):
                return False, (f"world={world_size} pattern={pattern} rank={rank}: "
                               f"expected list output, got {type(out)}")
            if len(out) != len(exp):
                return False, (f"world={world_size} pattern={pattern} rank={rank}: "
                               f"expected {len(exp)} tensors, got {len(out)}")
            for i, (o, e) in enumerate(zip(out, exp)):
                o_t = _unwrap(o).float() if hasattr(o, 'data') else o.float()
                e_t = e.float()
                if o_t.shape != e_t.shape:
                    return False, (f"world={world_size} pattern={pattern} rank={rank} "
                                   f"tensor[{i}]: shape {o_t.shape} != {e_t.shape}")
                if not torch.allclose(o_t, e_t, atol=1e-3, rtol=1e-3):
                    diff = (o_t - e_t).abs().max().item()
                    return False, (f"world={world_size} pattern={pattern} rank={rank} "
                                   f"tensor[{i}]: max_diff={diff:.6f}")
        else:
            out_t = _unwrap(out).float() if hasattr(out, 'data') else out.float()
            exp_t = exp.float()
            if out_t.shape != exp_t.shape:
                return False, (f"world={world_size} pattern={pattern} rank={rank}: "
                               f"shape {out_t.shape} != {exp_t.shape}")
            if not torch.allclose(out_t, exp_t, atol=1e-3, rtol=1e-3):
                diff = (out_t - exp_t).abs().max().item()
                return False, (f"world={world_size} pattern={pattern} rank={rank}: "
                               f"max_diff={diff:.6f}")

    return True, ""


def test_xla_candidate_bf16(problem, candidate_fn, num_nodes=1,
                            unsupported_primitives=None):
    """Test that a candidate works with bf16 inputs (as used in real training).

    Runs a single world_size=8 test with bf16 inputs to catch:
    - Hardcoded torch.float32 dtype in tensor creation
    - Operations that don't preserve dtype through the pipeline
    - Precision issues specific to bf16

    Returns: (passed: bool, details: str)
    """
    ws = 8
    num_devices = ws // 2
    cpd = 2

    for pattern in _get_patterns_for_problem(problem)[:2]:
        shard_size = 32
        test_case = problem.generate_test_case(ws, pattern, shard_size, seed=0)
        expected = test_case["expected"]

        sim = CollectiveSimulator(ws)

        # Convert inputs to bf16
        bf16_per_rank = []
        for rank_args in test_case["per_rank_args"]:
            bf16_args = {}
            for k, v in rank_args.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    bf16_args[k] = v.to(torch.bfloat16)
                elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                    bf16_args[k] = [t.to(torch.bfloat16) if t.is_floating_point() else t
                                    for t in v]
                else:
                    bf16_args[k] = v
            bf16_per_rank.append(bf16_args)

        for _pass in range(8):
            sim.set_phase("collect")
            for rank in range(ws):
                counter = TorchOpCounter()
                xm_mock = MockXM(sim, rank, counter,
                                 unsupported_primitives=unsupported_primitives)
                torch_mock = MockTorch(counter)
                rank_args = _wrap_rank_args(bf16_per_rank[rank], counter)
                try:
                    problem.call_candidate(
                        candidate_fn, rank_args,
                        test_case["shared_args"],
                        rank, ws, num_devices, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)
                except Exception as e:
                    return False, (
                        f"BF16 CRASH in collect pass {_pass}: world={ws} "
                        f"pattern={pattern} rank={rank}: "
                        f"{type(e).__name__}: {e}")
            sim.resolve()

        sim.set_phase("resolve")
        outputs = []
        for rank in range(ws):
            counter = TorchOpCounter()
            xm_mock = MockXM(sim, rank, counter,
                             unsupported_primitives=unsupported_primitives)
            torch_mock = MockTorch(counter)
            rank_args = _wrap_rank_args(bf16_per_rank[rank], counter)
            try:
                out = problem.call_candidate(
                    candidate_fn, rank_args,
                    test_case["shared_args"],
                    rank, ws, num_devices, cpd,
                    xm_mock, torch_mock, num_nodes=num_nodes)
                out_t = _unwrap_generic(out)
                outputs.append(out_t)
            except Exception as e:
                return False, (
                    f"BF16 CRASH in resolve phase: world={ws} "
                    f"pattern={pattern} rank={rank}: "
                    f"{type(e).__name__}: {e}")

        # Compare with bf16-cast expected (bf16 precision loss is expected)
        for rank in range(ws):
            out = outputs[rank]
            exp = expected[rank]
            if isinstance(exp, list):
                for i, (o, e) in enumerate(zip(out, exp)):
                    o_t = _unwrap(o).float() if hasattr(o, 'data') else o.float()
                    e_t = e.to(torch.bfloat16).float()
                    if o_t.shape != e_t.shape:
                        return False, (
                            f"BF16 SHAPE MISMATCH: world={ws} pattern={pattern} "
                            f"rank={rank} tensor[{i}]: {o_t.shape} != {e_t.shape}")
                    if not torch.allclose(o_t, e_t, atol=0.1, rtol=0.05):
                        diff = (o_t - e_t).abs().max().item()
                        return False, (
                            f"BF16 VALUE MISMATCH: world={ws} pattern={pattern} "
                            f"rank={rank} tensor[{i}]: max_diff={diff:.6f}")
            else:
                out_t = _unwrap(out).float() if hasattr(out, 'data') else out.float()
                exp_t = exp.to(torch.bfloat16).float()
                if out_t.shape != exp_t.shape:
                    return False, (
                        f"BF16 SHAPE MISMATCH: world={ws} pattern={pattern} "
                        f"rank={rank}: {out_t.shape} != {exp_t.shape}")
                if not torch.allclose(out_t, exp_t, atol=0.1, rtol=0.05):
                    diff = (out_t - exp_t).abs().max().item()
                    return False, (
                        f"BF16 VALUE MISMATCH: world={ws} pattern={pattern} "
                        f"rank={rank}: max_diff={diff:.6f}")

    return True, "BF16 correctness tests passed"
