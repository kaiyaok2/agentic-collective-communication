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

# Shape-changing ops that MATERIALIZE a new tensor with a different
# layout than their inputs. They break XLA's ability to fuse element-
# wise / reduction compute across them into a collective's HLO segment,
# because the post-barrier tensor has a different shape/strides than
# the pre-barrier tensor. Charged for their byte-proportional cost
# elsewhere; here we use them as fusion barriers when assigning the
# fusion-credit discount in `benchmark_xla_candidate_generic`.
_FUSION_BARRIER_OPS = frozenset({
    "stack", "cat",
})

# Local compute ops that, when adjacent to a collective (no fusion
# barrier between them), can be HLO-fused with the collective on
# Trainium — the local op then contributes near-zero marginal cost.
# Includes reductions and elementwise compute that XLA actually fuses
# in practice. NOT included: shape-changing or volume-scaled ops, which
# materialize tensors and prevent fusion.
_FUSION_ELIGIBLE_LOCAL_OPS = _FUSED_ELEMENTWISE_OPS | frozenset({
    "exp", "log", "sqrt", "rsqrt", "pow", "abs", "clamp",
    "sum", "mean", "amax", "amin",
    "gather", "where",
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
        t = _unwrap(tensor)
        try:
            _b = t.numel() * t.element_size()
        except Exception:
            _b = 0
        self.counter.record("collective_permute", _b)
        result = self.sim.collective_permute(t, pairs, self.rank, step)
        return TrackedTensor(result, self.counter)

    def all_gather(self, tensor, dim=0, groups=None):
        self._check_supported("all_gather")
        step = self._ag_step
        self._ag_step += 1
        t = _unwrap(tensor)
        try:
            _b = t.numel() * t.element_size()
        except Exception:
            _b = 0
        self.counter.record("all_gather", _b)
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
        t = _unwrap(input)
        try:
            _b = t.numel() * t.element_size()
        except Exception:
            _b = 0
        self.counter.record("reduce_scatter", _b)
        result = self.sim.reduce_scatter(
            t, reduce_type, scatter_dim,
            shard_count or self.sim.world_size, self.rank, step, scale)
        return TrackedTensor(result, self.counter)

    def all_reduce(self, reduce_type, tensor, groups=None):
        self._check_supported("all_reduce")
        step = getattr(self, '_ar_step', 0)
        self._ar_step = step + 1
        t = _unwrap(tensor)
        # Record payload bytes so the cost model can apply a bandwidth floor.
        try:
            _ar_bytes = t.numel() * t.element_size()
        except Exception:
            _ar_bytes = 0
        self.counter.record("all_reduce", _ar_bytes)
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
        out = torch.zeros(*args, dtype=dtype or torch.float32)
        # Record the allocation with its bytes; this surfaces large
        # intermediates as back-to-back-amortization barriers in the
        # cost loop (they materialize a tensor between ARs which XLA
        # cannot fuse across).
        try:
            self.counter.record("zeros", out.numel() * out.element_size())
        except Exception:
            self.counter.record("zeros")
        return TrackedTensor(out, self.counter)

    def ones(self, *args, device=None, dtype=None, **kwargs):
        out = torch.ones(*args, dtype=dtype or torch.float32)
        try:
            self.counter.record("ones", out.numel() * out.element_size())
        except Exception:
            self.counter.record("ones")
        return TrackedTensor(out, self.counter)

    def empty(self, *args, device=None, dtype=None, **kwargs):
        out = torch.empty(*args, dtype=dtype or torch.float32)
        try:
            self.counter.record("empty", out.numel() * out.element_size())
        except Exception:
            self.counter.record("empty")
        return TrackedTensor(out, self.counter)

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
        _b = t.numel() * t.element_size()
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "collective_permute",
                "step": step,
                "pairs": list(pairs),
                "tensor_bytes": _b,
            })
        self.counter.record("collective_permute", _b)
        return TrackedTensor(torch.zeros_like(t), self.counter)

    def all_gather(self, tensor, dim=0, groups=None):
        self._check_supported("all_gather")
        step = self._step
        self._step += 1
        t = _unwrap(tensor)
        _b = t.numel() * t.element_size()
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "all_gather",
                "step": step,
                "tensor_bytes": _b,
                "groups": list(groups) if groups else None,
            })
        self.counter.record("all_gather", _b)
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
        _b = t.numel() * t.element_size()
        sc = shard_count or self.profiler.world_size
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "reduce_scatter",
                "step": step,
                "tensor_bytes": _b,
                "shard_count": sc,
            })
        self.counter.record("reduce_scatter", _b)
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
        _b = t.numel() * t.element_size()
        if self.rank == 0:
            self.profiler.steps.append({
                "type": "all_reduce",
                "step": step,
                "tensor_bytes": _b,
                "grouped": groups is not None,
            })
        self.counter.record("all_reduce", _b)
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


# ======================================================================
# AST-based structural analysis of candidate source for the simulator's
# per-mark_step graph overhead and HBM peak-bytes terms.
#
# These two terms together close the simulator-vs-HW gap on:
#   (1) bundling-trick problems (fsdp_prefetch, tp_mlp, pp_send_recv) where
#       the per_mb baseline emits one mark_step graph per microbatch and
#       pays an N_MB x per_graph_neff_load_us launch tax, while a single
#       stacked-collective bundle pays only one graph load.
#   (2) huge-intermediate problems (grad_ar cat-AR) where a single
#       all-grads-cat tensor compiles fine on the correctness test but
#       OOMs at training scale. The bucketed bound caps each AR's peak
#       at a hardcoded constant (`bucket_bytes`) regardless of how many
#       grads exist.
#
# Both signals are read off the candidate's Python source via ast — no
# per-problem hardcoding, no name matching. The analyses are best-effort
# (None when source is unavailable or AST parsing fails) and the cost
# model degrades to the original event-stream-only path.
# ======================================================================

_COLLECTIVE_ATTR_NAMES = frozenset({
    "all_gather", "all_reduce", "reduce_scatter",
    "all_to_all", "collective_permute",
})


# Outer-loop iterables that the codebase's training driver wraps in a
# per-iteration `xm.mark_step()` boundary. When the candidate function
# is `for m in range(M):` (microbatches), each iteration is a separate
# NEFF graph at training scale, even though the candidate's sandbox
# doesn't emit an explicit mark_step. This is the structural pattern
# the cost model needs to charge against per-microbatch baselines.
_MICROBATCH_LOOP_VAR_NAMES = frozenset({
    "M", "N_MB", "num_microbatches", "n_microbatches",
    "num_mb", "n_mb",
})


def _ast_count_outer_collective_loops(source):
    """Count outer `for` loops in the candidate function whose iteration
    corresponds to a per-step mark_step boundary at training scale, and
    the total number of collective dispatches that fall under those
    graph-inducing loops.

    A loop counts as "graph-inducing" if either:

      (a) It contains an explicit `xm.mark_step()` (or `xm.step()`)
          call somewhere in its subtree, OR
      (b) It iterates over `range(M)` / `range(N_MB)` /
          `range(num_microbatches)` (the conventional microbatch
          loop variable) AND contains at least one collective.

    Loops that emit collectives but iterate over a non-microbatch
    structural quantity (e.g. `for chunk in kv_chunks`,
    `for g in rep_grads`, `for bk in buckets`) are NOT counted. In
    real training, those collectives fuse into a single back-to-back
    NEFF because the runtime calls the candidate function once per
    training step, not per loop iteration.

    Returns:
        (n_outer_microbatch_loops, total_collective_call_count,
         n_graph_inducing_collectives) or (None, None, None) if source
        could not be parsed.

        n_graph_inducing_collectives is the count of xm collective
        dispatch sites nested under graph-inducing outer loops. At
        training scale each such site becomes a distinct NEFF per
        iteration (M dispatches across M microbatches if the dispatch
        sits directly under the `for m in range(M):` body; M*L
        dispatches if a nested layer loop wraps the dispatch within
        the microbatch loop). This count drives the per-graph NEFF
        load tax in the cost model: a bundled candidate hoists its
        collective(s) OUT of the microbatch loop and pays zero
        graph-induced tax, while a per_mb candidate pays
        2 * n_graph_inducing_collectives * per_graph_neff_load_us.
    """
    import ast
    try:
        tree = ast.parse(source)
    except Exception:
        return None, None, None

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        return None, None, None

    def _node_has_collective(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr in _COLLECTIVE_ATTR_NAMES:
                    return True
        return False

    def _node_has_explicit_mark_step(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr in ("mark_step", "step"):
                    # Only mark_step / step on something xm-like;
                    # ignore other unrelated .step() calls.
                    if isinstance(sub.func.value, ast.Name) and sub.func.value.id in {
                        "xm", "xla",
                    }:
                        return True
        return False

    def _count_collectives(node):
        n = 0
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr in _COLLECTIVE_ATTR_NAMES:
                    n += 1
        return n

    def _iter_is_microbatch_range(iter_node):
        """True iff `iter_node` is `range(M)` / `range(N_MB)` / ...
        for one of the conventional microbatch-loop variables."""
        if isinstance(iter_node, ast.Call):
            if (isinstance(iter_node.func, ast.Name)
                    and iter_node.func.id == "range"
                    and len(iter_node.args) >= 1):
                last = iter_node.args[-1]
                if isinstance(last, ast.Name) and last.id in _MICROBATCH_LOOP_VAR_NAMES:
                    return True
        return False

    n_outer_loops = 0
    n_graph_inducing_collectives = 0
    for stmt in func_def.body:
        if not isinstance(stmt, ast.For):
            continue
        if not _node_has_collective(stmt):
            continue
        if (_node_has_explicit_mark_step(stmt) or
                _iter_is_microbatch_range(stmt.iter)):
            n_outer_loops += 1
            # Each xm collective syntactically nested under a graph-
            # inducing outer loop becomes one NEFF per iteration at
            # training scale. The total NEFF count is the number of
            # such dispatch *sites* (not iterations of any inner range)
            # times the observed dispatch multiplicity. The event-stream
            # counter handles the multiplicity (each iteration of any
            # inner loop emits its own dispatch event), so here we just
            # count the dispatch SITES inside the outer loop's body.
            # That count is then used together with the event-stream's
            # observed collective count: the actual number of NEFF
            # graphs equals the number of collective events that
            # originated inside a graph-inducing outer loop.
            n_graph_inducing_collectives += _count_collectives(stmt)

    total_coll_calls = _count_collectives(func_def)
    return n_outer_loops, total_coll_calls, n_graph_inducing_collectives


def _ast_detect_bucket_cap(source):
    """Detect a hardcoded byte-size cap in the candidate. Common patterns:

        bucket_bytes = 32 * 1024 * 1024
        BUCKET_BYTES = 32 << 20
        max_bucket_bytes = 64 * 1024 ** 2

    A bucketed algorithm clamps its peak intermediate at this cap
    regardless of how many params/microbatches it processes. At training
    scale, this is the structural property that distinguishes a paper-
    quality bucketed reduce from a naive cat-all-then-reduce.

    Returns the cap in bytes, or None if no such assignment is found.
    """
    import ast
    try:
        tree = ast.parse(source)
    except Exception:
        return None

    BUCKET_NAMES = {
        # Bucket-family names
        "bucket_bytes", "max_bucket_bytes", "bucket_size_bytes",
        "BUCKET_BYTES", "MAX_BUCKET_BYTES", "BUCKET_SIZE_BYTES",
        "bucket_byte_limit", "bucket_cap_bytes",
        # Chunk-family names
        "chunk_bytes", "max_chunk_bytes", "chunk_size_bytes",
        "CHUNK_BYTES", "MAX_CHUNK_BYTES", "CHUNK_SIZE_BYTES",
        "chunk_byte_limit", "chunk_cap_bytes",
        # Partition-family names
        "partition_bytes", "max_partition_bytes",
        "PARTITION_BYTES", "MAX_PARTITION_BYTES",
    }

    def _eval_const(node):
        """Try to evaluate a constant-arithmetic AST node (no names)."""
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, (int, float)) else None
        if isinstance(node, ast.BinOp):
            l = _eval_const(node.left)
            r = _eval_const(node.right)
            if l is None or r is None:
                return None
            try:
                if isinstance(node.op, ast.Add):  return l + r
                if isinstance(node.op, ast.Sub):  return l - r
                if isinstance(node.op, ast.Mult): return l * r
                if isinstance(node.op, ast.Div):  return l / r
                if isinstance(node.op, ast.LShift): return int(l) << int(r)
                if isinstance(node.op, ast.RShift): return int(l) >> int(r)
                if isinstance(node.op, ast.Pow):  return l ** r
            except Exception:
                return None
        return None

    cap = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in BUCKET_NAMES:
                    val = _eval_const(node.value)
                    if isinstance(val, (int, float)) and val > 0:
                        if cap is None or val > cap:
                            cap = int(val)
    return cap


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
                                     pipeline_amort_alpha1=0.30,
                                     pipeline_amort_alpha2=0.10,
                                     pipeline_amort_alpha3=0.02,
                                     training_scale_bytes_multiplier=1.0,
                                     min_local_op_us=1.0,
                                     view_op_us=None,
                                     memcpy_bytes_per_us=0.0,
                                     memcpy_seq_bytes_per_us=0.0,
                                     per_graph_neff_load_us=200.0,
                                     hbm_budget_bytes_per_core=2 * (1024 ** 3),
                                     hbm_penalty_scale_us=10000.0,
                                     candidate_source=None):
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
        per_graph_neff_load_us: per-mark_step NEFF load + launch tax
            paid each time the candidate's algorithm crosses a graph
            boundary at training scale. AST-detected: each outer `for`
            loop in the function body that contains a collective
            dispatch contributes one such graph boundary (the loop's
            body becomes a separately-cached NEFF). At training scale
            on the 7-node EFA cluster this is ~150-300us each.
        hbm_budget_bytes_per_core: per-rank HBM safety budget for a
            single intermediate tensor (default 2 GB). Beyond this, a
            smooth quadratic penalty is added to the candidate's
            sim_time_us. The penalty does NOT reject the candidate —
            it down-ranks it in proportion to overflow severity.
        hbm_penalty_scale_us: coefficient on the quadratic overflow
            term: `((peak_mb - budget_mb) / budget_mb)**2 *
            hbm_penalty_scale_us`. Tuned so that 2x overflow ~ +10ms
            (clearly the worst candidate); 1.2x overflow ~ +400us
            (marginal).
        candidate_source: optional Python source string for the
            candidate. When supplied, enables AST-based detection of
            (1) outer `for` loops containing collectives (per-graph
            tax) and (2) hardcoded bucket-bytes caps (bounds the
            simulated peak intermediate at training scale). When
            absent both terms degrade gracefully to zero contribution.
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

    # Detect a hardcoded byte-budget cap (e.g. `chunk_bytes = 64 * 1024 * 1024`)
    # so that cat/stack/reshape/contiguous costs of a *bucketed* algorithm get
    # clamped at the bucket size at training scale, not at multiplier*test_size.
    # Without this, a chunked algorithm's cat-within-bucket would charge for
    # the full ~30 GB scaled tensor instead of the actual 64 MB per-bucket cat.
    _src_for_cap = candidate_source
    if _src_for_cap is None:
        _src_for_cap = getattr(candidate_fn, "__candidate_source__", None)
    _bucket_cap = _ast_detect_bucket_cap(_src_for_cap) if _src_for_cap else None

    def _scaled_bytes(copy_bytes):
        """Bytes at training scale, clamped by detected bucket cap (if any)."""
        scaled = copy_bytes * float(training_scale_bytes_multiplier)
        if _bucket_cap is not None and _bucket_cap > 0:
            scaled = min(scaled, float(_bucket_cap))
        return scaled

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
            scaled = _scaled_bytes(copy_bytes)
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
            scaled = _scaled_bytes(copy_bytes)
            return max(scaled / float(memcpy_bytes_per_us), base_view_us)
        # Reshape / contiguous on a dense (full-storage permute) source:
        # predictable strided access that the compiler vectorizes,
        # charged at sequential memcpy bandwidth.
        if op_name in _MAYBE_COPY_OPS_DENSE:
            if copy_bytes <= 0 or seq_bw <= 0:
                return base_view_us
            scaled = _scaled_bytes(copy_bytes)
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
    # Three-tier EFA pipeline-fill model:
    #   * FIRST collective in a run pays full d_full (no pipelining yet).
    #   * FIRST follow-up (idx==1) pays ~30% of d_full (early pipeline fill).
    #   * Tier 2-7 (idx 2..7) pay ~10% of d_full (deep EFA pipeline).
    #   * Tier 8+ (idx >= 8) pay ~2% of d_full (intra-graph fusion regime;
    #     XLA fuses adjacent independent collectives at the HLO level into
    #     a single hardware operation, so the marginal per-issue cost is
    #     near zero past N=8 in a single mark_step graph).
    # Calibrated against the round-8 sim-vs-HW sweep across (DM, L) where
    # the existing 3-tier model overestimated bundling wins by 2-3x at L>=8.
    dispatch_partial_amort_us = max(0.0, dispatch_overhead_us * pipeline_amort_alpha1)
    dispatch_deep_amort_us    = max(0.0, dispatch_overhead_us * pipeline_amort_alpha2)
    dispatch_fused_amort_us   = max(0.0, dispatch_overhead_us * pipeline_amort_alpha3)

    # Per-collective BANDWIDTH FLOOR: a collective transferring N bytes is
    # physically bound by N / effective_bandwidth, regardless of how
    # pipelined the dispatch is. Below this floor is impossible. This is
    # what stops the simulator from amortizing N tiny ARs to ~zero cost
    # when each AR is actually bandwidth-bound at training scale.
    # Effective bandwidth is read off the topology:
    #   * For multi-node clusters, the inter-node EFA per node is the
    #     bottleneck (lower than NeuronLink intra-node).
    #   * For single-node, NeuronLink dominates.
    # The bytes-per-collective at training scale is clamped by the
    # AST-detected bucket_cap when present (bucketed algos reduce the
    # per-call payload to the bucket size).
    _topo_num_nodes = getattr(topology, "num_nodes", 1) if topology is not None else 1
    _topo_efa_bw = getattr(topology, "efa_bw", 12.5) if topology is not None else 12.5
    _topo_efa_adapters = getattr(topology, "efa_adapters", 8) if topology is not None else 8
    _topo_link_bw = getattr(topology, "link_bw", 192.0) if topology is not None else 192.0
    if _topo_num_nodes > 1:
        # Inter-node bottleneck: aggregate EFA per node (per-node bytes / us)
        _cluster_bw_bps = _topo_efa_bw * _topo_efa_adapters * 1024.0  # GB/s -> MB/s -> bytes/us
    else:
        # Intra-node: NeuronLink per device, scale by num_cores for parallel
        _cluster_bw_bps = _topo_link_bw * 1024.0  # GB/s -> bytes/us

    def _coll_bandwidth_floor_us(bytes_at_test):
        if _cluster_bw_bps <= 0:
            return 0.0
        scaled = bytes_at_test * float(training_scale_bytes_multiplier)
        if _bucket_cap is not None and _bucket_cap > 0:
            scaled = min(scaled, float(_bucket_cap))
        return scaled / _cluster_bw_bps

    total_us = 0.0
    coll_run_idx = 0  # 0 = no run; 1 = first follow-up; 2..7 = deep pipeline;
                       # 8+ = intra-graph fusion regime.
    for (op, b) in counter.events:
        if op in _COLLECTIVE_OPS:
            if coll_run_idx == 0:
                per_issue = dispatch_overhead_us
            elif coll_run_idx == 1:
                per_issue = dispatch_partial_amort_us
            elif coll_run_idx < 8:
                per_issue = dispatch_deep_amort_us
            else:
                per_issue = dispatch_fused_amort_us
            # max of amortized-dispatch and bandwidth-floor; doubled for fwd+bwd
            bw_us = _coll_bandwidth_floor_us(b)
            total_us += 2 * max(per_issue, bw_us)
            coll_run_idx += 1
        else:
            total_us += _op_cost(op, b)
            # Break the run only on a non-free op (free metadata ops between
            # collectives don't create a real data dependency).
            if op not in _FREE_XLA_OPS:
                coll_run_idx = 0

    # ------------------------------------------------------------------
    # Fusion-credit term (structural, not hardcoded)
    # ------------------------------------------------------------------
    # On Trainium, XLA fuses adjacent fusion-eligible local compute ops
    # into the same HLO segment as a collective when no fusion barrier
    # (stack, cat) separates them. The local op then contributes near-
    # zero marginal cost. The existing loop above charges each local op
    # at its full _op_cost; this pass adds a credit (negative cost) for
    # each fusion-eligible local op that is adjacent to a collective
    # with no barrier between.
    #
    # The fusion window: from each collective, walk outward (both
    # directions) skipping _FREE_XLA_OPS; credit fusion-eligible compute
    # ops; stop at the first fusion barrier OR non-eligible non-free op.
    #
    # Bundling-credit extension: a `cat` or `stack` that *directly
    # precedes* a collective (through only free-metadata ops) is the
    # input-materialisation step of a bundled collective pattern (e.g.
    # `stack(M*L partials) -> all_reduce` or `cat(grad_shards) ->
    # all_reduce`). On Trainium this lowers into a single HLO segment in
    # which the cat/stack becomes the collective's input-layout planner
    # and emits ONE HBM round-trip — the same round-trip the collective
    # itself already pays (and which the simulator charges via the
    # per-collective bandwidth floor, scaled by the AR's bytes). The
    # event-stream cost model otherwise charges the cat/stack byte-cost
    # a SECOND time at sequential memcpy bandwidth, which is faster than
    # the EFA cluster floor — so the bundled pattern looks more
    # expensive than the per-microbatch loop even at training scale
    # where the M*L dispatches dominate per-mb's wall time.
    #
    # The credit is the FULL `_op_cost(cat|stack)`: that byte cost is
    # already paid by the downstream collective's bandwidth-floor (which
    # scales linearly with the same bytes), so charging it twice double-
    # counts the HBM round-trip. The cat/stack still contributes to the
    # peak-intermediate bookkeeping below (HBM penalty) — a candidate
    # whose bundled tensor exceeds the per-rank HBM budget is still
    # penalised structurally regardless of this credit.
    #
    # This applies only to the BACKWARD walk (cat/stack preceding the
    # collective). Cat/stack AFTER a collective (e.g. an output
    # reorganisation cat in `ag_slice_cat`) is NOT credited because its
    # bytes flow into post-collective code, not into the collective.
    fusion_credit_frac = 0.30  # calibrated against 7-node bundled-vs-per_mb data
    n = len(counter.events)
    fusion_credit_us = 0.0
    credited = set()  # event indices already credited (avoid double-counting)
    for i, (op, _) in enumerate(counter.events):
        if op not in _COLLECTIVE_OPS:
            continue
        # Walk backward
        j = i - 1
        bundling_input_credited = False
        while j >= 0:
            op_j, b_j = counter.events[j]
            if op_j in _COLLECTIVE_OPS:
                break
            if op_j in _FUSION_BARRIER_OPS:
                # First cat/stack on the path back from the collective
                # is the bundling-input materialiser. Credit it fully and
                # stop the walk (this op is also a fusion barrier for any
                # further-back fusion-eligible compute).
                if (not bundling_input_credited) and j not in credited:
                    fusion_credit_us += _op_cost(op_j, b_j)
                    credited.add(j)
                    bundling_input_credited = True
                break
            if op_j in _FUSION_ELIGIBLE_LOCAL_OPS and j not in credited:
                fusion_credit_us += fusion_credit_frac * _op_cost(op_j, b_j)
                credited.add(j)
            if op_j not in _FREE_XLA_OPS and op_j not in _FUSION_ELIGIBLE_LOCAL_OPS:
                break
            j -= 1
        # Walk forward (no bundling credit — cat/stack AFTER a collective
        # is post-collective output reshaping, not bundling.)
        j = i + 1
        while j < n:
            op_j, b_j = counter.events[j]
            if op_j in _FUSION_BARRIER_OPS or op_j in _COLLECTIVE_OPS:
                break
            if op_j in _FUSION_ELIGIBLE_LOCAL_OPS and j not in credited:
                fusion_credit_us += fusion_credit_frac * _op_cost(op_j, b_j)
                credited.add(j)
            if op_j not in _FREE_XLA_OPS and op_j not in _FUSION_ELIGIBLE_LOCAL_OPS:
                break
            j += 1
    total_us -= fusion_credit_us
    if total_us < 0:
        total_us = 0.0

    # ------------------------------------------------------------------
    # STANDALONE_GRAPH_HW_CAL — cost floor for collective-free graphs
    # ------------------------------------------------------------------
    # Empirical Neuron 2-node 64-rank measurement (2026-08-11): a
    # graph with only local elementwise arithmetic on a small (16x16
    # or 32x32) position-based tensor takes a base kernel-launch cost
    # regardless of op count, and marginal per-op cost drops sharply
    # after 3-4 ops as XLA fuses.
    #
    # Measured (bit-decomposition workload, N=16):
    #   1 add:         343 us
    #   3 adds:        754 us
    #   7 adds:        818 us
    #   15 adds:       805 us
    # A pure constant-fold  graph: 122 us.
    #
    # The current per-op sum charges each _FUSED_ELEMENTWISE_OPS at
    # min_local_op_us (1 us) — off by 100-500x for this regime. Fusion
    # credit only fires next to a collective, so pure _bcast problems
    # never benefit. The fix: if the graph has zero collectives, replace
    # the sum-of-per-op elementwise cost with a saturating model
    # calibrated to the measurements above.
    #
    # Only applies when there are NO collective events and at least
    # one _FUSED_ELEMENTWISE_OPS event (otherwise it is a pure view/
    # metadata graph and the existing floor is already correct).
    HW_CAL_BASE_US = 340.0        # 1 arithmetic op after arange
    HW_CAL_MARGINAL_US = 100.0    # per additional op up to saturation
    HW_CAL_SATURATION_US = 800.0  # ~7+ ops
    HW_CAL_CONST_FOLD_US = 120.0  # torch.tensor(list) baked constant
    n_coll = sum(1 for (op, _) in counter.events if op in _COLLECTIVE_OPS)
    n_fused_arith = sum(1 for (op, _) in local_events if op in _FUSED_ELEMENTWISE_OPS)
    n_tensor_const = sum(1 for (op, _) in local_events if op == 'tensor')
    if n_coll == 0:
        # Recompute the elementwise-fused contribution using the
        # empirical model, then swap it into total_us.
        old_arith_us = 0.0
        for (op, b) in local_events:
            if op in _FUSED_ELEMENTWISE_OPS:
                old_arith_us += _op_cost(op, b)
        if n_fused_arith > 0:
            new_arith_us = min(HW_CAL_SATURATION_US,
                               HW_CAL_BASE_US + HW_CAL_MARGINAL_US * max(0, n_fused_arith - 1))
            total_us += new_arith_us - old_arith_us
        # A pure constant-fold graph: the tensor(list) op is currently
        # charged max(29, bytes/bw); the HW measurement says the
        # compile-time constant graph costs ~120 us end-to-end. If the
        # graph is dominated by ONE tensor() op with no other compute
        # (arange excepted), replace the tensor cost with the calibrated
        # constant-fold cost.
        if n_tensor_const == 1 and n_fused_arith == 0:
            old_tensor_us = 0.0
            for (op, b) in local_events:
                if op == 'tensor':
                    old_tensor_us += _op_cost(op, b)
            total_us += HW_CAL_CONST_FOLD_US - old_tensor_us
        if total_us < 0:
            total_us = 0.0

    # ------------------------------------------------------------------
    # Primitive-viability terms (structural, real HW failure modes)
    # ------------------------------------------------------------------
    # 1. collective_permute ring patterns SIGABRT at world_size > 64 on
    #    this cluster (documented in the cluster runbook). Any candidate
    #    that uses collective_permute at this scale is HW-invalid.
    # 2. all_reduce over a single payload exceeding the NRT per-collective
    #    tensor-size limit (~8 GB at training scale) triggers
    #    NRT_RESOURCE during LoadCollectives. Reject candidates that
    #    would issue such an AR.
    NRT_PER_COLLECTIVE_BYTES_LIMIT = 8 * (1024 ** 3)  # ~8 GB
    if world_size > 64:
        for (op, b) in counter.events:
            if op == "collective_permute":
                try:
                    counter.hw_invalid_reason = (
                        f"collective_permute used at world_size={world_size} "
                        f"(SIGABRT-known at ws>64 on this cluster)"
                    )
                except Exception:
                    pass
                return {"error": "hw_invalid: collective_permute at ws>64",
                        "sim_time_us": float("inf"),
                        "local_cost_us": 0.0,
                        "local_ops": 0,
                        "num_collective_permute": 1,
                        "num_all_gather": 0,
                        "num_all_reduce": 0}
    scale = float(training_scale_bytes_multiplier)
    for (op, b) in counter.events:
        if op == "all_reduce" and b * scale > NRT_PER_COLLECTIVE_BYTES_LIMIT:
            try:
                counter.hw_invalid_reason = (
                    f"all_reduce payload {b * scale} bytes > NRT limit "
                    f"{NRT_PER_COLLECTIVE_BYTES_LIMIT} bytes (LoadCollectives "
                    f"would fail at training scale)"
                )
            except Exception:
                pass
            return {"error": "hw_invalid: all_reduce payload exceeds NRT limit",
                    "sim_time_us": float("inf"),
                    "local_cost_us": 0.0,
                    "local_ops": 0,
                    "num_collective_permute": 0,
                    "num_all_gather": 0,
                    "num_all_reduce": 1}

    # ------------------------------------------------------------------
    # HBM peak-bytes penalty (structural, smooth, not a hard reject)
    # ------------------------------------------------------------------
    # Track the largest single intermediate tensor the candidate
    # materializes (in bytes; volume known from event stream). Three
    # principled corrections from the previous hard-reject formulation:
    #
    #   1. Smooth penalty instead of float('inf'). A candidate that
    #      modestly overflows the HBM budget at training scale (e.g.
    #      1.2x) is *worse* than one that fits, but should not be
    #      treated identically to a candidate that overflows by 10x.
    #      The quadratic term `((peak - budget) / budget)**2` gives
    #      gradient information to the search instead of binary
    #      pass/fail.
    #   2. Bucket-cap clamping when the candidate's source declares a
    #      `bucket_bytes = <const>` cap. At training scale the cap
    #      determines the real peak regardless of the multiplier,
    #      because the algorithm dynamically partitions to stay
    #      under-cap. AST-detected; falls back to the raw scaled
    #      peak if no cap is found.
    #   3. Reduced default budget (2 GB / rank, was 4 GB). Trainium
    #      HBM is 16 GB but the activation+optimizer+scratch
    #      footprint at ~10B params consumes most of it; the safety
    #      budget for a single collective-intermediate tensor is
    #      well under 2 GB on the 7-node cluster.
    peak_intermediate_bytes = 0
    for (op, b) in counter.events:
        if op in _COLLECTIVE_OPS or op in _FUSION_BARRIER_OPS or op == "zeros":
            peak_intermediate_bytes = max(peak_intermediate_bytes, b)
    raw_scaled_peak = peak_intermediate_bytes * float(training_scale_bytes_multiplier)

    # AST-detect a hardcoded bucket cap, if any, and clamp the
    # training-scale peak by it. A bucketed algorithm that says
    # `bucket_bytes = 32 << 20` has peak intermediate ~32 MB at
    # training scale no matter how many params there are.
    bucket_cap_bytes = None
    if candidate_source is None:
        candidate_source = getattr(candidate_fn, "__candidate_source__", None)
    if candidate_source is not None:
        bucket_cap_bytes = _ast_detect_bucket_cap(candidate_source)
    if bucket_cap_bytes is not None and bucket_cap_bytes > 0:
        scaled_peak = min(raw_scaled_peak, float(bucket_cap_bytes))
    else:
        scaled_peak = raw_scaled_peak

    budget = float(hbm_budget_bytes_per_core)
    if budget > 0 and scaled_peak > budget:
        overflow_ratio = (scaled_peak - budget) / budget
        hbm_penalty_us = overflow_ratio * overflow_ratio * float(hbm_penalty_scale_us)
    else:
        hbm_penalty_us = 0.0
    try:
        counter.oom_peak_bytes = scaled_peak
        counter.oom_budget_bytes = budget
    except Exception:
        pass

    # Preserve the legacy `local_cost_us` reporting term (only local ops).
    local_cost_us = sum(_op_cost(op, b) for (op, b) in local_events)
    latency = total_us * 1e-6

    # ------------------------------------------------------------------
    # Per-mark_step framework overhead (structural, AST-derived)
    # ------------------------------------------------------------------
    # The candidate runs inside one autograd.Function and pays one
    # framework graph-launch tax per forward+backward (the legacy
    # `graph_launch_overhead_us` term). On top of that, every outer
    # `for` loop whose iteration matches a *microbatch boundary* breaks
    # XLA's compilation unit into N separately-cached NEFFs at training
    # scale, even though the candidate's sandbox doesn't insert an
    # explicit mark_step. _ast_count_outer_collective_loops detects
    # this via two patterns:
    #   (i)  `for m in range(M):` (M/N_MB/num_microbatches), or
    #   (ii) any outer for-loop containing an explicit `xm.mark_step()`.
    # Loops over per-tensor or per-bucket inner data (e.g.
    # `for g in rep_grads`, `for bk in buckets`) are NOT charged: in
    # real training those collectives fuse into a single back-to-back
    # NEFF because the runtime calls the candidate function once per
    # step, not per loop iteration.
    #
    # IMPORTANT: the tax scales with the *number of NEFF graphs*, not
    # with the number of OUTER loops. A `for m in range(M): for L in
    # range(N_LAYERS): xm.all_reduce(...)` body produces M*L distinct
    # NEFFs at training scale (each (m, L) is its own mark_step graph),
    # not 1. We approximate the NEFF count by the number of collective
    # *events* recorded by the profiler that originated under graph-
    # inducing outer loops — which equals the total collective event
    # count when every collective in the candidate sits under such a
    # loop, and 0 when no collective does. This matches the actual
    # NEFF cache size that the training runtime pays for.
    n_steps = len(profiler.steps)
    if n_steps == 0:
        graph_launch_total_us = 0.0
    else:
        graph_launch_total_us = 2 * graph_launch_overhead_us

    n_outer_collective_loops = 0
    n_graph_inducing_coll_sites = 0
    n_total_coll_sites = 0
    if candidate_source is not None:
        n_loops, n_total_sites, n_grph_sites = _ast_count_outer_collective_loops(
            candidate_source)
        if n_loops is not None:
            n_outer_collective_loops = int(n_loops)
        if n_grph_sites is not None:
            n_graph_inducing_coll_sites = int(n_grph_sites)
        if n_total_sites is not None:
            n_total_coll_sites = int(n_total_sites)

    # Number of NEFF graphs charged per fwd pass: every collective EVENT
    # whose syntactic dispatch site lives under a graph-inducing outer
    # loop becomes its own NEFF at training scale. The event-stream
    # counter already produced one event per iteration of any inner
    # loop, so we read the multiplicity off `n_steps` directly when ALL
    # collective sites in the candidate are graph-inducing. When the
    # candidate has a mix (rare in practice — a candidate either bundles
    # or doesn't), we scale n_steps by the fraction of sites that are
    # graph-inducing. When no site is graph-inducing (bundled), the
    # charge is 0 regardless of n_steps.
    if n_outer_collective_loops > 0 and n_total_coll_sites > 0:
        if n_total_coll_sites == n_graph_inducing_coll_sites:
            n_neff_graphs = n_steps
        else:
            frac = float(n_graph_inducing_coll_sites) / float(n_total_coll_sites)
            n_neff_graphs = int(round(n_steps * frac))
    else:
        n_neff_graphs = 0

    # Doubled for forward + backward (each autograd.Function pass
    # rebuilds the same loop-induced NEFF chain).
    per_graph_total_us = (
        2.0 * n_neff_graphs * float(per_graph_neff_load_us)
    )

    latency += graph_launch_total_us * 1e-6
    latency += per_graph_total_us * 1e-6
    latency += hbm_penalty_us * 1e-6

    # NEFF compilation/load cost amortized over a training run. Driven by
    # the LARGEST single-collective tensor in the algorithm (graph size
    # tracks largest tensor). Scaled by training_scale_bytes_multiplier
    # because correctness-test inputs are smaller than real training tensors.
    compilation_amortized_us = 0.0
    if compilation_cost_samples:
        # NEFF compile-cost driver: the *total* bytes across all collectives
        # in the candidate's single forward NEFF graph, not the max-per-
        # step bytes. A NEFF graph with k AR sites of S bytes each compiles
        # to roughly the same HLO graph complexity as one AR of k*S bytes:
        # graph compile time is dominated by HLO pass structure, not by
        # how the data is split across calls. Using *max-per-step* over-
        # penalises a bundled candidate (one big AR) versus a sequential
        # candidate (k small ARs) that emit equivalent total work in a
        # single forward NEFF.
        per_step_total_bytes = sum(
            s.get("tensor_bytes", 0) for s in profiler.steps)
        per_step_max_bytes = max(
            (s.get("tensor_bytes", 0) for s in profiler.steps), default=0)
        # When the candidate emits multiple NEFFs at training scale
        # (the per_graph_total_us tax fires), each NEFF compiles its own
        # max-per-step graph independently; use the max-per-step bytes
        # instead of the total. The n_outer_collective_loops flag is the
        # signal that this candidate fragments into multiple NEFFs.
        if n_outer_collective_loops > 0 and n_total_coll_sites > 0:
            _compile_bytes = per_step_max_bytes
        else:
            _compile_bytes = per_step_total_bytes
        scaled_max_bytes = _compile_bytes * float(training_scale_bytes_multiplier)
        # Bucketed algorithms cap per-NEFF graph size at bucket_cap regardless
        # of total parameter count. Without this clamp, the log-linear
        # extrapolation past the agent's largest measured sample explodes
        # for bucketed candidates whose actual graph tensor stays small.
        if bucket_cap_bytes is not None and bucket_cap_bytes > 0:
            scaled_max_bytes = min(scaled_max_bytes, float(bucket_cap_bytes))
        # Clamp the bytes value used for compilation-cost interpolation at
        # the largest *non-spike* calibrated sample. The hardware-
        # measurement table includes one threshold sample at the top
        # (e.g. 50 MB -> 7.2 ms then a 100 MB -> 77 ms spike) which
        # represents a *threshold* marker, not the slope of a
        # continuous power law. Above the calibrated range the
        # log-linear extrapolation explodes and reads off this
        # threshold spike; that double-penalises bundled candidates
        # which also pay the HBM-peak penalty (a separate term above).
        # We clamp at the largest sample point whose neff_load_us
        # stays within 2x the floor sample - i.e. the calibrated
        # "flat" region - so the camort term saturates at the flat-
        # region maximum rather than exploding at the spike.
        if scaled_max_bytes > 0:
            _sorted = sorted(compilation_cost_samples,
                              key=lambda s: s.get("tensor_bytes", 0))
            if _sorted:
                _floor_us = float(_sorted[0].get("neff_load_us", 0.0))
                _flat_max_bytes = 0
                for s in _sorted:
                    if float(s.get("neff_load_us", 0.0)) <= 2.0 * _floor_us:
                        _flat_max_bytes = s.get("tensor_bytes", 0)
                if _flat_max_bytes > 0:
                    scaled_max_bytes = min(scaled_max_bytes, float(_flat_max_bytes))
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
        "per_graph_total_us": per_graph_total_us,
        "n_outer_collective_loops": n_outer_collective_loops,
        "n_neff_graphs": n_neff_graphs,
        "hbm_penalty_us": hbm_penalty_us,
        "hbm_peak_bytes_scaled": scaled_peak,
        "hbm_bucket_cap_bytes": bucket_cap_bytes,
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
    """Compare candidate outputs vs reference, recursively over nested
    list structures. # R9-ter v2: fully recursive compare with
    TrackedTensor unwrap at leaves."""

    def _leaf_to_tensor(x):
        # TrackedTensor wraps torch.Tensor; _unwrap returns the inner Tensor.
        if isinstance(x, TrackedTensor):
            return x._t.float()
        if isinstance(x, torch.Tensor):
            return x.float()
        return None  # signal: not tensor-like

    def _cmp(o, e, path):
        if isinstance(e, list):
            if not isinstance(o, list):
                return False, f"{path}: expected list, got {type(o).__name__}"
            if len(o) != len(e):
                return False, f"{path}: expected list of length {len(e)}, got {len(o)}"
            for i, (oo, ee) in enumerate(zip(o, e)):
                ok, err = _cmp(oo, ee, f"{path}[{i}]")
                if not ok:
                    return False, err
            return True, None
        # leaf: must be tensor-like
        o_t = _leaf_to_tensor(o)
        if o_t is None:
            return False, f"{path}: expected tensor, got {type(o).__name__}"
        e_t = e.float()
        if o_t.shape != e_t.shape:
            return False, f"{path}: shape {tuple(o_t.shape)} != {tuple(e_t.shape)}"
        if not torch.allclose(o_t, e_t, atol=1e-3, rtol=1e-3):
            diff = (o_t - e_t).abs().max().item()
            return False, f"{path}: max_diff={diff:.6f}"
        return True, None

    for rank in range(world_size):
        ok, err = _cmp(outputs[rank], expected[rank],
                       f"world={world_size} pattern={pattern} rank={rank}")
        if not ok:
            return False, err
    return True, None


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

        # Compare with bf16-cast expected (recursive over nested lists). # R9-ter v3
        def _bf16_leaf_to_tensor(x):
            if isinstance(x, TrackedTensor):
                return x._t.float()
            if isinstance(x, torch.Tensor):
                return x.float()
            return None

        def _bf16_cmp(o, e, path):
            if isinstance(e, list):
                if not isinstance(o, list):
                    return False, f"BF16 STRUCTURE: {path}: expected list, got {type(o).__name__}"
                if len(o) != len(e):
                    return False, f"BF16 STRUCTURE: {path}: expected list of length {len(e)}, got {len(o)}"
                for i, (oo, ee) in enumerate(zip(o, e)):
                    ok, err = _bf16_cmp(oo, ee, f"{path}[{i}]")
                    if not ok:
                        return False, err
                return True, None
            o_t = _bf16_leaf_to_tensor(o)
            if o_t is None:
                return False, f"BF16 STRUCTURE: {path}: expected tensor, got {type(o).__name__}"
            e_t = e.to(torch.bfloat16).float()
            if o_t.shape != e_t.shape:
                return False, f"BF16 SHAPE MISMATCH: {path}: {tuple(o_t.shape)} != {tuple(e_t.shape)}"
            if not torch.allclose(o_t, e_t, atol=0.1, rtol=0.05):
                diff = (o_t - e_t).abs().max().item()
                return False, f"BF16 VALUE MISMATCH: {path}: max_diff={diff:.6f}"
            return True, None

        for rank in range(ws):
            ok, err = _bf16_cmp(outputs[rank], expected[rank],
                                 f"world={ws} pattern={pattern} rank={rank}")
            if not ok:
                return False, err

    return True, "BF16 correctness tests passed"
