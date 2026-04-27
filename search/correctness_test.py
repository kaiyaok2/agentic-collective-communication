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

_FREE_XLA_OPS = frozenset({
    "view", "reshape", "unsqueeze", "squeeze", "flatten",
    "narrow", "transpose", "permute", "expand", "contiguous",
    "slice",
})

_COLLECTIVE_OPS = frozenset({
    "collective_permute", "all_gather", "all_to_all",
    "reduce_scatter", "all_reduce",
})


class TorchOpCounter:
    """Counts XLA IR ops generated by a candidate."""

    def __init__(self):
        self.ops = []

    def record(self, op_name, *args):
        self.ops.append(op_name)

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
        return TrackedTensor(self._t.contiguous(), self._counter)

    def permute(self, *dims):
        self._counter.record("permute")
        return TrackedTensor(self._t.permute(*dims), self._counter)

    def reshape(self, *shape):
        self._counter.record("reshape")
        return TrackedTensor(self._t.reshape(*shape), self._counter)

    def flatten(self, *args, **kwargs):
        self._counter.record("flatten")
        return TrackedTensor(self._t.flatten(*args, **kwargs), self._counter)

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

    @property
    def tensor(self):
        return self._t

    def __add__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(self._t + other_t, self._counter)

    def __radd__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(other_t + self._t, self._counter)

    def __sub__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(self._t - other_t, self._counter)

    def __rsub__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(other_t - self._t, self._counter)

    def __mul__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(self._t * other_t, self._counter)

    def __rmul__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(other_t * self._t, self._counter)

    def __truediv__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(self._t / other_t, self._counter)

    def __floordiv__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
        return TrackedTensor(self._t // other_t, self._counter)

    def __mod__(self, other):
        other_t = _unwrap(other) if isinstance(other, TrackedTensor) else other
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
        return TrackedTensor(torch.tensor(data, dtype=dtype), self.counter)

    def cat(self, tensors, dim=0):
        self.counter.record("cat")
        unwrapped = [_unwrap(t) for t in tensors]
        return TrackedTensor(torch.cat(unwrapped, dim=dim), self.counter)

    def index_select(self, input, dim, index):
        self.counter.record("index_select")
        inp = _unwrap(input)
        idx = _unwrap(index)
        return TrackedTensor(torch.index_select(inp, dim, idx), self.counter)

    def full(self, size, fill_value, device=None, dtype=None, **kwargs):
        return TrackedTensor(
            torch.full(size, fill_value, dtype=dtype or torch.float32),
            self.counter)

    def arange(self, *args, device=None, dtype=None, **kwargs):
        unwrapped = [_unwrap(a) for a in args]
        return TrackedTensor(
            torch.arange(*unwrapped, dtype=dtype or torch.float32), self.counter)

    def stack(self, tensors, dim=0):
        self.counter.record("stack")
        unwrapped = [_unwrap(t) for t in tensors]
        return TrackedTensor(torch.stack(unwrapped, dim=dim), self.counter)

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
        self.counter.record("reshape")
        inp = _unwrap(input)
        return TrackedTensor(torch.reshape(inp, shape), self.counter)

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
                         dispatch_overhead_s=100e-6, local_ops=0):
        """Estimate latency from collective dispatches + bandwidth + local ops.

        On Trainium, collective dispatch overhead dominates bandwidth for
        typical AllToAllV sizes. This model uses flat per-dispatch costs
        rather than simulating ring steps, matching real hardware behavior
        where all_gather latency is ~1050 us regardless of data size.

        Each collective dispatch costs 2x in a training context because the
        backward pass triggers an implicit inverse collective (e.g.
        all_gather.backward → reduce_scatter, reduce_scatter.backward →
        all_gather). This 2x factor is critical for predicting real training
        throughput rather than forward-only microbenchmark performance.

        Args:
            topology: TrainiumTopology
            local_op_overhead_s: Per-op XLA dispatch overhead (default 29us)
            dispatch_overhead_s: Per-collective dispatch overhead (default 100us)
            local_ops: Number of local torch ops (cat, index_select, etc.)
        """
        total_time = local_ops * local_op_overhead_s

        for step_info in self.steps:
            total_time += 2 * dispatch_overhead_s

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
    latency = profiler.estimate_latency(topology, local_ops=local_ops)
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


def benchmark_xla_candidate_generic(problem, candidate_fn, topology,
                                     send_counts_matrix, world_size,
                                     num_nodes=1,
                                     unsupported_primitives=None,
                                     op_costs=None,
                                     dispatch_overhead_us=100.0):
    """Benchmark any XLA collective candidate using the problem definition.

    Args:
        op_costs: dict mapping op_name -> cost_in_us (from agent profiling).
            If provided, per-op costs are used instead of a flat per-op overhead.
        dispatch_overhead_us: per-collective dispatch overhead in microseconds.
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

    local_ops_list = [op for op in counter.ops if op not in _COLLECTIVE_OPS]
    if op_costs:
        local_cost_us = sum(op_costs.get(op, 29.0) for op in local_ops_list)
    else:
        local_cost_us = counter.real_local_ops * 29.0
    dispatch_s = dispatch_overhead_us * 1e-6
    latency = local_cost_us * 1e-6 + len(profiler.steps) * 2 * dispatch_s
    total_bytes = sum(s.get("tensor_bytes", 0) for s in profiler.steps)
    num_cp = sum(1 for s in profiler.steps
                 if s["type"] in ("collective_permute",
                                  "collective_permute_implicit"))
    num_ag = sum(1 for s in profiler.steps if s["type"] == "all_gather")
    num_rs = sum(1 for s in profiler.steps if s["type"] == "reduce_scatter")
    num_ar = sum(1 for s in profiler.steps if s["type"] == "all_reduce")

    op_breakdown = {}
    for op in local_ops_list:
        cost = op_costs.get(op, 29.0) if op_costs else 29.0
        if op not in op_breakdown:
            op_breakdown[op] = {"count": 0, "per_op_us": cost, "total_us": 0.0}
        op_breakdown[op]["count"] += 1
        op_breakdown[op]["total_us"] += cost

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
    }


def _get_patterns_for_problem(problem):
    """Return appropriate test patterns for each problem type."""
    pattern_map = {
        "alltoallv": ["moe", "uniform", "skewed", "zero_some", "variable"],
        "uniform_a2a": ["uniform", "large", "small", "moe_capacity"],
        "fused_reducescatter": ["uniform", "mixed", "many_small", "few_large", "grad_buckets"],
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
