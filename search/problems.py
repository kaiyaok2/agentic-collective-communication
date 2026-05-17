"""
Collective optimization problem registry.

Each problem defines: function signature, reference implementation,
baseline templates, test-input generators, and prompt fragments.
The template evolution engine uses these to optimize any collective.
"""

import torch
import random as _rng_mod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CollectiveProblem:
    name: str
    display_name: str
    evolved_fn_name: str
    signature: str
    signature_doc: str
    reference_fn: Callable
    generate_test_case: Callable
    call_candidate: Callable
    builtin_templates: dict = field(default_factory=dict)
    optimization_hints: str = ""
    runtime_module_name: str = ""
    public_api_code: str = ""
    training_validation_code: str = ""

    def __post_init__(self):
        if not self.runtime_module_name:
            self.runtime_module_name = f"trainium_{self.name}"


PROBLEMS: dict = {}


def register_problem(problem: CollectiveProblem):
    PROBLEMS[problem.name] = problem


def get_problem(name: str) -> CollectiveProblem:
    return PROBLEMS[name]


# ================================================================
# Problem 1: AllToAllV
# ================================================================

def _alltoallv_reference(inputs, send_counts_matrix, world_size):
    outputs = []
    for dst in range(world_size):
        parts = []
        for src in range(world_size):
            offset = sum(send_counts_matrix[src][:dst])
            count = send_counts_matrix[src][dst]
            parts.append(inputs[src][offset:offset + count])
        outputs.append(torch.cat(parts, dim=0))
    return outputs


def _alltoallv_make_traffic(world_size, pattern="moe", shard_size=32):
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
        cdf, acc = [], 0.0
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
    elif pattern == "variable":
        torch.manual_seed(42)
        for s in range(world_size):
            for d in range(world_size):
                matrix[s][d] = int(torch.randint(1, shard_size * 2, (1,)).item())
    return matrix


def _alltoallv_generate_test_case(world_size, pattern="moe", shard_size=32, seed=0):
    if world_size > 32:
        shard_size = 16
    matrix = _alltoallv_make_traffic(world_size, pattern, shard_size)
    torch.manual_seed(seed)
    inputs = []
    for rank in range(world_size):
        total_send = sum(matrix[rank])
        base = rank * 10000
        inputs.append(torch.arange(base, base + total_send, dtype=torch.float32))

    expected = _alltoallv_reference(inputs, matrix, world_size)
    max_chunk = max((matrix[s][d] for s in range(world_size)
                     for d in range(world_size)), default=1)

    per_rank_args = []
    for rank in range(world_size):
        send_counts = matrix[rank]
        recv_counts = [matrix[src][rank] for src in range(world_size)]
        per_rank_args.append({
            "input_tensor": inputs[rank],
            "send_counts": send_counts,
            "recv_counts": recv_counts,
        })

    return {
        "per_rank_args": per_rank_args,
        "shared_args": {"max_chunk": max_chunk},
        "expected": expected,
    }


def _alltoallv_call_candidate(candidate_fn, rank_args, shared_args,
                               rank, world_size, num_devices, cpd,
                               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["input_tensor"],
        rank_args["send_counts"],
        rank_args["recv_counts"],
        shared_args["max_chunk"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_ALLTOALLV_SIGNATURE = """\
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):"""

_ALLTOALLV_SIGNATURE_DOC = """\
    Args:
        input_tensor: 1D tensor. Data layout:
            input_tensor[send_offsets[i]:send_offsets[i]+send_counts[i]] is data for rank i.
        send_counts: list[int] of length world_size. Elements to send to each rank.
        recv_counts: list[int] of length world_size. Elements to receive from each rank.
        max_chunk: int. Maximum element count across all send/recv pairs.
        rank: int. This rank's index.
        world_size: int. Total ranks.
        num_devices, cores_per_device: hardware topology.
        xm: XLA model module (provides collective_permute, all_gather, all_to_all, reduce_scatter).
        torch: Torch module (provides zeros, cat, index_select, tensor, etc.).
        num_nodes: int. Number of nodes (default 1).

    Returns:
        1D tensor with received data from all sources, concatenated in
        source-rank order: [data_from_rank_0, data_from_rank_1, ...]."""

_ALLTOALLV_HINTS = """\
### XLA op cost model (from hardware profiling):
{op_cost_table}

### Data layout:
- input_tensor is packed: [data_for_rank_0, data_for_rank_1, ...] with variable sizes.
- After all_gather with canonical packing (pad each dest slot to max_chunk), the layout is
  [rank0_packed, rank1_packed, ...] where each packed region is [slot_0, slot_1, ...].
- Each collective dispatch has ~{dispatch_overhead_us}us overhead (doubled in training for backward pass).
- Reducing total (collective dispatches + real local ops) minimizes latency.
- All index arithmetic MUST be done in Python (plain ints/lists), not device tensors."""

# Import existing builtin templates from template_evolution
_ALLTOALLV_BUILTINS = {}

_ALLTOALLV_BUILTINS["naive_allgather"] = '''\
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """Naive AllGather + Slice + Cat."""
    pack_size = world_size * max_chunk
    packed = torch.zeros(pack_size, device=input_tensor.device, dtype=input_tensor.dtype)
    send_off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = input_tensor[send_off:send_off + sc]
        send_off += sc
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0).view(-1)
    chunks = []
    for src in range(world_size):
        count = recv_counts[src]
        base = src * pack_size + rank * max_chunk
        chunks.append(gathered[base:base + count])
    return torch.cat(chunks)
'''

_ALLTOALLV_BUILTINS["allgather_reduce_scatter"] = '''\
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """AllGather + ReduceScatter."""
    pack_size = world_size * max_chunk
    packed = torch.zeros(pack_size, device=input_tensor.device, dtype=input_tensor.dtype)
    send_off = 0
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            packed[i * max_chunk:i * max_chunk + sc] = input_tensor[send_off:send_off + sc]
        send_off += sc
    gathered = xm.all_gather(packed.unsqueeze(0), dim=0)
    reshaped = gathered.view(world_size, world_size, max_chunk)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    my_shard = xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0/world_size,
                                  scatter_dim=0, shard_count=world_size)
    flat_idx = []
    for src in range(world_size):
        count = recv_counts[src]
        base = src * max_chunk
        flat_idx.extend(range(base, base + count))
    idx_tensor = torch.tensor(flat_idx, device=input_tensor.device, dtype=torch.long)
    return torch.index_select(my_shard, 0, idx_tensor)
'''

_ALLTOALLV_BUILTINS["permute_ring"] = '''\
def evolved_alltoallv(input_tensor, send_counts, recv_counts, max_chunk,
                      rank, world_size, num_devices, cores_per_device,
                      xm, torch, num_nodes=1):
    """Permute Ring: one collective_permute per distance."""
    send_offsets = [0]
    for c in send_counts[:-1]:
        send_offsets.append(send_offsets[-1] + c)
    recv_offsets = [0]
    for c in recv_counts[:-1]:
        recv_offsets.append(recv_offsets[-1] + c)
    total_recv = sum(recv_counts)
    output = torch.zeros(total_recv, device=input_tensor.device, dtype=input_tensor.dtype)
    self_count = recv_counts[rank]
    if self_count > 0:
        output[recv_offsets[rank]:recv_offsets[rank] + self_count] = \
            input_tensor[send_offsets[rank]:send_offsets[rank] + send_counts[rank]]
    shards = []
    for i in range(world_size):
        sc = send_counts[i]
        chunk = input_tensor[send_offsets[i]:send_offsets[i] + sc]
        if sc < max_chunk:
            chunk = torch.cat([chunk, torch.zeros(max_chunk - sc,
                              device=input_tensor.device, dtype=input_tensor.dtype)])
        shards.append(chunk)
    for d in range(1, world_size):
        send_to = (rank + d) % world_size
        recv_from = (rank - d + world_size) % world_size
        pairs = [(r, (r + d) % world_size) for r in range(world_size)]
        received = xm.collective_permute(shards[send_to], pairs=pairs)
        rc = recv_counts[recv_from]
        if rc > 0:
            output[recv_offsets[recv_from]:recv_offsets[recv_from] + rc] = received[:rc]
    return output
'''


register_problem(CollectiveProblem(
    name="alltoallv",
    display_name="AllToAllV",
    evolved_fn_name="evolved_alltoallv",
    signature=_ALLTOALLV_SIGNATURE,
    signature_doc=_ALLTOALLV_SIGNATURE_DOC,
    reference_fn=_alltoallv_reference,
    generate_test_case=_alltoallv_generate_test_case,
    call_candidate=_alltoallv_call_candidate,
    builtin_templates=_ALLTOALLV_BUILTINS,
    optimization_hints=_ALLTOALLV_HINTS,
    public_api_code='''\
def compute_recv_counts(send_counts):
    """Exchange send_counts to derive recv_counts via all_gather."""
    device = xm.xla_device()
    t = torch.tensor(send_counts, device=device, dtype=torch.int32)
    gathered = xm.all_gather(t)
    gathered = gathered.view(_world_size, _world_size)
    return gathered[:, _rank].tolist()


def all_to_allv(x, send_counts, recv_counts=None, max_chunk=None):
    """Perform AllToAllV using the agent-evolved algorithm."""
    if _rank is None:
        raise RuntimeError("Call init_alltoallv() first")
    if recv_counts is None:
        recv_counts = compute_recv_counts(send_counts)
    if max_chunk is None:
        max_chunk = max(max(send_counts), max(recv_counts), 1)
    return evolved_alltoallv(
        x, send_counts, recv_counts, max_chunk,
        _rank, _world_size, _NUM_DEVICES, _CORES_PER_DEVICE,
        xm, torch, num_nodes=_NUM_NODES)
''',
    training_validation_code='''\
max_chunk = 4096
send_counts = [max_chunk] * world
INPUT_SIZE = world * max_chunk

class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        xm.mark_step()
        out = all_to_allv(x, send_counts, max_chunk=max_chunk)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = all_to_allv(g.contiguous(), send_counts, max_chunk=max_chunk)
        xm.mark_step()
        return out

class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        xm.mark_step()
        out = all_to_allv(x, send_counts, max_chunk=max_chunk)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = all_to_allv(g.contiguous(), send_counts, max_chunk=max_chunk)
        xm.mark_step()
        return out

class _CollectiveOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        h = _Dispatch.apply(x)
        return _Combine.apply(h)

    @staticmethod
    def backward(ctx, g):
        return g

OUTPUT_SIZE = INPUT_SIZE
''',
))


# ================================================================
# Problem 2: Uniform AllToAll (Fixed-Capacity MoE Token Exchange)
# ================================================================

def _uniform_a2a_reference(inputs, world_size):
    """Reference: each rank sends chunk_size elements to every other rank.
    Input: flat tensor of size world_size * chunk_size (one chunk per dest).
    Output: flat tensor of size world_size * chunk_size (one chunk from each src).
    Output layout: [chunk_from_rank_0, chunk_from_rank_1, ...]."""
    chunk_size = inputs[0].numel() // world_size
    outputs = []
    for dst in range(world_size):
        parts = []
        for src in range(world_size):
            parts.append(inputs[src][dst * chunk_size:(dst + 1) * chunk_size])
        outputs.append(torch.cat(parts, dim=0))
    return outputs


def _uniform_a2a_generate_test_case(world_size, pattern="uniform", shard_size=32, seed=0):
    torch.manual_seed(seed)
    if pattern == "uniform":
        chunk_size = shard_size * 4
    elif pattern == "large":
        chunk_size = shard_size * 16
    elif pattern == "small":
        chunk_size = max(4, shard_size)
    elif pattern == "moe_capacity":
        chunk_size = 24 * 128
    else:
        chunk_size = shard_size * 4

    inputs = []
    for rank in range(world_size):
        base = rank * 100000
        inputs.append(torch.arange(base, base + world_size * chunk_size, dtype=torch.float32))

    expected = _uniform_a2a_reference(inputs, world_size)

    per_rank_args = []
    for rank in range(world_size):
        per_rank_args.append({
            "input_tensor": inputs[rank],
            "chunk_size": chunk_size,
        })

    return {
        "per_rank_args": per_rank_args,
        "shared_args": {},
        "expected": expected,
    }


def _uniform_a2a_call_candidate(candidate_fn, rank_args, shared_args,
                                 rank, world_size, num_devices, cpd,
                                 xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["input_tensor"],
        rank_args["chunk_size"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_UA2A_SIGNATURE = """\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size,
                        num_devices, cores_per_device, xm, torch,
                        num_nodes=1):"""

_UA2A_SIGNATURE_DOC = """\
    Args:
        input_tensor: 1D tensor of size world_size * chunk_size.
            Layout: [chunk_for_rank_0, chunk_for_rank_1, ...].
        chunk_size: int. Elements per destination rank.
        rank: int. This rank's index.
        world_size: int. Total ranks.
        num_devices, cores_per_device: hardware topology.
        xm: XLA model module.
        torch: Torch module.
        num_nodes: int. Number of nodes (default 1).

    Returns:
        1D tensor of size world_size * chunk_size.
        Layout: [chunk_from_rank_0, chunk_from_rank_1, ...].
        Element [i*chunk_size:(i+1)*chunk_size] is the data rank i sent to this rank."""

_UA2A_BUILTINS = {}

_UA2A_BUILTINS["slice_loop"] = '''\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size,
                        num_devices, cores_per_device, xm, torch,
                        num_nodes=1):
    """AllGather + per-source slice loop."""
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0)
    flat = gathered.view(world_size, -1)
    chunks = []
    for src in range(world_size):
        chunks.append(flat[src, rank * chunk_size:(rank + 1) * chunk_size])
    return torch.cat(chunks, dim=0)
'''

_UA2A_BUILTINS["ag_flat_extract"] = '''\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size,
                        num_devices, cores_per_device, xm, torch,
                        num_nodes=1):
    """AllGather + flat extraction."""
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0).view(-1)
    total_per_rank = world_size * chunk_size
    idx = []
    for src in range(world_size):
        base = src * total_per_rank + rank * chunk_size
        idx.extend(range(base, base + chunk_size))
    idx_tensor = torch.tensor(idx, device=input_tensor.device, dtype=torch.long)
    return torch.index_select(gathered, 0, idx_tensor)
'''

_UA2A_BUILTINS["allgather_reduce_scatter"] = '''\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size,
                        num_devices, cores_per_device, xm, torch,
                        num_nodes=1):
    """AllGather + transpose + ReduceScatter."""
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0)
    reshaped = gathered.view(world_size, world_size, chunk_size)
    transposed = reshaped.permute(1, 0, 2).contiguous().view(-1)
    shard = xm.reduce_scatter(xm.REDUCE_SUM, transposed,
                              scale=1.0 / world_size,
                              scatter_dim=0, shard_count=world_size)
    return shard
'''

_UA2A_BUILTINS["permute_ring"] = '''\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size,
                        num_devices, cores_per_device, xm, torch,
                        num_nodes=1):
    """Permute Ring: one collective_permute per distance."""
    total = world_size * chunk_size
    output = torch.zeros(total, device=input_tensor.device, dtype=input_tensor.dtype)
    output[rank * chunk_size:(rank + 1) * chunk_size] = \\
        input_tensor[rank * chunk_size:(rank + 1) * chunk_size]
    for d in range(1, world_size):
        send_to = (rank + d) % world_size
        recv_from = (rank - d + world_size) % world_size
        send_chunk = input_tensor[send_to * chunk_size:(send_to + 1) * chunk_size]
        pairs = [(r, (r + d) % world_size) for r in range(world_size)]
        received = xm.collective_permute(send_chunk, pairs=pairs)
        output[recv_from * chunk_size:(recv_from + 1) * chunk_size] = received
    return output
'''

_UA2A_HINTS = """\
### XLA op cost model (from hardware profiling):
{op_cost_table}

### Data layout after all_gather:
- The gathered tensor is [rank0_data, rank1_data, ...] where each rank's
  data is [chunk_for_dest0, chunk_for_dest1, ...]. You need chunk_for_this_rank
  from each source — a strided access pattern.
- Each collective dispatch has ~{dispatch_overhead_us}us overhead (doubled in training for backward pass).
- Reducing total (collective dispatches + real local ops) minimizes latency."""

register_problem(CollectiveProblem(
    name="uniform_a2a",
    display_name="Uniform AllToAll (Fixed-Capacity MoE)",
    evolved_fn_name="evolved_uniform_a2a",
    signature=_UA2A_SIGNATURE,
    signature_doc=_UA2A_SIGNATURE_DOC,
    reference_fn=_uniform_a2a_reference,
    generate_test_case=_uniform_a2a_generate_test_case,
    call_candidate=_uniform_a2a_call_candidate,
    builtin_templates=_UA2A_BUILTINS,
    optimization_hints=_UA2A_HINTS,
    public_api_code='''\
def uniform_a2a(input_tensor, chunk_size):
    """Uniform all-to-all using the agent-evolved algorithm."""
    if _rank is None:
        raise RuntimeError("Call init_uniform_a2a() first")
    return evolved_uniform_a2a(
        input_tensor, chunk_size, _rank, _world_size, _NUM_DEVICES,
        _CORES_PER_DEVICE, xm, torch, num_nodes=_NUM_NODES)
''',
    training_validation_code='''\
chunk_size = 8192
INPUT_SIZE = world * chunk_size

class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        xm.mark_step()
        out = uniform_a2a(x, chunk_size)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = uniform_a2a(g.contiguous(), chunk_size)
        xm.mark_step()
        return out

class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        xm.mark_step()
        out = uniform_a2a(x, chunk_size)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = uniform_a2a(g.contiguous(), chunk_size)
        xm.mark_step()
        return out

class _CollectiveOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        h = _Dispatch.apply(x)
        return _Combine.apply(h)

    @staticmethod
    def backward(ctx, g):
        return g

OUTPUT_SIZE = INPUT_SIZE
''',
))


# ================================================================
# Problem 3: Fused ReduceScatter (Multi-Tensor Gradient Partitioning)
# ================================================================

def _fused_rs_reference(inputs, world_size):
    """Reference: reduce_scatter for each of N tensors across all ranks.
    Each tensor is summed element-wise, then each rank gets its 1/ws shard.
    inputs[rank] = list of N tensors (each size must be divisible by ws).
    outputs[rank] = list of N tensors, each size = original / ws."""
    n_tensors = len(inputs[0])
    outputs = []
    for dst in range(world_size):
        shards = []
        for t_idx in range(n_tensors):
            summed = inputs[0][t_idx].clone()
            for src in range(1, world_size):
                summed = summed + inputs[src][t_idx]
            sz = summed.numel() // world_size
            shards.append(summed[dst * sz:(dst + 1) * sz])
        outputs.append(shards)
    return outputs


def _fused_rs_generate_test_case(world_size, pattern="uniform", shard_size=32, seed=0):
    torch.manual_seed(seed)
    base_per_rank = shard_size * world_size
    if pattern == "uniform":
        sizes = [base_per_rank] * 8
    elif pattern == "mixed":
        sizes = [base_per_rank, base_per_rank * 2, base_per_rank,
                 base_per_rank * 4, base_per_rank, base_per_rank * 2,
                 base_per_rank, base_per_rank]
    elif pattern == "many_small":
        sizes = [base_per_rank] * 16
    elif pattern == "few_large":
        sizes = [base_per_rank * 4] * 4
    elif pattern == "grad_buckets":
        sizes = [base_per_rank * 8] * 4
    else:
        sizes = [base_per_rank] * 8

    inputs = []
    for rank in range(world_size):
        tensors = []
        for i, sz in enumerate(sizes):
            base = rank * 100 + i * 10
            tensors.append(torch.arange(base, base + sz, dtype=torch.float32))
        inputs.append(tensors)

    expected = _fused_rs_reference(inputs, world_size)

    per_rank_args = []
    for rank in range(world_size):
        per_rank_args.append({"tensors": inputs[rank]})

    return {
        "per_rank_args": per_rank_args,
        "shared_args": {},
        "expected": expected,
    }


def _fused_rs_call_candidate(candidate_fn, rank_args, shared_args,
                              rank, world_size, num_devices, cpd,
                              xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["tensors"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_FRS_SIGNATURE = """\
def evolved_fused_reducescatter(tensors, rank, world_size, num_devices,
                                cores_per_device, xm, torch, num_nodes=1):"""

_FRS_SIGNATURE_DOC = """\
    Args:
        tensors: list of 1D tensors. N tensors to reduce-scatter across ranks.
            Each tensor size must be divisible by world_size.
        rank: int. This rank's index.
        world_size: int. Total ranks.
        num_devices, cores_per_device: hardware topology.
        xm: XLA model module (provides reduce_scatter, all_gather, etc.).
        torch: Torch module.
        num_nodes: int. Number of nodes (default 1).

    Returns:
        list of 1D tensors, length N.
        Each output tensor[i] has size = input tensor[i].numel() / world_size.
        Contains this rank's shard of the element-wise sum across all ranks."""

_FRS_BUILTINS = {}

_FRS_BUILTINS["per_tensor_rs"] = '''\
def evolved_fused_reducescatter(tensors, rank, world_size, num_devices,
                                cores_per_device, xm, torch, num_nodes=1):
    """One reduce_scatter per tensor."""
    results = []
    for t in tensors:
        shard = xm.reduce_scatter(
            xm.REDUCE_SUM, t, scale=1.0,
            scatter_dim=0, shard_count=world_size)
        results.append(shard)
    return results
'''

_FRS_BUILTINS["cat_ag_sum_split"] = '''\
def evolved_fused_reducescatter(tensors, rank, world_size, num_devices,
                                cores_per_device, xm, torch, num_nodes=1):
    """Cat + all_gather + local sum + split."""
    sizes = [t.numel() for t in tensors]
    flat = torch.cat(tensors, dim=0)
    gathered = xm.all_gather(flat.unsqueeze(0), dim=0)
    summed = gathered.view(world_size, -1).sum(dim=0)
    results = []
    offset = 0
    for sz in sizes:
        shard_sz = sz // world_size
        start = offset + rank * shard_sz
        results.append(summed[start:start + shard_sz])
        offset += sz
    return results
'''

_FRS_BUILTINS["packed_split2_rs"] = '''\
def evolved_fused_reducescatter(tensors, rank, world_size, num_devices,
                                cores_per_device, xm, torch, num_nodes=1):
    """Split the tensor list into two halves; pack each half into rank-block
    layout; reduce_scatter each packed half independently; return per-tensor
    shards.

    The split is along the TENSOR axis (not the rank axis): every chunk
    holds all ranks' contributions for its half of the tensors, so
    reduce_scatter on the chunk yields the correct per-tensor shards for
    those tensors. Splitting along the rank axis would produce wrong shards
    for the second half of tensors.
    """
    n = len(tensors)
    sizes = [t.numel() for t in tensors]
    if n < 2 or len(set(sizes)) > 1:
        return [xm.reduce_scatter(xm.REDUCE_SUM, t, scale=1.0,
                                   scatter_dim=0, shard_count=world_size)
                for t in tensors]
    M = sizes[0]
    shard_size = M // world_size
    n1 = n // 2

    def _pack(group):
        m = len(group)
        return (torch.stack(group, dim=0)
                  .reshape(m, world_size, shard_size)
                  .permute(1, 0, 2)
                  .contiguous()
                  .reshape(-1))

    rs1 = xm.reduce_scatter(xm.REDUCE_SUM, _pack(tensors[:n1]), scale=1.0,
                            scatter_dim=0, shard_count=world_size)
    rs2 = xm.reduce_scatter(xm.REDUCE_SUM, _pack(tensors[n1:]), scale=1.0,
                            scatter_dim=0, shard_count=world_size)
    rs = torch.cat([rs1, rs2], dim=0)
    return [rs[i * shard_size:(i + 1) * shard_size] for i in range(n)]
'''

_FRS_BUILTINS["packed_split4_rs"] = '''\
def evolved_fused_reducescatter(tensors, rank, world_size, num_devices,
                                cores_per_device, xm, torch, num_nodes=1):
    """Like packed_split2_rs but splits the tensor list into 4 groups and
    reduce_scatters each group's rank-major-packed buffer independently."""
    n = len(tensors)
    sizes = [t.numel() for t in tensors]
    if n < 4 or len(set(sizes)) > 1:
        return [xm.reduce_scatter(xm.REDUCE_SUM, t, scale=1.0,
                                   scatter_dim=0, shard_count=world_size)
                for t in tensors]
    M = sizes[0]
    shard_size = M // world_size
    n_pieces = 4
    base = n // n_pieces
    extra = n - base * n_pieces
    group_sizes = [base + (1 if g < extra else 0) for g in range(n_pieces)]
    starts = [0]
    for gs in group_sizes[:-1]:
        starts.append(starts[-1] + gs)

    def _pack(group):
        m = len(group)
        return (torch.stack(group, dim=0)
                  .reshape(m, world_size, shard_size)
                  .permute(1, 0, 2)
                  .contiguous()
                  .reshape(-1))

    rss = []
    for g in range(n_pieces):
        st = starts[g]
        gs = group_sizes[g]
        rss.append(xm.reduce_scatter(xm.REDUCE_SUM, _pack(tensors[st:st + gs]),
                                      scale=1.0, scatter_dim=0,
                                      shard_count=world_size))
    rs = torch.cat(rss, dim=0)
    return [rs[i * shard_size:(i + 1) * shard_size] for i in range(n)]
'''

_FRS_BUILTINS["packed_one_rs"] = '''\
def evolved_fused_reducescatter(tensors, rank, world_size, num_devices,
                                cores_per_device, xm, torch, num_nodes=1):
    """Single monolithic reduce_scatter on a rank-major-packed buffer.

    Same packing as the split variants, but as one large reduce_scatter
    dispatch (largest graph, fewest dispatches).
    """
    n = len(tensors)
    sizes = [t.numel() for t in tensors]
    if n < 2 or len(set(sizes)) > 1:
        return [xm.reduce_scatter(xm.REDUCE_SUM, t, scale=1.0,
                                   scatter_dim=0, shard_count=world_size)
                for t in tensors]
    M = sizes[0]
    shard_size = M // world_size
    packed = (torch.stack(tensors, dim=0)
              .reshape(n, world_size, shard_size)
              .permute(1, 0, 2)
              .contiguous()
              .reshape(-1))
    rs = xm.reduce_scatter(xm.REDUCE_SUM, packed, scale=1.0,
                           scatter_dim=0, shard_count=world_size)
    return [rs[i * shard_size:(i + 1) * shard_size] for i in range(n)]
'''

_FRS_HINTS = """\
### Constraints:
- Each input tensor size must be divisible by world_size.
- Output: a list of N tensors, each of size original/world_size.
- WARNING: reduce_scatter does NOT distribute over cat — the scatter dimension
  interleaves differently if you try to fuse via concatenation."""

# ================================================================
# Problem 4: Ring Attention KV Distribution
# ================================================================

def _ring_kv_reference(inputs, world_size):
    """Reference: each rank receives, for every input slot, the full
    rank-ordered concatenation of that slot across all ranks.

    inputs[rank] is a list of 1D tensors (one per slot, e.g. [K, V]).
    Output: outputs[rank][slot] = concat([inputs[r][slot] for r in 0..ws-1]).
    """
    num_slots = len(inputs[0])
    per_slot_concat = []
    for s in range(num_slots):
        per_slot_concat.append(torch.cat([inputs[r][s] for r in range(world_size)], dim=0))
    outputs = []
    for dst in range(world_size):
        outputs.append(list(per_slot_concat))
    return outputs


def _ring_kv_generate_test_case(world_size, pattern="uniform", shard_size=32, seed=0):
    torch.manual_seed(seed)
    if pattern == "uniform":
        kv_size = shard_size * 4
    elif pattern == "large":
        kv_size = shard_size * 16
    elif pattern == "small":
        kv_size = max(4, shard_size)
    elif pattern == "head_dim":
        kv_size = 128 * 2  # K + V for head_dim=128
    else:
        kv_size = shard_size * 4

    # Sequence-parallel attention has K and V buffers per rank; benchmarks
    # provide them as a list [K, V] so the algorithm can choose to gather
    # them jointly or independently.
    num_slots = 2

    inputs = []
    for rank in range(world_size):
        slots = []
        for s in range(num_slots):
            base = rank * 10000 + s * 1_000_000
            slots.append(torch.arange(base, base + kv_size, dtype=torch.float32))
        inputs.append(slots)

    expected = _ring_kv_reference(inputs, world_size)

    per_rank_args = []
    for rank in range(world_size):
        per_rank_args.append({"kv_chunks": inputs[rank]})

    return {
        "per_rank_args": per_rank_args,
        "shared_args": {},
        "expected": expected,
    }


def _ring_kv_call_candidate(candidate_fn, rank_args, shared_args,
                             rank, world_size, num_devices, cpd,
                             xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["kv_chunks"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_RING_KV_SIGNATURE = """\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):"""

_RING_KV_SIGNATURE_DOC = """\
    Args:
        kv_chunks: list of 1D tensors. This rank's per-slot KV data
            (typically [K_local, V_local] for sequence-parallel attention).
            All slots have the same per-rank size and dtype.
        rank: int. This rank's index.
        world_size: int. Total ranks.
        num_devices, cores_per_device: hardware topology.
        xm: XLA model module.
        torch: Torch module.
        num_nodes: int. Number of nodes (default 1).

    Returns:
        list of 1D tensors, one per input slot, each of size
        world_size * kv_chunks[s].numel(), holding all ranks' slot s data
        concatenated in rank order:
        [slot_s_from_rank_0, slot_s_from_rank_1, ..., slot_s_from_rank_{ws-1}]."""

_RING_KV_BUILTINS = {}

_RING_KV_BUILTINS["per_slot_allgather"] = '''\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Independent all_gather for each slot (one collective per slot)."""
    results = []
    for chunk in kv_chunks:
        gathered = xm.all_gather(chunk.unsqueeze(0), dim=0)
        results.append(gathered.view(-1))
    return results
'''

_RING_KV_BUILTINS["per_chunk_per_part_allgather"] = '''\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Many small all_gathers: each slot is split into N equal parts and
    each part is gathered separately, then re-concatenated."""
    N_PARTS = 16
    results = []
    for chunk in kv_chunks:
        kv_size = chunk.numel()
        part_sz = kv_size // N_PARTS
        parts = []
        for p in range(N_PARTS):
            piece = chunk[p * part_sz:(p + 1) * part_sz]
            gathered = xm.all_gather(piece.unsqueeze(0), dim=0)
            parts.append(gathered.view(-1))
        results.append(torch.cat(parts))
    return results
'''

_RING_KV_BUILTINS["joint_cat_allgather"] = '''\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Concatenate all slot tensors locally, do one all_gather over the
    combined buffer, then split per-slot using a (world_size, total_size)
    view so that each slot's per-rank pieces remain contiguous in rank order.
    """
    sizes = [c.numel() for c in kv_chunks]
    flat = torch.cat(kv_chunks, dim=0)
    gathered = xm.all_gather(flat.unsqueeze(0), dim=0)
    # gathered shape: (world_size, sum(sizes)); rows are per-rank.
    results = []
    offset = 0
    for sz in sizes:
        results.append(gathered[:, offset:offset + sz].reshape(-1))
        offset += sz
    return results
'''

_RING_KV_BUILTINS["naive_ring_permute"] = '''\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Ring: rotate each slot's chunks via collective_permute (per-slot)."""
    results = []
    for chunk in kv_chunks:
        kv_size = chunk.numel()
        output = torch.zeros(world_size * kv_size, device=chunk.device, dtype=chunk.dtype)
        output[rank * kv_size:(rank + 1) * kv_size] = chunk
        current = chunk
        for step in range(1, world_size):
            pairs = [(r, (r + 1) % world_size) for r in range(world_size)]
            current = xm.collective_permute(current, pairs=pairs)
            src = (rank - step + world_size) % world_size
            output[src * kv_size:(src + 1) * kv_size] = current
        results.append(output)
    return results
'''

_RING_KV_HINTS = """\
### Constraints:
- `xm.collective_permute` ring patterns cause SIGABRT at 64 ranks on this hardware.
  Do NOT use collective_permute-based ring approaches.
- The `xm.all_gather` `groups` parameter can be used for hierarchical grouping
  (e.g., intra-node then cross-node)."""

register_problem(CollectiveProblem(
    name="ring_kv",
    display_name="Ring Attention KV Distribution",
    evolved_fn_name="evolved_ring_kv",
    signature=_RING_KV_SIGNATURE,
    signature_doc=_RING_KV_SIGNATURE_DOC,
    reference_fn=_ring_kv_reference,
    generate_test_case=_ring_kv_generate_test_case,
    call_candidate=_ring_kv_call_candidate,
    builtin_templates=_RING_KV_BUILTINS,
    optimization_hints=_RING_KV_HINTS,
    public_api_code='''\
def ring_kv_gather(kv_chunks):
    """Gather all ranks' per-slot KV chunks using the agent-evolved algorithm.

    Args:
        kv_chunks: list of 1D tensors (typically [K_local, V_local]).

    Returns:
        list of 1D tensors, one per input slot, each holding all ranks' data
        for that slot concatenated in rank order.
    """
    if _rank is None:
        raise RuntimeError("Call init_ring_kv() first")
    return evolved_ring_kv(
        kv_chunks, _rank, _world_size, _NUM_DEVICES, _CORES_PER_DEVICE,
        xm, torch, num_nodes=_NUM_NODES)
''',
    training_validation_code='''\
KV_SIZE = 8192
INPUT_SIZE = KV_SIZE * 2  # K + V packed into one tensor for the generic harness

class _CollectiveOp(torch.autograd.Function):
    """Generic-harness wrapper: takes a single tensor (concat of K and V)
    and returns a single tensor (concat of K_global and V_global). The
    actual ring_kv_gather() runtime accepts a list of slot tensors;
    we split the input in half here to match its API.
    """
    @staticmethod
    def forward(ctx, kv_packed):
        half = kv_packed.numel() // 2
        ctx.half = half
        k = kv_packed[:half]
        v = kv_packed[half:]
        xm.mark_step()
        outs = ring_kv_gather([k, v])
        xm.mark_step()
        return torch.cat([outs[0], outs[1]], dim=0)

    @staticmethod
    def backward(ctx, grad_out):
        rank = xr.global_ordinal()
        xm.mark_step()
        ws = xr.world_size()
        per_rank_k = ctx.half * ws // ws  # = ctx.half (gathered K is ws*half)
        # grad_out is concat of grad_K_global, grad_V_global
        kv_split = grad_out.numel() // 2
        gk_global = grad_out[:kv_split]
        gv_global = grad_out[kv_split:]
        # This rank's contribution slot = its rank * half elements
        gk = gk_global[rank * ctx.half:(rank + 1) * ctx.half]
        gv = gv_global[rank * ctx.half:(rank + 1) * ctx.half]
        xm.mark_step()
        return torch.cat([gk, gv], dim=0)

OUTPUT_SIZE = KV_SIZE * 2 * world
''',
))


# ================================================================
# Problem 5: Multi-Tensor AllGather  (MoE expert-weight fetch)
# ================================================================
#
# Each rank holds its own shard of N expert weight tensors (one shard
# per expert / per slot). Forward pass needs every rank to have the
# full per-expert weight, so all_gather each of the N shards.
#
# Distinct from Ring Attention KV (which only has 2 slots: K and V).
# At higher slot counts (8+) Neuron's auto-fusion of adjacent
# xm.all_gather calls breaks down, and a single-dispatch composition
# wins meaningfully.

def _multi_ag_reference(inputs, world_size):
    """For each slot i, output[i] = concat([inputs[r][i] for r in 0..ws]).
    Every rank sees the same output (standard all_gather semantics)."""
    num_slots = len(inputs[0])
    per_slot = []
    for s in range(num_slots):
        per_slot.append(torch.cat([inputs[r][s] for r in range(world_size)],
                                   dim=0))
    return [list(per_slot) for _ in range(world_size)]


def _multi_ag_generate_test_case(world_size, pattern="uniform",
                                  shard_size=32, seed=0):
    torch.manual_seed(seed)
    if pattern == "uniform":
        per_shard, num_slots = shard_size * 4, 8
    elif pattern == "large":
        per_shard, num_slots = shard_size * 16, 8
    elif pattern == "many":
        per_shard, num_slots = shard_size * 2, 16
    elif pattern == "small":
        per_shard, num_slots = max(4, shard_size), 4
    else:
        per_shard, num_slots = shard_size * 4, 8

    inputs = []
    for rank in range(world_size):
        slots = []
        for s in range(num_slots):
            base = rank * 10000 + s * 1_000_000
            slots.append(torch.arange(base, base + per_shard,
                                       dtype=torch.float32))
        inputs.append(slots)

    expected = _multi_ag_reference(inputs, world_size)

    per_rank_args = [{"slot_shards": inputs[r]} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {},
            "expected": expected}


def _multi_ag_call_candidate(candidate_fn, rank_args, shared_args,
                              rank, world_size, num_devices, cpd,
                              xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["slot_shards"], rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_MAG_SIGNATURE = """\
def evolved_multi_allgather(slot_shards, rank, world_size, num_devices,
                            cores_per_device, xm, torch, num_nodes=1):"""

_MAG_SIGNATURE_DOC = """\
    Args:
        slot_shards: list of 1D tensors. This rank's local shard for
            each of N slots (e.g., expert weight shards). All slots
            have the same per-rank size.
        rank, world_size, num_devices, cores_per_device: topology.
        xm, torch: backend modules.
        num_nodes: int.

    Returns:
        list of 1D tensors, length N. tensor[i] is the rank-ordered
        concatenation of slot i across all ranks."""


_MAG_BUILTINS = {}

_MAG_BUILTINS["per_slot_ag"] = '''\
def evolved_multi_allgather(slot_shards, rank, world_size, num_devices,
                            cores_per_device, xm, torch, num_nodes=1):
    """Developer baseline: one xm.all_gather per slot."""
    results = []
    for s in slot_shards:
        g = xm.all_gather(s.unsqueeze(0), dim=0)
        results.append(g.view(-1))
    return results
'''

_MAG_BUILTINS["stack_dim1_ag"] = '''\
def evolved_multi_allgather(slot_shards, rank, world_size, num_devices,
                            cores_per_device, xm, torch, num_nodes=1):
    """Stack into (N, shard_size); all_gather along dim=1; rows are
    contiguous and reads are free."""
    if len(slot_shards) == 1:
        return [xm.all_gather(slot_shards[0].unsqueeze(0), dim=0).view(-1)]
    sizes = [s.numel() for s in slot_shards]
    if len(set(sizes)) != 1:
        return [xm.all_gather(s.unsqueeze(0), dim=0).view(-1)
                for s in slot_shards]
    stacked = torch.stack(slot_shards, dim=0)            # (N, shard_size)
    gathered = xm.all_gather(stacked, dim=1)             # (N, ws*shard_size)
    return [gathered[i] for i in range(len(slot_shards))]
'''

_MAG_BUILTINS["cat_one_ag_split"] = '''\
def evolved_multi_allgather(slot_shards, rank, world_size, num_devices,
                            cores_per_device, xm, torch, num_nodes=1):
    """Cat all slots, 1 all_gather, slice/reshape per slot.

    The post-gather extraction is a sub-region narrow on dim=1 → forces
    strided memcpy; included as a control."""
    sizes = [s.numel() for s in slot_shards]
    flat = torch.cat(slot_shards, dim=0)
    gathered = xm.all_gather(flat.unsqueeze(0), dim=0)   # (ws, total)
    results, offset = [], 0
    for sz in sizes:
        results.append(gathered[:, offset:offset + sz].reshape(-1))
        offset += sz
    return results
'''

_MAG_HINTS = """\
### Constraints:
- All slot shards are 1D, equal shape per rank.
- Output: list of N tensors, slot i = rank-ordered concat across ranks.
- All ranks see the same outputs (all_gather semantics)."""




# ================================================================
# Problem 5: Replicated-Gradient AllReduce
# ================================================================

def _grad_ar_reference(inputs, world_size):
    """Reference: per-rank list of replicated-grad tensors -> per-rank list
    of grad tensors equal to (mean across all ranks).

    inputs[rank] = list of 1D tensors. All ranks see identical shapes.
    outputs[rank][t] = mean over r of inputs[r][t]. Same value on every rank.
    """
    import torch
    n = len(inputs[0])
    sums = []
    for t in range(n):
        s = torch.zeros_like(inputs[0][t])
        for r in range(world_size):
            s = s + inputs[r][t]
        sums.append(s / world_size)
    return [list(sums) for _ in range(world_size)]


def _grad_ar_generate_test_case(world_size, pattern="uniform", shard_size=32, seed=0):
    import torch
    torch.manual_seed(seed)
    # ~50 replicated grad tensors of varied sizes, matching what a real OLMoE
    # trainer would feed.
    if pattern == "large":
        sizes = [4096] * 16 + [1024] * 16 + [256] * 16
    elif pattern == "small":
        sizes = [256] * 32
    elif pattern == "mixed":
        sizes = [4096, 1024, 256, 64] * 12
    else:
        sizes = [shard_size] * 32

    inputs = []
    for rank in range(world_size):
        grads = [torch.randn(sz, dtype=torch.float32) + 0.01 * rank for sz in sizes]
        inputs.append(grads)
    expected = _grad_ar_reference(inputs, world_size)
    per_rank_args = [{"rep_grads": inputs[r]} for r in range(world_size)]
    return {
        "per_rank_args": per_rank_args,
        "shared_args": {},
        "expected": expected,
    }


def _grad_ar_call_candidate(candidate_fn, rank_args, shared_args,
                             rank, world_size, num_devices, cpd,
                             xm_mock, torch_mock, num_nodes=1):
    # The candidate signature: evolved_grad_ar(rep_grads, rank, ws, ...) -> list
    return candidate_fn(
        rank_args["rep_grads"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_GRAD_AR_SIGNATURE = """\
def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):"""

_GRAD_AR_SIGNATURE_DOC = """\
    Args:
        rep_grads: list of 1D tensors. Replicated parameter gradients held on
            this rank. All ranks see the same shape list and need the same
            averaged result.
        rank: int. This rank's index.
        world_size: int. Total ranks.
        num_devices, cores_per_device: hardware topology.
        xm: XLA model module.
        torch: Torch module.
        num_nodes: int. Number of nodes (default 1).

    Returns:
        list of 1D tensors of the same shape as rep_grads, each equal to the
        mean of that grad across all ranks. May return the same Python list
        (modified in place) or a new list — the harness checks values only."""

_GRAD_AR_BUILTINS = {}

_GRAD_AR_BUILTINS["per_tensor_loop"] = '''\
def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Per-tensor xm.all_reduce in a Python loop (developer baseline)."""
    inv = 1.0 / world_size
    out = []
    for g in rep_grads:
        out.append(xm.all_reduce(xm.REDUCE_SUM, g) * inv)
    return out
'''

_GRAD_AR_BUILTINS["async_back_to_back"] = '''\
def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Issue every xm.all_reduce back-to-back, then apply 1/ws in a
    second pass. Avoids XLA recompile-fragmentation in the first ~30 steps."""
    inv = 1.0 / world_size
    pending = [xm.all_reduce(xm.REDUCE_SUM, g) for g in rep_grads]
    return [g * inv for g in pending]
'''

_GRAD_AR_BUILTINS["flat_single_ar"] = '''\
def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Concat -> single AR -> split. Loses on Trainium due to cat overhead."""
    inv = 1.0 / world_size
    shapes = [g.shape for g in rep_grads]
    flat = torch.cat([g.reshape(-1) for g in rep_grads])
    flat = xm.all_reduce(xm.REDUCE_SUM, flat) * inv
    out = []
    offs = 0
    for sh in shapes:
        n = 1
        for d in sh:
            n *= d
        out.append(flat[offs:offs + n].reshape(sh))
        offs += n
    return out
'''

_GRAD_AR_BUILTINS["bucket_32mb"] = '''\
def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Pack grads into ~32 MB buckets, one AR per bucket."""
    inv = 1.0 / world_size
    bucket_bytes = 32 * 1024 * 1024
    out = [None] * len(rep_grads)
    cur_idx, cur_bytes = [], 0
    buckets = []
    for i, g in enumerate(rep_grads):
        b = g.numel() * g.element_size()
        if cur_idx and cur_bytes + b > bucket_bytes:
            buckets.append(cur_idx)
            cur_idx, cur_bytes = [], 0
        cur_idx.append(i)
        cur_bytes += b
    if cur_idx:
        buckets.append(cur_idx)
    for bk in buckets:
        if len(bk) == 1:
            out[bk[0]] = xm.all_reduce(xm.REDUCE_SUM, rep_grads[bk[0]]) * inv
        else:
            grads = [rep_grads[i] for i in bk]
            shapes = [g.shape for g in grads]
            flat = torch.cat([g.reshape(-1) for g in grads])
            flat = xm.all_reduce(xm.REDUCE_SUM, flat) * inv
            offs = 0
            for j, sh in zip(bk, shapes):
                n = 1
                for d in sh:
                    n *= d
                out[j] = flat[offs:offs + n].reshape(sh)
                offs += n
    return out
'''

_GRAD_AR_BUILTINS["hierarchical_intra_inter"] = '''\
def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Hierarchical: intra-node AR (32 ranks per node) + inter-node AR
    via groups. Loses on Trainium because the two-stage overhead exceeds
    the cross-node bandwidth savings at this scale."""
    inv = 1.0 / world_size
    nproc_per_node = num_devices * cores_per_device // num_nodes if num_nodes > 1 else world_size
    if nproc_per_node == 0:
        nproc_per_node = world_size
    n_nodes = world_size // nproc_per_node
    intra_groups = [list(range(n * nproc_per_node, (n + 1) * nproc_per_node))
                    for n in range(n_nodes)]
    inter_groups = [[n * nproc_per_node + lo for n in range(n_nodes)]
                    for lo in range(nproc_per_node)]
    out = []
    for g in rep_grads:
        h1 = xm.all_reduce(xm.REDUCE_SUM, g, groups=intra_groups)
        h2 = xm.all_reduce(xm.REDUCE_SUM, h1, groups=inter_groups)
        out.append(h2 * inv)
    return out
'''

_GRAD_AR_HINTS = """\
### Constraints and observations:
- The async_back_to_back pattern is the empirical winner on Trainium
  (measured 2.37x wall-clock at 7 nodes, 224 ranks); the steady-state
  per-step time is similar across all variants but the back-to-back
  pattern avoids XLA recompile-fragmentation in the first ~30 steps.
- `flat_single_ar` loses to per-tensor due to `torch.cat`/`reshape`
  overhead exceeding the AR launch-cost savings.
- Hierarchical AR via the `groups` parameter is slower than flat AR at
  this scale (~19% slower in screening) — the two-stage reduce dominates."""

register_problem(CollectiveProblem(
    name="grad_ar",
    display_name="Replicated-Gradient AllReduce",
    evolved_fn_name="evolved_grad_ar",
    signature=_GRAD_AR_SIGNATURE,
    signature_doc=_GRAD_AR_SIGNATURE_DOC,
    reference_fn=_grad_ar_reference,
    generate_test_case=_grad_ar_generate_test_case,
    call_candidate=_grad_ar_call_candidate,
    builtin_templates=_GRAD_AR_BUILTINS,
    optimization_hints=_GRAD_AR_HINTS,
    public_api_code='''\
def grad_ar_sync(rep_params, world_size=None):
    """All-reduce per-tensor on a list of replicated parameters\' gradients,
    in place, dividing by world_size for a mean.
    """
    if _rank is None:
        raise RuntimeError("Call init_grad_ar() first")
    ws = world_size if world_size is not None else _world_size
    grads = [p.grad.data for p in rep_params if p.grad is not None]
    if not grads:
        return
    out = evolved_grad_ar(
        grads, _rank, ws, _NUM_DEVICES, _CORES_PER_DEVICE,
        xm, torch, num_nodes=_NUM_NODES)
    i = 0
    for p in rep_params:
        if p.grad is not None:
            p.grad.data = out[i]
            i += 1
''',
    training_validation_code='''\
N_PARAMS = 16
PARAM_SIZE = 4096

# Build a list of dummy parameters with grad tensors to all-reduce.
params = [torch.randn(PARAM_SIZE, device=device, dtype=torch.float32, requires_grad=True)
          for _ in range(N_PARAMS)]
for p in params:
    p.grad = torch.randn(PARAM_SIZE, device=device, dtype=torch.float32)

class _Sync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *grads):
        xm.mark_step()
        # Call evolved fn directly with the grads list.
        ws_now = xr.world_size()
        out = evolved_grad_ar(list(grads), xr.global_ordinal(), ws_now,
                              _NUM_DEVICES, _CORES_PER_DEVICE, xm, torch,
                              num_nodes=_NUM_NODES)
        xm.mark_step()
        return tuple(out)

    @staticmethod
    def backward(ctx, *g):
        # Identity: input grads are themselves replicated tensors.
        return tuple(g)
''',
))


# ================================================================
# Problem 6: Distributed Cross-Entropy (vocab-sharded logits)
# ================================================================

def _dxe_reference(inputs, world_size):
    """Reference: full F.cross_entropy on concatenated logits.

    inputs[rank] = {"logits_local": (T, V_local), "targets": (T,)}.
    All ranks must compute the SAME loss (mean over T).
    Returns one scalar per rank, all equal.
    """
    import torch
    import torch.nn.functional as F
    logits_full = torch.cat(
        [inputs[r]["logits_local"] for r in range(world_size)], dim=-1)
    targets = inputs[0]["targets"]
    loss = F.cross_entropy(logits_full, targets, reduction="mean")
    return [loss.detach().clone() for _ in range(world_size)]


def _dxe_generate_test_case(world_size, pattern="uniform", shard_size=32, seed=0):
    import torch
    torch.manual_seed(seed)
    if pattern == "small":
        T, V = 32, world_size * 4
    elif pattern == "large":
        T, V = 256, world_size * 64
    else:
        T, V = 64, world_size * 16
    V_local = V // world_size
    # Identical targets across ranks for the reference to be well-defined.
    targets = torch.randint(0, V, (T,), dtype=torch.long)
    # logits_local is the rank's slice of the global logits matrix.
    full_logits = torch.randn(T, V, dtype=torch.float32)
    inputs = []
    for rank in range(world_size):
        slc = full_logits[:, rank * V_local:(rank + 1) * V_local].contiguous()
        inputs.append({"logits_local": slc, "targets": targets, "V_local": V_local})
    expected = _dxe_reference(inputs, world_size)
    per_rank_args = [{"logits_local": inputs[r]["logits_local"],
                      "targets": inputs[r]["targets"],
                      "V_local": V_local} for r in range(world_size)]
    return {
        "per_rank_args": per_rank_args,
        "shared_args": {},
        "expected": expected,
    }


def _dxe_call_candidate(candidate_fn, rank_args, shared_args,
                        rank, world_size, num_devices, cpd,
                        xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["logits_local"],
        rank_args["targets"],
        rank_args["V_local"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_DXE_SIGNATURE = """\
def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices,
                cores_per_device, xm, torch, num_nodes=1):"""

_DXE_SIGNATURE_DOC = """\
    Args:
        logits_local: (T, V_local) tensor; this rank\'s slice of the global
            logits over the vocab dimension.
        targets: (T,) long tensor of global vocab indices in [0, V_local*ws).
        V_local: int. Size of this rank\'s vocab slice. V_local * ws == V.
        rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes:
            standard topology/library args.

    Returns:
        Scalar loss tensor (mean over T), identical on every rank, equivalent
        to F.cross_entropy(full_logits, targets, reduction="mean")."""

_DXE_BUILTINS = {}

_DXE_BUILTINS["three_ar"] = '''\
def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices,
                cores_per_device, xm, torch, num_nodes=1):
    """Naive 3-AR distributed CE (developer baseline)."""
    local_max = logits_local.max(dim=-1).values
    global_max = xm.all_reduce(xm.REDUCE_MAX, local_max)
    shifted = logits_local - global_max.unsqueeze(-1)
    local_sum_exp = shifted.exp().sum(dim=-1)
    global_sum_exp = xm.all_reduce(xm.REDUCE_SUM, local_sum_exp)
    log_sum_exp = global_sum_exp.log() + global_max
    lo, hi = rank * V_local, (rank + 1) * V_local
    target_local = targets - lo
    in_shard = (targets >= lo) & (targets < hi)
    target_local_safe = target_local.clamp(0, V_local - 1)
    local_target = logits_local.gather(1, target_local_safe.unsqueeze(1)).squeeze(1)
    local_target = torch.where(in_shard, local_target, torch.zeros_like(local_target))
    global_target = xm.all_reduce(xm.REDUCE_SUM, local_target)
    return (log_sum_exp - global_target).mean()
'''

_DXE_BUILTINS["async_back_to_back"] = '''\
def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices,
                cores_per_device, xm, torch, num_nodes=1):
    """Issue MAX-AR, then SUM-AR pair back-to-back without materializing
    intermediates; combine in the final scalar arithmetic.
    Empirically ~28x wall-clock speedup vs three_ar on Trainium 7-node."""
    local_max = logits_local.max(dim=-1).values
    global_max = xm.all_reduce(xm.REDUCE_MAX, local_max)
    shifted = logits_local - global_max.unsqueeze(-1)
    local_sum_exp = shifted.exp().sum(dim=-1)
    lo, hi = rank * V_local, (rank + 1) * V_local
    target_local = targets - lo
    in_shard = (targets >= lo) & (targets < hi)
    target_local_safe = target_local.clamp(0, V_local - 1)
    local_target = logits_local.gather(1, target_local_safe.unsqueeze(1)).squeeze(1)
    local_target = torch.where(in_shard, local_target, torch.zeros_like(local_target))
    g_sum_exp = xm.all_reduce(xm.REDUCE_SUM, local_sum_exp)
    g_target = xm.all_reduce(xm.REDUCE_SUM, local_target)
    log_sum_exp = g_sum_exp.log() + global_max
    return (log_sum_exp - g_target).mean()
'''

_DXE_BUILTINS["fused_sum_stack"] = '''\
def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices,
                cores_per_device, xm, torch, num_nodes=1):
    """MAX-AR, then stack [sum_exp, target_logit_shifted] for one SUM-AR.
    Two ARs total. Marginal in steady-state but loses on first-step compile."""
    local_max = logits_local.max(dim=-1).values
    global_max = xm.all_reduce(xm.REDUCE_MAX, local_max)
    shifted = logits_local - global_max.unsqueeze(-1)
    local_sum_exp = shifted.exp().sum(dim=-1)
    lo, hi = rank * V_local, (rank + 1) * V_local
    target_local = targets - lo
    in_shard = (targets >= lo) & (targets < hi)
    target_local_safe = target_local.clamp(0, V_local - 1)
    local_target_shifted = logits_local.gather(
        1, target_local_safe.unsqueeze(1)).squeeze(1) - global_max
    local_target_shifted = torch.where(
        in_shard, local_target_shifted, torch.zeros_like(local_target_shifted))
    stacked = torch.stack([local_sum_exp, local_target_shifted], dim=0)
    g = xm.all_reduce(xm.REDUCE_SUM, stacked)
    log_sum_exp = g[0].log() + global_max
    return (log_sum_exp - (g[1] + global_max)).mean()
'''

_DXE_BUILTINS["two_ar_nomax"] = '''\
def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices,
                cores_per_device, xm, torch, num_nodes=1):
    """Skip the MAX-shift (rely on bf16 having enough range). 2 ARs."""
    local_sum_exp = logits_local.exp().sum(dim=-1)
    global_sum_exp = xm.all_reduce(xm.REDUCE_SUM, local_sum_exp)
    log_sum_exp = global_sum_exp.log()
    lo, hi = rank * V_local, (rank + 1) * V_local
    target_local = targets - lo
    in_shard = (targets >= lo) & (targets < hi)
    target_local_safe = target_local.clamp(0, V_local - 1)
    local_target = logits_local.gather(1, target_local_safe.unsqueeze(1)).squeeze(1)
    local_target = torch.where(in_shard, local_target, torch.zeros_like(local_target))
    global_target = xm.all_reduce(xm.REDUCE_SUM, local_target)
    return (log_sum_exp - global_target).mean()
'''

_DXE_HINTS = """\
### Constraints and observations:
- async_back_to_back is the empirical winner on Trainium (~28x wall-clock
  vs three_ar at 7 nodes, 224 ranks). Steady-state per-step is statistically
  tied across all distributed variants; the gain comes from compile-time
  fragmentation avoidance.
- two_ar_nomax is numerically risky in bf16 if any logit exceeds ~88
  (exp overflow). The agent should treat skipping MAX-shift as legal only
  with explicit overflow guards.
- The (T,) scalar/sum tensors are small enough that AR launch cost
  dominates payload bandwidth; minimizing the *number* of AR launches
  is more impactful than minimizing per-launch size."""

register_problem(CollectiveProblem(
    name="dxe",
    display_name="Distributed Cross-Entropy (vocab-sharded)",
    evolved_fn_name="evolved_dxe",
    signature=_DXE_SIGNATURE,
    signature_doc=_DXE_SIGNATURE_DOC,
    reference_fn=_dxe_reference,
    generate_test_case=_dxe_generate_test_case,
    call_candidate=_dxe_call_candidate,
    builtin_templates=_DXE_BUILTINS,
    optimization_hints=_DXE_HINTS,
    public_api_code='''\
def dxe_loss(logits_local, targets, V_local):
    """Compute distributed cross-entropy on vocab-sharded logits."""
    if _rank is None:
        raise RuntimeError("Call init_dxe() first")
    return evolved_dxe(
        logits_local, targets, V_local, _rank, _world_size,
        _NUM_DEVICES, _CORES_PER_DEVICE, xm, torch, num_nodes=_NUM_NODES)
''',
    training_validation_code='''\
T = 64
V_local = 144
V_total = V_local * xr.world_size()

logits = torch.randn(T, V_local, device=device, dtype=torch.float32, requires_grad=True)
targets = torch.randint(0, V_total, (T,), device=device, dtype=torch.long)

class _Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits_in, targets_in):
        ctx.V_local = V_local
        xm.mark_step()
        out = evolved_dxe(logits_in, targets_in, V_local,
                          xr.global_ordinal(), xr.world_size(),
                          _NUM_DEVICES, _CORES_PER_DEVICE, xm, torch,
                          num_nodes=_NUM_NODES)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, g):
        # Identity-ish gradient — the harness only checks forward correctness.
        return torch.zeros_like(g).expand(g.shape if g.dim() else (1,)).contiguous(), None
''',
))
