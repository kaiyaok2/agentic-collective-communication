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

_FRS_HINTS = """\
### Constraints:
- Each input tensor size must be divisible by world_size.
- Output: a list of N tensors, each of size original/world_size.
- WARNING: reduce_scatter does NOT distribute over cat — the scatter dimension
  interleaves differently if you try to fuse via concatenation."""

register_problem(CollectiveProblem(
    name="fused_reducescatter",
    display_name="Fused ReduceScatter (Multi-Tensor Gradient Partition)",
    evolved_fn_name="evolved_fused_reducescatter",
    signature=_FRS_SIGNATURE,
    signature_doc=_FRS_SIGNATURE_DOC,
    reference_fn=_fused_rs_reference,
    generate_test_case=_fused_rs_generate_test_case,
    call_candidate=_fused_rs_call_candidate,
    builtin_templates=_FRS_BUILTINS,
    optimization_hints=_FRS_HINTS,
    public_api_code='''\
def fused_reducescatter(tensors):
    """ReduceScatter multiple tensors using the agent-evolved fused algorithm."""
    if _rank is None:
        raise RuntimeError("Call init_fused_reducescatter() first")
    return evolved_fused_reducescatter(
        tensors, _rank, _world_size, _NUM_DEVICES, _CORES_PER_DEVICE,
        xm, torch, num_nodes=_NUM_NODES)
''',
    training_validation_code='''\
N_SHARDS = 8
SHARD_SIZE = 1024 * world
INPUT_SIZE = N_SHARDS * SHARD_SIZE

class _CollectiveOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tensors = list(x.split(SHARD_SIZE))
        xm.mark_step()
        shards = fused_reducescatter(tensors)
        xm.mark_step()
        return torch.cat(shards, dim=0)

    @staticmethod
    def backward(ctx, g):
        xm.mark_step()
        out = xm.all_gather(g, dim=0)
        xm.mark_step()
        return out

OUTPUT_SIZE = INPUT_SIZE // world
''',
))


# ================================================================
# Problem 4: Ring Attention KV Distribution
# ================================================================

def _ring_kv_reference(inputs, world_size):
    """Reference: all_gather - each rank gets all KV chunks in rank order."""
    outputs = []
    for dst in range(world_size):
        outputs.append(torch.cat(inputs, dim=0))
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

    inputs = []
    for rank in range(world_size):
        base = rank * 10000
        inputs.append(torch.arange(base, base + kv_size, dtype=torch.float32))

    expected = _ring_kv_reference(inputs, world_size)

    per_rank_args = []
    for rank in range(world_size):
        per_rank_args.append({"kv_chunk": inputs[rank]})

    return {
        "per_rank_args": per_rank_args,
        "shared_args": {},
        "expected": expected,
    }


def _ring_kv_call_candidate(candidate_fn, rank_args, shared_args,
                             rank, world_size, num_devices, cpd,
                             xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(
        rank_args["kv_chunk"],
        rank, world_size, num_devices, cpd,
        xm_mock, torch_mock, num_nodes=num_nodes)


_RING_KV_SIGNATURE = """\
def evolved_ring_kv(kv_chunk, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):"""

_RING_KV_SIGNATURE_DOC = """\
    Args:
        kv_chunk: 1D tensor. This rank's KV data (key-value cache for attention).
        rank: int. This rank's index.
        world_size: int. Total ranks.
        num_devices, cores_per_device: hardware topology.
        xm: XLA model module.
        torch: Torch module.
        num_nodes: int. Number of nodes (default 1).

    Returns:
        1D tensor of size world_size * len(kv_chunk).
        Contains all ranks' KV chunks concatenated in rank order:
        [kv_from_rank_0, kv_from_rank_1, ..., kv_from_rank_{world_size-1}]."""

_RING_KV_BUILTINS = {}

_RING_KV_BUILTINS["naive_ring_permute"] = '''\
def evolved_ring_kv(kv_chunk, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Ring: rotate KV chunks via collective_permute."""
    kv_size = kv_chunk.numel()
    output = torch.zeros(world_size * kv_size, device=kv_chunk.device, dtype=kv_chunk.dtype)
    output[rank * kv_size:(rank + 1) * kv_size] = kv_chunk
    current = kv_chunk
    for step in range(1, world_size):
        pairs = [(r, (r + 1) % world_size) for r in range(world_size)]
        current = xm.collective_permute(current, pairs=pairs)
        src = (rank - step + world_size) % world_size
        output[src * kv_size:(src + 1) * kv_size] = current
    return output
'''

_RING_KV_BUILTINS["flat_allgather"] = '''\
def evolved_ring_kv(kv_chunk, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    """Flat all_gather approach."""
    gathered = xm.all_gather(kv_chunk.unsqueeze(0), dim=0)
    return gathered.view(-1)
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
def ring_kv_gather(kv_chunk):
    """Gather all ranks' KV chunks using the agent-evolved algorithm."""
    if _rank is None:
        raise RuntimeError("Call init_ring_kv() first")
    return evolved_ring_kv(
        kv_chunk, _rank, _world_size, _NUM_DEVICES, _CORES_PER_DEVICE,
        xm, torch, num_nodes=_NUM_NODES)
''',
    training_validation_code='''\
KV_SIZE = 8192
INPUT_SIZE = KV_SIZE

class _CollectiveOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv):
        ctx.kv_size = kv.numel()
        xm.mark_step()
        out = ring_kv_gather(kv)
        xm.mark_step()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rank = xr.global_ordinal()
        kv_size = ctx.kv_size
        xm.mark_step()
        grad_local = grad_out[rank * kv_size:(rank + 1) * kv_size]
        xm.mark_step()
        return grad_local

OUTPUT_SIZE = KV_SIZE * world
''',
))
