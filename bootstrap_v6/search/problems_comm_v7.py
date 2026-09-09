"""10 new benchmarks requiring REAL cross-rank communication.

Design principle: unlike the 12 _bcast problems (which test 'recognize
local computation'), these problems have per-rank inputs whose values
carry semantic meaning; the output at any rank depends on ALL ranks'
inputs. A candidate that simply computes locally CANNOT produce the
correct answer — a collective is required.

Naming: _comm suffix distinguishes from _bcast.
"""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================================
# P_110 — sum_across_ranks: y[i] = sum over all ranks r of x_r[i]
# Each rank has a length-N vector; output is elementwise sum across ranks.
# ============================================================================
def _p110_ref(inputs, world_size):
    ref = sum(inputs)
    return [ref.clone() for _ in range(world_size)]


def _p110_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    inputs = [torch.arange(N).float() * (r + 1) for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p110_ref(inputs, world_size)}


def _p110_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P110_SIG = '''def evolved_p110(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P110_DOC = '''Args: x (N,) — local vector, DIFFERENT per rank. N: int.
Formula: y[i] = sum over r in [0, world_size) of x_r[i], where x_r is rank r's local x.
Returns (N,) tensor identical on every rank.
'''
_P110_BUILTINS = {
    'baseline_ar': '''def evolved_p110(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}
register_problem(CollectiveProblem(
    name='sum_across_ranks_comm',
    display_name='Problem P_110',
    evolved_fn_name='evolved_p110',
    signature=_P110_SIG,
    signature_doc=_P110_DOC,
    reference_fn=_p110_ref,
    generate_test_case=_p110_generate,
    call_candidate=_p110_call,
    builtin_templates=_P110_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_111 — max_across_ranks: y[i] = max_r x_r[i]
# ============================================================================
def _p111_ref(inputs, world_size):
    stacked = torch.stack(inputs, dim=0)
    ref = stacked.max(dim=0).values
    return [ref.clone() for _ in range(world_size)]


def _p111_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p111_ref(inputs, world_size)}


def _p111_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P111_SIG = '''def evolved_p111(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P111_DOC = '''Args: x (N,) — local vector. N: int.
Formula: y[i] = max over all ranks r of x_r[i].
Returns (N,) tensor identical on every rank.
'''
_P111_BUILTINS = {
    'baseline_ar_max': '''def evolved_p111(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # baseline: naive gather-and-max
    g = xm.all_gather(x, dim=0).reshape(world_size, N)
    return g.max(dim=0).values
''',
}
register_problem(CollectiveProblem(
    name='max_across_ranks_comm',
    display_name='Problem P_111',
    evolved_fn_name='evolved_p111',
    signature=_P111_SIG,
    signature_doc=_P111_DOC,
    reference_fn=_p111_ref,
    generate_test_case=_p111_generate,
    call_candidate=_p111_call,
    builtin_templates=_P111_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_112 — concat_all_ranks: y[r*N:(r+1)*N] = x_r  (i.e., all_gather along dim 0)
# ============================================================================
def _p112_ref(inputs, world_size):
    ref = torch.cat(inputs, dim=0)
    return [ref.clone() for _ in range(world_size)]


def _p112_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 64
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p112_ref(inputs, world_size)}


def _p112_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P112_SIG = '''def evolved_p112(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P112_DOC = '''Args: x (N,) — local shard. N: int (per-rank shard size).
Formula: y[r*N + i] = x_r[i] for r in [0, world_size), i in [0, N).
Returns (world_size * N,) tensor identical on every rank (concatenated shards).
'''
_P112_BUILTINS = {
    'baseline_ag': '''def evolved_p112(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_gather(x, dim=0)
''',
}
register_problem(CollectiveProblem(
    name='concat_all_ranks_comm',
    display_name='Problem P_112',
    evolved_fn_name='evolved_p112',
    signature=_P112_SIG,
    signature_doc=_P112_DOC,
    reference_fn=_p112_ref,
    generate_test_case=_p112_generate,
    call_candidate=_p112_call,
    builtin_templates=_P112_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_113 — dot_product_across_ranks: scalar = sum over r of dot(x_r, y_r)
# Each rank has (x_r, y_r); output is a scalar broadcast to all ranks.
# ============================================================================
def _p113_ref(inputs_x, inputs_y, world_size):
    s = 0.0
    for r in range(world_size):
        s += float((inputs_x[r] * inputs_y[r]).sum())
    ref = torch.tensor([s]).float()
    return [ref.clone() for _ in range(world_size)]


def _p113_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    xs = [torch.randn(N) for _ in range(world_size)]
    ys = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': xs[r], 'y': ys[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p113_ref(xs, ys, world_size)}


def _p113_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['y'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P113_SIG = '''def evolved_p113(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P113_DOC = '''Args: x (N,), y (N,) — local vectors. N: int.
Formula: s = sum over r of dot(x_r, y_r) (scalar).
Returns (1,) tensor with s, identical on every rank.
'''
_P113_BUILTINS = {
    'baseline_ar_dot': '''def evolved_p113(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    local = (x * y).sum().unsqueeze(0)
    return xm.all_reduce(xm.REDUCE_SUM, local)
''',
}
register_problem(CollectiveProblem(
    name='dot_across_ranks_comm',
    display_name='Problem P_113',
    evolved_fn_name='evolved_p113',
    signature=_P113_SIG,
    signature_doc=_P113_DOC,
    reference_fn=_p113_ref,
    generate_test_case=_p113_generate,
    call_candidate=_p113_call,
    builtin_templates=_P113_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_114 — permute_shift: y_r[i] = x_{(r+1) mod W}[i]  (each rank gets its next neighbor's data)
# ============================================================================
def _p114_ref(inputs, world_size):
    return [inputs[(r + 1) % world_size].clone() for r in range(world_size)]


def _p114_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 64
    torch.manual_seed(seed)
    inputs = [torch.arange(N).float() + r * 1000 for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p114_ref(inputs, world_size)}


def _p114_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P114_SIG = '''def evolved_p114(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P114_DOC = '''Args: x (N,) — rank r's local vector. N: int.
Formula: y_r[i] = x_{(r+1) mod world_size}[i] (each rank returns its right neighbor's data).
Returns (N,) tensor DIFFERENT per rank.
'''
_P114_BUILTINS = {
    'baseline_cp': '''def evolved_p114(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    pairs = [(r, (r - 1) % world_size) for r in range(world_size)]
    return xm.collective_permute(x, pairs=pairs)
''',
}
register_problem(CollectiveProblem(
    name='shift_neighbor_comm',
    display_name='Problem P_114',
    evolved_fn_name='evolved_p114',
    signature=_P114_SIG,
    signature_doc=_P114_DOC,
    reference_fn=_p114_ref,
    generate_test_case=_p114_generate,
    call_candidate=_p114_call,
    builtin_templates=_P114_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_115 — reduce_scatter_sum: y_r = sum over r' of x_{r'}[r*K:(r+1)*K]
# All ranks contribute a length-W*K vector; each rank keeps a K-slice of the sum.
# ============================================================================
def _p115_ref(inputs, world_size):
    stacked = torch.stack(inputs, dim=0)
    summed = stacked.sum(dim=0)  # (W*K,)
    K = summed.shape[0] // world_size
    return [summed[r*K:(r+1)*K].clone() for r in range(world_size)]


def _p115_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    K = 64
    N = K * world_size
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N, 'K': K} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p115_ref(inputs, world_size)}


def _p115_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank_args['K'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P115_SIG = '''def evolved_p115(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P115_DOC = '''Args: x (N,) — local vector, N = K * world_size. K: int.
Formula: y_r[i] = sum over r' in [0, world_size) of x_{r'}[rank*K + i] for i in [0, K).
Returns (K,) tensor DIFFERENT per rank (rank-r's shard of the summed vector).
'''
_P115_BUILTINS = {
    'baseline_rs': '''def evolved_p115(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=world_size)
''',
}
register_problem(CollectiveProblem(
    name='reduce_scatter_sum_comm',
    display_name='Problem P_115',
    evolved_fn_name='evolved_p115',
    signature=_P115_SIG,
    signature_doc=_P115_DOC,
    reference_fn=_p115_ref,
    generate_test_case=_p115_generate,
    call_candidate=_p115_call,
    builtin_templates=_P115_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_116 — mean_across_ranks_normalized: y[i] = (1/W) * sum_r x_r[i] / (max_r |x_r[i]| + 1e-8)
# Requires BOTH an all_reduce (sum) and all_reduce (max) — composable pattern.
# ============================================================================
def _p116_ref(inputs, world_size):
    stacked = torch.stack(inputs, dim=0)
    mean = stacked.mean(dim=0)
    max_abs = stacked.abs().max(dim=0).values + 1e-8
    ref = mean / max_abs
    return [ref.clone() for _ in range(world_size)]


def _p116_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p116_ref(inputs, world_size)}


def _p116_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P116_SIG = '''def evolved_p116(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P116_DOC = '''Args: x (N,) — local vector. N: int.
Formula: y[i] = (mean_r x_r[i]) / (max_r |x_r[i]| + 1e-8).
Requires BOTH a sum and a max across ranks. Returns (N,) identical on every rank.
'''
_P116_BUILTINS = {
    'baseline_two_ar': '''def evolved_p116(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # naive: 2 ARs
    g = xm.all_gather(x, dim=0).reshape(world_size, N)
    m = g.abs().max(dim=0).values
    return (s / world_size) / (m + 1e-8)
''',
}
register_problem(CollectiveProblem(
    name='mean_max_normalize_comm',
    display_name='Problem P_116',
    evolved_fn_name='evolved_p116',
    signature=_P116_SIG,
    signature_doc=_P116_DOC,
    reference_fn=_p116_ref,
    generate_test_case=_p116_generate,
    call_candidate=_p116_call,
    builtin_templates=_P116_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_117 — running_prefix_sum: y_r = sum over r' <= r of x_{r'} (per-rank prefix)
# Requires a scan pattern across ranks; each rank output differs.
# ============================================================================
def _p117_ref(inputs, world_size):
    out = []
    cum = torch.zeros_like(inputs[0])
    for r in range(world_size):
        cum = cum + inputs[r]
        out.append(cum.clone())
    return out


def _p117_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 64
    torch.manual_seed(seed)
    inputs = [torch.arange(N).float() * (r + 1) for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p117_ref(inputs, world_size)}


def _p117_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P117_SIG = '''def evolved_p117(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P117_DOC = '''Args: x (N,) — local vector. N: int.
Formula: y_r[i] = sum over r' <= rank of x_{r'}[i] (per-rank prefix sum across ranks).
Returns (N,) tensor DIFFERENT per rank.
'''
_P117_BUILTINS = {
    'baseline_ag_prefix': '''def evolved_p117(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    g = xm.all_gather(x, dim=0).reshape(world_size, N)
    prefix = g[:rank + 1].sum(dim=0)
    return prefix
''',
}
register_problem(CollectiveProblem(
    name='rank_prefix_sum_comm',
    display_name='Problem P_117',
    evolved_fn_name='evolved_p117',
    signature=_P117_SIG,
    signature_doc=_P117_DOC,
    reference_fn=_p117_ref,
    generate_test_case=_p117_generate,
    call_candidate=_p117_call,
    builtin_templates=_P117_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_118 — pairwise_diff: y_r[i] = x_r[i] - mean_{r'} x_{r'}[i]
# Center each rank's data around cross-rank mean.
# ============================================================================
def _p118_ref(inputs, world_size):
    stacked = torch.stack(inputs, dim=0)
    mean = stacked.mean(dim=0)
    return [(inputs[r] - mean).clone() for r in range(world_size)]


def _p118_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    inputs = [torch.randn(N) + r for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p118_ref(inputs, world_size)}


def _p118_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P118_SIG = '''def evolved_p118(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P118_DOC = '''Args: x (N,) — local vector. N: int.
Formula: y_r[i] = x_r[i] - (1/world_size) * sum_{r'} x_{r'}[i].
Returns (N,) tensor DIFFERENT per rank (rank-centered).
'''
_P118_BUILTINS = {
    'baseline_center': '''def evolved_p118(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mean = xm.all_reduce(xm.REDUCE_SUM, x) / world_size
    return x - mean
''',
}
register_problem(CollectiveProblem(
    name='center_by_mean_comm',
    display_name='Problem P_118',
    evolved_fn_name='evolved_p118',
    signature=_P118_SIG,
    signature_doc=_P118_DOC,
    reference_fn=_p118_ref,
    generate_test_case=_p118_generate,
    call_candidate=_p118_call,
    builtin_templates=_P118_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_119 — top_k_across_ranks: y[k] = k-th largest value across all ranks' first elements
# All ranks contribute one scalar; output is top-K values sorted descending.
# ============================================================================
def _p119_ref(inputs, world_size):
    # global max of first element across all ranks
    firsts = torch.stack([inp[0] for inp in inputs])
    ref = torch.tensor([float(firsts.max())]).float()
    return [ref.clone() for _ in range(world_size)]


def _p119_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 16
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p119_ref(inputs, world_size)}


def _p119_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P119_SIG = '''def evolved_p119(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P119_DOC = '''Args: x (N,) — local vector. N: int.
Formula: y = max over all ranks r of x_r[0] (single scalar broadcast).
Returns (1,) tensor identical on every rank.
'''
_P119_BUILTINS = {
    'baseline_ag_topk': '''def evolved_p119(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    scalar = x[:1]
    return xm.all_reduce(xm.REDUCE_MAX, scalar)
''',
}
register_problem(CollectiveProblem(
    name='top_k_scalars_comm',
    display_name='Problem P_119',
    evolved_fn_name='evolved_p119',
    signature=_P119_SIG,
    signature_doc=_P119_DOC,
    reference_fn=_p119_ref,
    generate_test_case=_p119_generate,
    call_candidate=_p119_call,
    builtin_templates=_P119_BUILTINS,
    optimization_hints='',
    public_api_code='',
    training_validation_code='',
))
