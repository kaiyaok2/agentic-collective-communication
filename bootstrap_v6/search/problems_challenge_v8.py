"""10 challenging comm problems with real optimization tension.

Design principle: each problem has AT LEAST TWO plausible strategies where the
'optimal' choice is NOT obvious from the signature alone. Kiss vs strat is a
real tie-breaker: they may converge to different solutions, and sim vs HW
verdicts may disagree (like OverlayCCL's grad_ar naive vs bucketed).

Naming: _chal suffix.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _p120_ref(inputs_list, world_size):
    n_grads = len(inputs_list[0])
    outputs = []
    for gi in range(n_grads):
        s = sum(inputs_list[r][gi] for r in range(world_size))
        outputs.append(s)
    return [list(outputs) for _ in range(world_size)]


def _p120_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    sizes = [512, 1024, 256, 2048, 128, 1024, 512, 768]
    inputs = []
    for r in range(world_size):
        rank_grads = [torch.randn(s) * (r + 1) for s in sizes]
        inputs.append(rank_grads)
    per_rank_args = [{'rep_grads': inputs[r]} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p120_ref(inputs, world_size)}


def _p120_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['rep_grads'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P120_SIG = '''def evolved_p120(rep_grads, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P120_DOC = '''Args: rep_grads - list of 8 gradient tensors of varying sizes.
Formula: y[gi][i] = sum over ranks r of rep_grads_r[gi][i], for each of the 8 grads.
Returns: list of 8 tensors, each identical on every rank.

NON-OBVIOUS: 8 per-tensor all_reduces vs 1 bucketed cat+AR+split. Bucketing
saves dispatch overhead but pays cat/split HBM cost.
'''
_P120_BUILTINS = {
    'naive_per_tensor_ar': '''def evolved_p120(rep_grads, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return [xm.all_reduce(xm.REDUCE_SUM, g) for g in rep_grads]
''',
}
register_problem(CollectiveProblem(
    name='multi_grad_ar_chal',
    display_name='Problem P_120',
    evolved_fn_name='evolved_p120',
    signature=_P120_SIG,
    signature_doc=_P120_DOC,
    reference_fn=_p120_ref,
    generate_test_case=_p120_generate,
    call_candidate=_p120_call,
    builtin_templates=_P120_BUILTINS,
))


def _p121_ref(inputs, world_size):
    K = inputs[0].shape[0]
    out = []
    for r in range(world_size):
        s = sum(inputs)
        out.append(s[:, r].clone())
    return out


def _p121_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    K = 64
    torch.manual_seed(seed)
    inputs = [torch.randn(K, world_size) * (r + 1) for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'K': K} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p121_ref(inputs, world_size)}


def _p121_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['K'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P121_SIG = '''def evolved_p121(x, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P121_DOC = '''Args: x (K, world_size) - local matrix. K: int.
Formula: y_r = sum over ranks r' of x_{r'}[:, r] - rank r keeps summed column r.
Returns: (K,) tensor DIFFERENT per rank.
NON-OBVIOUS: (a) AR full matrix then slice column, or (b) reduce_scatter along dim 1.
'''
_P121_BUILTINS = {
    'naive_ar_slice': '''def evolved_p121(x, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    return s[:, rank]
''',
}
register_problem(CollectiveProblem(
    name='ag_then_rs_chal',
    display_name='Problem P_121',
    evolved_fn_name='evolved_p121',
    signature=_P121_SIG,
    signature_doc=_P121_DOC,
    reference_fn=_p121_ref,
    generate_test_case=_p121_generate,
    call_candidate=_p121_call,
    builtin_templates=_P121_BUILTINS,
))


def _p122_ref(inputs, world_size):
    n_layers = len(inputs[0])
    outputs = []
    for li in range(n_layers):
        s = sum(inputs[r][li] for r in range(world_size))
        outputs.append(s + li)
    return [list(outputs) for _ in range(world_size)]


def _p122_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    inputs = []
    for r in range(world_size):
        rank_layers = [torch.randn(N) * (r + 1) for _ in range(4)]
        inputs.append(rank_layers)
    per_rank_args = [{'layers': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p122_ref(inputs, world_size)}


def _p122_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['layers'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P122_SIG = '''def evolved_p122(layers, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P122_DOC = '''Args: layers - list of 4 (N,) tensors. N: int.
Formula: y[li] = sum_r layers_r[li] + li, for li in 0..3.
Returns: list of 4 (N,) tensors identical on every rank.
NON-OBVIOUS: 4x AR (naive) vs bucketed 1x AR + split. Amortization matters.
'''
_P122_BUILTINS = {
    'naive_per_layer_ar': '''def evolved_p122(layers, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    out = []
    for li, g in enumerate(layers):
        out.append(xm.all_reduce(xm.REDUCE_SUM, g) + li)
    return out
''',
}
register_problem(CollectiveProblem(
    name='multi_layer_ar_chal',
    display_name='Problem P_122',
    evolved_fn_name='evolved_p122',
    signature=_P122_SIG,
    signature_doc=_P122_DOC,
    reference_fn=_p122_ref,
    generate_test_case=_p122_generate,
    call_candidate=_p122_call,
    builtin_templates=_P122_BUILTINS,
))


def _p123_ref(inputs, world_size):
    stacked = torch.stack(inputs, dim=0)
    s = stacked.sum(dim=0)
    m = stacked.max(dim=0).values
    ref = torch.cat([s, m], dim=0)
    return [ref.clone() for _ in range(world_size)]


def _p123_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p123_ref(inputs, world_size)}


def _p123_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P123_SIG = '''def evolved_p123(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P123_DOC = '''Args: x (N,) - local vector. N: int.
Formula: y = concat(sum_r x_r, max_r x_r) along dim 0. Length 2*N.
NON-OBVIOUS: 2 ARs (SUM + MAX) vs single AR with pre-computed local extremes.
'''
_P123_BUILTINS = {
    'naive_two_ar': '''def evolved_p123(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    m = xm.all_reduce(xm.REDUCE_MAX, x)
    return torch.cat([s, m], dim=0)
''',
}
register_problem(CollectiveProblem(
    name='double_reduction_chal',
    display_name='Problem P_123',
    evolved_fn_name='evolved_p123',
    signature=_P123_SIG,
    signature_doc=_P123_DOC,
    reference_fn=_p123_ref,
    generate_test_case=_p123_generate,
    call_candidate=_p123_call,
    builtin_templates=_P123_BUILTINS,
))


def _p124_ref(inputs, world_size):
    ref = sum(inputs)
    return [ref.clone() for _ in range(world_size)]


def _p124_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 512
    torch.manual_seed(seed)
    inputs = [torch.randn(N) * (r + 1) for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p124_ref(inputs, world_size)}


def _p124_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P124_SIG = '''def evolved_p124(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P124_DOC = '''Args: x (N,) - local vector. N: int (=512).
Formula: y = sum_r x_r, identical on every rank.
Cluster topology: 2 nodes x 32 ranks. Intra-node NeuronLink ~15x faster than inter-node EFA.
NON-OBVIOUS: (a) flat 64-rank AR or (b) intra-node AR + inter-node exchange.
'''
_P124_BUILTINS = {
    'naive_flat_ar': '''def evolved_p124(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}
register_problem(CollectiveProblem(
    name='hierarchical_ar_chal',
    display_name='Problem P_124',
    evolved_fn_name='evolved_p124',
    signature=_P124_SIG,
    signature_doc=_P124_DOC,
    reference_fn=_p124_ref,
    generate_test_case=_p124_generate,
    call_candidate=_p124_call,
    builtin_templates=_P124_BUILTINS,
))


def _p125_ref(inputs, world_size):
    K = 8
    all_vals = torch.cat(inputs)
    top = torch.topk(all_vals, K).values.sort(descending=True).values
    return [top.clone() for _ in range(world_size)]


def _p125_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p125_ref(inputs, world_size)}


def _p125_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P125_SIG = '''def evolved_p125(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P125_DOC = '''Args: x (N,) - local vector. N: int.
Formula: y = top-8 largest values across all ranks concatenation, sorted descending.
Returns (8,) tensor identical on every rank.
NON-OBVIOUS: (a) all_gather full x then local top-K, or (b) local top-K then all_gather K per rank then global top-K. Note: sort/topk may be unsupported on Neuron.
'''
_P125_BUILTINS = {
    'naive_ag_topk': '''def evolved_p125(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Use all_reduce MAX K times as a workaround if topk unsupported
    g = xm.all_gather(x, dim=0)
    return torch.topk(g, k=8).values
''',
}
register_problem(CollectiveProblem(
    name='sparse_topk_chal',
    display_name='Problem P_125',
    evolved_fn_name='evolved_p125',
    signature=_P125_SIG,
    signature_doc=_P125_DOC,
    reference_fn=_p125_ref,
    generate_test_case=_p125_generate,
    call_candidate=_p125_call,
    builtin_templates=_P125_BUILTINS,
))


def _p126_ref(inputs, world_size):
    num = sum(inp.abs() * inp for inp in inputs)
    den = sum(inp.abs() for inp in inputs)
    ref = num / (den + 1e-8)
    return [ref.clone() for _ in range(world_size)]


def _p126_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    inputs = [torch.randn(N) for _ in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p126_ref(inputs, world_size)}


def _p126_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P126_SIG = '''def evolved_p126(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P126_DOC = '''Args: x (N,) - local vector. N: int.
Formula: y[i] = (sum_r |x_r[i]| * x_r[i]) / (sum_r |x_r[i]| + 1e-8).
NON-OBVIOUS: 2 ARs (numerator, denominator), or concat local num+den into 1 AR of 2N bytes then split, or all_gather x then local weighted mean.
'''
_P126_BUILTINS = {
    'naive_two_ar': '''def evolved_p126(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = x.abs()
    num = xm.all_reduce(xm.REDUCE_SUM, ax * x)
    den = xm.all_reduce(xm.REDUCE_SUM, ax)
    return num / (den + 1e-8)
''',
}
register_problem(CollectiveProblem(
    name='weighted_mean_chal',
    display_name='Problem P_126',
    evolved_fn_name='evolved_p126',
    signature=_P126_SIG,
    signature_doc=_P126_DOC,
    reference_fn=_p126_ref,
    generate_test_case=_p126_generate,
    call_candidate=_p126_call,
    builtin_templates=_P126_BUILTINS,
))


def _p127_ref(inputs_q, inputs_k, world_size):
    K_full = torch.stack(inputs_k, dim=0)
    ref = [inputs_q[r] @ K_full.t() for r in range(world_size)]
    return ref


def _p127_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N, D = 32, 64
    torch.manual_seed(seed)
    Qs = [torch.randn(N, D) for _ in range(world_size)]
    Ks = [torch.randn(D) * (r + 1) for r in range(world_size)]
    per_rank_args = [{'Q': Qs[r], 'k_row': Ks[r], 'N': N, 'D': D} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p127_ref(Qs, Ks, world_size)}


def _p127_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['Q'], rank_args['k_row'], rank_args['N'], rank_args['D'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P127_SIG = '''def evolved_p127(Q, k_row, N, D, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P127_DOC = '''Args: Q (N, D) - local queries. k_row (D,) - this ranks row of K. N, D: ints.
Formula: y = Q @ K^T where K is (world_size, D) from all ranks k_row.
Returns (N, world_size) DIFFERENT per rank.
NON-OBVIOUS: (a) all_gather k_row then matmul, (b) reduce_scatter matmul chunks, (c) ring-tile.
'''
_P127_BUILTINS = {
    'naive_ag_matmul': '''def evolved_p127(Q, k_row, N, D, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    K = xm.all_gather(k_row.unsqueeze(0), dim=0)
    return Q @ K.t()
''',
}
register_problem(CollectiveProblem(
    name='layered_matmul_chal',
    display_name='Problem P_127',
    evolved_fn_name='evolved_p127',
    signature=_P127_SIG,
    signature_doc=_P127_DOC,
    reference_fn=_p127_ref,
    generate_test_case=_p127_generate,
    call_candidate=_p127_call,
    builtin_templates=_P127_BUILTINS,
))


def _p128_ref(inputs, world_size):
    ref = sum(inp.float() for inp in inputs)
    return [ref.clone() for _ in range(world_size)]


def _p128_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    inputs = [torch.randn(N) * (r + 1) for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p128_ref(inputs, world_size)}


def _p128_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P128_SIG = '''def evolved_p128(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P128_DOC = '''Args: x (N,) - local vector (float32). N: int.
Formula: y[i] = sum_r x_r[i] (in float32 precision).
NON-OBVIOUS: (a) upcast to float32 locally then AR (large payload), (b) AR at native precision then upcast, (c) reduce_scatter partial float32 sums then all_gather.
'''
_P128_BUILTINS = {
    'naive_upcast_then_ar': '''def evolved_p128(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x.float())
''',
}
register_problem(CollectiveProblem(
    name='mixed_precision_ar_chal',
    display_name='Problem P_128',
    evolved_fn_name='evolved_p128',
    signature=_P128_SIG,
    signature_doc=_P128_DOC,
    reference_fn=_p128_ref,
    generate_test_case=_p128_generate,
    call_candidate=_p128_call,
    builtin_templates=_P128_BUILTINS,
))


def _p129_ref(inputs, world_size):
    return [torch.cat([inputs[(r + 1) % world_size], inputs[(r + 2) % world_size]], dim=0) for r in range(world_size)]


def _p129_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 128
    torch.manual_seed(seed)
    inputs = [torch.arange(N).float() + r * 1000 for r in range(world_size)]
    per_rank_args = [{'x': inputs[r], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p129_ref(inputs, world_size)}


def _p129_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P129_SIG = '''def evolved_p129(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P129_DOC = '''Args: x (N,) - local vector. N: int.
Formula: y_r = concat(x_{(r+1) % W}, x_{(r+2) % W}) - each rank gets 2 neighbors.
Returns (2*N,) tensor DIFFERENT per rank.
NON-OBVIOUS: (a) all_gather full ring + 2 slices, (b) 2 collective_permutes (+1 and +2) then concat.
'''
_P129_BUILTINS = {
    'naive_ag_slice': '''def evolved_p129(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    g = xm.all_gather(x, dim=0).reshape(world_size, N)
    plus1 = g[(rank + 1) % world_size]
    plus2 = g[(rank + 2) % world_size]
    return torch.cat([plus1, plus2], dim=0)
''',
}
register_problem(CollectiveProblem(
    name='rotating_shuffle_chal',
    display_name='Problem P_129',
    evolved_fn_name='evolved_p129',
    signature=_P129_SIG,
    signature_doc=_P129_DOC,
    reference_fn=_p129_ref,
    generate_test_case=_p129_generate,
    call_candidate=_p129_call,
    builtin_templates=_P129_BUILTINS,
))


def _p130_ref(inputs_grads, inputs_scales, world_size):
    n_grads = len(inputs_grads[0])
    outputs = []
    for gi in range(n_grads):
        s = sum(inputs_grads[r][gi] * inputs_scales[r][gi] for r in range(world_size))
        outputs.append(s)
    return [list(outputs) for _ in range(world_size)]


def _p130_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    sizes = [128, 256, 512, 128, 256]
    grads = []
    scales = []
    for r in range(world_size):
        rg = [torch.randn(s) for s in sizes]
        rs = [torch.tensor([0.5 + r * 0.01 + i * 0.1]).float() for i in range(len(sizes))]
        grads.append(rg)
        scales.append(rs)
    per_rank_args = [{'grads': grads[r], 'scales': scales[r]} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p130_ref(grads, scales, world_size)}


def _p130_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['grads'], rank_args['scales'], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P130_SIG = '''def evolved_p130(grads, scales, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P130_DOC = '''Args: grads - list of 5 tensors, scales - list of 5 scalars.
Formula: y[gi][i] = sum_r grads_r[gi][i] * scales_r[gi] for each gi in 0..4.
NON-OBVIOUS: 5 per-tensor scaled ARs vs local scale then batch-AR vs cat all grads+scales then 1 AR then split.
'''
_P130_BUILTINS = {
    'naive_per_tensor_scaled_ar': '''def evolved_p130(grads, scales, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    out = []
    for g, s in zip(grads, scales):
        out.append(xm.all_reduce(xm.REDUCE_SUM, g * s))
    return out
''',
}
register_problem(CollectiveProblem(
    name='batched_ar_scale_chal',
    display_name='Problem P_130',
    evolved_fn_name='evolved_p130',
    signature=_P130_SIG,
    signature_doc=_P130_DOC,
    reference_fn=_p130_ref,
    generate_test_case=_p130_generate,
    call_candidate=_p130_call,
    builtin_templates=_P130_BUILTINS,
))
