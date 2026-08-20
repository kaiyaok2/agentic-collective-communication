"""Diverse Round 15 (V20): more mixed-collective and CSE."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6700: per_row_ar_M128 — huge M
# ============================================================
def _p6700_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6700_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(128, 512)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6700_ref(per_rank_args, world_size)}

def _p6700_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6700_SIG = '''def evolved_p6700(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6700_DOC = '''Local x (128, 512). AR full. Baseline ARs 128 rows.'''

_P6700_BUILTINS = {'per_row_ar_M128': '''def evolved_p6700(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M128_chal',
    display_name='Problem P_6700',
    evolved_fn_name='evolved_p6700',
    signature=_P6700_SIG,
    signature_doc=_P6700_DOC,
    reference_fn=_p6700_ref,
    generate_test_case=_p6700_generate,
    call_candidate=_p6700_call,
    builtin_templates=_P6700_BUILTINS,
))


# ============================================================
# P_6701: cse_across_reduce_ops — AR(SUM, x) used to compute both mean and total
# Baseline: two ARs (once for total, once for mean/W)
# Sorcar: one AR, divide locally
# ============================================================
def _p6701_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    W = world_size
    return [(ax + ax / W).clone() for _ in range(world_size)]

def _p6701_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6701_ref(per_rank_args, world_size)}

def _p6701_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6701_SIG = '''def evolved_p6701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6701_DOC = '''Local x (N,). N=65536.
Compute: y = total_sum + mean = AR(x) + AR(x)/W.
Return (N,).'''

_P6701_BUILTINS = {'total_and_mean_ars': '''def evolved_p6701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    total = xm.all_reduce(xm.REDUCE_SUM, x)
    mean = xm.all_reduce(xm.REDUCE_SUM, x) / world_size
    return total + mean
'''}

register_problem(CollectiveProblem(
    name='total_and_mean_ars_chal',
    display_name='Problem P_6701',
    evolved_fn_name='evolved_p6701',
    signature=_P6701_SIG,
    signature_doc=_P6701_DOC,
    reference_fn=_p6701_ref,
    generate_test_case=_p6701_generate,
    call_candidate=_p6701_call,
    builtin_templates=_P6701_BUILTINS,
))


# ============================================================
# P_6702: sum_of_scaled_bcast — 4 ARs of scaled same x + broadcast constant
# ============================================================
def _p6702_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    W = world_size
    # coef = 2 + 3 + 5 + 7 = 17; plus per-rank constant 1 → sum = W
    return [(ax * 17 + W).clone() for _ in range(world_size)]

def _p6702_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6702_ref(per_rank_args, world_size)}

def _p6702_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6702_SIG = '''def evolved_p6702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6702_DOC = '''Local x (N,). N=65536.
Compute: y = 2*AR(x) + 3*AR(x) + 5*AR(x) + 7*AR(x) + AR(ones).
Result: 17*AR(x) + W. Return (N,).'''

_P6702_BUILTINS = {'four_ar_scaled_plus_bcast': '''def evolved_p6702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = 2 * xm.all_reduce(xm.REDUCE_SUM, x)
    b = 3 * xm.all_reduce(xm.REDUCE_SUM, x)
    c = 5 * xm.all_reduce(xm.REDUCE_SUM, x)
    d = 7 * xm.all_reduce(xm.REDUCE_SUM, x)
    ones = torch.ones_like(x)
    e = xm.all_reduce(xm.REDUCE_SUM, ones)
    return a + b + c + d + e
'''}

register_problem(CollectiveProblem(
    name='four_scaled_plus_bcast_ar_chal',
    display_name='Problem P_6702',
    evolved_fn_name='evolved_p6702',
    signature=_P6702_SIG,
    signature_doc=_P6702_DOC,
    reference_fn=_p6702_ref,
    generate_test_case=_p6702_generate,
    call_candidate=_p6702_call,
    builtin_templates=_P6702_BUILTINS,
))


# ============================================================
# P_6703: ar_then_reduce_dim_broadcast — AR of 2D, then sum-dim, then broadcast
# Baseline over-communicates by keeping full 2D shape.
# ============================================================
def _p6703_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.sum(dim=0, keepdim=True).expand_as(ax).clone() for _ in range(world_size)]

def _p6703_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(16, 4096)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6703_ref(per_rank_args, world_size)}

def _p6703_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6703_SIG = '''def evolved_p6703(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6703_DOC = '''Local x (16, 4096). Compute AR(x), sum along dim 0, broadcast to shape of x.
Return (16, 4096) with each row = sum-along-M of AR(x).'''

_P6703_BUILTINS = {'ar_then_sum_dim_expand': '''def evolved_p6703(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    reduced = ax.sum(dim=0, keepdim=True)
    return reduced.expand_as(ax)
'''}

register_problem(CollectiveProblem(
    name='ar_sum_dim_expand_chal',
    display_name='Problem P_6703',
    evolved_fn_name='evolved_p6703',
    signature=_P6703_SIG,
    signature_doc=_P6703_DOC,
    reference_fn=_p6703_ref,
    generate_test_case=_p6703_generate,
    call_candidate=_p6703_call,
    builtin_templates=_P6703_BUILTINS,
))
