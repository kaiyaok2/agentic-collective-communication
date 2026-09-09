"""Diverse Round 15 (V16): more variety."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6300: seven_ars_same_input — 7 ARs of same x (CSE at scale)
# Coefficients such that sum = 100
# ============================================================
def _p6300_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    total_coef = 1 + 5 + 10 + 15 + 20 + 25 + 24  # = 100
    return [(ax * total_coef).clone() for _ in range(world_size)]

def _p6300_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6300_ref(per_rank_args, world_size)}

def _p6300_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6300_SIG = '''def evolved_p6300(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6300_DOC = '''Local x (N,). N=65536.
Compute: y = 1*AR(x) + 5*AR(x) + 10*AR(x) + 15*AR(x) + 20*AR(x) + 25*AR(x) + 24*AR(x) = 100*AR(x).
Return (N,).'''

_P6300_BUILTINS = {'seven_ar_same_input': '''def evolved_p6300(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = 1 * xm.all_reduce(xm.REDUCE_SUM, x)
    a2 = 5 * xm.all_reduce(xm.REDUCE_SUM, x)
    a3 = 10 * xm.all_reduce(xm.REDUCE_SUM, x)
    a4 = 15 * xm.all_reduce(xm.REDUCE_SUM, x)
    a5 = 20 * xm.all_reduce(xm.REDUCE_SUM, x)
    a6 = 25 * xm.all_reduce(xm.REDUCE_SUM, x)
    a7 = 24 * xm.all_reduce(xm.REDUCE_SUM, x)
    return a1 + a2 + a3 + a4 + a5 + a6 + a7
'''}

register_problem(CollectiveProblem(
    name='seven_ar_same_input_chal',
    display_name='Problem P_6300',
    evolved_fn_name='evolved_p6300',
    signature=_P6300_SIG,
    signature_doc=_P6300_DOC,
    reference_fn=_p6300_ref,
    generate_test_case=_p6300_generate,
    call_candidate=_p6300_call,
    builtin_templates=_P6300_BUILTINS,
))


# ============================================================
# P_6301: nested_reduce_min_via_max — MIN via negated MAX with baseline having 3 ARs
# ============================================================
def _p6301_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    mn = xs[0]
    for x in xs[1:]: mn = torch.minimum(mn, x)
    return [mn.clone() for _ in range(world_size)]

def _p6301_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6301_ref(per_rank_args, world_size)}

def _p6301_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6301_SIG = '''def evolved_p6301(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6301_DOC = '''Local x (N,). N=65536.
Compute: elementwise MIN across ranks.
Baseline uses MIN via -MAX(-x) with extra verify pass.
Sorcar: single REDUCE_MIN AR.'''

_P6301_BUILTINS = {'min_neg_max_dead_verify': '''def evolved_p6301(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    neg = -x
    m = xm.all_reduce(xm.REDUCE_MAX, neg)
    min1 = -m
    # verify via another MIN AR (redundant)
    verify = xm.all_reduce(xm.REDUCE_MIN, min1)
    return verify
'''}

register_problem(CollectiveProblem(
    name='min_neg_max_dead_verify_chal',
    display_name='Problem P_6301',
    evolved_fn_name='evolved_p6301',
    signature=_P6301_SIG,
    signature_doc=_P6301_DOC,
    reference_fn=_p6301_ref,
    generate_test_case=_p6301_generate,
    call_candidate=_p6301_call,
    builtin_templates=_P6301_BUILTINS,
))


# ============================================================
# P_6302: reduce_split_by_rank_class — even ranks do MAX, odd do SUM  
# Baseline: multiple ARs. Sorcar: no clean optimization.
# Instead: baseline does AR(x) then AR(y-x) where y=1  — really AR(x)+W-W*AR(x)= W-... too complex
# 
# Try: ar_pipeline_two_indep — 2 AR calls that share NO dep
# Baseline uses .item() between them (breaks compilation)
# Sorcar: skip the .item()
# Skip this since .item() might not work in mock.
# 
# Do: identical_ars_via_different_paths — AR(x) computed twice via
# different but equivalent paths
# ============================================================
def _p6302_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax * 4).clone() for _ in range(world_size)]

def _p6302_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6302_ref(per_rank_args, world_size)}

def _p6302_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6302_SIG = '''def evolved_p6302(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6302_DOC = '''Local x (N,). N=65536.
Compute: y = 4 * AR(x). Return (N,).
Baseline computes 2 * AR(x) then 2 * AR(x) via different variables and adds.'''

_P6302_BUILTINS = {'ar_via_two_paths': '''def evolved_p6302(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Path 1: 2 * AR(x)
    a = xm.all_reduce(xm.REDUCE_SUM, x)
    path1 = 2 * a
    # Path 2: AR(2*x) which is 2 * AR(x)
    scaled_x = 2 * x
    b = xm.all_reduce(xm.REDUCE_SUM, scaled_x)
    path2 = b
    return path1 + path2
'''}

register_problem(CollectiveProblem(
    name='ar_via_two_paths_chal',
    display_name='Problem P_6302',
    evolved_fn_name='evolved_p6302',
    signature=_P6302_SIG,
    signature_doc=_P6302_DOC,
    reference_fn=_p6302_ref,
    generate_test_case=_p6302_generate,
    call_candidate=_p6302_call,
    builtin_templates=_P6302_BUILTINS,
))


# ============================================================
# P_6303: 2d_col_ar_and_reduce — 2D reduce along different dims
# Baseline: AR(x[:, k]) for each column k separately, W dispatches
# ============================================================
def _p6303_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6303_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(1024, 8)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6303_ref(per_rank_args, world_size)}

def _p6303_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6303_SIG = '''def evolved_p6303(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6303_DOC = '''Local x (1024, 8). AR the whole tensor.
Baseline ARs each of 8 columns separately. Sorcar: single AR of full 2D.'''

_P6303_BUILTINS = {'per_column_ar': '''def evolved_p6303(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    cols = [xm.all_reduce(xm.REDUCE_SUM, x[:, k]) for k in range(8)]
    return torch.stack(cols, dim=1)
'''}

register_problem(CollectiveProblem(
    name='per_column_ar_chal',
    display_name='Problem P_6303',
    evolved_fn_name='evolved_p6303',
    signature=_P6303_SIG,
    signature_doc=_P6303_DOC,
    reference_fn=_p6303_ref,
    generate_test_case=_p6303_generate,
    call_candidate=_p6303_call,
    builtin_templates=_P6303_BUILTINS,
))
