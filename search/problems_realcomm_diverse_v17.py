"""Diverse Round 15 (V17): more variety."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6400: nine_ars_same_input — 9 ARs (CSE at even bigger scale)  
# ============================================================
def _p6400_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    total = sum(range(1, 10))  # 45
    return [(ax * total).clone() for _ in range(world_size)]

def _p6400_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 32768
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6400_ref(per_rank_args, world_size)}

def _p6400_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6400_SIG = '''def evolved_p6400(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6400_DOC = '''Local x (N,). N=32768.
Compute: sum from k=1 to 9 of k*AR(x) = 45*AR(x). Return (N,).'''

_P6400_BUILTINS = {'nine_ar_same_input': '''def evolved_p6400(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    r = 0
    for k in range(1, 10):
        r = r + k * xm.all_reduce(xm.REDUCE_SUM, x)
    return r
'''}

register_problem(CollectiveProblem(
    name='nine_ar_same_input_chal',
    display_name='Problem P_6400',
    evolved_fn_name='evolved_p6400',
    signature=_P6400_SIG,
    signature_doc=_P6400_DOC,
    reference_fn=_p6400_ref,
    generate_test_case=_p6400_generate,
    call_candidate=_p6400_call,
    builtin_templates=_P6400_BUILTINS,
))


# ============================================================
# P_6401: per_row_max_ar — 2D MAX AR per row (baseline slow)
# ============================================================
def _p6401_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    result = xs[0]
    for x in xs[1:]: result = torch.maximum(result, x)
    return [result.clone() for _ in range(world_size)]

def _p6401_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(16, 4096)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6401_ref(per_rank_args, world_size)}

def _p6401_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6401_SIG = '''def evolved_p6401(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6401_DOC = '''Local x (16, 4096). Compute AR(MAX) of full x.
Baseline ARs each row separately (16 dispatches). Sorcar: 1 AR of full 2D.'''

_P6401_BUILTINS = {'per_row_max_ar': '''def evolved_p6401(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_MAX, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_max_ar_chal',
    display_name='Problem P_6401',
    evolved_fn_name='evolved_p6401',
    signature=_P6401_SIG,
    signature_doc=_P6401_DOC,
    reference_fn=_p6401_ref,
    generate_test_case=_p6401_generate,
    call_candidate=_p6401_call,
    builtin_templates=_P6401_BUILTINS,
))


# ============================================================
# P_6402: chained_sums — sum(x) then AR then sum
# Baseline: local sum, AR full tensor, sum result
# Actually that's inefficient. Skip.
# 
# Try: alternating_sum_sub — AR(x) + AR(x) - AR(x) - AR(x) = 0
# But actually equals 0 = trivial. Make it y * 0 = 0.
# Actually make it: alternating with variable coefficients that sum to zero.
# Baseline computes 4 ARs and adds; result = 0.
# Sorcar: recognize algebraic zero.
# ============================================================
def _p6402_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [torch.zeros_like(ax).clone() for _ in range(world_size)]  # coefs sum to 0

def _p6402_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6402_ref(per_rank_args, world_size)}

def _p6402_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6402_SIG = '''def evolved_p6402(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6402_DOC = '''Local x (N,). N=65536.
Compute: y = 2*AR(x) + 3*AR(x) - AR(x) - 4*AR(x) (coefs 2+3-1-4 = 0).
Result should be zero(s) tensor of shape (N,). Sorcar should recognize.'''

_P6402_BUILTINS = {'four_ar_sum_zero': '''def evolved_p6402(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = 2 * xm.all_reduce(xm.REDUCE_SUM, x)
    b = 3 * xm.all_reduce(xm.REDUCE_SUM, x)
    c = xm.all_reduce(xm.REDUCE_SUM, x)
    d = 4 * xm.all_reduce(xm.REDUCE_SUM, x)
    return a + b - c - d
'''}

register_problem(CollectiveProblem(
    name='four_ar_sum_zero_chal',
    display_name='Problem P_6402',
    evolved_fn_name='evolved_p6402',
    signature=_P6402_SIG,
    signature_doc=_P6402_DOC,
    reference_fn=_p6402_ref,
    generate_test_case=_p6402_generate,
    call_candidate=_p6402_call,
    builtin_templates=_P6402_BUILTINS,
))


# ============================================================
# P_6403: mixed_ar_sums_pooled — 4 ARs of different inputs, pooled at end
# Test if Sorcar can pool independent ARs (like V5 test but bigger N)
# ============================================================
def _p6403_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1, 5)]
    return [sum(axs).clone() for _ in range(world_size)]

def _p6403_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 131072
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.2) for i in range(1, 5)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6403_ref(per_rank_args, world_size)}

def _p6403_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6403_SIG = '''def evolved_p6403(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6403_DOC = '''4 local vectors x1..x4 each (N,). N=131072 (LARGE).
Compute: y = AR(x1) + AR(x2) + AR(x3) + AR(x4).
Return (N,).'''

_P6403_BUILTINS = {'four_ar_indep_large_N': '''def evolved_p6403(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    return a1 + a2 + a3 + a4
'''}

register_problem(CollectiveProblem(
    name='four_ar_indep_large_N_chal',
    display_name='Problem P_6403',
    evolved_fn_name='evolved_p6403',
    signature=_P6403_SIG,
    signature_doc=_P6403_DOC,
    reference_fn=_p6403_ref,
    generate_test_case=_p6403_generate,
    call_candidate=_p6403_call,
    builtin_templates=_P6403_BUILTINS,
))
