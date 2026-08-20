"""Diverse Round 15 (V10): more distinct classes."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5700: ar_of_expression_with_common_subterm — CSE within AR arg
# Baseline: AR(x + f(x)) + AR(y + f(x)) where f(x) is a common subterm
# Sorcar: compute f(x) once, AR both.
# Actually simpler: baseline stacks (2, N) with each row containing f(x) mixed in,
# ARs everything. Sorcar: no way to avoid because f(x) is inside AR arg.
# Actually make it: AR(f(x) + a) + AR(f(x) + b) where f(x) is rank-common.
# ============================================================
def _p5700_ref(inputs, world_size):
    # f(x) is rank-common (uses only rank 0's data), but let's make it a shared scalar.
    # Simpler: (x + s) all_reduced + (y + s) all_reduced where s is scalar
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    W = world_size
    return [(ax + 3.0 * W + ay + 3.0 * W).clone() for _ in range(world_size)]

def _p5700_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5700_ref(per_rank_args, world_size)}

def _p5700_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5700_SIG = '''def evolved_p5700(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5700_DOC = '''Local x, y (N,). N=65536.
Compute: y_out = AR(x + 3.0) + AR(y + 3.0). Return (N,) identical.
Result: sum_r(x) + W*3 + sum_r(y) + W*3. Sorcar could combine to AR(x+y+6.0).'''

_P5700_BUILTINS = {'two_ars_with_common_scalar': '''def evolved_p5700(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x + 3.0)
    b = xm.all_reduce(xm.REDUCE_SUM, y + 3.0)
    return a + b
'''}

register_problem(CollectiveProblem(
    name='two_ars_common_scalar_chal',
    display_name='Problem P_5700',
    evolved_fn_name='evolved_p5700',
    signature=_P5700_SIG,
    signature_doc=_P5700_DOC,
    reference_fn=_p5700_ref,
    generate_test_case=_p5700_generate,
    call_candidate=_p5700_call,
    builtin_templates=_P5700_BUILTINS,
))


# ============================================================
# P_5701: sum_of_ars_alternating_ops — AR(SUM)+AR(SUM) but with intermediate transpose
# Baseline: transpose x, AR, transpose back, AR again — 2 ARs of same shape
# Sorcar: recognize transpose is free
# ============================================================
def _p5701_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax * 2).clone() for _ in range(world_size)]  # 2 ARs of same tensor

def _p5701_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(128, 512)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5701_ref(per_rank_args, world_size)}

def _p5701_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5701_SIG = '''def evolved_p5701(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5701_DOC = '''Local x (128, 512).
Compute: 2 * AR(x). Return (128, 512).
Baseline does AR(x) + AR(x) with intervening transposes (net-op is identity).'''

_P5701_BUILTINS = {'two_ars_transposed': '''def evolved_p5701(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x)
    xt = x.t()
    at = xm.all_reduce(xm.REDUCE_SUM, xt)
    return a + at.t()
'''}

register_problem(CollectiveProblem(
    name='two_ars_transposed_chal',
    display_name='Problem P_5701',
    evolved_fn_name='evolved_p5701',
    signature=_P5701_SIG,
    signature_doc=_P5701_DOC,
    reference_fn=_p5701_ref,
    generate_test_case=_p5701_generate,
    call_candidate=_p5701_call,
    builtin_templates=_P5701_BUILTINS,
))


# ============================================================
# P_5702: chained_pow_ar — AR(x**2) then AR(result)  
# Baseline: 2 sequential ARs where the 2nd input depends on 1st AR output
# Sorcar: hard case — can't easily reduce to 1 AR because x**2 is per-rank
# BUT: (AR(x**2))**2 is not the same as AR(x**4) — so second AR is redundant if we
# only want AR(x**2). Make baseline compute AR(x**2) then AR(same) = same.
# ============================================================
def _p5702_ref(inputs, world_size):
    ax_sq = sum(inp['x']**2 for inp in inputs)
    return [ax_sq.clone() for _ in range(world_size)]

def _p5702_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5702_ref(per_rank_args, world_size)}

def _p5702_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5702_SIG = '''def evolved_p5702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5702_DOC = '''Local x (N,). N=65536.
Compute: y = AR(SUM, x**2). Return (N,) identical on every rank.
Baseline computes AR(x**2) then re-ARs the result (which is idempotent for SUM
of an already-reduced value, but only up to a factor of W).'''

_P5702_BUILTINS = {'pow_ar_double': '''def evolved_p5702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x * x)
    # The following AR is REDUNDANT: ax is already reduced across ranks
    # (all ranks have same value). AR(SUM) of a rank-common value = W * value.
    # But we divide by W to undo. So it's dead code.
    verify = xm.all_reduce(xm.REDUCE_SUM, ax) / world_size
    return verify
'''}

register_problem(CollectiveProblem(
    name='pow_ar_double_verify_chal',
    display_name='Problem P_5702',
    evolved_fn_name='evolved_p5702',
    signature=_P5702_SIG,
    signature_doc=_P5702_DOC,
    reference_fn=_p5702_ref,
    generate_test_case=_p5702_generate,
    call_candidate=_p5702_call,
    builtin_templates=_P5702_BUILTINS,
))


# ============================================================
# P_5703: 2d_row_ar — AR each row separately (baseline) vs full 2D AR
# ============================================================
def _p5703_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5703_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    M, N = 8, 8192
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(M, N)*(r+1), 'M': M, 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5703_ref(per_rank_args, world_size)}

def _p5703_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['M'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5703_SIG = '''def evolved_p5703(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5703_DOC = '''Local x (M, N). M=8, N=8192.
Compute: AR(x) over 2D tensor. Return (M, N) identical on every rank.
Baseline ARs each row separately (M dispatches); Sorcar should AR full 2D at once.'''

_P5703_BUILTINS = {'per_row_ar': '''def evolved_p5703(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(M)]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M8_chal',
    display_name='Problem P_5703',
    evolved_fn_name='evolved_p5703',
    signature=_P5703_SIG,
    signature_doc=_P5703_DOC,
    reference_fn=_p5703_ref,
    generate_test_case=_p5703_generate,
    call_candidate=_p5703_call,
    builtin_templates=_P5703_BUILTINS,
))
