"""Diverse Round 15 (V12): more classes."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5900: nested_conditional_ar — condition that's always true (rank-common)
# Baseline: if some_condition(rank, W): AR(x) + something; else: AR(x)
# Sorcar: recognize the branch selection is fixed at compile time.
# But the branch is rank-dependent... skip.
# 
# Instead: baseline uses  — mask is applied to same AR
# ============================================================
def _p5900_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5900_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5900_ref(per_rank_args, world_size)}

def _p5900_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5900_SIG = '''def evolved_p5900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5900_DOC = '''Local x (N,). N=65536.
Compute: AR(x). Return (N,) identical on every rank.
Baseline uses  where ar1 and ar2 are both AR(x)
— redundant branch. Both branches identical.'''

_P5900_BUILTINS = {'ar_where_dead': '''def evolved_p5900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ar1 = xm.all_reduce(xm.REDUCE_SUM, x)
    ar2 = xm.all_reduce(xm.REDUCE_SUM, x)
    mask = torch.ones(N, dtype=torch.bool, device=x.device)
    return torch.where(mask, ar1, ar2)
'''}

register_problem(CollectiveProblem(
    name='ar_where_dead_chal',
    display_name='Problem P_5900',
    evolved_fn_name='evolved_p5900',
    signature=_P5900_SIG,
    signature_doc=_P5900_DOC,
    reference_fn=_p5900_ref,
    generate_test_case=_p5900_generate,
    call_candidate=_p5900_call,
    builtin_templates=_P5900_BUILTINS,
))


# ============================================================
# P_5901: ar_of_mask_applied — AR of masked tensor where mask is rank-specific
# Baseline: mask x by rank_id then AR — different data per rank
# Sorcar: same as baseline (mask is rank-specific → can't optimize).
# Control test.
# 
# Actually let me flip: baseline applies mask, ARs, then applies same mask
# Sorcar: recognize the mask outside AR is redundant if AR(SUM) with mask is done inside
# ============================================================
def _p5901_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    # ax[::2] masked = keep every other, other = 0
    result = ax.clone()
    result[1::2] = 0
    return [result.clone() for _ in range(world_size)]

def _p5901_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5901_ref(per_rank_args, world_size)}

def _p5901_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5901_SIG = '''def evolved_p5901(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5901_DOC = '''Local x (N,). N=65536.
Compute: apply zero-out-odd-indices mask, then AR. Result: zeros at odd indices
of AR(x) — since mask is rank-common (fixed pattern), applying before/after AR
gives same result.
Return (N,) identical on every rank.'''

_P5901_BUILTINS = {'mask_after_ar_dead': '''def evolved_p5901(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # First apply mask (kill odd indices before AR)
    xm_ = x.clone()
    xm_[1::2] = 0
    a = xm.all_reduce(xm.REDUCE_SUM, xm_)
    # Then re-apply mask (redundant — already zero at those indices)
    a[1::2] = 0
    return a
'''}

register_problem(CollectiveProblem(
    name='mask_apply_twice_around_ar_chal',
    display_name='Problem P_5901',
    evolved_fn_name='evolved_p5901',
    signature=_P5901_SIG,
    signature_doc=_P5901_DOC,
    reference_fn=_p5901_ref,
    generate_test_case=_p5901_generate,
    call_candidate=_p5901_call,
    builtin_templates=_P5901_BUILTINS,
))


# ============================================================
# P_5902: ar_of_ar_squared — AR(x^2) computed via 2 sequential ARs
# Baseline: y = AR(x); return y**2 (WRONG: this gives (sum_r x)^2, not sum_r x^2)
# But if we want AR(x^2): sorcar should compute x**2 locally then AR.
# ============================================================
def _p5902_ref(inputs, world_size):
    # y = sum_r x_r^2
    total = sum(inp['x'].pow(2) for inp in inputs)
    return [total.clone() for _ in range(world_size)]

def _p5902_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5902_ref(per_rank_args, world_size)}

def _p5902_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5902_SIG = '''def evolved_p5902(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5902_DOC = '''Local x (N,). N=65536.
Compute: y = sum_r x_r^2 (sum of squares across ranks). Return (N,).
Baseline squares AFTER AR which is arithmetically wrong (gives (sum)^2 not sum(^2))
but might use  pattern. Sorcar: local square, then AR.'''

# Actually baseline should be: AR(x**2) — that's already correct. Add wasteful ops.
_P5902_BUILTINS = {'square_before_ar_wasteful': '''def evolved_p5902(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    sq = x * x
    a = xm.all_reduce(xm.REDUCE_SUM, sq)
    # Wasteful: sqrt then square is identity for non-negative values
    root = a.abs().sqrt()
    return root * root
'''}

register_problem(CollectiveProblem(
    name='sqrt_square_after_ar_chal',
    display_name='Problem P_5902',
    evolved_fn_name='evolved_p5902',
    signature=_P5902_SIG,
    signature_doc=_P5902_DOC,
    reference_fn=_p5902_ref,
    generate_test_case=_p5902_generate,
    call_candidate=_p5902_call,
    builtin_templates=_P5902_BUILTINS,
))


# ============================================================
# P_5903: two_ars_intersected_by_prefix_sum — 2 sequential ARs with 
# local cumulative sum between. Cumulative sum requires the AR-1 result.
# ============================================================
def _p5903_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)  # AR of x
    # cumsum locally (rank-common on the reduced tensor)
    csum = torch.cumsum(ax, dim=0)
    ay = sum(inp['y'] for inp in inputs)
    return [(csum + ay).clone() for _ in range(world_size)]

def _p5903_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 1024  # cumsum can be slow on Neuron
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5903_ref(per_rank_args, world_size)}

def _p5903_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5903_SIG = '''def evolved_p5903(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5903_DOC = '''Local x, y (N,). N=1024.
Compute: y_out = cumsum(AR(x)) + AR(y). Return (N,).'''

_P5903_BUILTINS = {'two_ars_prefix_sum': '''def evolved_p5903(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    csum = torch.cumsum(ax, dim=0)
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    return csum + ay
'''}

register_problem(CollectiveProblem(
    name='two_ars_prefix_sum_chal',
    display_name='Problem P_5903',
    evolved_fn_name='evolved_p5903',
    signature=_P5903_SIG,
    signature_doc=_P5903_DOC,
    reference_fn=_p5903_ref,
    generate_test_case=_p5903_generate,
    call_candidate=_p5903_call,
    builtin_templates=_P5903_BUILTINS,
))
