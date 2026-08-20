"""Diverse Round 15 (V14): more patterns."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6100: sum_squared_norm — global L2 norm squared via ARs
# Baseline: AR(x**2) then sum. Sorcar: sum locally, AR scalar.
# But with a twist: baseline stores intermediate in a tensor then re-ARs.
# ============================================================
def _p6100_ref(inputs, world_size):
    total = sum(inp['x'].pow(2).sum() for inp in inputs)
    return [total.clone() for _ in range(world_size)]

def _p6100_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 131072
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6100_ref(per_rank_args, world_size)}

def _p6100_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6100_SIG = '''def evolved_p6100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6100_DOC = '''Local x (N,). N=131072.
Compute: L2 norm squared globally: sum_i,r (x_r[i]^2). Return scalar.'''

_P6100_BUILTINS = {'ar_before_sum_norm': '''def evolved_p6100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    sq = x.pow(2)
    ar = xm.all_reduce(xm.REDUCE_SUM, sq)   # N bytes
    return ar.sum()
'''}

register_problem(CollectiveProblem(
    name='global_l2_norm_sq_chal',
    display_name='Problem P_6100',
    evolved_fn_name='evolved_p6100',
    signature=_P6100_SIG,
    signature_doc=_P6100_DOC,
    reference_fn=_p6100_ref,
    generate_test_case=_p6100_generate,
    call_candidate=_p6100_call,
    builtin_templates=_P6100_BUILTINS,
))


# ============================================================
# P_6101: rearranged_dot_product — global dot product
# Baseline: AR(x*y) then sum. Sorcar: (x*y).sum() then AR scalar.
# ============================================================
def _p6101_ref(inputs, world_size):
    total = sum((inp['x'] * inp['y']).sum() for inp in inputs)
    return [total.clone() for _ in range(world_size)]

def _p6101_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 131072
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6101_ref(per_rank_args, world_size)}

def _p6101_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6101_SIG = '''def evolved_p6101(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6101_DOC = '''Local x, y (N,). N=131072.
Compute: global dot product = sum_r sum_i x_r[i] * y_r[i]. Return scalar.'''

_P6101_BUILTINS = {'ar_then_dot_sum': '''def evolved_p6101(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    prod = x * y
    ar = xm.all_reduce(xm.REDUCE_SUM, prod)   # N bytes
    return ar.sum()
'''}

register_problem(CollectiveProblem(
    name='global_dot_product_chal',
    display_name='Problem P_6101',
    evolved_fn_name='evolved_p6101',
    signature=_P6101_SIG,
    signature_doc=_P6101_DOC,
    reference_fn=_p6101_ref,
    generate_test_case=_p6101_generate,
    call_candidate=_p6101_call,
    builtin_templates=_P6101_BUILTINS,
))


# ============================================================
# P_6102: ar_max_then_relu — max then relu
# Baseline: AR(x) then relu. Sorcar: no change (relu is not linear).
# But baseline has redundant clamp: AR(x).clamp(0) — same as relu.
# Let sorcar recognize the chained no-op.
# ============================================================
def _p6102_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [torch.relu(ax).clone() for _ in range(world_size)]

def _p6102_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6102_ref(per_rank_args, world_size)}

def _p6102_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6102_SIG = '''def evolved_p6102(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6102_DOC = '''Local x (N,). N=65536.
Compute: y = relu(AR(x)). Return (N,) identical on every rank.
Baseline applies redundant chained clamp/relu operations. Sorcar: single relu.'''

_P6102_BUILTINS = {'ar_relu_dead_chain': '''def evolved_p6102(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    step1 = torch.relu(ax)
    step2 = step1.clamp(min=0)      # dead: already >= 0
    step3 = torch.relu(step2)       # dead: already >= 0
    return step3
'''}

register_problem(CollectiveProblem(
    name='ar_relu_dead_chain_chal',
    display_name='Problem P_6102',
    evolved_fn_name='evolved_p6102',
    signature=_P6102_SIG,
    signature_doc=_P6102_DOC,
    reference_fn=_p6102_ref,
    generate_test_case=_p6102_generate,
    call_candidate=_p6102_call,
    builtin_templates=_P6102_BUILTINS,
))


# ============================================================
# P_6103: ar_by_masked_sum — baseline uses masked_fill(mask, 0) then AR.
# Both baseline and sorcar likely converge; skip.
# 
# Instead: ar_of_matmul_output — AR after matmul
# Baseline: AR of matmul output (matmul-then-AR). Sorcar: AR-then-matmul reduces bandwidth.
# But matmul has different local costs. Just test.
# 
# Skip; do ar_add_scalar_ar_sub_scalar — commutative reordering
# ============================================================
def _p6103_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax * 4).clone() for _ in range(world_size)]

def _p6103_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6103_ref(per_rank_args, world_size)}

def _p6103_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6103_SIG = '''def evolved_p6103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6103_DOC = '''Local x (N,). N=65536.
Compute: y = 4 * AR(x). Return (N,) identical on every rank.
Baseline chains: AR(x) then repeated add/subtract of same scalar (cancels).'''

_P6103_BUILTINS = {'ar_scalar_cancel_chain': '''def evolved_p6103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    y = 4 * ax
    y = y + 100.0      # add
    y = y - 100.0      # subtract (cancels)
    y = y + 50.0
    y = y - 50.0
    return y
'''}

register_problem(CollectiveProblem(
    name='ar_scalar_cancel_chain_chal',
    display_name='Problem P_6103',
    evolved_fn_name='evolved_p6103',
    signature=_P6103_SIG,
    signature_doc=_P6103_DOC,
    reference_fn=_p6103_ref,
    generate_test_case=_p6103_generate,
    call_candidate=_p6103_call,
    builtin_templates=_P6103_BUILTINS,
))
