"""Diverse Round 15 (V4): more distinct classes."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5100: ag_then_reduce_local — allgather then compute local reduction
# Baseline: AG all ranks' x, then sum for each rank's row-slice
# Sorcar: AR gives same result without AG
# ============================================================
def _p5100_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5100_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 8192
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5100_ref(per_rank_args, world_size)}

def _p5100_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5100_SIG = '''def evolved_p5100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5100_DOC = '''Local x (N,). N=8192.
Compute: y = sum across all ranks of x = AR(x).
Baseline uses AG then local sum, transferring W*N bytes instead of 2*N/(W).
Return (N,) identical on every rank.'''

_P5100_BUILTINS = {'ag_then_sum': '''def evolved_p5100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    g = xm.all_gather(x, dim=0)           # (world_size * N,) — expensive AG
    return g.reshape(world_size, N).sum(dim=0)   # local sum-across-ranks
'''}

register_problem(CollectiveProblem(
    name='ag_then_local_sum_chal',
    display_name='Problem P_5100',
    evolved_fn_name='evolved_p5100',
    signature=_P5100_SIG,
    signature_doc=_P5100_DOC,
    reference_fn=_p5100_ref,
    generate_test_case=_p5100_generate,
    call_candidate=_p5100_call,
    builtin_templates=_P5100_BUILTINS,
))


# ============================================================
# P_5101: nested_scalar_mul — the AR result gets multiplied by
# rank-common scalar TWO times: c1 * (c2 * AR(x))
# Sorcar: fuse the scalars — return (c1*c2) * AR(x)
# Tests scalar fusion opportunity.
# ============================================================
def _p5101_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax * 6).clone() for _ in range(world_size)]  # 2*3 = 6

def _p5101_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5101_ref(per_rank_args, world_size)}

def _p5101_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5101_SIG = '''def evolved_p5101(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5101_DOC = '''Local x (N,). N=65536.
Compute: y = 2 * (3 * AR(x)) = 6 * AR(x).
Return (N,) identical on every rank.'''

_P5101_BUILTINS = {'nested_scalar': '''def evolved_p5101(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    intermediate = 3 * xm.all_reduce(xm.REDUCE_SUM, x)
    return 2 * intermediate
'''}

register_problem(CollectiveProblem(
    name='nested_scalar_mul_chal',
    display_name='Problem P_5101',
    evolved_fn_name='evolved_p5101',
    signature=_P5101_SIG,
    signature_doc=_P5101_DOC,
    reference_fn=_p5101_ref,
    generate_test_case=_p5101_generate,
    call_candidate=_p5101_call,
    builtin_templates=_P5101_BUILTINS,
))


# ============================================================
# P_5102: reduce_over_ranks_and_dim — 2D tensor, reduce across ranks AND dim0
# Baseline: AR(x) then x.sum(0) — same as P_4902 but sim missed it. Try N=65K.
# ============================================================
def _p5102_ref(inputs, world_size):
    return [sum(inp['x'].sum(dim=0) for inp in inputs).clone() for _ in range(world_size)]

def _p5102_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    M, N = 128, 4096  # 128*4096 = 512K bytes per rank
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(M, N)*(r+1), 'M': M, 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5102_ref(per_rank_args, world_size)}

def _p5102_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['M'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5102_SIG = '''def evolved_p5102(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5102_DOC = '''Local x (M, N). M=128, N=4096.
Compute: y = sum_over_ranks(sum_over_M(x_r)).
Baseline AR-then-sum transfers M*N bytes; sum-then-AR transfers only N bytes.
Return (N,) identical on every rank.'''

_P5102_BUILTINS = {'ar_before_local_reduce': '''def evolved_p5102(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)   # transfers M*N bytes
    return ax.sum(dim=0)
'''}

register_problem(CollectiveProblem(
    name='ar_before_local_reduce_M128_chal',
    display_name='Problem P_5102',
    evolved_fn_name='evolved_p5102',
    signature=_P5102_SIG,
    signature_doc=_P5102_DOC,
    reference_fn=_p5102_ref,
    generate_test_case=_p5102_generate,
    call_candidate=_p5102_call,
    builtin_templates=_P5102_BUILTINS,
))


# ============================================================
# P_5103: idempotent_reduce_max — MAX of MAX is same MAX
# Baseline: nested MAX all_reduces
# Sorcar: drop the outer redundant one
# ============================================================
def _p5103_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    max_x = xs[0]
    for x in xs[1:]:
        max_x = torch.maximum(max_x, x)
    return [max_x.clone() for _ in range(world_size)]

def _p5103_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5103_ref(per_rank_args, world_size)}

def _p5103_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5103_SIG = '''def evolved_p5103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5103_DOC = '''Local x (N,). N=65536.
Compute: elementwise max of x across all ranks.
Baseline sandwiches the MAX AR between two "verification" scalar operations
and MAX AR calls. Idempotence: max(a, max(b, c)) = max(a, b, c) — outer MAX-AR
of an already-reduced tensor is redundant.
Return (N,) identical on every rank.'''

_P5103_BUILTINS = {'three_max_ars': '''def evolved_p5103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    m1 = xm.all_reduce(xm.REDUCE_MAX, x)
    scaled = m1 * 1.0                                  # rank-common no-op
    m2 = xm.all_reduce(xm.REDUCE_MAX, scaled)          # redundant — same result
    m3 = xm.all_reduce(xm.REDUCE_MAX, m2)              # redundant
    return m3
'''}

register_problem(CollectiveProblem(
    name='idempotent_reduce_max_chal',
    display_name='Problem P_5103',
    evolved_fn_name='evolved_p5103',
    signature=_P5103_SIG,
    signature_doc=_P5103_DOC,
    reference_fn=_p5103_ref,
    generate_test_case=_p5103_generate,
    call_candidate=_p5103_call,
    builtin_templates=_P5103_BUILTINS,
))
