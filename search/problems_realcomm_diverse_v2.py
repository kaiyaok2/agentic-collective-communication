"""Diverse Round 15 (V2): more distinct optimization classes."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_4900: reduce_scatter_from_ar — AR + slice = RS
# Baseline: AR full tensor then keep own slice
# Sorcar: use reduce_scatter to only get own portion (1/W bandwidth)
# ============================================================
def _p4900_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)  # (world_size * N,)
    N = inputs[0]['N']
    return [ax.narrow(0, r * N, N).clone() for r in range(world_size)]

def _p4900_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 16384
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(world_size * N)*(r+1), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4900_ref(per_rank_args, world_size)}

def _p4900_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4900_SIG = '''def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4900_DOC = '''Local x (world_size * N,). N=16384 (per-rank slice size).
Compute: reduce SUM across all ranks, then each rank keeps only its own slice
(x[rank*N:(rank+1)*N] of the reduced tensor).
Return (N,) — own slice of the reduced.'''

_P4900_BUILTINS = {'ar_then_narrow': '''def evolved_p4900(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)     # bandwidth = world_size * N
    return ax.narrow(0, rank * N, N)         # only need N bytes but transferred W*N
'''}

register_problem(CollectiveProblem(
    name='reduce_scatter_from_ar_chal',
    display_name='Problem P_4900',
    evolved_fn_name='evolved_p4900',
    signature=_P4900_SIG,
    signature_doc=_P4900_DOC,
    reference_fn=_p4900_ref,
    generate_test_case=_p4900_generate,
    call_candidate=_p4900_call,
    builtin_templates=_P4900_BUILTINS,
))


# ============================================================
# P_4901: constant_add_folding — compile-time constant + AR
# Baseline: c = torch.full((N,), 5.0); return AR(x + c)
# Sorcar: AR(x) + W*5  — pull the constant out of the AR.
# ============================================================
def _p4901_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    N = inputs[0]['N']
    return [(ax + 5.0 * world_size).clone() for _ in range(world_size)]

def _p4901_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4901_ref(per_rank_args, world_size)}

def _p4901_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4901_SIG = '''def evolved_p4901(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4901_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x + 5.0) where 5.0 is a compile-time constant added elementwise
before the reduce. Result: sum_r(x_r) + world_size * 5.0.
Return (N,) identical on every rank.'''

_P4901_BUILTINS = {'ar_of_x_plus_const': '''def evolved_p4901(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x + 5.0)
'''}

register_problem(CollectiveProblem(
    name='constant_add_folding_chal',
    display_name='Problem P_4901',
    evolved_fn_name='evolved_p4901',
    signature=_P4901_SIG,
    signature_doc=_P4901_DOC,
    reference_fn=_p4901_ref,
    generate_test_case=_p4901_generate,
    call_candidate=_p4901_call,
    builtin_templates=_P4901_BUILTINS,
))


# ============================================================
# P_4902: local_reduce_then_ar — sum over a local dim first, then AR
# Baseline: AR of full 2D, then sum over dim 0 locally
# Sorcar: sum locally first (reduces per-rank bytes), then AR the vector
# ============================================================
def _p4902_ref(inputs, world_size):
    # x is (M, N) per rank; result = sum_over_ranks(sum_over_M(x))
    return [sum(inp['x'].sum(dim=0) for inp in inputs).clone() for _ in range(world_size)]

def _p4902_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    M, N = 32, 4096  # per-rank 2D
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(M, N)*(r+1), 'M': M, 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4902_ref(per_rank_args, world_size)}

def _p4902_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['M'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4902_SIG = '''def evolved_p4902(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4902_DOC = '''Local x (M, N). M=32, N=4096.
Compute: y = sum_over_ranks(sum_over_M(x_r)) — sum along dim 0 then reduce across ranks.
Baseline does AR of full (M, N) then sums; efficient version sums locally first.
Return (N,) identical on every rank.'''

_P4902_BUILTINS = {'ar_full_then_reduce': '''def evolved_p4902(x, M, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)   # transfer M*N bytes
    return ax.sum(dim=0)                   # sum-M happens post-AR
'''}

register_problem(CollectiveProblem(
    name='local_reduce_then_ar_chal',
    display_name='Problem P_4902',
    evolved_fn_name='evolved_p4902',
    signature=_P4902_SIG,
    signature_doc=_P4902_DOC,
    reference_fn=_p4902_ref,
    generate_test_case=_p4902_generate,
    call_candidate=_p4902_call,
    builtin_templates=_P4902_BUILTINS,
))


# ============================================================
# P_4903: consecutive_scaled_ars — c1*AR(x) then c2*AR(y) then c3*AR(z)
#   but c1, c2, c3 are RANK-SPECIFIC scalars (broadcast constants)
# Rank-common intermediates get compiler-fused; rank-specific don't.
# Test whether Sorcar handles rank-dependent coefficient patterns.
# ============================================================
def _p4903_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(2*ax + 3*ay + 5*az).clone() for _ in range(world_size)]

def _p4903_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4903_ref(per_rank_args, world_size)}

def _p4903_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4903_SIG = '''def evolved_p4903(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4903_DOC = '''Local x, y, z (N,). N=65536.
Compute: r = 2*AR(x) + 3*AR(y) + 5*AR(z).
Baseline scaffolds intermediate accumulator with 
which is inefficient; the sum can be done pre-AR.
Return (N,) identical on every rank.'''

_P4903_BUILTINS = {'ar_stack_narrow': '''def evolved_p4903(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    stacked = torch.stack([2*x, 3*y, 5*z])
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    return reduced[0] + reduced[1] + reduced[2]
'''}

register_problem(CollectiveProblem(
    name='ar_stack_narrow_chal',
    display_name='Problem P_4903',
    evolved_fn_name='evolved_p4903',
    signature=_P4903_SIG,
    signature_doc=_P4903_DOC,
    reference_fn=_p4903_ref,
    generate_test_case=_p4903_generate,
    call_candidate=_p4903_call,
    builtin_templates=_P4903_BUILTINS,
))
