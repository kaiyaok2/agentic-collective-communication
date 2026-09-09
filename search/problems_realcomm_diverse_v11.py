"""Diverse Round 15 (V11): pattern shrinkage / expansion."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5800: bmm_before_ar — batched matmul then AR — vs AR of inputs then bmm.
# Baseline: local bmm then AR (transfers M*K bytes)
# Sorcar: AR inputs (2 ARs) then bmm — cheaper if N is small. Actually AR of 
# outputs is usually simpler. Skip; test something else.
# 
# Instead: ar_of_norm — AR(||x||^2) then divide by W
# Baseline uses per-elem AR + then sum. Sorcar: sum locally first.
# ============================================================
def _p5800_ref(inputs, world_size):
    # y = sum_over_ranks(x**2 summed over N)
    total = sum(inp['x'].pow(2).sum() for inp in inputs)
    return [total.clone() for _ in range(world_size)]

def _p5800_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5800_ref(per_rank_args, world_size)}

def _p5800_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5800_SIG = '''def evolved_p5800(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5800_DOC = '''Local x (N,). N=65536.
Compute: y = scalar sum of x**2 across all elements across all ranks.
Baseline ARs the elementwise x**2 (N bytes), then sums.
Sorcar: sum locally then AR scalar (1 element).'''

_P5800_BUILTINS = {'ar_before_sum_scalar': '''def evolved_p5800(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    sq = x.pow(2)
    ar = xm.all_reduce(xm.REDUCE_SUM, sq)   # N bytes AR
    return ar.sum()                          # then sum
'''}

register_problem(CollectiveProblem(
    name='ar_before_scalar_reduce_chal',
    display_name='Problem P_5800',
    evolved_fn_name='evolved_p5800',
    signature=_P5800_SIG,
    signature_doc=_P5800_DOC,
    reference_fn=_p5800_ref,
    generate_test_case=_p5800_generate,
    call_candidate=_p5800_call,
    builtin_templates=_P5800_BUILTINS,
))


# ============================================================
# P_5801: ar_after_scalar_reduce — same as above but reverse pattern
# Baseline already sums-then-ARs (efficient). Sorcar should match — control test.
# ============================================================
def _p5801_ref(inputs, world_size):
    total = sum(inp['x'].sum() for inp in inputs)
    return [total.clone() for _ in range(world_size)]

def _p5801_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 262144
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5801_ref(per_rank_args, world_size)}

def _p5801_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5801_SIG = '''def evolved_p5801(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5801_DOC = '''Local x (N,). N=262144.
Compute: scalar sum across all ranks + all elements.
Baseline does: AR(x) then sum. Sorcar can: sum then AR the scalar (much less bytes).'''

_P5801_BUILTINS = {'ar_then_scalar_reduce_large_N': '''def evolved_p5801(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ar = xm.all_reduce(xm.REDUCE_SUM, x)   # N bytes
    return ar.sum()
'''}

register_problem(CollectiveProblem(
    name='ar_then_scalar_reduce_largeN_chal',
    display_name='Problem P_5801',
    evolved_fn_name='evolved_p5801',
    signature=_P5801_SIG,
    signature_doc=_P5801_DOC,
    reference_fn=_p5801_ref,
    generate_test_case=_p5801_generate,
    call_candidate=_p5801_call,
    builtin_templates=_P5801_BUILTINS,
))


# ============================================================
# P_5802: ar_output_multiplied_by_worldsize — AR(x) * W then / W = AR(x)
# Baseline includes redundant multiplication.
# ============================================================
def _p5802_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5802_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5802_ref(per_rank_args, world_size)}

def _p5802_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5802_SIG = '''def evolved_p5802(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5802_DOC = '''Local x (N,). N=65536.
Compute: AR(x). Return (N,) identical on every rank.
Baseline decorates AR with  (algebraic no-op).'''

_P5802_BUILTINS = {'ar_useless_scale': '''def evolved_p5802(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    scaled = ax * world_size    # algebraic no-op combined with next
    return scaled / world_size
'''}

register_problem(CollectiveProblem(
    name='ar_useless_scale_chal',
    display_name='Problem P_5802',
    evolved_fn_name='evolved_p5802',
    signature=_P5802_SIG,
    signature_doc=_P5802_DOC,
    reference_fn=_p5802_ref,
    generate_test_case=_p5802_generate,
    call_candidate=_p5802_call,
    builtin_templates=_P5802_BUILTINS,
))


# ============================================================
# P_5803: partitioned_reduce — AR of x split into odd/even indices
# Baseline: split x into odd/even, AR each separately, interleave.
# Sorcar: recognize concat is meaningless — AR full x.
# ============================================================
def _p5803_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5803_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5803_ref(per_rank_args, world_size)}

def _p5803_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5803_SIG = '''def evolved_p5803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5803_DOC = '''Local x (N,). N=65536.
Compute: AR(x). Return (N,) identical on every rank.
Baseline splits x into even/odd indices, ARs each half, reassembles.'''

_P5803_BUILTINS = {'even_odd_split_ar': '''def evolved_p5803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    even = x[::2]                                # N/2 elements
    odd = x[1::2]
    a_even = xm.all_reduce(xm.REDUCE_SUM, even)
    a_odd = xm.all_reduce(xm.REDUCE_SUM, odd)
    # Interleave back
    result = torch.zeros_like(x)
    result[::2] = a_even
    result[1::2] = a_odd
    return result
'''}

register_problem(CollectiveProblem(
    name='even_odd_split_ar_chal',
    display_name='Problem P_5803',
    evolved_fn_name='evolved_p5803',
    signature=_P5803_SIG,
    signature_doc=_P5803_DOC,
    reference_fn=_p5803_ref,
    generate_test_case=_p5803_generate,
    call_candidate=_p5803_call,
    builtin_templates=_P5803_BUILTINS,
))
