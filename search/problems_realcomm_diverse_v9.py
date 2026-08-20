"""Diverse Round 15 (V9): collective_permute-heavy patterns."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5600: cp_ring_rotate — collective_permute forms a ring. If used W times
# it's a full rotation → no-op. Sorcar: recognize.
# ============================================================
def _p5600_ref(inputs, world_size):
    # W collective_permutes each shifts by 1 → full cycle → identity
    return [inp['x'].clone() for inp in inputs]

def _p5600_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 32768
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5600_ref(per_rank_args, world_size)}

def _p5600_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5600_SIG = '''def evolved_p5600(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5600_DOC = '''Local x (N,). N=32768.
Compute: identity operation on x — each rank should return its own original x.
Baseline uses W collective_permutes forming a full ring cycle (net identity).
Sorcar should return x directly.'''

# Note: collective_permute in xm.
_P5600_BUILTINS = {'cp_full_cycle': '''def evolved_p5600(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Build ring: rank i sends to (i+1)%W, receives from (i-1)%W
    pairs = [(i, (i + 1) % world_size) for i in range(world_size)]
    # world_size sequential collective_permutes = full cycle
    y = x
    for _ in range(world_size):
        y = xm.collective_permute(y, pairs)
    return y
'''}

register_problem(CollectiveProblem(
    name='cp_ring_full_cycle_chal',
    display_name='Problem P_5600',
    evolved_fn_name='evolved_p5600',
    signature=_P5600_SIG,
    signature_doc=_P5600_DOC,
    reference_fn=_p5600_ref,
    generate_test_case=_p5600_generate,
    call_candidate=_p5600_call,
    builtin_templates=_P5600_BUILTINS,
))


# ============================================================
# P_5601: cp_pair_swap_twice — swap ranks 0<->1, then swap again = no-op
# ============================================================
def _p5601_ref(inputs, world_size):
    return [inp['x'].clone() for inp in inputs]

def _p5601_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 32768
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5601_ref(per_rank_args, world_size)}

def _p5601_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5601_SIG = '''def evolved_p5601(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5601_DOC = '''Local x (N,). N=32768.
Compute: identity operation.
Baseline swaps pairs (0<->1, 2<->3, ...) twice via collective_permute — net identity.
Sorcar should return x.'''

_P5601_BUILTINS = {'cp_double_swap': '''def evolved_p5601(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    pairs = [(i, i ^ 1) for i in range(world_size)]  # pair swap
    y = xm.collective_permute(x, pairs)
    y = xm.collective_permute(y, pairs)  # swap again = identity
    return y
'''}

register_problem(CollectiveProblem(
    name='cp_double_swap_chal',
    display_name='Problem P_5601',
    evolved_fn_name='evolved_p5601',
    signature=_P5601_SIG,
    signature_doc=_P5601_DOC,
    reference_fn=_p5601_ref,
    generate_test_case=_p5601_generate,
    call_candidate=_p5601_call,
    builtin_templates=_P5601_BUILTINS,
))


# ============================================================
# P_5602: multi_ar_multi_input_same_op — 5 ARs of different inputs
# but result gets summed at end. Compare to concat-then-single-AR.
# Extends P_5001 pattern but with different inputs.
# ============================================================
def _p5602_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1, 6)]
    return [sum(axs).clone() for _ in range(world_size)]

def _p5602_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.2) for i in range(1, 6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5602_ref(per_rank_args, world_size)}

def _p5602_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5602_SIG = '''def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5602_DOC = '''5 local vectors x1..x5 each (N,). N=65536.
Compute: y = AR(x1) + AR(x2) + AR(x3) + AR(x4) + AR(x5) = AR(x1+x2+x3+x4+x5).
Return (N,) identical on every rank.'''

_P5602_BUILTINS = {'five_ar_indep': '''def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    a5 = xm.all_reduce(xm.REDUCE_SUM, x5)
    return a1 + a2 + a3 + a4 + a5
'''}

register_problem(CollectiveProblem(
    name='five_ar_indep_sumatend_chal',
    display_name='Problem P_5602',
    evolved_fn_name='evolved_p5602',
    signature=_P5602_SIG,
    signature_doc=_P5602_DOC,
    reference_fn=_p5602_ref,
    generate_test_case=_p5602_generate,
    call_candidate=_p5602_call,
    builtin_templates=_P5602_BUILTINS,
))


# ============================================================
# P_5603: ar_reduce_op_min_negate — AR(MIN, -x) = -AR(MAX, x)
# Baseline: computes min via -MAX(-x) trick, but with the double negation
# Sorcar: direct MIN AR
# ============================================================
def _p5603_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    mn = xs[0]
    for x in xs[1:]: mn = torch.minimum(mn, x)
    return [mn.clone() for _ in range(world_size)]

def _p5603_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5603_ref(per_rank_args, world_size)}

def _p5603_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5603_SIG = '''def evolved_p5603(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5603_DOC = '''Local x (N,). N=65536.
Compute: elementwise MIN across ranks.
Baseline uses -MAX(-x) trick with extra negation operations.
Return (N,) identical on every rank.'''

_P5603_BUILTINS = {'min_via_max_neg': '''def evolved_p5603(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    neg = -x
    m = xm.all_reduce(xm.REDUCE_MAX, neg)
    return -m
'''}

register_problem(CollectiveProblem(
    name='min_via_max_neg_chal',
    display_name='Problem P_5603',
    evolved_fn_name='evolved_p5603',
    signature=_P5603_SIG,
    signature_doc=_P5603_DOC,
    reference_fn=_p5603_ref,
    generate_test_case=_p5603_generate,
    call_candidate=_p5603_call,
    builtin_templates=_P5603_BUILTINS,
))
