"""Diverse Round 15 (V15): more variations."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6200: bigar_repeated — same big AR result gets used in 3 different sums
# Baseline: computes AR(x) once, uses in 3 places. Sorcar: no change.
# But then adds "verification" AR that recomputes. Sorcar: recognize.
# ============================================================
def _p6200_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax + ax * 2 + ax * 3).clone() for _ in range(world_size)]  # = 6 * ax

def _p6200_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6200_ref(per_rank_args, world_size)}

def _p6200_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6200_SIG = '''def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6200_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x) + 2*AR(x) + 3*AR(x) = 6*AR(x). Return (N,).
Baseline calls AR 3 times inline. Sorcar should call once.'''

_P6200_BUILTINS = {'three_inline_ars': '''def evolved_p6200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x)
    b = 2 * xm.all_reduce(xm.REDUCE_SUM, x)
    c = 3 * xm.all_reduce(xm.REDUCE_SUM, x)
    return a + b + c
'''}

register_problem(CollectiveProblem(
    name='three_inline_ars_chal',
    display_name='Problem P_6200',
    evolved_fn_name='evolved_p6200',
    signature=_P6200_SIG,
    signature_doc=_P6200_DOC,
    reference_fn=_p6200_ref,
    generate_test_case=_p6200_generate,
    call_candidate=_p6200_call,
    builtin_templates=_P6200_BUILTINS,
))


# ============================================================
# P_6201: nested_max — MAX(MAX(x, y), z) via ARs
# Baseline: AR(MAX, x), AR(MAX, y) then MAX pair-wise
# Sorcar: single AR(MAX, MAX(x, MAX(y, z))) or stack
# ============================================================
def _p6201_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    ys = [inp['y'] for inp in inputs]
    zs = [inp['z'] for inp in inputs]
    result = xs[0]
    for a in xs[1:] + ys + zs: result = torch.maximum(result, a)
    return [result.clone() for _ in range(world_size)]

def _p6201_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6201_ref(per_rank_args, world_size)}

def _p6201_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6201_SIG = '''def evolved_p6201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6201_DOC = '''Local x, y, z (N,). N=65536.
Compute: elementwise max across ranks of the max of (x, y, z).
Baseline does 3 separate REDUCE_MAX ARs then pairwise max. Sorcar: 1 AR of local max.'''

_P6201_BUILTINS = {'three_max_ars_pairwise': '''def evolved_p6201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    my = xm.all_reduce(xm.REDUCE_MAX, y)
    mz = xm.all_reduce(xm.REDUCE_MAX, z)
    return torch.maximum(torch.maximum(mx, my), mz)
'''}

register_problem(CollectiveProblem(
    name='three_max_ars_pairwise_chal',
    display_name='Problem P_6201',
    evolved_fn_name='evolved_p6201',
    signature=_P6201_SIG,
    signature_doc=_P6201_DOC,
    reference_fn=_p6201_ref,
    generate_test_case=_p6201_generate,
    call_candidate=_p6201_call,
    builtin_templates=_P6201_BUILTINS,
))


# ============================================================
# P_6202: alternate_add_sub_indep — a-b+c-d+e where each is AR of different input
# Baseline: 5 ARs sequential. Sorcar: linear combination.
# But make it different by using ADD/SUB alternating.
# ============================================================
def _p6202_ref(inputs, world_size):
    ax = sum(inp['x1'] for inp in inputs)
    ay = sum(inp['x2'] for inp in inputs)
    az = sum(inp['x3'] for inp in inputs)
    aw = sum(inp['x4'] for inp in inputs)
    av = sum(inp['x5'] for inp in inputs)
    return [(ax - ay + az - aw + av).clone() for _ in range(world_size)]

def _p6202_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.2) for i in range(1, 6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6202_ref(per_rank_args, world_size)}

def _p6202_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6202_SIG = '''def evolved_p6202(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6202_DOC = '''5 local vectors x1..x5 each (N,). N=65536.
Compute: y = AR(x1) - AR(x2) + AR(x3) - AR(x4) + AR(x5).
Return (N,) identical on every rank.'''

_P6202_BUILTINS = {'alternating_indep_ars': '''def evolved_p6202(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    a5 = xm.all_reduce(xm.REDUCE_SUM, x5)
    return a1 - a2 + a3 - a4 + a5
'''}

register_problem(CollectiveProblem(
    name='alternating_indep_ars_chal',
    display_name='Problem P_6202',
    evolved_fn_name='evolved_p6202',
    signature=_P6202_SIG,
    signature_doc=_P6202_DOC,
    reference_fn=_p6202_ref,
    generate_test_case=_p6202_generate,
    call_candidate=_p6202_call,
    builtin_templates=_P6202_BUILTINS,
))


# ============================================================
# P_6203: partition_reduce_then_ar_combine — different logic per rank
# Different ranks contribute to different indices; sum across ranks.
# Baseline uses masking + AR. Sorcar: no clear optimization; control.
# 
# Instead: multi_reduce_op_dead_chain
# Baseline: AR(SUM), AR(SUM) of scaled, AR(SUM) again
# All 3 ARs are of scaled(x) = c*x → single AR
# ============================================================
def _p6203_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax * 5).clone() for _ in range(world_size)]  # 5 = 1+2+2

def _p6203_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6203_ref(per_rank_args, world_size)}

def _p6203_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6203_SIG = '''def evolved_p6203(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6203_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x) + AR(2*x) + AR(2*x). 
= AR(x) + 2*AR(x) + 2*AR(x) = 5*AR(x). Return (N,).'''

_P6203_BUILTINS = {'three_scaled_indep': '''def evolved_p6203(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x)
    b = xm.all_reduce(xm.REDUCE_SUM, 2 * x)
    c = xm.all_reduce(xm.REDUCE_SUM, 2 * x)
    return a + b + c
'''}

register_problem(CollectiveProblem(
    name='three_scaled_x_ars_chal',
    display_name='Problem P_6203',
    evolved_fn_name='evolved_p6203',
    signature=_P6203_SIG,
    signature_doc=_P6203_DOC,
    reference_fn=_p6203_ref,
    generate_test_case=_p6203_generate,
    call_candidate=_p6203_call,
    builtin_templates=_P6203_BUILTINS,
))
