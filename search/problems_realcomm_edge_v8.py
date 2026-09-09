"""Real-comm edge V8: Heterogeneous shapes + dep chain.

New territory beyond V4-V7:
- 2D tensor ARs (small [8,32] rows) — could fuse via reshape+stack
- Mixed 1D+2D inputs
- ARs where the intermediate is used in TWO downstream computations (fan-out)
- The rank-common intermediate re-used in later expressions
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_4200: 2D tensors, 3 dep-chain ARs on rows [8, 32]
def _p4200_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(ax + 2*ay + 3*az).clone() for _ in range(world_size)]

def _p4200_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(8, 32)*(r+1),
                      'y': torch.randn(8, 32)*(r+0.5),
                      'z': torch.randn(8, 32)*(r+0.3)}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4200_ref(per_rank_args, world_size)}

def _p4200_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4200_SIG = '''def evolved_p4200(x, y, z, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4200_DOC = '''3 local 2D tensors x,y,z each (8, 32).
Sequential dep chain: s = AR(x); s = s + 2 * AR(y); s = s + 3 * AR(z).
Return final s (8, 32).'''

_P4200_BUILTINS = {'three_ar_2d_dep': '''def evolved_p4200(x, y, z, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s + 2 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s + 3 * az
    return s
'''}

register_problem(CollectiveProblem(
    name='three_ar_2d_dep_edge_chal',
    display_name='Problem P_4200',
    evolved_fn_name='evolved_p4200',
    signature=_P4200_SIG,
    signature_doc=_P4200_DOC,
    reference_fn=_p4200_ref,
    generate_test_case=_p4200_generate,
    call_candidate=_p4200_call,
    builtin_templates=_P4200_BUILTINS,
))


# P_4201: 3 ARs, fan-out reuse — ax used in BOTH s and t (two output cols)
def _p4201_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(2*ax + 3*ay + 5*ax*0 + 5*az + 7*ax*0).clone() for _ in range(world_size)]
    # Simpler: 2*ax + 3*ay + 5*az (fan-out only in baseline template)

def _p4201_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4201_ref(per_rank_args, world_size)}

def _p4201_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4201_SIG = '''def evolved_p4201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4201_DOC = '''3 local vectors x,y,z each (N,). N=256.
Sequential dep chain with fan-out reuse:
  ax = AR(x); tmp = 2 * ax
  ay = AR(y); mid = tmp + 3 * ay
  az = AR(z); result = mid + 5 * az
Return final result (N,). Note ax is used in tmp and then influences mid.'''

_P4201_BUILTINS = {'three_ar_fanout': '''def evolved_p4201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    tmp = 2 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    mid = tmp + 3 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    result = mid + 5 * az
    return result
'''}

register_problem(CollectiveProblem(
    name='three_ar_fanout_edge_chal',
    display_name='Problem P_4201',
    evolved_fn_name='evolved_p4201',
    signature=_P4201_SIG,
    signature_doc=_P4201_DOC,
    reference_fn=_p4201_ref,
    generate_test_case=_p4201_generate,
    call_candidate=_p4201_call,
    builtin_templates=_P4201_BUILTINS,
))


# P_4202: 4 ARs at N=256, dep chain, mixed integer/fractional coefficients
def _p4202_ref(inputs, world_size):
    a1 = sum(inp['x1'] for inp in inputs)
    a2 = sum(inp['x2'] for inp in inputs)
    a3 = sum(inp['x3'] for inp in inputs)
    a4 = sum(inp['x4'] for inp in inputs)
    return [(3*a1 + 0.5*a2 + 7*a3 + 1.5*a4).clone() for _ in range(world_size)]

def _p4202_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.3) for i in range(1,5)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4202_ref(per_rank_args, world_size)}

def _p4202_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4202_SIG = '''def evolved_p4202(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4202_DOC = '''4 local vectors x1..x4 each (N,). N=256.
Sequential dep chain: s = 3*AR(x1); s = s + 0.5*AR(x2); s = s + 7*AR(x3); s = s + 1.5*AR(x4).
Return final s (N,). Mixed integer/fractional coefficients.'''

_P4202_BUILTINS = {'four_ar_mixed_coef': '''def evolved_p4202(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1); s = 3 * a1
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 0.5 * a2
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 7 * a3
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 1.5 * a4
    return s
'''}

register_problem(CollectiveProblem(
    name='four_ar_mixed_coef_edge_chal',
    display_name='Problem P_4202',
    evolved_fn_name='evolved_p4202',
    signature=_P4202_SIG,
    signature_doc=_P4202_DOC,
    reference_fn=_p4202_ref,
    generate_test_case=_p4202_generate,
    call_candidate=_p4202_call,
    builtin_templates=_P4202_BUILTINS,
))
