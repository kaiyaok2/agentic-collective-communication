"""Real-comm edge V7: Sweet-spot pattern (N=256, dep-chain intermediates).

Rules learned from V4/V5/V6 RT verification:
- N=256 is the divergence sweet spot (N>=384 causes baseline auto-fusion)
- Baseline MUST have per-step intermediate ops (not bulk-declared ARs)
- Sorcar's win: single stacked AR + local math

V7 tries new intermediate patterns:
- Non-integer coefficients (compiler struggles more)
- Mixed reduce ops: AR + AR-Squared (nonlinear breaks fusion, wait — that's a problem)
  Actually keep everything linear-in-AR; nonlinear breaks the algebraic fusion.
- Broader tuple shapes: 4 dep-chain ARs, tuples reused
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_4100: 4 ARs with fractional coefficients
def _p4100_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    aw = sum(inp['w'] for inp in inputs)
    return [(0.7*ax + 1.3*ay + 2.1*az + 3.7*aw).clone() for _ in range(world_size)]

def _p4100_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'w': torch.randn(N)*(r+0.7),
                      'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4100_ref(per_rank_args, world_size)}

def _p4100_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['w'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4100_SIG = '''def evolved_p4100(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4100_DOC = '''4 local vectors x,y,z,w each (N,). N=256.
Sequential dep chain: s = 0.7 * AR(x); s = s + 1.3 * AR(y); s = s + 2.1 * AR(z); s = s + 3.7 * AR(w).
Return final s (N,).'''

_P4100_BUILTINS = {'four_ar_fractional': '''def evolved_p4100(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = 0.7 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s + 1.3 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s + 2.1 * az
    aw = xm.all_reduce(xm.REDUCE_SUM, w)
    s = s + 3.7 * aw
    return s
'''}

register_problem(CollectiveProblem(
    name='four_ar_fractional_edge_chal',
    display_name='Problem P_4100',
    evolved_fn_name='evolved_p4100',
    signature=_P4100_SIG,
    signature_doc=_P4100_DOC,
    reference_fn=_p4100_ref,
    generate_test_case=_p4100_generate,
    call_candidate=_p4100_call,
    builtin_templates=_P4100_BUILTINS,
))


# P_4101: 5 ARs, mixed positive/negative coefficients, dep chain
def _p4101_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,6)]
    coefs = [1.0, -0.5, 2.5, -1.5, 3.0]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4101_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.5) for i in range(1,6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4101_ref(per_rank_args, world_size)}

def _p4101_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4101_SIG = '''def evolved_p4101(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4101_DOC = '''5 local vectors x1..x5 each (N,). N=256.
Sequential dep chain: s = AR(x1); s = s - 0.5*AR(x2); s = s + 2.5*AR(x3); s = s - 1.5*AR(x4); s = s + 3*AR(x5).
Return final s (N,).'''

_P4101_BUILTINS = {'five_ar_mixed_sign': '''def evolved_p4101(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1); s = a1
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2); s = s - 0.5 * a2
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 2.5 * a3
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4); s = s - 1.5 * a4
    a5 = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 3.0 * a5
    return s
'''}

register_problem(CollectiveProblem(
    name='five_ar_mixed_sign_edge_chal',
    display_name='Problem P_4101',
    evolved_fn_name='evolved_p4101',
    signature=_P4101_SIG,
    signature_doc=_P4101_DOC,
    reference_fn=_p4101_ref,
    generate_test_case=_p4101_generate,
    call_candidate=_p4101_call,
    builtin_templates=_P4101_BUILTINS,
))


# P_4102: 3 ARs at N=256, dep chain with per-step .mul_ style ops
def _p4102_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    # s = 2.5 * ax; then s = s * 1.0 + 3.5 * ay; then s = s + 7.5 * az
    # Simpler: 2.5*ax + 3.5*ay + 7.5*az
    return [(2.5*ax + 3.5*ay + 7.5*az).clone() for _ in range(world_size)]

def _p4102_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4102_ref(per_rank_args, world_size)}

def _p4102_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4102_SIG = '''def evolved_p4102(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4102_DOC = '''3 local vectors x,y,z each (N,). N=256.
Sequential dep chain: s = 2.5 * AR(x); s = s + 3.5 * AR(y); s = s + 7.5 * AR(z).
Return final s (N,).'''

_P4102_BUILTINS = {'three_ar_frac_dep': '''def evolved_p4102(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = 2.5 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s + 3.5 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s + 7.5 * az
    return s
'''}

register_problem(CollectiveProblem(
    name='three_ar_frac_dep_edge_chal',
    display_name='Problem P_4102',
    evolved_fn_name='evolved_p4102',
    signature=_P4102_SIG,
    signature_doc=_P4102_DOC,
    reference_fn=_p4102_ref,
    generate_test_case=_p4102_generate,
    call_candidate=_p4102_call,
    builtin_templates=_P4102_BUILTINS,
))
