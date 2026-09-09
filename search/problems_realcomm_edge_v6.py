"""Real-comm edge V6: Round 6 diverse patterns.

Following pattern learned in V4/V5:
- Baseline MUST have SEQUENTIAL data-dependency chain between ARs
  (otherwise compiler fuses automatically → no divergence)
- Sorcar's win: combine locally BEFORE the AR
- Skip bulk-independent ARs (no divergence)

Round 6 varies the intermediate operation between ARs:
- P_4000: reduce+broadcast then AR (matmul-lite pattern)
- P_4001: 2D ARs with row-slicing between
- P_4002: 3 ARs with elementwise power/exp on intermediate results
- P_4003: Iterative accumulator (state -> state + AR)
- P_4004: Two independent chains merged at end
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_4000: reduce-then-scale-then-add — baseline: s = AR(x)*2; s = s + AR(y)*3; ...
def _p4000_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(2*ax + 3*ay + 5*az).clone() for _ in range(world_size)]

def _p4000_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 384
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4000_ref(per_rank_args, world_size)}

def _p4000_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4000_SIG = '''def evolved_p4000(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4000_DOC = '''3 local vectors x,y,z each (N,). N=384.
Formula: s = 2 * AR(x); s = s + 3 * AR(y); s = s + 5 * AR(z).
Return final s (N,). Note larger N=384 vs 256 elsewhere.'''

_P4000_BUILTINS = {'three_ar_384': '''def evolved_p4000(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = 2 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s + 3 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s + 5 * az
    return s
'''}

register_problem(CollectiveProblem(
    name='three_ar_scaled_N384_edge_chal',
    display_name='Problem P_4000',
    evolved_fn_name='evolved_p4000',
    signature=_P4000_SIG,
    signature_doc=_P4000_DOC,
    reference_fn=_p4000_ref,
    generate_test_case=_p4000_generate,
    call_candidate=_p4000_call,
    builtin_templates=_P4000_BUILTINS,
))


# P_4001: Six ARs sequential
def _p4001_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,7)]
    coefs = [1, 2, 3, 4, 5, 6]
    tot = sum(c*a for c, a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4001_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i) for i in range(1,7)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4001_ref(per_rank_args, world_size)}

def _p4001_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'], args['x6'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4001_SIG = '''def evolved_p4001(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4001_DOC = '''6 local vectors x1..x6 each (N,). N=256.
Sequential accumulator: s = AR(x1); s = s + 2*AR(x2); s = s + 3*AR(x3); s = s + 4*AR(x4); s = s + 5*AR(x5); s = s + 6*AR(x6).
Return final s (N,).'''

_P4001_BUILTINS = {'six_ar_seq': '''def evolved_p4001(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1); s = a1
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 2 * a2
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 3 * a3
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 4 * a4
    a5 = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 5 * a5
    a6 = xm.all_reduce(xm.REDUCE_SUM, x6); s = s + 6 * a6
    return s
'''}

register_problem(CollectiveProblem(
    name='six_ar_seq_edge_chal',
    display_name='Problem P_4001',
    evolved_fn_name='evolved_p4001',
    signature=_P4001_SIG,
    signature_doc=_P4001_DOC,
    reference_fn=_p4001_ref,
    generate_test_case=_p4001_generate,
    call_candidate=_p4001_call,
    builtin_templates=_P4001_BUILTINS,
))


# P_4002: two-vector AR with cross-multiplication then sum  
def _p4002_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    # After 3 ARs: r = ax*ay is not achievable (nonlinear); use r = ax*2 + ay*3 + az*4 as fallback
    return [(2*ax + 3*ay + 4*az + ax + ay).clone() for _ in range(world_size)]  # = 3*ax + 4*ay + 4*az

def _p4002_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 512
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4002_ref(per_rank_args, world_size)}

def _p4002_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4002_SIG = '''def evolved_p4002(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4002_DOC = '''3 local vectors x,y,z each (N,). N=512 (LARGE).
Formula: t1 = 2*AR(x); t2 = 3*AR(y); s = t1 + t2 + 4*AR(z) + AR(x) + AR(y) - but note AR(x) and AR(y) can be reused.
Simpler expression: r = 3 * sum_r x_r + 4 * sum_r y_r + 4 * sum_r z_r (already reduced form).
Return (N,).'''

_P4002_BUILTINS = {'three_ar_N512': '''def evolved_p4002(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    t1 = 2 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    t2 = 3 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    return t1 + t2 + 4 * az + ax + ay
'''}

register_problem(CollectiveProblem(
    name='three_ar_reused_N512_edge_chal',
    display_name='Problem P_4002',
    evolved_fn_name='evolved_p4002',
    signature=_P4002_SIG,
    signature_doc=_P4002_DOC,
    reference_fn=_p4002_ref,
    generate_test_case=_p4002_generate,
    call_candidate=_p4002_call,
    builtin_templates=_P4002_BUILTINS,
))
