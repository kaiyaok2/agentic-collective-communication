"""Real-comm edge problems Round 4 — algebraic-fusion opportunities.

Each problem has:
- baseline template = N naive sequential ARs (heavy)
- Sorcar can find: collapse N ARs into 1 AR of a linear/polynomial combination

Pattern extends Round 3 (P_3700/3701) which produced 1.12-1.40x RT wins.
"""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_3800: quadruple_ar_polynomial — y = a*sum(x) + b*sum(y) + c*sum(z) + d*sum(w)
# ============================================================
def _p3800_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    aw = sum(inp['w'] for inp in inputs)
    return [(2*ax + 3*ay + 5*az + 7*aw).clone() for _ in range(world_size)]

def _p3800_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'w': torch.randn(N)*(r+0.7),
                      'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3800_ref(per_rank_args, world_size)}

def _p3800_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['w'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P3800_SIG = '''def evolved_p3800(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3800_DOC = '''4 local vectors x,y,z,w each (N,). N=256.
Formula: result = 2*sum_r x_r + 3*sum_r y_r + 5*sum_r z_r + 7*sum_r w_r.
Return (N,) tensor equal on every rank.'''

_P3800_BUILTINS = {'four_ar_naive': '''def evolved_p3800(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    aw = xm.all_reduce(xm.REDUCE_SUM, w)
    return 2*ax + 3*ay + 5*az + 7*aw
'''}

register_problem(CollectiveProblem(
    name='quadruple_ar_poly_edge_chal',
    display_name='Problem P_3800',
    evolved_fn_name='evolved_p3800',
    signature=_P3800_SIG,
    signature_doc=_P3800_DOC,
    reference_fn=_p3800_ref,
    generate_test_case=_p3800_generate,
    call_candidate=_p3800_call,
    builtin_templates=_P3800_BUILTINS,
))


# ============================================================
# P_3801: ar_scalar_chain — 3 ARs where output of AR is scaled by a rank-common scalar between
# ============================================================
def _p3801_ref(inputs, world_size):
    W = world_size
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    # y_new = 2 * ax; z_new = 3 * ay; out = 5 * az + y_new + z_new
    # But actual test: 5 * az + 2 * ax + 3 * ay
    return [(2*ax + 3*ay + 5*az).clone() for _ in range(world_size)]

def _p3801_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3801_ref(per_rank_args, world_size)}

def _p3801_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P3801_SIG = '''def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3801_DOC = '''Compute r = 2*AR(x) + 3*AR(y) + 5*AR(z) sequentially.
Return (N,) tensor equal on every rank.'''

_P3801_BUILTINS = {'three_scaled_ar': '''def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    a2 = 2 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    a3 = 3 * ay + a2
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    return 5 * az + a3
'''}

register_problem(CollectiveProblem(
    name='ar_scalar_chain_edge_chal',
    display_name='Problem P_3801',
    evolved_fn_name='evolved_p3801',
    signature=_P3801_SIG,
    signature_doc=_P3801_DOC,
    reference_fn=_p3801_ref,
    generate_test_case=_p3801_generate,
    call_candidate=_p3801_call,
    builtin_templates=_P3801_BUILTINS,
))


# ============================================================
# P_3802: ar_mixed_dims — 3 ARs on tensors of same shape, coefficients depend on rank/world const
# ============================================================
def _p3802_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    W = world_size
    coef_a, coef_b, coef_c = W, W - 1, 2
    return [(coef_a*ax + coef_b*ay + coef_c*az).clone() for _ in range(world_size)]

def _p3802_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3802_ref(per_rank_args, world_size)}

def _p3802_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P3802_SIG = '''def evolved_p3802(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3802_DOC = '''Compute r = W*AR(x) + (W-1)*AR(y) + 2*AR(z), where W = world_size.
Coefficients depend on world_size which is a compile-time constant.
Return (N,) tensor equal on every rank.'''

_P3802_BUILTINS = {'three_ar_worldsize_coef': '''def evolved_p3802(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    return world_size * ax + (world_size - 1) * ay + 2 * az
'''}

register_problem(CollectiveProblem(
    name='ar_worldsize_coef_edge_chal',
    display_name='Problem P_3802',
    evolved_fn_name='evolved_p3802',
    signature=_P3802_SIG,
    signature_doc=_P3802_DOC,
    reference_fn=_p3802_ref,
    generate_test_case=_p3802_generate,
    call_candidate=_p3802_call,
    builtin_templates=_P3802_BUILTINS,
))
