"""Real-comm edge V5: 'data-dependency-chain' pattern.

Key insight from V4: 'ar_scalar_chain' won 1.18x RT because baseline had
sequential intermediate ops between ARs (a2 = 2*ax; ay = AR(y); a3 = 3*ay + a2; ...),
which prevents Neuron compiler from parallelizing/fusing the ARs.
Sorcar's stack-then-single-AR pattern removes that dependency chain.

Contrast: quadruple_ar_poly baseline declares all 4 ARs upfront (independent)
so compiler already fuses them → no divergence.

Round 5: create MORE problems with the winning 'sequential dep chain' pattern.
"""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_3900: seq_dep_chain_5 — 5 ARs in a sequential dep chain
# baseline: ax1 = AR(x1); s1 = ax1 * c1; ax2 = AR(x2); s2 = s1 + ax2*c2; ...
# sorcar can: stack all 5, single AR, then unpack
# ============================================================
def _p3900_ref(inputs, world_size):
    ax1 = sum(inp['x1'] for inp in inputs)
    ax2 = sum(inp['x2'] for inp in inputs)
    ax3 = sum(inp['x3'] for inp in inputs)
    ax4 = sum(inp['x4'] for inp in inputs)
    ax5 = sum(inp['x5'] for inp in inputs)
    return [(ax1 + 2*ax2 + 3*ax3 + 4*ax4 + 5*ax5).clone() for _ in range(world_size)]

def _p3900_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i) for i in range(1,6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3900_ref(per_rank_args, world_size)}

def _p3900_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P3900_SIG = '''def evolved_p3900(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3900_DOC = '''5 local vectors x1..x5 each (N,). N=256.
Formula: result = AR(x1) + 2*AR(x2) + 3*AR(x3) + 4*AR(x4) + 5*AR(x5).
Return (N,) tensor identical on every rank.'''

_P3900_BUILTINS = {'five_ar_seq_dep': '''def evolved_p3900(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    s = a1
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    s = s + 2 * a2
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    s = s + 3 * a3
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    s = s + 4 * a4
    a5 = xm.all_reduce(xm.REDUCE_SUM, x5)
    s = s + 5 * a5
    return s
'''}

register_problem(CollectiveProblem(
    name='seq_dep_chain5_edge_chal',
    display_name='Problem P_3900',
    evolved_fn_name='evolved_p3900',
    signature=_P3900_SIG,
    signature_doc=_P3900_DOC,
    reference_fn=_p3900_ref,
    generate_test_case=_p3900_generate,
    call_candidate=_p3900_call,
    builtin_templates=_P3900_BUILTINS,
))


# ============================================================
# P_3901: seq_dep_chain_with_scalars — same but with per-step scaling
# ============================================================
def _p3901_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    aw = sum(inp['w'] for inp in inputs)
    # step-by-step: s = ax * 1.5; s = s + ay * 2.5; s = s + az * 3.5; s = s + aw * 4.5
    return [(1.5*ax + 2.5*ay + 3.5*az + 4.5*aw).clone() for _ in range(world_size)]

def _p3901_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'w': torch.randn(N)*(r+0.7),
                      'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3901_ref(per_rank_args, world_size)}

def _p3901_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['w'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P3901_SIG = '''def evolved_p3901(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3901_DOC = '''4 local vectors x,y,z,w each (N,). N=256.
Formula: sequential accumulator. Start with s = 1.5 * AR(x). Then s = s + 2.5 * AR(y).
Then s = s + 3.5 * AR(z). Then s = s + 4.5 * AR(w). Return final s (N,).'''

_P3901_BUILTINS = {'four_ar_seq_scaled': '''def evolved_p3901(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = 1.5 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s + 2.5 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s + 3.5 * az
    aw = xm.all_reduce(xm.REDUCE_SUM, w)
    s = s + 4.5 * aw
    return s
'''}

register_problem(CollectiveProblem(
    name='seq_dep_chain4_scaled_edge_chal',
    display_name='Problem P_3901',
    evolved_fn_name='evolved_p3901',
    signature=_P3901_SIG,
    signature_doc=_P3901_DOC,
    reference_fn=_p3901_ref,
    generate_test_case=_p3901_generate,
    call_candidate=_p3901_call,
    builtin_templates=_P3901_BUILTINS,
))


# ============================================================
# P_3902: alternating_add_sub — signs alternate
# baseline: s = AR(x1); s = s - AR(x2); s = s + AR(x3); s = s - AR(x4)
# sorcar: stack with signs
# ============================================================
def _p3902_ref(inputs, world_size):
    a1 = sum(inp['x1'] for inp in inputs)
    a2 = sum(inp['x2'] for inp in inputs)
    a3 = sum(inp['x3'] for inp in inputs)
    a4 = sum(inp['x4'] for inp in inputs)
    return [(a1 - a2 + a3 - a4).clone() for _ in range(world_size)]

def _p3902_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i) for i in range(1,5)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3902_ref(per_rank_args, world_size)}

def _p3902_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P3902_SIG = '''def evolved_p3902(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3902_DOC = '''4 local vectors x1..x4 each (N,). N=256.
Formula: alternating accumulator. s = AR(x1); s = s - AR(x2); s = s + AR(x3); s = s - AR(x4).
Return final s (N,).'''

_P3902_BUILTINS = {'four_ar_alt': '''def evolved_p3902(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    s = a1
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    s = s - a2
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    s = s + a3
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    s = s - a4
    return s
'''}

register_problem(CollectiveProblem(
    name='alt_add_sub_edge_chal',
    display_name='Problem P_3902',
    evolved_fn_name='evolved_p3902',
    signature=_P3902_SIG,
    signature_doc=_P3902_DOC,
    reference_fn=_p3902_ref,
    generate_test_case=_p3902_generate,
    call_candidate=_p3902_call,
    builtin_templates=_P3902_BUILTINS,
))
