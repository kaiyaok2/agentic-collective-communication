"""Real-comm edge V12: More sweet-spot variants at N=256, 3-6 ARs.

Boundary conditions confirmed after V11:
- 9 ARs is too many (compiler fuses fully → no divergence)
- Sweet spot is 3-8 ARs at N=256 with per-step accumulator
- Sorcar 12+ wins so far all match this pattern
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_4600: 3 ARs, negative-only coefficients
def _p4600_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(-2*ax - 3*ay - 5*az).clone() for _ in range(world_size)]

def _p4600_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4600_ref(per_rank_args, world_size)}

def _p4600_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4600_SIG = '''def evolved_p4600(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4600_DOC = '''3 local vectors x,y,z each (N,). N=256.
Sequential dep chain: s = -2*AR(x); s = s - 3*AR(y); s = s - 5*AR(z).
Return final s (N,). All negative coefficients.'''

_P4600_BUILTINS = {'three_ar_neg': '''def evolved_p4600(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = -2 * ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s - 3 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s - 5 * az
    return s
'''}

register_problem(CollectiveProblem(
    name='three_ar_neg_edge_chal',
    display_name='Problem P_4600',
    evolved_fn_name='evolved_p4600',
    signature=_P4600_SIG,
    signature_doc=_P4600_DOC,
    reference_fn=_p4600_ref,
    generate_test_case=_p4600_generate,
    call_candidate=_p4600_call,
    builtin_templates=_P4600_BUILTINS,
))


# P_4601: 5 ARs, tiny half-integer coefficients
def _p4601_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,6)]
    coefs = [0.5, 0.5, 0.5, 0.5, 0.5]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4601_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.3) for i in range(1,6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4601_ref(per_rank_args, world_size)}

def _p4601_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4601_SIG = '''def evolved_p4601(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4601_DOC = '''5 local vectors x1..x5 each (N,). N=256.
Sequential dep chain: s = 0.5*AR(x1); s = s + 0.5*AR(x2); ...; s = s + 0.5*AR(x5).
Return final s (N,). Uniform 0.5 coefficient.'''

_P4601_BUILTINS = {'five_ar_half_uniform': '''def evolved_p4601(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 0.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 0.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 0.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 0.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 0.5 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='five_ar_half_uniform_edge_chal',
    display_name='Problem P_4601',
    evolved_fn_name='evolved_p4601',
    signature=_P4601_SIG,
    signature_doc=_P4601_DOC,
    reference_fn=_p4601_ref,
    generate_test_case=_p4601_generate,
    call_candidate=_p4601_call,
    builtin_templates=_P4601_BUILTINS,
))


# P_4602: 4 ARs, geometric coefficients
def _p4602_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,5)]
    coefs = [1, 2, 4, 8]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4602_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.4) for i in range(1,5)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4602_ref(per_rank_args, world_size)}

def _p4602_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4602_SIG = '''def evolved_p4602(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4602_DOC = '''4 local vectors x1..x4 each (N,). N=256.
Sequential dep chain: s = 1*AR(x1); s = s + 2*AR(x2); s = s + 4*AR(x3); s = s + 8*AR(x4).
Return final s (N,). Powers of 2.'''

_P4602_BUILTINS = {'four_ar_pow2': '''def evolved_p4602(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 4 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 8 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='four_ar_pow2_edge_chal',
    display_name='Problem P_4602',
    evolved_fn_name='evolved_p4602',
    signature=_P4602_SIG,
    signature_doc=_P4602_DOC,
    reference_fn=_p4602_ref,
    generate_test_case=_p4602_generate,
    call_candidate=_p4602_call,
    builtin_templates=_P4602_BUILTINS,
))
