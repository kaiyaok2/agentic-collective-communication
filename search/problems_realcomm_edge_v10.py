"""Real-comm edge V10: More sweet-spot variants.

Extend the N=256, 4-6 dep-chain-AR winning pattern:
- P_4400: 5 ARs with float coefficients (variant of P_4101 winner)
- P_4401: 6 ARs with tiny int coefficients
- P_4402: 4 ARs at N=224 (smaller than sweet spot 256; check divergence still holds)
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_4400: 5 ARs, float coefs 0.5..2.5
def _p4400_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,6)]
    coefs = [0.5, 1.0, 1.5, 2.0, 2.5]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4400_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.3) for i in range(1,6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4400_ref(per_rank_args, world_size)}

def _p4400_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4400_SIG = '''def evolved_p4400(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4400_DOC = '''5 local vectors x1..x5 each (N,). N=256.
Sequential dep chain: s = 0.5*AR(x1); s = s + 1.0*AR(x2); s = s + 1.5*AR(x3); s = s + 2.0*AR(x4); s = s + 2.5*AR(x5).
Return final s (N,).'''

_P4400_BUILTINS = {'five_ar_arith_prog': '''def evolved_p4400(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 0.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 1.0 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 1.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 2.0 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 2.5 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='five_ar_arith_prog_edge_chal',
    display_name='Problem P_4400',
    evolved_fn_name='evolved_p4400',
    signature=_P4400_SIG,
    signature_doc=_P4400_DOC,
    reference_fn=_p4400_ref,
    generate_test_case=_p4400_generate,
    call_candidate=_p4400_call,
    builtin_templates=_P4400_BUILTINS,
))


# P_4401: 8 ARs, half-integer coefs
def _p4401_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,9)]
    coefs = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4401_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.15) for i in range(1,9)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4401_ref(per_rank_args, world_size)}

def _p4401_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'], args['x6'], args['x7'], args['x8'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4401_SIG = '''def evolved_p4401(x1, x2, x3, x4, x5, x6, x7, x8, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4401_DOC = '''8 local vectors x1..x8 each (N,). N=256.
Sequential dep chain: s = 0.5*AR(x1); s = s + 1.5*AR(x2); s = s + 2.5*AR(x3); ...; s = s + 7.5*AR(x8).
Return final s (N,). Half-integer coefficients.'''

_P4401_BUILTINS = {'eight_ar_half_ints': '''def evolved_p4401(x1, x2, x3, x4, x5, x6, x7, x8, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 0.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 1.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 2.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 3.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 4.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x6); s = s + 5.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x7); s = s + 6.5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x8); s = s + 7.5 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='eight_ar_half_ints_edge_chal',
    display_name='Problem P_4401',
    evolved_fn_name='evolved_p4401',
    signature=_P4401_SIG,
    signature_doc=_P4401_DOC,
    reference_fn=_p4401_ref,
    generate_test_case=_p4401_generate,
    call_candidate=_p4401_call,
    builtin_templates=_P4401_BUILTINS,
))


# P_4402: 4 ARs at N=224
def _p4402_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,5)]
    coefs = [2, 3, 5, 7]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4402_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 224
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.4) for i in range(1,5)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4402_ref(per_rank_args, world_size)}

def _p4402_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4402_SIG = '''def evolved_p4402(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4402_DOC = '''4 local vectors x1..x4 each (N,). N=224.
Sequential dep chain: s = 2*AR(x1); s = s + 3*AR(x2); s = s + 5*AR(x3); s = s + 7*AR(x4).
Return final s (N,).'''

_P4402_BUILTINS = {'four_ar_N224': '''def evolved_p4402(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 7 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='four_ar_N224_edge_chal',
    display_name='Problem P_4402',
    evolved_fn_name='evolved_p4402',
    signature=_P4402_SIG,
    signature_doc=_P4402_DOC,
    reference_fn=_p4402_ref,
    generate_test_case=_p4402_generate,
    call_candidate=_p4402_call,
    builtin_templates=_P4402_BUILTINS,
))
