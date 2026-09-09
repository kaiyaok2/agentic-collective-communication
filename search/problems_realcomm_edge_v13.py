"""Real-comm edge V13: try 3-5 ARs with even more sweet-spot variants."""
import torch
from .problems import CollectiveProblem, register_problem


# P_4700: 4 ARs at N=256 with tiny-integer coefs
def _p4700_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,5)]
    coefs = [2, 4, 6, 8]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4700_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.35) for i in range(1,5)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4700_ref(per_rank_args, world_size)}

def _p4700_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4700_SIG = '''def evolved_p4700(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4700_DOC = '''4 local vectors x1..x4 each (N,). N=256.
Sequential dep chain: s = 2*AR(x1); s = s + 4*AR(x2); s = s + 6*AR(x3); s = s + 8*AR(x4).
Return final s (N,). Even integer coefs.'''

_P4700_BUILTINS = {'four_ar_evens': '''def evolved_p4700(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 4 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 6 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 8 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='four_ar_evens_edge_chal',
    display_name='Problem P_4700',
    evolved_fn_name='evolved_p4700',
    signature=_P4700_SIG,
    signature_doc=_P4700_DOC,
    reference_fn=_p4700_ref,
    generate_test_case=_p4700_generate,
    call_candidate=_p4700_call,
    builtin_templates=_P4700_BUILTINS,
))


# P_4701: 3 ARs at N=256 with coefficients 1, 10, 100
def _p4701_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(ax + 10*ay + 100*az).clone() for _ in range(world_size)]

def _p4701_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1)*0.1,
                      'y': torch.randn(N)*(r+0.5)*0.05,
                      'z': torch.randn(N)*(r+0.3)*0.01, 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4701_ref(per_rank_args, world_size)}

def _p4701_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4701_SIG = '''def evolved_p4701(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4701_DOC = '''3 local vectors x,y,z each (N,). N=256.
Sequential dep chain: s = AR(x); s = s + 10*AR(y); s = s + 100*AR(z).
Return final s (N,). Powers of 10 coefficients.'''

_P4701_BUILTINS = {'three_ar_pow10': '''def evolved_p4701(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    s = ax
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    s = s + 10 * ay
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    s = s + 100 * az
    return s
'''}

register_problem(CollectiveProblem(
    name='three_ar_pow10_edge_chal',
    display_name='Problem P_4701',
    evolved_fn_name='evolved_p4701',
    signature=_P4701_SIG,
    signature_doc=_P4701_DOC,
    reference_fn=_p4701_ref,
    generate_test_case=_p4701_generate,
    call_candidate=_p4701_call,
    builtin_templates=_P4701_BUILTINS,
))


# P_4702: 6 ARs at N=256, coefs 1..6 (arithmetic progression)
def _p4702_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,7)]
    coefs = [1, 2, 3, 4, 5, 6]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4702_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.25) for i in range(1,7)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4702_ref(per_rank_args, world_size)}

def _p4702_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'], args['x6'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4702_SIG = '''def evolved_p4702(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4702_DOC = '''6 local vectors x1..x6 each (N,). N=256.
Sequential dep chain: s = AR(x1); s = s + 2*AR(x2); s = s + 3*AR(x3); ...; s = s + 6*AR(x6).
Return final s (N,). Arithmetic progression 1..6.'''

_P4702_BUILTINS = {'six_ar_arith': '''def evolved_p4702(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 4 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x6); s = s + 6 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='six_ar_arith_edge_chal',
    display_name='Problem P_4702',
    evolved_fn_name='evolved_p4702',
    signature=_P4702_SIG,
    signature_doc=_P4702_DOC,
    reference_fn=_p4702_ref,
    generate_test_case=_p4702_generate,
    call_candidate=_p4702_call,
    builtin_templates=_P4702_BUILTINS,
))
