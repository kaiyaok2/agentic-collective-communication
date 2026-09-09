"""Real-comm edge V11: N=256 mixed patterns, up to 9-AR chains.

V4-V10 winners (11 real-comm edge wins) all follow: N=256 + dep-chain ARs.
V11 adds:
- P_4500: 9-AR chain with arithmetic-progression coefs (test deep chains)  
- P_4501: 5 ARs with unrelated non-consecutive coefs (0.3, 4.7, 1.9, ...)
- P_4502: 3-AR with reused single input (x reused twice in accumulator)
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_4500: 9-AR dep chain (deep)
def _p4500_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,10)]
    coefs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4500_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.15) for i in range(1,10)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4500_ref(per_rank_args, world_size)}

def _p4500_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'], args['x6'], args['x7'], args['x8'], args['x9'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4500_SIG = '''def evolved_p4500(x1, x2, x3, x4, x5, x6, x7, x8, x9, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4500_DOC = '''9 local vectors x1..x9 each (N,). N=256.
Sequential dep chain: s = AR(x1); s = s + 2*AR(x2); ...; s = s + 9*AR(x9).
Return final s (N,).'''

_P4500_BUILTINS = {'nine_ar_seq': '''def evolved_p4500(x1, x2, x3, x4, x5, x6, x7, x8, x9, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 4 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x6); s = s + 6 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x7); s = s + 7 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x8); s = s + 8 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x9); s = s + 9 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='nine_ar_seq_edge_chal',
    display_name='Problem P_4500',
    evolved_fn_name='evolved_p4500',
    signature=_P4500_SIG,
    signature_doc=_P4500_DOC,
    reference_fn=_p4500_ref,
    generate_test_case=_p4500_generate,
    call_candidate=_p4500_call,
    builtin_templates=_P4500_BUILTINS,
))


# P_4501: 5 ARs, non-obvious coefs
def _p4501_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,6)]
    coefs = [0.3, 4.7, 1.9, 2.8, 3.4]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4501_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.3) for i in range(1,6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4501_ref(per_rank_args, world_size)}

def _p4501_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4501_SIG = '''def evolved_p4501(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4501_DOC = '''5 local vectors x1..x5 each (N,). N=256.
Sequential dep chain: s = 0.3*AR(x1); s = s + 4.7*AR(x2); s = s + 1.9*AR(x3); s = s + 2.8*AR(x4); s = s + 3.4*AR(x5).
Return final s (N,).'''

_P4501_BUILTINS = {'five_ar_random_coef': '''def evolved_p4501(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 0.3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 4.7 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 1.9 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 2.8 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 3.4 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='five_ar_random_coef_edge_chal',
    display_name='Problem P_4501',
    evolved_fn_name='evolved_p4501',
    signature=_P4501_SIG,
    signature_doc=_P4501_DOC,
    reference_fn=_p4501_ref,
    generate_test_case=_p4501_generate,
    call_candidate=_p4501_call,
    builtin_templates=_P4501_BUILTINS,
))


# P_4502: 6 ARs at N=256 with mixed pos/neg
def _p4502_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,7)]
    coefs = [3, -2, 5, -4, 7, -6]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4502_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.25) for i in range(1,7)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4502_ref(per_rank_args, world_size)}

def _p4502_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'], args['x6'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4502_SIG = '''def evolved_p4502(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4502_DOC = '''6 local vectors x1..x6 each (N,). N=256.
Sequential dep chain: s = 3*AR(x1); s = s - 2*AR(x2); s = s + 5*AR(x3); s = s - 4*AR(x4); s = s + 7*AR(x5); s = s - 6*AR(x6).
Return final s (N,).'''

_P4502_BUILTINS = {'six_ar_altsign': '''def evolved_p4502(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s - 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s - 4 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 7 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x6); s = s - 6 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='six_ar_altsign_edge_chal',
    display_name='Problem P_4502',
    evolved_fn_name='evolved_p4502',
    signature=_P4502_SIG,
    signature_doc=_P4502_DOC,
    reference_fn=_p4502_ref,
    generate_test_case=_p4502_generate,
    call_candidate=_p4502_call,
    builtin_templates=_P4502_BUILTINS,
))
