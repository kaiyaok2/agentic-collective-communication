"""Real-comm edge V9: 3 variants + a heterogeneous size problem.

Rules re-affirmed by V4-V8:
- Winners at N=256, 3-6 dep-chain ARs
- Losers: bulk-declared ARs (auto-fused), large N (auto-fused), fan-out reuse
  (breaks Sorcar's stack-then-one-AR pattern because the reused intermediate 
  can't be captured in a single AR call)

V9 targets: 
- 7 ARs to push beyond V5's 5-AR and V6's 6-AR
- Different scalar coefficients per step
- 3 ARs with subtract-chain (test if Sorcar handles negatives OK)
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_4300: 7 ARs dep chain
def _p4300_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,8)]
    coefs = [1, 2, 3, 4, 5, 6, 7]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4300_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.2) for i in range(1,8)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4300_ref(per_rank_args, world_size)}

def _p4300_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'], args['x6'], args['x7'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4300_SIG = '''def evolved_p4300(x1, x2, x3, x4, x5, x6, x7, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4300_DOC = '''7 local vectors x1..x7 each (N,). N=256.
Sequential dep chain: s = AR(x1); s = s + 2*AR(x2); ...; s = s + 7*AR(x7).
Return final s (N,).'''

_P4300_BUILTINS = {'seven_ar_seq': '''def evolved_p4300(x1, x2, x3, x4, x5, x6, x7, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 4 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 5 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x6); s = s + 6 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x7); s = s + 7 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='seven_ar_seq_edge_chal',
    display_name='Problem P_4300',
    evolved_fn_name='evolved_p4300',
    signature=_P4300_SIG,
    signature_doc=_P4300_DOC,
    reference_fn=_p4300_ref,
    generate_test_case=_p4300_generate,
    call_candidate=_p4300_call,
    builtin_templates=_P4300_BUILTINS,
))


# P_4301: prime-number coefficients (harder for compiler patterns)
def _p4301_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,6)]
    coefs = [7, 11, 13, 17, 19]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4301_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.4) for i in range(1,6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4301_ref(per_rank_args, world_size)}

def _p4301_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4301_SIG = '''def evolved_p4301(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4301_DOC = '''5 local vectors x1..x5 each (N,). N=256.
Sequential dep chain: s = 7*AR(x1); s = s + 11*AR(x2); s = s + 13*AR(x3); s = s + 17*AR(x4); s = s + 19*AR(x5).
Return final s (N,). Prime coefficients.'''

_P4301_BUILTINS = {'five_ar_primes': '''def evolved_p4301(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 7 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 11 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 13 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 17 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x5); s = s + 19 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='five_ar_primes_edge_chal',
    display_name='Problem P_4301',
    evolved_fn_name='evolved_p4301',
    signature=_P4301_SIG,
    signature_doc=_P4301_DOC,
    reference_fn=_p4301_ref,
    generate_test_case=_p4301_generate,
    call_candidate=_p4301_call,
    builtin_templates=_P4301_BUILTINS,
))


# P_4302: N=192 (smaller than 256, different territory) + 4 ARs dep chain
def _p4302_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1,5)]
    coefs = [1.1, 2.2, 3.3, 4.4]
    tot = sum(c*a for c,a in zip(coefs, axs))
    return [tot.clone() for _ in range(world_size)]

def _p4302_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 192
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.5) for i in range(1,5)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4302_ref(per_rank_args, world_size)}

def _p4302_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4302_SIG = '''def evolved_p4302(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4302_DOC = '''4 local vectors x1..x4 each (N,). N=192.
Sequential dep chain: s = 1.1*AR(x1); s = s + 2.2*AR(x2); s = s + 3.3*AR(x3); s = s + 4.4*AR(x4).
Return final s (N,).'''

_P4302_BUILTINS = {'four_ar_N192': '''def evolved_p4302(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x1); s = 1.1 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x2); s = s + 2.2 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x3); s = s + 3.3 * a
    a = xm.all_reduce(xm.REDUCE_SUM, x4); s = s + 4.4 * a
    return s
'''}

register_problem(CollectiveProblem(
    name='four_ar_N192_edge_chal',
    display_name='Problem P_4302',
    evolved_fn_name='evolved_p4302',
    signature=_P4302_SIG,
    signature_doc=_P4302_DOC,
    reference_fn=_p4302_ref,
    generate_test_case=_p4302_generate,
    call_candidate=_p4302_call,
    builtin_templates=_P4302_BUILTINS,
))
