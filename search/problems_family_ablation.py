"""Family-ablation catalog: registered CollectiveProblem defs for the
F6/F7 e2e family sites that previously existed only as extras (no
catalog entry). Used by the 2026-09 ablation study (sim deltas / AI
discovery loop / adversarial testing) on the 8 problems mapped from the
10B TP e2e family sites.

Doc formulas are stated as WHAT to compute, never HOW (no-leak)."""
import torch
from .problems import CollectiveProblem, register_problem


# P_9001 (F6): mixmaxmin — interleaved MAX/MIN reductions, scaled sum
def _p9001_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    xmax = xs[0]
    xmin = xs[0]
    for x in xs[1:]:
        xmax = torch.maximum(xmax, x)
        xmin = torch.minimum(xmin, x)
    sum_max = sum((i + 1) * 0.1 for i in range(8))
    sum_min = sum((i + 1) * 0.05 for i in range(8))
    out = xmax * sum_max + xmin * sum_min
    return [out.clone() for _ in range(world_size)]

def _p9001_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(4096) * (r + 1)}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p9001_ref(per_rank_args, world_size)}

def _p9001_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P9001_SIG = '''def evolved_p9001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P9001_DOC = '''Local x (4096,).
Compute: a = zeros; for i in 0..7:
  a += MAXreduce_over_ranks(x) * ((i+1)*0.1)
  a += MINreduce_over_ranks(x) * ((i+1)*0.05)
Return a; identical on every rank.'''

_P9001_BUILTINS = {'mixmaxmin_loop': '''def evolved_p9001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = torch.zeros_like(x)
    for i in range(8):
        a = a + xm.all_reduce(xm.REDUCE_MAX, x) * ((i + 1) * 0.1)
        a = a + xm.all_reduce(xm.REDUCE_MIN, x) * ((i + 1) * 0.05)
    return a
'''}

register_problem(CollectiveProblem(
    name='mixmaxmin_chal',
    display_name='Problem P_9001',
    evolved_fn_name='evolved_p9001',
    signature=_P9001_SIG,
    signature_doc=_P9001_DOC,
    reference_fn=_p9001_ref,
    generate_test_case=_p9001_generate,
    call_candidate=_p9001_call,
    builtin_templates=_P9001_BUILTINS,
))


# P_9002 (F7): eightslab — contiguous slabs reduced separately, re-cat
def _p9002_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p9002_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(8192) * (r + 1)}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p9002_ref(per_rank_args, world_size)}

def _p9002_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P9002_SIG = '''def evolved_p9002(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P9002_DOC = '''Local x (8192,). Compute SUMreduce_over_ranks(x): the
elementwise sum of every rank's x. Return (8192,), identical on every
rank.'''

_P9002_BUILTINS = {'eight_slab_loop': '''def evolved_p9002(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N = x.shape[0] // 8
    parts = [xm.all_reduce(xm.REDUCE_SUM, x[i*N:(i+1)*N]) for i in range(8)]
    return torch.cat(parts, dim=0)
'''}

register_problem(CollectiveProblem(
    name='eightslab_chal',
    display_name='Problem P_9002',
    evolved_fn_name='evolved_p9002',
    signature=_P9002_SIG,
    signature_doc=_P9002_DOC,
    reference_fn=_p9002_ref,
    generate_test_case=_p9002_generate,
    call_candidate=_p9002_call,
    builtin_templates=_P9002_BUILTINS,
))
