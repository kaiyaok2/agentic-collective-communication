"""Diverse Round 15 (V26): mid-range M variants."""
import torch
from .problems import CollectiveProblem, register_problem


# P_7300: per_row_ar_M48
def _p7300_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p7300_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(48, 1024)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7300_ref(per_rank_args, world_size)}

def _p7300_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7300_SIG = '''def evolved_p7300(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7300_DOC = '''Local x (48, 1024). AR full.'''

_P7300_BUILTINS = {'per_row_ar_M48': '''def evolved_p7300(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M48_chal',
    display_name='Problem P_7300',
    evolved_fn_name='evolved_p7300',
    signature=_P7300_SIG,
    signature_doc=_P7300_DOC,
    reference_fn=_p7300_ref,
    generate_test_case=_p7300_generate,
    call_candidate=_p7300_call,
    builtin_templates=_P7300_BUILTINS,
))


# P_7301: per_row_min_ar_M16 (variant of per_row_min)
def _p7301_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    result = xs[0]
    for x in xs[1:]: result = torch.minimum(result, x)
    return [result.clone() for _ in range(world_size)]

def _p7301_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(32, 2048)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7301_ref(per_rank_args, world_size)}

def _p7301_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7301_SIG = '''def evolved_p7301(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7301_DOC = '''Local x (32, 2048). AR(MIN) full. Per-row MIN.'''

_P7301_BUILTINS = {'per_row_min_ar_M32': '''def evolved_p7301(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_MIN, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_min_ar_M32_chal',
    display_name='Problem P_7301',
    evolved_fn_name='evolved_p7301',
    signature=_P7301_SIG,
    signature_doc=_P7301_DOC,
    reference_fn=_p7301_ref,
    generate_test_case=_p7301_generate,
    call_candidate=_p7301_call,
    builtin_templates=_P7301_BUILTINS,
))
