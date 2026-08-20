"""Diverse Round 15 (V22): more per-row/col high-yield."""
import torch
from .problems import CollectiveProblem, register_problem


# P_6900: per_row_ar_M96
def _p6900_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6900_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(96, 512)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6900_ref(per_rank_args, world_size)}

def _p6900_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6900_SIG = '''def evolved_p6900(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6900_DOC = '''Local x (96, 512). AR full.'''

_P6900_BUILTINS = {'per_row_ar_M96': '''def evolved_p6900(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M96_chal',
    display_name='Problem P_6900',
    evolved_fn_name='evolved_p6900',
    signature=_P6900_SIG,
    signature_doc=_P6900_DOC,
    reference_fn=_p6900_ref,
    generate_test_case=_p6900_generate,
    call_candidate=_p6900_call,
    builtin_templates=_P6900_BUILTINS,
))


# P_6901: per_row_max_ar_M32
def _p6901_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    result = xs[0]
    for x in xs[1:]: result = torch.maximum(result, x)
    return [result.clone() for _ in range(world_size)]

def _p6901_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(32, 2048)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6901_ref(per_rank_args, world_size)}

def _p6901_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6901_SIG = '''def evolved_p6901(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6901_DOC = '''Local x (32, 2048). AR(MAX) full.'''

_P6901_BUILTINS = {'per_row_max_ar_M32': '''def evolved_p6901(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_MAX, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_max_ar_M32_chal',
    display_name='Problem P_6901',
    evolved_fn_name='evolved_p6901',
    signature=_P6901_SIG,
    signature_doc=_P6901_DOC,
    reference_fn=_p6901_ref,
    generate_test_case=_p6901_generate,
    call_candidate=_p6901_call,
    builtin_templates=_P6901_BUILTINS,
))
