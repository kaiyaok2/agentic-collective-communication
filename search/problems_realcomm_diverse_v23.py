"""Diverse Round 15 (V23): higher-M per-row variants."""
import torch
from .problems import CollectiveProblem, register_problem


# P_7000: per_row_ar_M512 - extreme
def _p7000_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p7000_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(512, 128)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7000_ref(per_rank_args, world_size)}

def _p7000_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7000_SIG = '''def evolved_p7000(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7000_DOC = '''Local x (512, 128). AR full.'''

_P7000_BUILTINS = {'per_row_ar_M512': '''def evolved_p7000(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M512_chal',
    display_name='Problem P_7000',
    evolved_fn_name='evolved_p7000',
    signature=_P7000_SIG,
    signature_doc=_P7000_DOC,
    reference_fn=_p7000_ref,
    generate_test_case=_p7000_generate,
    call_candidate=_p7000_call,
    builtin_templates=_P7000_BUILTINS,
))


# P_7001: per_col_ar_C64
def _p7001_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p7001_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(128, 64)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7001_ref(per_rank_args, world_size)}

def _p7001_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7001_SIG = '''def evolved_p7001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7001_DOC = '''Local x (128, 64). AR full. Baseline ARs 64 cols.'''

_P7001_BUILTINS = {'per_column_ar_C64': '''def evolved_p7001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    cols = [xm.all_reduce(xm.REDUCE_SUM, x[:, k]) for k in range(x.shape[1])]
    return torch.stack(cols, dim=1)
'''}

register_problem(CollectiveProblem(
    name='per_column_ar_C64_chal',
    display_name='Problem P_7001',
    evolved_fn_name='evolved_p7001',
    signature=_P7001_SIG,
    signature_doc=_P7001_DOC,
    reference_fn=_p7001_ref,
    generate_test_case=_p7001_generate,
    call_candidate=_p7001_call,
    builtin_templates=_P7001_BUILTINS,
))
