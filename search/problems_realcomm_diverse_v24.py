"""Diverse Round 15 (V24): per-column MAX/MIN variants."""
import torch
from .problems import CollectiveProblem, register_problem


# P_7100: per_column_max_ar
def _p7100_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    result = xs[0]
    for x in xs[1:]: result = torch.maximum(result, x)
    return [result.clone() for _ in range(world_size)]

def _p7100_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(512, 16)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7100_ref(per_rank_args, world_size)}

def _p7100_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7100_SIG = '''def evolved_p7100(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7100_DOC = '''Local x (512, 16). AR(MAX) full. Baseline: per-col MAX AR.'''

_P7100_BUILTINS = {'per_column_max_ar': '''def evolved_p7100(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    cols = [xm.all_reduce(xm.REDUCE_MAX, x[:, k]) for k in range(x.shape[1])]
    return torch.stack(cols, dim=1)
'''}

register_problem(CollectiveProblem(
    name='per_column_max_ar_chal',
    display_name='Problem P_7100',
    evolved_fn_name='evolved_p7100',
    signature=_P7100_SIG,
    signature_doc=_P7100_DOC,
    reference_fn=_p7100_ref,
    generate_test_case=_p7100_generate,
    call_candidate=_p7100_call,
    builtin_templates=_P7100_BUILTINS,
))


# P_7101: 3d_batch_ar - per-batch AR
def _p7101_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p7101_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(8, 16, 512)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7101_ref(per_rank_args, world_size)}

def _p7101_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7101_SIG = '''def evolved_p7101(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7101_DOC = '''Local x (8, 16, 512). AR full. Baseline: per-batch AR (8 dispatches).'''

_P7101_BUILTINS = {'per_batch_ar': '''def evolved_p7101(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    batches = [xm.all_reduce(xm.REDUCE_SUM, x[b]) for b in range(x.shape[0])]
    return torch.stack(batches, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_batch_ar_3d_chal',
    display_name='Problem P_7101',
    evolved_fn_name='evolved_p7101',
    signature=_P7101_SIG,
    signature_doc=_P7101_DOC,
    reference_fn=_p7101_ref,
    generate_test_case=_p7101_generate,
    call_candidate=_p7101_call,
    builtin_templates=_P7101_BUILTINS,
))
