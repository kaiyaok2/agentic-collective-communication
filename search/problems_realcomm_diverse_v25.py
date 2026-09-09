"""Diverse Round 15 (V25): more high-yield patterns."""
import torch
from .problems import CollectiveProblem, register_problem


# P_7200: per_row_ar_M1024 - massive dispatch overhead saving
def _p7200_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p7200_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(1024, 64)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7200_ref(per_rank_args, world_size)}

def _p7200_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7200_SIG = '''def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7200_DOC = '''Local x (1024, 64). AR full.'''

_P7200_BUILTINS = {'per_row_ar_M1024': '''def evolved_p7200(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M1024_chal',
    display_name='Problem P_7200',
    evolved_fn_name='evolved_p7200',
    signature=_P7200_SIG,
    signature_doc=_P7200_DOC,
    reference_fn=_p7200_ref,
    generate_test_case=_p7200_generate,
    call_candidate=_p7200_call,
    builtin_templates=_P7200_BUILTINS,
))


# P_7201: three_group_alt - 3 AR groups: MAX+SUM+MIN, redundant verifies
def _p7201_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    ys = [inp['y'] for inp in inputs]
    zs = [inp['z'] for inp in inputs]
    mx = xs[0]
    for x in xs[1:]: mx = torch.maximum(mx, x)
    sy = sum(ys)
    mz = zs[0]
    for z in zs[1:]: mz = torch.minimum(mz, z)
    return [(mx + sy + mz).clone() for _ in range(world_size)]

def _p7201_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p7201_ref(per_rank_args, world_size)}

def _p7201_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P7201_SIG = '''def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P7201_DOC = '''Local x, y, z (N,). N=65536.
Compute: max_r(x) + sum_r(y) + min_r(z).
Baseline uses 6 ARs (3 ops * 2 with verify). Sorcar: 3 ARs.'''

_P7201_BUILTINS = {'three_group_dead_verify': '''def evolved_p7201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    mx_v = xm.all_reduce(xm.REDUCE_MAX, mx)
    sy = xm.all_reduce(xm.REDUCE_SUM, y)
    sy_v = xm.all_reduce(xm.REDUCE_SUM, sy) / world_size
    mz = xm.all_reduce(xm.REDUCE_MIN, z)
    mz_v = xm.all_reduce(xm.REDUCE_MIN, mz)
    return mx_v + sy_v + mz_v
'''}

register_problem(CollectiveProblem(
    name='three_group_dead_verify_chal',
    display_name='Problem P_7201',
    evolved_fn_name='evolved_p7201',
    signature=_P7201_SIG,
    signature_doc=_P7201_DOC,
    reference_fn=_p7201_ref,
    generate_test_case=_p7201_generate,
    call_candidate=_p7201_call,
    builtin_templates=_P7201_BUILTINS,
))
