"""Diverse Round 15 (V19): high-yield per-row and CSE patterns."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6600: per_row_ar_M64 — bigger M
# ============================================================
def _p6600_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6600_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(64, 1024)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6600_ref(per_rank_args, world_size)}

def _p6600_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6600_SIG = '''def evolved_p6600(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6600_DOC = '''Local x (64, 1024). AR full. Baseline ARs 64 rows separately.'''

_P6600_BUILTINS = {'per_row_ar_M64': '''def evolved_p6600(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M64_chal',
    display_name='Problem P_6600',
    evolved_fn_name='evolved_p6600',
    signature=_P6600_SIG,
    signature_doc=_P6600_DOC,
    reference_fn=_p6600_ref,
    generate_test_case=_p6600_generate,
    call_candidate=_p6600_call,
    builtin_templates=_P6600_BUILTINS,
))


# ============================================================
# P_6601: per_column_ar_C16 — 16 cols
# ============================================================
def _p6601_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6601_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(512, 16)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6601_ref(per_rank_args, world_size)}

def _p6601_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6601_SIG = '''def evolved_p6601(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6601_DOC = '''Local x (512, 16). AR full. Baseline ARs 16 columns separately.'''

_P6601_BUILTINS = {'per_column_ar_C16': '''def evolved_p6601(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    cols = [xm.all_reduce(xm.REDUCE_SUM, x[:, k]) for k in range(x.shape[1])]
    return torch.stack(cols, dim=1)
'''}

register_problem(CollectiveProblem(
    name='per_column_ar_C16_chal',
    display_name='Problem P_6601',
    evolved_fn_name='evolved_p6601',
    signature=_P6601_SIG,
    signature_doc=_P6601_DOC,
    reference_fn=_p6601_ref,
    generate_test_case=_p6601_generate,
    call_candidate=_p6601_call,
    builtin_templates=_P6601_BUILTINS,
))


# ============================================================
# P_6602: ten_ar_alternate_sign — 10 sequential inline ARs
# alternating sign, all of x
# = (1-1+1-1+1-1+1-1+1-1) * AR(x) = 0
# ============================================================
def _p6602_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [torch.zeros_like(ax).clone() for _ in range(world_size)]

def _p6602_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6602_ref(per_rank_args, world_size)}

def _p6602_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6602_SIG = '''def evolved_p6602(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6602_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x) - AR(x) + AR(x) - AR(x) + ... (10 terms alternating)  = 0.
Return zeros (N,). Sorcar should recognize this simplifies to zero.'''

_P6602_BUILTINS = {'ten_ar_alternate_sign': '''def evolved_p6602(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return (xm.all_reduce(xm.REDUCE_SUM, x)
          - xm.all_reduce(xm.REDUCE_SUM, x)
          + xm.all_reduce(xm.REDUCE_SUM, x)
          - xm.all_reduce(xm.REDUCE_SUM, x)
          + xm.all_reduce(xm.REDUCE_SUM, x)
          - xm.all_reduce(xm.REDUCE_SUM, x)
          + xm.all_reduce(xm.REDUCE_SUM, x)
          - xm.all_reduce(xm.REDUCE_SUM, x)
          + xm.all_reduce(xm.REDUCE_SUM, x)
          - xm.all_reduce(xm.REDUCE_SUM, x))
'''}

register_problem(CollectiveProblem(
    name='ten_ar_alt_sign_zero_chal',
    display_name='Problem P_6602',
    evolved_fn_name='evolved_p6602',
    signature=_P6602_SIG,
    signature_doc=_P6602_DOC,
    reference_fn=_p6602_ref,
    generate_test_case=_p6602_generate,
    call_candidate=_p6602_call,
    builtin_templates=_P6602_BUILTINS,
))


# ============================================================
# P_6603: six_ar_indep_pool_sum — 6 independent inputs summed
# ============================================================
def _p6603_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1, 7)]
    return [sum(axs).clone() for _ in range(world_size)]

def _p6603_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.2) for i in range(1, 7)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6603_ref(per_rank_args, world_size)}

def _p6603_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'], args['x6'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6603_SIG = '''def evolved_p6603(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6603_DOC = '''6 local vectors x1..x6 each (N,). N=65536.
Compute: y = AR(x1) + AR(x2) + AR(x3) + AR(x4) + AR(x5) + AR(x6).
Return (N,).'''

_P6603_BUILTINS = {'six_ar_indep_pool': '''def evolved_p6603(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x1)
    a2 = xm.all_reduce(xm.REDUCE_SUM, x2)
    a3 = xm.all_reduce(xm.REDUCE_SUM, x3)
    a4 = xm.all_reduce(xm.REDUCE_SUM, x4)
    a5 = xm.all_reduce(xm.REDUCE_SUM, x5)
    a6 = xm.all_reduce(xm.REDUCE_SUM, x6)
    return a1 + a2 + a3 + a4 + a5 + a6
'''}

register_problem(CollectiveProblem(
    name='six_ar_indep_pool_chal',
    display_name='Problem P_6603',
    evolved_fn_name='evolved_p6603',
    signature=_P6603_SIG,
    signature_doc=_P6603_DOC,
    reference_fn=_p6603_ref,
    generate_test_case=_p6603_generate,
    call_candidate=_p6603_call,
    builtin_templates=_P6603_BUILTINS,
))
