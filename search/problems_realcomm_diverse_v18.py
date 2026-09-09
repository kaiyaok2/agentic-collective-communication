"""Diverse Round 15 (V18): more high-yield patterns."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6500: per_row_min_ar — MIN AR per row of 2D
# ============================================================
def _p6500_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    result = xs[0]
    for x in xs[1:]: result = torch.minimum(result, x)
    return [result.clone() for _ in range(world_size)]

def _p6500_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(16, 4096)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6500_ref(per_rank_args, world_size)}

def _p6500_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6500_SIG = '''def evolved_p6500(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6500_DOC = '''Local x (16, 4096). Compute AR(MIN) of full x. Baseline ARs each row.'''

_P6500_BUILTINS = {'per_row_min_ar': '''def evolved_p6500(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_MIN, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_min_ar_chal',
    display_name='Problem P_6500',
    evolved_fn_name='evolved_p6500',
    signature=_P6500_SIG,
    signature_doc=_P6500_DOC,
    reference_fn=_p6500_ref,
    generate_test_case=_p6500_generate,
    call_candidate=_p6500_call,
    builtin_templates=_P6500_BUILTINS,
))


# ============================================================
# P_6501: per_row_ar_M32 — bigger M, more overhead to save
# ============================================================
def _p6501_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6501_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(32, 2048)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6501_ref(per_rank_args, world_size)}

def _p6501_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6501_SIG = '''def evolved_p6501(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6501_DOC = '''Local x (32, 2048). Compute AR of full x.
Baseline ARs each of 32 rows (32 dispatches). Sorcar: 1 AR of full 2D.'''

_P6501_BUILTINS = {'per_row_ar_M32': '''def evolved_p6501(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M32_chal',
    display_name='Problem P_6501',
    evolved_fn_name='evolved_p6501',
    signature=_P6501_SIG,
    signature_doc=_P6501_DOC,
    reference_fn=_p6501_ref,
    generate_test_case=_p6501_generate,
    call_candidate=_p6501_call,
    builtin_templates=_P6501_BUILTINS,
))


# ============================================================
# P_6502: five_ar_zero_coefs — mix of ARs with zero coefficients
# Baseline: 5 ARs. Sorcar: drop 3 that have coef zero → 2 ARs → 1 combined.
# ============================================================
def _p6502_ref(inputs, world_size):
    axs = [sum(inp[f'x{i}'] for inp in inputs) for i in range(1, 6)]
    # coefs = [3, 0, 5, 0, 2] → 3*a1 + 5*a3 + 2*a5
    return [(3 * axs[0] + 5 * axs[2] + 2 * axs[4]).clone() for _ in range(world_size)]

def _p6502_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{f'x{i}': torch.randn(N)*(r+i*0.2) for i in range(1, 6)}
                     for r in range(world_size)]
    for a in per_rank_args:
        a['N'] = N
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6502_ref(per_rank_args, world_size)}

def _p6502_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x1'], args['x2'], args['x3'], args['x4'], args['x5'],
              args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6502_SIG = '''def evolved_p6502(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6502_DOC = '''5 local vectors x1..x5 each (N,). N=65536.
Compute: y = 3*AR(x1) + 0*AR(x2) + 5*AR(x3) + 0*AR(x4) + 2*AR(x5) = 3*AR(x1) + 5*AR(x3) + 2*AR(x5).
Return (N,) identical on every rank.'''

_P6502_BUILTINS = {'five_ar_zero_coefs': '''def evolved_p6502(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = 3 * xm.all_reduce(xm.REDUCE_SUM, x1)
    a2 = 0 * xm.all_reduce(xm.REDUCE_SUM, x2)   # dead AR!
    a3 = 5 * xm.all_reduce(xm.REDUCE_SUM, x3)
    a4 = 0 * xm.all_reduce(xm.REDUCE_SUM, x4)   # dead AR!
    a5 = 2 * xm.all_reduce(xm.REDUCE_SUM, x5)
    return a1 + a2 + a3 + a4 + a5
'''}

register_problem(CollectiveProblem(
    name='five_ar_zero_coefs_chal',
    display_name='Problem P_6502',
    evolved_fn_name='evolved_p6502',
    signature=_P6502_SIG,
    signature_doc=_P6502_DOC,
    reference_fn=_p6502_ref,
    generate_test_case=_p6502_generate,
    call_candidate=_p6502_call,
    builtin_templates=_P6502_BUILTINS,
))


# ============================================================
# P_6503: ar_of_addition_of_repeated — AR(x+x+x+x+x) = AR(5x) = 5*AR(x)
# Baseline computes x+x+x+x+x locally then ARs. Sorcar: just AR(x) * 5.
# But baseline already efficient. Try differently:
# Baseline uses 5 separate AR(x) then adds. Sorcar: 1 AR of 5*x.
# Wait, that's same as previous. Let's do: AR of average
# ============================================================
def _p6503_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax / world_size).clone() for _ in range(world_size)]

def _p6503_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6503_ref(per_rank_args, world_size)}

def _p6503_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6503_SIG = '''def evolved_p6503(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6503_DOC = '''Local x (N,). N=65536.
Compute: y = mean of x across ranks = AR(x) / W. Return (N,).
Baseline: AR(x) * some_scalar then divide, doubled up.'''

_P6503_BUILTINS = {'ar_double_scale_div': '''def evolved_p6503(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    step1 = ax * world_size                # useless
    step2 = step1 / world_size             # cancels step1
    step3 = step2 / world_size             # this is the real /W
    return step3
'''}

register_problem(CollectiveProblem(
    name='ar_double_scale_div_chal',
    display_name='Problem P_6503',
    evolved_fn_name='evolved_p6503',
    signature=_P6503_SIG,
    signature_doc=_P6503_DOC,
    reference_fn=_p6503_ref,
    generate_test_case=_p6503_generate,
    call_candidate=_p6503_call,
    builtin_templates=_P6503_BUILTINS,
))
