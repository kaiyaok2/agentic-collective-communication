"""Diverse Round 15 (V8): edge cases + boundary conditions."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5500: cat_before_ar_bytes_bloat — cat with zero tensor doubles bytes
# Baseline: cat(x, zeros_like(x)) then AR — 2x the bytes
# Sorcar: AR only x, then cat with zeros locally (zeros are rank-common)
# ============================================================
def _p5500_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    N = inputs[0]['N']
    return [torch.cat([ax, torch.zeros(N)]).clone() for _ in range(world_size)]

def _p5500_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5500_ref(per_rank_args, world_size)}

def _p5500_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5500_SIG = '''def evolved_p5500(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5500_DOC = '''Local x (N,). N=65536.
Compute: y = cat([AR(x), zeros(N)]) — reduce, then append N zeros.
Baseline pads x with zeros first then ARs (transferring 2N bytes needlessly).
Return (2N,).'''

_P5500_BUILTINS = {'cat_zero_then_ar': '''def evolved_p5500(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    padded = torch.cat([x, torch.zeros_like(x)])
    return xm.all_reduce(xm.REDUCE_SUM, padded)      # transfers 2N bytes
'''}

register_problem(CollectiveProblem(
    name='cat_zero_before_ar_chal',
    display_name='Problem P_5500',
    evolved_fn_name='evolved_p5500',
    signature=_P5500_SIG,
    signature_doc=_P5500_DOC,
    reference_fn=_p5500_ref,
    generate_test_case=_p5500_generate,
    call_candidate=_p5500_call,
    builtin_templates=_P5500_BUILTINS,
))


# ============================================================
# P_5501: cast_before_ar_bloat — cast fp32 x to fp64 before AR
# Baseline: AR of fp64 (2x bytes)
# Sorcar: AR of fp32, cast to fp64 after
# ============================================================
def _p5501_ref(inputs, world_size):
    ax = sum(inp['x'].double() for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5501_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 32768
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5501_ref(per_rank_args, world_size)}

def _p5501_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5501_SIG = '''def evolved_p5501(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5501_DOC = '''Local x (N,) fp32. N=32768.
Compute: y = AR(x) cast to fp64. Return (N,) fp64.
Baseline upcasts to fp64 before AR (doubles bytes over EFA).'''

_P5501_BUILTINS = {'upcast_before_ar': '''def evolved_p5501(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    x64 = x.double()
    return xm.all_reduce(xm.REDUCE_SUM, x64)      # 2x bytes
'''}

register_problem(CollectiveProblem(
    name='cast_upbefore_ar_chal',
    display_name='Problem P_5501',
    evolved_fn_name='evolved_p5501',
    signature=_P5501_SIG,
    signature_doc=_P5501_DOC,
    reference_fn=_p5501_ref,
    generate_test_case=_p5501_generate,
    call_candidate=_p5501_call,
    builtin_templates=_P5501_BUILTINS,
))


# ============================================================
# P_5502: three_ars_in_conditional — control flow around AR
# Baseline: if rank_even: AR(x)+AR(y); else: AR(x)+AR(y)+AR(z)
# But since branches diverge, they're both compiled; wasteful.
# Sorcar: AR only what's needed based on rank-common condition.
# But rank_even is rank-specific so can't be rank-common...
# Actually let me flip: condition on world_size (compile-time constant)
# ============================================================
def _p5502_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    if world_size >= 2:
        result = ax + ay + az
    else:
        result = ax
    return [result.clone() for _ in range(world_size)]

def _p5502_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5502_ref(per_rank_args, world_size)}

def _p5502_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5502_SIG = '''def evolved_p5502(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5502_DOC = '''Local x, y, z (N,). N=65536.
Compute: if world_size >= 2 return AR(x)+AR(y)+AR(z) else AR(x).
world_size is a compile-time constant (typically 64 for 2-node cluster).'''

_P5502_BUILTINS = {'conditional_ars': '''def evolved_p5502(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    if world_size >= 2:
        return ax + ay + az
    else:
        return ax
'''}

register_problem(CollectiveProblem(
    name='conditional_ars_chal',
    display_name='Problem P_5502',
    evolved_fn_name='evolved_p5502',
    signature=_P5502_SIG,
    signature_doc=_P5502_DOC,
    reference_fn=_p5502_ref,
    generate_test_case=_p5502_generate,
    call_candidate=_p5502_call,
    builtin_templates=_P5502_BUILTINS,
))


# ============================================================
# P_5503: ar_padded_with_useless_dim — baseline unsqueezes to (1, N)
#   and ARs. Sorcar: unsqueeze is free but the AR bytes shouldn't differ.
# Actually simpler: use AR of stacked pair when only 1 is needed
# ============================================================
def _p5503_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5503_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5503_ref(per_rank_args, world_size)}

def _p5503_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5503_SIG = '''def evolved_p5503(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5503_DOC = '''Local x, y (N,). N=65536.
Compute: y_out = AR(x). Return (N,) identical on every rank.
Baseline stacks [x, y] and ARs both, then discards the y result.
Sorcar should AR only x.'''

_P5503_BUILTINS = {'stack_unused_ar': '''def evolved_p5503(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    stacked = torch.stack([x, y])
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)   # 2N bytes but only need N
    return reduced[0]                                  # discard reduced y
'''}

register_problem(CollectiveProblem(
    name='ar_extra_dim_unused_chal',
    display_name='Problem P_5503',
    evolved_fn_name='evolved_p5503',
    signature=_P5503_SIG,
    signature_doc=_P5503_DOC,
    reference_fn=_p5503_ref,
    generate_test_case=_p5503_generate,
    call_candidate=_p5503_call,
    builtin_templates=_P5503_BUILTINS,
))
