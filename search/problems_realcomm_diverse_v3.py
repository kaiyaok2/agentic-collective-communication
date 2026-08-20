"""Diverse Round 15 (V3): more distinct classes.
Rules: NOT sequential-AR-linearity. Different optimization axes.
"""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5000: max_reduce_swap — REDUCE_SUM used when REDUCE_MAX is sufficient
# Baseline: uses AR(SUM) then divides by W — actually computes mean
# But problem asks for MAX. Sorcar should use REDUCE_MAX.
# ============================================================
def _p5000_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    max_x = xs[0]
    for x in xs[1:]:
        max_x = torch.maximum(max_x, x)
    return [max_x.clone() for _ in range(world_size)]

def _p5000_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5000_ref(per_rank_args, world_size)}

def _p5000_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5000_SIG = '''def evolved_p5000(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5000_DOC = '''Local x (N,). N=65536.
Compute elementwise max of x across all ranks: y[i] = max_r(x_r[i]).
Baseline uses AR(SUM) + comparison hack; correct answer requires REDUCE_MAX.
Return (N,) identical on every rank.'''

# Baseline uses ridiculous exp-then-sum-then-log to compute max via SUM
_P5000_BUILTINS = {'sum_hack_max': '''def evolved_p5000(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Baseline: use max-softlog trick (wrong for our purposes) — actually just use AR of exponentiated
    # For correctness we hack: assume we know max, but that requires MAX collective.
    # Simpler baseline: compute max via a sequence of pairwise MAX AR calls
    # (still redundant; sorcar should use single REDUCE_MAX)
    m = xm.all_reduce(xm.REDUCE_MAX, x)
    verify = xm.all_reduce(xm.REDUCE_MAX, m)  # redundant second AR
    return verify
'''}

register_problem(CollectiveProblem(
    name='max_reduce_redundant_chal',
    display_name='Problem P_5000',
    evolved_fn_name='evolved_p5000',
    signature=_P5000_SIG,
    signature_doc=_P5000_DOC,
    reference_fn=_p5000_ref,
    generate_test_case=_p5000_generate,
    call_candidate=_p5000_call,
    builtin_templates=_P5000_BUILTINS,
))


# ============================================================
# P_5001: ar_reused_via_variable — 3 ARs of same value stored in variable
# Baseline: y = AR(x); a = y * 2; b = xm.all_reduce(SUM, x) * 3; c = ...
# Different than P_4800 which had inline AR calls.
# ============================================================
def _p5001_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax * 15).clone() for _ in range(world_size)]  # 2+3+4+6 = 15

def _p5001_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5001_ref(per_rank_args, world_size)}

def _p5001_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5001_SIG = '''def evolved_p5001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5001_DOC = '''Local x (N,). N=65536.
Compute: sum of four scaled all-reduces of the SAME input x.
Formula: a = 2*AR(x); b = 3*AR(x); c = 4*AR(x); d = 6*AR(x); return a+b+c+d = 15*AR(x).
Return (N,).'''

_P5001_BUILTINS = {'four_ar_same_input': '''def evolved_p5001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x) * 2
    b = xm.all_reduce(xm.REDUCE_SUM, x) * 3
    c = xm.all_reduce(xm.REDUCE_SUM, x) * 4
    d = xm.all_reduce(xm.REDUCE_SUM, x) * 6
    return a + b + c + d
'''}

register_problem(CollectiveProblem(
    name='four_ar_same_input_chal',
    display_name='Problem P_5001',
    evolved_fn_name='evolved_p5001',
    signature=_P5001_SIG,
    signature_doc=_P5001_DOC,
    reference_fn=_p5001_ref,
    generate_test_case=_p5001_generate,
    call_candidate=_p5001_call,
    builtin_templates=_P5001_BUILTINS,
))


# ============================================================
# P_5002: ar_indexing_pattern — AR then broadcast via index_select
# Baseline: AR full tensor, then use torch.gather with rank-common indices
#   The gather is dead code (all indices same → return same tensor)
# Sorcar: recognize the gather is a no-op and skip it.
# ============================================================
def _p5002_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)  # (N,)
    return [ax.clone() for _ in range(world_size)]

def _p5002_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5002_ref(per_rank_args, world_size)}

def _p5002_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5002_SIG = '''def evolved_p5002(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5002_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x). Return (N,) identical on every rank.
Baseline decorates the AR with a redundant gather/index_select that
produces the same tensor.'''

_P5002_BUILTINS = {'ar_then_dead_gather': '''def evolved_p5002(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N, device=x.device)  # identity index
    gathered = ax[idx]                       # dead reindex — same result
    return gathered
'''}

register_problem(CollectiveProblem(
    name='ar_dead_gather_chal',
    display_name='Problem P_5002',
    evolved_fn_name='evolved_p5002',
    signature=_P5002_SIG,
    signature_doc=_P5002_DOC,
    reference_fn=_p5002_ref,
    generate_test_case=_p5002_generate,
    call_candidate=_p5002_call,
    builtin_templates=_P5002_BUILTINS,
))


# ============================================================
# P_5003: cp_broadcast_pattern — collective_permute used naively
# Baseline: uses W-1 collective_permutes to broadcast rank 0's data to all
# Sorcar: use all_gather / broadcast (single collective)
# ============================================================
def _p5003_ref(inputs, world_size):
    return [inputs[0]['x'].clone() for _ in range(world_size)]  # rank 0's tensor everywhere

def _p5003_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 4096
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5003_ref(per_rank_args, world_size)}

def _p5003_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5003_SIG = '''def evolved_p5003(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5003_DOC = '''Local x (N,). N=4096.
Broadcast rank 0's x to all ranks. Return (N,) equal to x_rank0 on every rank.
Baseline uses zero-mask + AR trick: on non-zero ranks, zero out x, then AR — that
gives sum of just rank 0's x. Sorcar could use a cleaner broadcast.'''

_P5003_BUILTINS = {'zero_mask_broadcast': '''def evolved_p5003(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # baseline: on non-zero ranks, zero out x; AR(SUM) then equals x_0
    if rank != 0:
        x = torch.zeros_like(x)
    return xm.all_reduce(xm.REDUCE_SUM, x)
'''}

register_problem(CollectiveProblem(
    name='rank0_broadcast_via_mask_chal',
    display_name='Problem P_5003',
    evolved_fn_name='evolved_p5003',
    signature=_P5003_SIG,
    signature_doc=_P5003_DOC,
    reference_fn=_p5003_ref,
    generate_test_case=_p5003_generate,
    call_candidate=_p5003_call,
    builtin_templates=_P5003_BUILTINS,
))
