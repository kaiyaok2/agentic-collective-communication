"""Diverse Round 15 (V6): still more classes."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5300: ar_result_broadcast — after AR everyone has result;
#         but baseline does an extra all_gather just to confirm consistency
# Sorcar: recognize AR already broadcasts.
# ============================================================
def _p5300_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5300_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5300_ref(per_rank_args, world_size)}

def _p5300_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5300_SIG = '''def evolved_p5300(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5300_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x). Return (N,) identical on every rank.
Baseline does AR followed by all_gather + narrow (all ranks have same value,
so the gather is dead).'''

_P5300_BUILTINS = {'ar_then_dead_ag': '''def evolved_p5300(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    g = xm.all_gather(ax, dim=0)      # dead: all ranks have same ax
    return g.narrow(0, 0, N)
'''}

register_problem(CollectiveProblem(
    name='ar_dead_gather_verify_chal',
    display_name='Problem P_5300',
    evolved_fn_name='evolved_p5300',
    signature=_P5300_SIG,
    signature_doc=_P5300_DOC,
    reference_fn=_p5300_ref,
    generate_test_case=_p5300_generate,
    call_candidate=_p5300_call,
    builtin_templates=_P5300_BUILTINS,
))


# ============================================================
# P_5301: rs_then_ag — reduce_scatter followed by all_gather = all_reduce
# Baseline: RS + AG (which is equivalent to AR but 2 collectives)
# Sorcar: use single AR
# ============================================================
def _p5301_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5301_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 8192  # small per-rank so W*N is reasonable
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N * world_size)*(r+1), 'N': N * world_size}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5301_ref(per_rank_args, world_size)}

def _p5301_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5301_SIG = '''def evolved_p5301(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5301_DOC = '''Local x (N,). N=8192 * world_size (full tensor per rank).
Compute: y = AR(x) — reduce-sum across all ranks.
Baseline uses RS then AG which totals same bytes as AR but 2 dispatches.
Return (N,) identical on every rank.'''

_P5301_BUILTINS = {'rs_ag_pattern': '''def evolved_p5301(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N_per_rank = N // world_size
    rs = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0,
                           shard_count=world_size, groups=None)  # (N_per_rank,)
    return xm.all_gather(rs, dim=0)  # (N,)
'''}

register_problem(CollectiveProblem(
    name='rs_then_ag_chal',
    display_name='Problem P_5301',
    evolved_fn_name='evolved_p5301',
    signature=_P5301_SIG,
    signature_doc=_P5301_DOC,
    reference_fn=_p5301_ref,
    generate_test_case=_p5301_generate,
    call_candidate=_p5301_call,
    builtin_templates=_P5301_BUILTINS,
))


# ============================================================
# P_5302: torch_dist_reduce_wrap — baseline uses torch.distributed style
#   with a wrap that becomes multiple ARs.
# Not sure this will hit sim; skip if problem.
# Instead: ar_ab_swap — swap two rank-specific values via ARs
# Baseline uses complex zero-mask + AR + subtract to do a swap.
# ============================================================
# Actually let's do: comparison_of_ars: (AR(x) > AR(y)) which needs both
# ARs; can they be stacked? Yes.
def _p5302_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    return [(ax > ay).to(torch.float32).clone() for _ in range(world_size)]

def _p5302_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5302_ref(per_rank_args, world_size)}

def _p5302_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5302_SIG = '''def evolved_p5302(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5302_DOC = '''Local x, y (N,). N=65536.
Compute: mask[i] = 1 if AR(x)[i] > AR(y)[i] else 0. Return as fp32 (N,).
Baseline computes AR(x) and AR(y) separately; Sorcar can stack them into
a single AR of (2, N) tensor.'''

_P5302_BUILTINS = {'compare_two_ars': '''def evolved_p5302(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    return (ax > ay).to(torch.float32)
'''}

register_problem(CollectiveProblem(
    name='compare_two_ars_chal',
    display_name='Problem P_5302',
    evolved_fn_name='evolved_p5302',
    signature=_P5302_SIG,
    signature_doc=_P5302_DOC,
    reference_fn=_p5302_ref,
    generate_test_case=_p5302_generate,
    call_candidate=_p5302_call,
    builtin_templates=_P5302_BUILTINS,
))


# ============================================================
# P_5303: ar_selfcancel — baseline uses AR twice in a way that cancels
# 1st AR then subtract same AR = 0 result.
# Sorcar: recognize the cancellation and skip both.
# ============================================================
def _p5303_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]  # result = AR(x), not zero

def _p5303_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5303_ref(per_rank_args, world_size)}

def _p5303_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5303_SIG = '''def evolved_p5303(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5303_DOC = '''Local x (N,). N=65536.
Compute: y = 2*AR(x) - AR(x). Return (N,) identical on every rank.
Simplifies to AR(x). Baseline calls AR twice; Sorcar should call once.'''

_P5303_BUILTINS = {'ar_selfcancel_baseline': '''def evolved_p5303(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = 2 * xm.all_reduce(xm.REDUCE_SUM, x)
    b = xm.all_reduce(xm.REDUCE_SUM, x)
    return a - b
'''}

register_problem(CollectiveProblem(
    name='ar_selfcancel_chal',
    display_name='Problem P_5303',
    evolved_fn_name='evolved_p5303',
    signature=_P5303_SIG,
    signature_doc=_P5303_DOC,
    reference_fn=_p5303_ref,
    generate_test_case=_p5303_generate,
    call_candidate=_p5303_call,
    builtin_templates=_P5303_BUILTINS,
))
