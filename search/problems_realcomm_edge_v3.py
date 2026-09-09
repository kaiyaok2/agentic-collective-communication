"""Real-comm edge V3: MORE algebraic-fusion problems (following sequential_ar_chain success)."""
import torch
from .problems import CollectiveProblem, register_problem


# P_3700 triple_ar_linear: 3 ARs, all fusable into 1 stacked AR
def _p3700_ref(inputs, world_size):
    """y = 3 * sum(x) + 5 * sum(y) + 7 * sum(z). Linear combo of 3 ARs."""
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(3 * ax + 5 * ay + 7 * az).clone() for _ in range(world_size)]


def _p3700_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N) * (r + 1), 'y': torch.randn(N) * (r + 0.5), 'z': torch.randn(N) * (r + 0.3), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3700_ref(per_rank_args, world_size)}


def _p3700_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)


_P3700_SIG = '''def evolved_p3700(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3700_DOC = '''Args: x (N,), y (N,), z (N,) local. N=256.
Formula: result = 3 * sum_r x_r + 5 * sum_r y_r + 7 * sum_r z_r. Linear combo of 3 all-reduces.
Returns: (N,) tensor identical on every rank.

DIVERGENCE POINT:
(a) Naive: 3 sequential xm.all_reduce, then linear combine. Sim: 3 ARs pipelined ~5350us.
(b) Stacked: torch.stack([3*x, 5*y, 7*z]) then 1 AR then sum. Sim: 1 AR (5160us) + local.
    ~5200us. WORSE than (a) — but wait — stack is a local op, sum locally is trivial.
    Actually stacked wins by ~150us in sim.
(c) Fused with pre-scaling: same as (b).
'''
_P3700_BUILTINS = {'three_seq_ar': '''def evolved_p3700(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    ay = xm.all_reduce(xm.REDUCE_SUM, y)
    az = xm.all_reduce(xm.REDUCE_SUM, z)
    return 3 * ax + 5 * ay + 7 * az
''',}
register_problem(CollectiveProblem(
    name='triple_ar_linear_edge_chal',
    display_name='Problem P_3700',
    evolved_fn_name='evolved_p3700',
    signature=_P3700_SIG,
    signature_doc=_P3700_DOC,
    reference_fn=_p3700_ref,
    generate_test_case=_p3700_generate,
    call_candidate=_p3700_call,
    builtin_templates=_P3700_BUILTINS,
))


# P_3701 chained_ar_nested: AR(AR(x + ax_p) + ax_p2) — 3 sequential ARs
def _p3701_ref(inputs, world_size):
    """Highly sequential: each AR feeds next. Same algebraic structure though."""
    x_list = [inp['x'] for inp in inputs]
    W = world_size
    ax1 = sum(x_list)  # sum of x
    # each rank has ax1 (same value)
    y1 = ax1 * 2  # local
    # AR of (x + y1): but y1 is common on all ranks, x differs
    # AR(x + y1) = sum(x) + W*y1 = ax1 + W*y1 (all same)
    ax2 = ax1 + W * y1  # this is what AR would produce
    y2 = ax2 + 3  # local
    ax3 = ax1 + W * y2  # third AR
    return [ax3.clone() for _ in range(world_size)]


def _p3701_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N) * (r + 1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3701_ref(per_rank_args, world_size)}


def _p3701_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)


_P3701_SIG = '''def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3701_DOC = '''Args: x (N,) local. N=256.
Formula: 3 sequential all-reduces with local post-processing:
   ax1 = sum_r x_r
   y1 = ax1 * 2  (local)
   ax2 = sum_r (x_r + y1)  (2nd AR; note y1 same on all ranks)
   y2 = ax2 + 3  (local)
   ax3 = sum_r (x_r + y2)  (3rd AR)
Returns: (N,) tensor identical on every rank (ax3).

DIVERGENCE POINT: All 3 ARs are equivalent to ax1 + W*y_k because y_k is on every rank.
So: ax2 = ax1 + W*y1, ax3 = ax1 + W*y2. Only ONE actual AR is needed (of x).
(a) Naive: 3 sequential ARs (~5350us pipelined)
(b) Clever: 1 AR (of x), rest local (~5160us) — Sorcar-style algebraic simplification.
'''
_P3701_BUILTINS = {'three_ar_naive': '''def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax1 = xm.all_reduce(xm.REDUCE_SUM, x)
    y1 = ax1 * 2
    ax2 = xm.all_reduce(xm.REDUCE_SUM, x + y1)
    y2 = ax2 + 3
    ax3 = xm.all_reduce(xm.REDUCE_SUM, x + y2)
    return ax3
''',}
register_problem(CollectiveProblem(
    name='chained_ar_nested_edge_chal',
    display_name='Problem P_3701',
    evolved_fn_name='evolved_p3701',
    signature=_P3701_SIG,
    signature_doc=_P3701_DOC,
    reference_fn=_p3701_ref,
    generate_test_case=_p3701_generate,
    call_candidate=_p3701_call,
    builtin_templates=_P3701_BUILTINS,
))
