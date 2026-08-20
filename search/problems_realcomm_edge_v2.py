"""Real-comm edge V2: NEFF-shape divergence.

Design problems where different implementations produce different NEFF sizes 
which triggers different NEFF compile/launch costs.
"""
import torch
from .problems import CollectiveProblem, register_problem


# P_3600 sequential_ar_chain: y = AR(AR(AR(x))). Wait — AR is idempotent for values.
# Better: y = f_3(AR(f_2(AR(f_1(AR(x)))))). Different local ops between.
# But local ops don't create separate NEFFs.
# 
# Different angle: y = AR(x)+AR(y) where y = f(AR(x)). Sequential dependency.
def _p3600_ref(inputs, world_size):
    x_list = [inp['x'] for inp in inputs]
    y_list = [inp['y'] for inp in inputs]
    ax = sum(x_list)  # AR(x)
    # local: ax_processed = ax * 2 (on every rank, same result)
    ax_p = ax * 2
    # AR(y + ax_p): each rank has y + ax_p, then AR
    ay = sum(y_list) + world_size * ax_p
    return [ay.clone() for _ in range(world_size)]


def _p3600_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 512
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N) * (r + 1), 'y': torch.randn(N) * (r + 0.5), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3600_ref(per_rank_args, world_size)}


def _p3600_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)


_P3600_SIG = '''def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3600_DOC = '''Args: x (N,), y (N,) local. N=512.
Formula: 
  Step 1: ax = sum_r x_r  (all-reduce x)
  Step 2: ax_p = ax * 2  (local on every rank)
  Step 3: contrib = y + ax_p  (local; y is per-rank input, ax_p is same on all ranks)
  Step 4: result = sum_r contrib_r  (all-reduce contrib)
Returns: (N,) tensor identical on every rank.

DIVERGENCE POINT: 2 SEQUENTIAL ARs with dependency (step 4 depends on step 1).
Cannot merge into 1 AR. But choice of local ops between (elementwise * 2, add) affects
fusion into surrounding HLOs.
(a) Vanilla: 2 xm.all_reduce calls back-to-back with intermediate local math.
    Sim: dispatch(100) + amort(30) = 130us + 2*bw_floor, total ~5290us.
(b) Same but agent might try to combine via all_gather + local reduce (world*N intermediate).
    Sim: 2 AG (2*1050us) + local = ~2200us. 
Wait — (b) is 2.4× cheaper! Interesting.
'''
_P3600_BUILTINS = {'sequential_two_ar': '''def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    ax_p = ax * 2
    contrib = y + ax_p
    return xm.all_reduce(xm.REDUCE_SUM, contrib)
''',}
register_problem(CollectiveProblem(
    name='sequential_ar_chain_edge_chal',
    display_name='Problem P_3600',
    evolved_fn_name='evolved_p3600',
    signature=_P3600_SIG,
    signature_doc=_P3600_DOC,
    reference_fn=_p3600_ref,
    generate_test_case=_p3600_generate,
    call_candidate=_p3600_call,
    builtin_templates=_P3600_BUILTINS,
))


# P_3601 conditional_ar_shape: y depends on rank's role
# Half of ranks do AR on x. Other half do CP-based exchange with mates.
# Two NEFF paths possible.
def _p3601_ref(inputs, world_size):
    """
    Ranks r < W/2: y_r = sum_r' x_r'[r%N + ...] — subset of AR
    Ranks r >= W/2: y_r = x_{r-W/2} — receive from mate (CP)
    """
    N = 256
    half = world_size // 2
    outs = []
    total = sum(inp['x'] for inp in inputs[:half])  # only first half contributes
    for r in range(world_size):
        if r < half:
            outs.append(total.clone())  # AR result
        else:
            outs.append(inputs[r - half]['x'].clone())  # CP from mate
    return outs


def _p3601_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 256
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N) * (r + 1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p3601_ref(per_rank_args, world_size)}


def _p3601_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)


_P3601_SIG = '''def evolved_p3601(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P3601_DOC = '''Args: x (N,) local. N=256. Let half = world_size // 2.
Formula:
  - If rank < half: y_r = sum over r' in [0, half) of x_{r'}  (partial AR)
  - If rank >= half: y_r = x_{r - half}  (receive from paired rank in first half)
Returns: (N,) tensor DIFFERENT per rank.

DIVERGENCE POINT: two different communication patterns needed:
(a) partial AR (over half of ranks) — masked AR of x*(rank<half)
    Sim: 1 AR (5160us).
(b) CP for the receive-from-mate half — 1 CP for those ranks.
Combined into ONE algorithm: might use masked AR (rank>=half zeroes contribution) then 
    each rank compares its position and copies the AR-result (if in first half) or 
    reads from mate (via CP).
Naive: AR + CP + local select = 5160+90+small ~5250us.
Smart: single AR is enough — first half sees actual sum, second half also sees full sum 
    (which happens to equal partial sum only if second half x=0 — WRONG for this formula).
    Actually masked AR: multiply x by (rank<half) mask, then AR — first half gets partial sum ✓.
    Second half: they need x from first half via CP.
Two collectives are needed: 1 masked AR + 1 CP. Sim: ~5250us total.
'''
_P3601_BUILTINS = {'masked_ar_plus_cp': '''def evolved_p3601(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    half = world_size // 2
    # Masked AR: only first half contributes
    if rank < half:
        contrib = x
    else:
        contrib = torch.zeros_like(x)
    partial_sum = xm.all_reduce(xm.REDUCE_SUM, contrib)
    # CP: mates in second half receive x from first half
    pairs = [(r, r + half) for r in range(half)] + [(r + half, r + half) for r in range(half)]
    cp_result = xm.collective_permute(x, pairs)
    if rank < half:
        return partial_sum
    else:
        return cp_result
''',}
register_problem(CollectiveProblem(
    name='conditional_ar_cp_edge_chal',
    display_name='Problem P_3601',
    evolved_fn_name='evolved_p3601',
    signature=_P3601_SIG,
    signature_doc=_P3601_DOC,
    reference_fn=_p3601_ref,
    generate_test_case=_p3601_generate,
    call_candidate=_p3601_call,
    builtin_templates=_P3601_BUILTINS,
))
