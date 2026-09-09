"""Round-18 problems: fusion-resistant multi-collective patterns.

Design principle: use MULTIPLE DIFFERENT collective types (AR + RS + AG,
etc.) in a single algorithm so Neuron compiler cannot fuse plausible
implementations into identical NEFFs. This preserves the sim-vs-RT
signal that round-17 lost.

Smoke-tested on 2-node 64-rank 2026-08-13:
- P150 dual_reduce_shard: 34% RT range across 3 solutions
- P151 topk_from_sum: 12% RT range
- P152 offset_shift_window: 9% RT range

Baseline seed is chosen as the WORSE solution so kiss/strat has room
to improve. No answer leaking: signature_doc describes the formula only.
"""
import torch
from .problems import CollectiveProblem, register_problem


# =============================================================================
# P_150 dual_reduce_shard — mix of AR and RS, forces distinct primitives
# =============================================================================

def _p150_ref(inputs, world_size):
    """Reference:
    z1[i] = sum over ranks r of x_r[i], for i in 0..N-1  (shape N)
    z2[i] = sum over ranks r of y_r[my_shard_indices], where my_shard_indices
            is the r-th shard of y (shape N)
    Each rank returns z1 concat z2 (shape 2N).

    Input shapes: x_r has shape (N,); y_r has shape (N * world_size,).
    """
    N = inputs[0]['x'].shape[0]
    W = world_size
    z1 = sum(inputs[r]['x'] for r in range(W))
    y_all_reduced = sum(inputs[r]['y'] for r in range(W))
    outs = []
    for r in range(W):
        z2 = y_all_reduced[r*N:(r+1)*N]
        outs.append(torch.cat([z1, z2], dim=0))
    return outs


def _p150_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 1024
    torch.manual_seed(seed)
    inputs = []
    for r in range(world_size):
        x = torch.randn(N) * (r + 1)
        y = torch.randn(N * world_size) * (r + 1) * 0.5
        inputs.append({'x': x, 'y': y})
    per_rank_args = [{'x': inputs[r]['x'], 'y': inputs[r]['y'], 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p150_ref(inputs, world_size)}


def _p150_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['y'], rank_args['N'],
                        rank, ws, nd, cpd, xm_mock, torch_mock,
                        num_nodes=num_nodes)


_P150_SIG = '''def evolved_p150(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P150_DOC = '''Args: x (N,) - local vector. y (N*world_size,) - local
larger vector. N: int.

Formula: return concat(z1, z2) where
  z1[i] = sum_r x_r[i]                  (shape N, same on every rank)
  z2[i] = sum_r y_r[rank*N + i]         (shape N, per-rank shard of AR(y))

NON-OBVIOUS: three plausible strategies —
  (a) all_reduce(x, SUM) + reduce_scatter(y, SUM)  — 2 different primitives
  (b) all_gather(x) + all_gather(y) then local reduce — 2 AGs (heavier)
  (c) all_reduce(x, SUM) + all_reduce(y, SUM) then slice y_full to shard
Strategy (c) does two same-type ARs plus a metadata narrow — potentially
fastest on Neuron because two ARs can co-fuse on the collective launch,
but naive reasoning suggests (a) since RS avoids sending full y to all.
Baseline seed is (b) two AGs — clearly worst.
'''
_P150_BUILTINS = {
    'naive_two_ag': '''def evolved_p150(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ag_x = xm.all_gather(x, dim=0).reshape(world_size, N)
    z1 = ag_x.sum(0)
    ag_y = xm.all_gather(y, dim=0).reshape(world_size, N * world_size)
    z2 = ag_y[:, rank*N:(rank+1)*N].sum(0)
    return torch.cat([z1, z2], dim=0)
''',
}
register_problem(CollectiveProblem(
    name='dual_reduce_shard',
    display_name='Problem P_150',
    evolved_fn_name='evolved_p150',
    signature=_P150_SIG,
    signature_doc=_P150_DOC,
    reference_fn=_p150_ref,
    generate_test_case=_p150_generate,
    call_candidate=_p150_call,
    builtin_templates=_P150_BUILTINS,
))


# =============================================================================
# P_151 topk_from_sum — reduce + top-k, mixed primitives
# =============================================================================

def _p151_ref(inputs, world_size):
    """Reference: y[k] = k-th largest value of sum_r x_r, for k in 0..K-1.
    K = 8, x_r shape (N * world_size,).
    """
    K = 8
    x_sum = sum(inputs[r]['x'] for r in range(world_size))
    sorted_vals, _ = torch.sort(x_sum, descending=True)
    top_k = sorted_vals[:K]
    return [top_k.clone() for _ in range(world_size)]


def _p151_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 1024
    torch.manual_seed(seed)
    inputs = [{'x': torch.randn(N * world_size) * (r + 1)}
              for r in range(world_size)]
    per_rank_args = [{'x': inputs[r]['x'], 'N': N, 'K': 8}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p151_ref(inputs, world_size)}


def _p151_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank_args['K'],
                        rank, ws, nd, cpd, xm_mock, torch_mock,
                        num_nodes=num_nodes)


_P151_SIG = '''def evolved_p151(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P151_DOC = '''Args: x (N*world_size,) - local vector. N=1024, K=8.

Formula: y[k] = k-th largest value of (sum_r x_r), for k in 0..K-1.
Output shape (K,), same on every rank.

NON-OBVIOUS: torch.topk unsupported on Neuron — must be emulated. Two
plausible strategies:
  (a) all_reduce(x, SUM) then iterative max-K-times locally on N*world
      elements                                            [1 big collective]
  (b) reduce_scatter(x, SUM) then local iterative max on N elements,
      all_gather local top-K, then iterative max on world*K globally
                                                          [2 collectives]

Baseline is a slight rearrangement of (b) — the "obvious" divide-and-conquer.
'''
_P151_BUILTINS = {
    'naive_rs_topk_ag': '''def evolved_p151(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    shard = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0,
                              shard_count=world_size)
    vals = []
    tmp = shard.clone()
    for _ in range(K):
        v = tmp.max()
        vals.append(v)
        tmp = torch.where(tmp >= v - 1e-6, torch.tensor(-1e9, device=tmp.device), tmp)
    local_top = torch.stack(vals)
    ag = xm.all_gather(local_top, dim=0).reshape(world_size, K)
    all_top = ag.reshape(-1)
    global_vals = []
    tmp = all_top.clone()
    for _ in range(K):
        v = tmp.max()
        global_vals.append(v)
        tmp = torch.where(tmp >= v - 1e-6, torch.tensor(-1e9, device=tmp.device), tmp)
    return torch.stack(global_vals)
''',
}
register_problem(CollectiveProblem(
    name='topk_from_sum',
    display_name='Problem P_151',
    evolved_fn_name='evolved_p151',
    signature=_P151_SIG,
    signature_doc=_P151_DOC,
    reference_fn=_p151_ref,
    generate_test_case=_p151_generate,
    call_candidate=_p151_call,
    builtin_templates=_P151_BUILTINS,
))


# =============================================================================
# P_152 offset_shift_window — data-dependent slice offset
# =============================================================================

def _p152_ref(inputs, world_size):
    """Reference: rank r returns window of length N starting at offset
    (r*13) % (N*world - N) from the all-gathered concat of all ranks' x.
    x_r has shape (N,).
    """
    N = inputs[0]['x'].shape[0]
    W = world_size
    concat_all = torch.cat([inputs[r]['x'] for r in range(W)], dim=0)
    outs = []
    total = N * W
    for r in range(W):
        offset = (r * 13) % (total - N)
        outs.append(concat_all[offset:offset+N].clone())
    return outs


def _p152_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 1024
    torch.manual_seed(seed)
    inputs = [{'x': torch.randn(N) * (r + 1)} for r in range(world_size)]
    per_rank_args = [{'x': inputs[r]['x'], 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p152_ref(inputs, world_size)}


def _p152_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)


_P152_SIG = '''def evolved_p152(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P152_DOC = '''Args: x (N,) - local vector.
Formula: y = concat_all_ranks(x)[offset : offset+N] where
         offset = (rank * 13) % (N*world_size - N).
Output shape (N,), unique per rank (data-dependent window position).

NON-OBVIOUS: three plausible ways to extract the window from the
all_gathered buffer —
  (a) reshape to (world_size, N) then advanced index by row (only works
      if offset happens to be a multiple of N)
  (b) linear slice ag[offset:offset+N] — general
  (c) torch.narrow(ag, 0, offset, N) — view-op equivalent to (b)
Neuron's XLA lowering treats slice vs narrow differently; the "obvious"
reshape+index only works for aligned offsets and is otherwise wrong.
'''
_P152_BUILTINS = {
    'naive_reshape_index': '''def evolved_p152(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ag = xm.all_gather(x, dim=0).reshape(world_size, N)
    idx = (rank * 13 // N) % world_size
    return ag[idx]
''',
}
register_problem(CollectiveProblem(
    name='offset_shift_window',
    display_name='Problem P_152',
    evolved_fn_name='evolved_p152',
    signature=_P152_SIG,
    signature_doc=_P152_DOC,
    reference_fn=_p152_ref,
    generate_test_case=_p152_generate,
    call_candidate=_p152_call,
    builtin_templates=_P152_BUILTINS,
))
