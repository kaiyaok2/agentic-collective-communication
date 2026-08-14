"""Round-18b additional problems: fusion-resistant.

P_153 double_shard_reduce — 2 vectors need per-rank shard of AR-SUM.
Smoke-test showed 4% divergence between 2xRS vs cat+1RS vs 2xAR+narrow.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _p153_ref(inputs, world_size):
    """Reference: rank r returns concat(shard_r(sum(x)), shard_r(sum(y))).
    x_r, y_r shape (N*world_size,).
    """
    N = inputs[0]['N']
    x_sum = sum(inputs[r]['x'] for r in range(world_size))
    y_sum = sum(inputs[r]['y'] for r in range(world_size))
    outs = []
    for r in range(world_size):
        zx = x_sum[r*N:(r+1)*N]
        zy = y_sum[r*N:(r+1)*N]
        outs.append(torch.cat([zx, zy], dim=0))
    return outs


def _p153_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 1024
    torch.manual_seed(seed)
    inputs = []
    for r in range(world_size):
        x = torch.randn(N * world_size) * (r + 1)
        y = torch.randn(N * world_size) * (r + 1) * 0.5
        inputs.append({'x': x, 'y': y, 'N': N})
    per_rank_args = [inputs[r] for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p153_ref(inputs, world_size)}


def _p153_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['y'], rank_args['N'],
                        rank, ws, nd, cpd, xm_mock, torch_mock,
                        num_nodes=num_nodes)


_P153_SIG = '''def evolved_p153(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P153_DOC = '''Args: x, y - each shape (N*world_size,). N=1024.
Formula: rank r returns concat(zx, zy) where
  zx[i] = sum_r x_r[rank*N + i]  (rank's shard of AR-SUM of x)
  zy[i] = sum_r y_r[rank*N + i]  (rank's shard of AR-SUM of y)

NON-OBVIOUS: three plausible strategies —
  (a) 2× reduce_scatter — one per vector [naive, 2 collectives]
  (b) concat(x, y) then 1 reduce_scatter, then narrow-split
      [needs careful cat pattern; smoke-tested best]
  (c) 2× all_reduce then narrow to shard   [2 same-type collectives + narrow]

Baseline seed: naive 2× reduce_scatter.
'''
_P153_BUILTINS = {
    'naive_two_rs': '''def evolved_p153(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    zx = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0,
                            shard_count=world_size)
    zy = xm.reduce_scatter(xm.REDUCE_SUM, y, scale=1.0, scatter_dim=0,
                            shard_count=world_size)
    return torch.cat([zx, zy], dim=0)
''',
}
register_problem(CollectiveProblem(
    name='double_shard_reduce',
    display_name='Problem P_153',
    evolved_fn_name='evolved_p153',
    signature=_P153_SIG,
    signature_doc=_P153_DOC,
    reference_fn=_p153_ref,
    generate_test_case=_p153_generate,
    call_candidate=_p153_call,
    builtin_templates=_P153_BUILTINS,
))
