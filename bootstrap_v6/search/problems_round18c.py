"""Round-18c: grad_sync-inspired triple-tensor AR problem.

P_154 triple_grad_ar — 3 tensors need AR-SUM.
Smoke-test RT ordering (2-node 64-rank, warm cache):
  three_ar (naive):        5.73 ms
  cat_ar_split:            5.46 ms  (5% faster)
  cat_ar_narrow (winner):  5.23 ms  (8% faster)

The narrow-vs-split difference is the same trick that made kiss win on
batched_ar_scale_chal. Here it's a smaller/cleaner problem.

Baseline seed: three per-tensor AR (naive).
"""
import torch
from .problems import CollectiveProblem, register_problem


def _p154_ref(inputs, world_size):
    """3 tensors summed across ranks. Each rank returns all 3.
    x, y, z all shape (N*2,).
    """
    x_sum = sum(inputs[r]['x'] for r in range(world_size))
    y_sum = sum(inputs[r]['y'] for r in range(world_size))
    z_sum = sum(inputs[r]['z'] for r in range(world_size))
    outs = []
    for r in range(world_size):
        outs.append([x_sum.clone(), y_sum.clone(), z_sum.clone()])
    return outs


def _p154_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 1024
    torch.manual_seed(seed)
    inputs = []
    for r in range(world_size):
        x = torch.randn(N * 2) * (r + 1)
        y = torch.randn(N * 2) * (r + 1) * 0.5
        z = torch.randn(N * 2) * (r + 1) * 0.25
        inputs.append({'x': x, 'y': y, 'z': z, 'N': N})
    per_rank_args = [inputs[r] for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p154_ref(inputs, world_size)}


def _p154_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['y'], rank_args['z'],
                        rank_args['N'], rank, ws, nd, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)


_P154_SIG = '''def evolved_p154(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P154_DOC = '''Args: x, y, z - three tensors each of shape (2N,). N=1024.
Formula: return [sum_r x_r, sum_r y_r, sum_r z_r] — each shape (2N,).

NON-OBVIOUS: three plausible strategies —
  (a) 3 separate all_reduce(SUM) calls              [naive, 3 collectives]
  (b) cat(x, y, z) then 1 all_reduce, then torch.split into 3 tensors
      [1 collective + 3 slices]
  (c) cat(x, y, z) then 1 all_reduce, then torch.narrow x3
      [1 collective + 3 metadata views]

Difference between (b) and (c): torch.narrow is a metadata-only view op
on Neuron; torch.split copies memory. The RT ordering is c < b < a.
Baseline seed: naive 3-AR.
'''
_P154_BUILTINS = {
    'naive_three_ar': '''def evolved_p154(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return [xm.all_reduce(xm.REDUCE_SUM, x),
            xm.all_reduce(xm.REDUCE_SUM, y),
            xm.all_reduce(xm.REDUCE_SUM, z)]
''',
}
register_problem(CollectiveProblem(
    name='triple_grad_ar',
    display_name='Problem P_154',
    evolved_fn_name='evolved_p154',
    signature=_P154_SIG,
    signature_doc=_P154_DOC,
    reference_fn=_p154_ref,
    generate_test_case=_p154_generate,
    call_candidate=_p154_call,
    builtin_templates=_P154_BUILTINS,
))
