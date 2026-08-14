"""Round-17 problems: designed with multiple plausible solutions where
developer intuition may guess WRONG.

Based on smoke-tests on 2-node 64-rank cluster us-east-1d 2026-08-13:
- P140 bidi_grad_ar: 2-AR "wasteful" pattern beats 1-AR (2.19x RT). Naive
  ag_local approach ties 1-AR (5.35 ms/iter). Kiss might find the 2-AR
  trick; strat's templates probably won't.
- P141 segmented_ar: full-cat (1 AR) beats per-tensor (64 ARs) by 65%
  and beats size-bucketed (2 ARs) by 5.6%. Developer would likely
  bucket-by-size; correct answer is fuse-all.

Both problems use per-tensor / naive-bucket as SEED baseline.
"""
import torch
from .problems import CollectiveProblem, register_problem


# =============================================================================
# P_140 bidi_grad_ar — half of ranks want SUM, other half want MAX
# =============================================================================

def _p140_ref(inputs, world_size):
    """Reference: rank r returns SUM(x_0..x_{W-1}) if r < W/2 else MAX(x_0..x_{W-1})"""
    xs = list(inputs)
    s = sum(xs)
    m = xs[0].clone()
    for x in xs[1:]:
        m = torch.maximum(m, x)
    out = []
    half = world_size // 2
    for r in range(world_size):
        out.append(s if r < half else m)
    return out


def _p140_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 1024
    torch.manual_seed(seed)
    inputs = [torch.randn(N) * (r + 1) for r in range(world_size)]
    per_rank_args = [{'x': inputs[r]} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p140_ref(inputs, world_size)}


def _p140_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank, ws, nd, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)


_P140_SIG = '''def evolved_p140(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P140_DOC = '''Args: x (N,) - local vector, N=1024.
Formula: y = sum_r x_r if rank < world_size/2 else max_r x_r.
NON-OBVIOUS: The obvious approach is to compute the value your rank needs
(one AR-SUM or one AR-MAX). But doing BOTH ARs and branching at output can
be faster due to compiler tricks (fusion, dead-code elimination that
still participates in collective). Naive baseline is 1-AR per rank. Also
consider all_gather + local reduce.
'''
_P140_BUILTINS = {
    # Seed baseline: 1 AR based on rank's need
    'naive_conditional_ar': '''def evolved_p140(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    if rank < world_size // 2:
        return xm.all_reduce(xm.REDUCE_SUM, x)
    else:
        return xm.all_reduce(xm.REDUCE_MAX, x)
''',
}
register_problem(CollectiveProblem(
    name='bidi_grad_ar',
    display_name='Problem P_140',
    evolved_fn_name='evolved_p140',
    signature=_P140_SIG,
    signature_doc=_P140_DOC,
    reference_fn=_p140_ref,
    generate_test_case=_p140_generate,
    call_candidate=_p140_call,
    builtin_templates=_P140_BUILTINS,
))


# =============================================================================
# P_141 segmented_ar — mix of large + tiny tensors
# =============================================================================

def _p141_ref(inputs_list, world_size):
    """Reduce SUM over all 64 tensors."""
    n_tensors = len(inputs_list[0])
    outputs = []
    for ti in range(n_tensors):
        s = sum(inputs_list[r][ti] for r in range(world_size))
        outputs.append(s)
    return [list(outputs) for _ in range(world_size)]


def _p141_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    # 16 large (8192 elts = 32KB fp32) + 48 tiny (4 elts = 16B fp32)
    large_size = 8192
    tiny_size = 4
    n_large = 16
    n_tiny = 48
    inputs = []
    for r in range(world_size):
        rank_tensors = ([torch.randn(large_size) * (r + 1)
                         for _ in range(n_large)] +
                        [torch.randn(tiny_size) * (r + 1)
                         for _ in range(n_tiny)])
        inputs.append(rank_tensors)
    per_rank_args = [{'tensors': inputs[r]} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p141_ref(inputs, world_size)}


def _p141_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
               xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['tensors'], rank, ws, nd, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)


_P141_SIG = '''def evolved_p141(tensors, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P141_DOC = '''Args: tensors - list of 64 tensors. First 16 are size 8192
(32KB fp32 each). Last 48 are size 4 (16B each). Mixed-size gradients.
Formula: y[i] = sum over ranks of tensors[i]. Return all 64.
NON-OBVIOUS: 64 per-tensor ARs is baseline. Options: (a) full-cat into
1 giant AR, (b) size-bucket (large group + small group, 2 ARs), (c)
hierarchical inter-node then intra-node, (d) large per-tensor + small
cat. Developer would likely pick (b) size-bucket by intuition, but (a)
full-cat wins by 5-6% on 2-node HW. Per-tensor is 65% slower.
'''
_P141_BUILTINS = {
    # Seed baseline: naive per-tensor (worst)
    'naive_per_tensor_ar': '''def evolved_p141(tensors, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return [xm.all_reduce(xm.REDUCE_SUM, t) for t in tensors]
''',
}
register_problem(CollectiveProblem(
    name='segmented_ar',
    display_name='Problem P_141',
    evolved_fn_name='evolved_p141',
    signature=_P141_SIG,
    signature_doc=_P141_DOC,
    reference_fn=_p141_ref,
    generate_test_case=_p141_generate,
    call_candidate=_p141_call,
    builtin_templates=_P141_BUILTINS,
))
