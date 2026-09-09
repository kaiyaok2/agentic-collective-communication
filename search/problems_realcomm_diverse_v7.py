"""Diverse Round 15 (V7): more classes."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5400: batched_ar_split — baseline splits x into 4 pieces then AR each
# Sorcar: AR the whole thing, then split (or don't split at all)
# ============================================================
def _p5400_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5400_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5400_ref(per_rank_args, world_size)}

def _p5400_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5400_SIG = '''def evolved_p5400(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5400_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x). Return (N,) identical on every rank.
Baseline splits x into 4 chunks, ARs each independently, then concatenates.
Sorcar: single AR on the full tensor.'''

_P5400_BUILTINS = {'ar_4chunk_pattern': '''def evolved_p5400(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    quarter = N // 4
    chunks = [x[i*quarter:(i+1)*quarter] for i in range(4)]
    reduced = [xm.all_reduce(xm.REDUCE_SUM, c) for c in chunks]
    return torch.cat(reduced, dim=0)
'''}

register_problem(CollectiveProblem(
    name='ar_4chunk_pattern_chal',
    display_name='Problem P_5400',
    evolved_fn_name='evolved_p5400',
    signature=_P5400_SIG,
    signature_doc=_P5400_DOC,
    reference_fn=_p5400_ref,
    generate_test_case=_p5400_generate,
    call_candidate=_p5400_call,
    builtin_templates=_P5400_BUILTINS,
))


# ============================================================
# P_5401: transpose_around_ar — .T is free, so wrapping AR with T.T works
# Baseline: transposes, ARs (with transposed layout), transposes back
# Sorcar: skip the transposes
# ============================================================
def _p5401_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5401_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(128, 256)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5401_ref(per_rank_args, world_size)}

def _p5401_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5401_SIG = '''def evolved_p5401(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5401_DOC = '''Local x (128, 256).
Compute: AR of x. Return (128, 256) identical on every rank.
Baseline transposes x, ARs, transposes back.'''

_P5401_BUILTINS = {'ar_transposed': '''def evolved_p5401(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    xt = x.t()
    axt = xm.all_reduce(xm.REDUCE_SUM, xt)
    return axt.t()
'''}

register_problem(CollectiveProblem(
    name='ar_transposed_chal',
    display_name='Problem P_5401',
    evolved_fn_name='evolved_p5401',
    signature=_P5401_SIG,
    signature_doc=_P5401_DOC,
    reference_fn=_p5401_ref,
    generate_test_case=_p5401_generate,
    call_candidate=_p5401_call,
    builtin_templates=_P5401_BUILTINS,
))


# ============================================================
# P_5402: masked_selection_from_ar — baseline uses expensive gather
#   on rank-common indices to select from AR output.
# Sorcar: if rank-common, entire selection is deterministic → just compute
# ============================================================
def _p5402_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    # Select first half of AR output
    N = inputs[0]['N']
    return [ax[:N//2].clone() for _ in range(world_size)]

def _p5402_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 131072
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5402_ref(per_rank_args, world_size)}

def _p5402_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5402_SIG = '''def evolved_p5402(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5402_DOC = '''Local x (N,). N=131072.
Compute: y = AR(x)[:N/2] — first half of AR output. Return (N/2,).
Baseline ARs the full N bytes then slices; more efficient: only AR the first half.'''

_P5402_BUILTINS = {'ar_full_then_slice_half': '''def evolved_p5402(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)   # transfers N bytes
    return ax[:N//2]                       # only need N/2
'''}

register_problem(CollectiveProblem(
    name='ar_slice_half_chal',
    display_name='Problem P_5402',
    evolved_fn_name='evolved_p5402',
    signature=_P5402_SIG,
    signature_doc=_P5402_DOC,
    reference_fn=_p5402_ref,
    generate_test_case=_p5402_generate,
    call_candidate=_p5402_call,
    builtin_templates=_P5402_BUILTINS,
))


# ============================================================
# P_5403: ar_of_broadcast — baseline broadcasts a scalar to N then ARs
# Sorcar: AR the scalar, then broadcast (N×smaller comm)
# ============================================================
def _p5403_ref(inputs, world_size):
    total = sum(inp['s'] for inp in inputs)
    N = inputs[0]['N']
    result = torch.full((N,), float(total))
    return [result.clone() for _ in range(world_size)]

def _p5403_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'s': torch.tensor(float(r + 1)), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5403_ref(per_rank_args, world_size)}

def _p5403_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['s'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5403_SIG = '''def evolved_p5403(s, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5403_DOC = '''Local scalar s (0-d). N=65536.
Compute: y = full-N tensor of sum_r(s_r). Return (N,) filled with sum-of-scalars.
Baseline broadcasts s to (N,) first then ARs — wasteful. Sorcar should AR the
scalar (1 element) then broadcast.'''

_P5403_BUILTINS = {'broadcast_before_ar': '''def evolved_p5403(s, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    tensor_N = s.expand(N).contiguous()               # (N,)
    ax = xm.all_reduce(xm.REDUCE_SUM, tensor_N)       # N bytes AR — wasteful
    return ax
'''}

register_problem(CollectiveProblem(
    name='ar_of_broadcast_scalar_chal',
    display_name='Problem P_5403',
    evolved_fn_name='evolved_p5403',
    signature=_P5403_SIG,
    signature_doc=_P5403_DOC,
    reference_fn=_p5403_ref,
    generate_test_case=_p5403_generate,
    call_candidate=_p5403_call,
    builtin_templates=_P5403_BUILTINS,
))
