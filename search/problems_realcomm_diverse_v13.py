"""Diverse Round 15 (V13): pattern rewrites."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6000: repeated_ar_of_same_x_N_times — 5 ARs of same input x
# Baseline: 5 sequential ARs. Sorcar: 1 AR + multiply.
# Similar to P_5001 but different signature and coefficient constants.
# Explicitly checks CSE with different coefficient sums.
# ============================================================
def _p6000_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    # 5 different coefficients each scaling AR(x); sum them
    return [(ax * (1.5 + 2.5 + 3.5 + 0.5 + 1.5)).clone() for _ in range(world_size)]

def _p6000_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6000_ref(per_rank_args, world_size)}

def _p6000_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6000_SIG = '''def evolved_p6000(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6000_DOC = '''Local x (N,). N=65536.
Compute: y = 1.5*AR(x) + 2.5*AR(x) + 3.5*AR(x) + 0.5*AR(x) + 1.5*AR(x).
Return (N,) identical on every rank. Sum of coefficients = 9.5, so result = 9.5*AR(x).'''

_P6000_BUILTINS = {'ar_five_scaled_same': '''def evolved_p6000(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return (1.5 * xm.all_reduce(xm.REDUCE_SUM, x)
          + 2.5 * xm.all_reduce(xm.REDUCE_SUM, x)
          + 3.5 * xm.all_reduce(xm.REDUCE_SUM, x)
          + 0.5 * xm.all_reduce(xm.REDUCE_SUM, x)
          + 1.5 * xm.all_reduce(xm.REDUCE_SUM, x))
'''}

register_problem(CollectiveProblem(
    name='five_ar_scaled_same_input_chal',
    display_name='Problem P_6000',
    evolved_fn_name='evolved_p6000',
    signature=_P6000_SIG,
    signature_doc=_P6000_DOC,
    reference_fn=_p6000_ref,
    generate_test_case=_p6000_generate,
    call_candidate=_p6000_call,
    builtin_templates=_P6000_BUILTINS,
))


# ============================================================
# P_6001: ar_of_slice_within — AR of x[a:b] where slice is fixed
# Baseline: pad zeros to match N, AR full. Sorcar: AR the slice.
# ============================================================
def _p6001_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    N = inputs[0]['N']
    # slice [N/4:3N/4] — reduce middle half
    mid = ax[N//4:3*N//4]
    return [mid.clone() for _ in range(world_size)]

def _p6001_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 131072
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6001_ref(per_rank_args, world_size)}

def _p6001_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6001_SIG = '''def evolved_p6001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6001_DOC = '''Local x (N,). N=131072.
Compute: y = AR(x[N/4:3N/4]) — reduce only the middle half.
Return (N/2,) identical on every rank.
Baseline ARs full x then slices. Sorcar: slice before AR (half the bytes over EFA).'''

_P6001_BUILTINS = {'ar_full_then_slice_mid': '''def evolved_p6001(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)   # N bytes
    return ax[N//4:3*N//4]                  # only need half
'''}

register_problem(CollectiveProblem(
    name='ar_slice_middle_chal',
    display_name='Problem P_6001',
    evolved_fn_name='evolved_p6001',
    signature=_P6001_SIG,
    signature_doc=_P6001_DOC,
    reference_fn=_p6001_ref,
    generate_test_case=_p6001_generate,
    call_candidate=_p6001_call,
    builtin_templates=_P6001_BUILTINS,
))


# ============================================================
# P_6002: repeat_input_before_ar — expand x to (W, N) via .expand
# Baseline: expand + AR of (W, N) = W*N bytes. Sorcar: AR of N, expand after.
# ============================================================
def _p6002_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.unsqueeze(0).expand(world_size, -1).contiguous().clone() for _ in range(world_size)]

def _p6002_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 8192
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6002_ref(per_rank_args, world_size)}

def _p6002_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6002_SIG = '''def evolved_p6002(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6002_DOC = '''Local x (N,). N=8192.
Compute: y = expand AR(x) to (W, N). Return (W, N) identical on every rank.
Baseline expands x first (materializes W*N bytes) then ARs. Sorcar: AR then expand.'''

_P6002_BUILTINS = {'expand_before_ar': '''def evolved_p6002(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    expanded = x.unsqueeze(0).expand(world_size, N).contiguous()
    return xm.all_reduce(xm.REDUCE_SUM, expanded) / world_size
'''}

register_problem(CollectiveProblem(
    name='expand_before_ar_chal',
    display_name='Problem P_6002',
    evolved_fn_name='evolved_p6002',
    signature=_P6002_SIG,
    signature_doc=_P6002_DOC,
    reference_fn=_p6002_ref,
    generate_test_case=_p6002_generate,
    call_candidate=_p6002_call,
    builtin_templates=_P6002_BUILTINS,
))


# ============================================================
# P_6003: multi_ar_with_zeros — 3 ARs on 3 tensors, but 2 of them are all-zeros
# Baseline: 3 ARs. Sorcar: drop the AR(zero) calls (they contribute 0).
# ============================================================
def _p6003_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6003_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6003_ref(per_rank_args, world_size)}

def _p6003_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6003_SIG = '''def evolved_p6003(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6003_DOC = '''Local x (N,). N=65536.
Compute: y = AR(x) + AR(zeros) + AR(zeros).
Return (N,) identical on every rank.
Two ARs of zero tensors are dead code — Sorcar should drop them.'''

_P6003_BUILTINS = {'three_ars_two_zero': '''def evolved_p6003(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a1 = xm.all_reduce(xm.REDUCE_SUM, x)
    zeros1 = torch.zeros_like(x)
    a2 = xm.all_reduce(xm.REDUCE_SUM, zeros1)
    zeros2 = torch.zeros_like(x)
    a3 = xm.all_reduce(xm.REDUCE_SUM, zeros2)
    return a1 + a2 + a3
'''}

register_problem(CollectiveProblem(
    name='three_ars_two_zero_chal',
    display_name='Problem P_6003',
    evolved_fn_name='evolved_p6003',
    signature=_P6003_SIG,
    signature_doc=_P6003_DOC,
    reference_fn=_p6003_ref,
    generate_test_case=_p6003_generate,
    call_candidate=_p6003_call,
    builtin_templates=_P6003_BUILTINS,
))
