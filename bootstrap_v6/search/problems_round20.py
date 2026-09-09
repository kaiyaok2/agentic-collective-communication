"""Round-20: 8 additional no-comm _bcast problems + 4 mixed challenge.

Design principle: rank 0 has the reference; other ranks zeros. Position-
based formulas that kiss will find via 'local recompute' but strat's
Phase-3 LLM often misses (because it enumerates collective strategies
by default).

All 8 use torch.arange + arithmetic — no bincount, no scatter_add,
no cumsum, no sort. Formulas designed so const-fold list-comp fallback
also works.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _bcast_generate(ref_fn, N, ndim, dtype=torch.float32):
    """Standard bcast generator: rank 0 has correct N-D tensor, others zero."""
    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        ref = ref_fn(N)
        shape = tuple([N] * ndim)
        per_rank = []
        for r in range(world_size):
            x = ref.clone() if r == 0 else torch.zeros(shape, dtype=dtype)
            per_rank.append({'x': x, 'N': N})
        expected = [ref.clone() for _ in range(world_size)]
        return {'per_rank_args': per_rank, 'shared_args': {},
                'expected': expected}
    return gen


def _bcast_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd,
                xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args['x'], rank_args['N'], rank, ws, nd, cpd,
                        xm_mock, torch_mock, num_nodes=num_nodes)


_SIG_1D = '''def evolved_{fn}(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_SIG_2D = _SIG_1D
_BASELINE = '''def evolved_{fn}(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
'''


# P_160 fib_mod_bcast: y[i] = (fib(i) % 13) as float, i=0..N-1
def _fib_ref(N):
    fib = [0, 1]
    for _ in range(N - 2):
        fib.append((fib[-1] + fib[-2]))
    return torch.tensor([f % 13 for f in fib[:N]], dtype=torch.float32)


register_problem(CollectiveProblem(
    name='fib_mod_bcast',
    display_name='Problem P_160',
    evolved_fn_name='evolved_p160',
    signature=_SIG_1D.format(fn='p160'),
    signature_doc='''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = fib(i) % 13, where fib(0)=0, fib(1)=1, fib(i)=fib(i-1)+fib(i-2).
Return shape (N,).
NON-OBVIOUS: recursive definition — hard to vectorize with torch.arange
alone. Consider Python list-comprehension + torch.tensor([...]).''',
    reference_fn=lambda inputs, ws: [_fib_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_fib_ref, 64, 1),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p160')},
))


# P_161 lucas_bcast: y[i] = (2*i + 1) if i even else (3*i - 2)
def _lucas_ref(N):
    return torch.tensor(
        [2 * i + 1 if i % 2 == 0 else 3 * i - 2 for i in range(N)],
        dtype=torch.float32)


register_problem(CollectiveProblem(
    name='lucas_bcast',
    display_name='Problem P_161',
    evolved_fn_name='evolved_p161',
    signature=_SIG_1D.format(fn='p161'),
    signature_doc='''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = 2*i + 1 if i%2==0 else 3*i - 2.
Return shape (N,).
NON-OBVIOUS: piecewise formula, vectorizable via torch.where + torch.arange.''',
    reference_fn=lambda inputs, ws: [_lucas_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_lucas_ref, 128, 1),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p161')},
))


# P_162 checkerboard_bcast: y[i,j] = (i + j) % 2
def _check_ref(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1)
    jj = idx.unsqueeze(0)
    return ((ii + jj) % 2).to(torch.float32).expand(N, N).contiguous()


register_problem(CollectiveProblem(
    name='checkerboard_bcast',
    display_name='Problem P_162',
    evolved_fn_name='evolved_p162',
    signature=_SIG_2D.format(fn='p162'),
    signature_doc='''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i + j) % 2. Checkerboard pattern.
Return shape (N, N).
NON-OBVIOUS: position-based, vectorizable via broadcast (torch.arange +
unsqueeze).''',
    reference_fn=lambda inputs, ws: [_check_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_check_ref, 32, 2),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p162')},
))


# P_163 diag_dist_bcast: y[i,j] = |i - j|
def _diag_dist_ref(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (ii - jj).abs().to(torch.float32).contiguous()


register_problem(CollectiveProblem(
    name='diag_dist_bcast',
    display_name='Problem P_163',
    evolved_fn_name='evolved_p163',
    signature=_SIG_2D.format(fn='p163'),
    signature_doc='''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = |i - j|. Manhattan-like diagonal distance.
Return shape (N, N).
NON-OBVIOUS: position-based, vectorizable via (arange - arange.T).abs().''',
    reference_fn=lambda inputs, ws: [_diag_dist_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_diag_dist_ref, 32, 2),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p163')},
))


# P_164 max_ij_bcast: y[i,j] = max(i, j)
def _max_ij_ref(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.maximum(ii, jj).to(torch.float32).contiguous()


register_problem(CollectiveProblem(
    name='max_ij_bcast',
    display_name='Problem P_164',
    evolved_fn_name='evolved_p164',
    signature=_SIG_2D.format(fn='p164'),
    signature_doc='''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = max(i, j).
Return shape (N, N).
NON-OBVIOUS: torch.maximum(arange.T, arange) — no collective needed.''',
    reference_fn=lambda inputs, ws: [_max_ij_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_max_ij_ref, 32, 2),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p164')},
))


# P_165 or_ij_bcast: y[i,j] = i | j (bitwise OR)
def _or_ij_ref(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_or(ii, jj).to(torch.float32).contiguous()


register_problem(CollectiveProblem(
    name='or_ij_bcast',
    display_name='Problem P_165',
    evolved_fn_name='evolved_p165',
    signature=_SIG_2D.format(fn='p165'),
    signature_doc='''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = i BITWISE-OR j.
Return shape (N, N).
NON-OBVIOUS: torch.bitwise_or on integer arange broadcast, cast to fp.''',
    reference_fn=lambda inputs, ws: [_or_ij_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_or_ij_ref, 32, 2),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p165')},
))


# P_166 and_ij_bcast: y[i,j] = i & j
def _and_ij_ref(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_and(ii, jj).to(torch.float32).contiguous()


register_problem(CollectiveProblem(
    name='and_ij_bcast',
    display_name='Problem P_166',
    evolved_fn_name='evolved_p166',
    signature=_SIG_2D.format(fn='p166'),
    signature_doc='''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = i BITWISE-AND j.
Return shape (N, N).
NON-OBVIOUS: torch.bitwise_and on arange broadcast.''',
    reference_fn=lambda inputs, ws: [_and_ij_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_and_ij_ref, 32, 2),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p166')},
))


# P_167 sq_diff_bcast: y[i] = (i * i) - i
def _sq_diff_ref(N):
    idx = torch.arange(N)
    return (idx * idx - idx).to(torch.float32)


register_problem(CollectiveProblem(
    name='sq_diff_bcast',
    display_name='Problem P_167',
    evolved_fn_name='evolved_p167',
    signature=_SIG_1D.format(fn='p167'),
    signature_doc='''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = i*i - i.
Return shape (N,).
NON-OBVIOUS: quadratic polynomial, simplest vectorization
torch.arange(N).float().pow(2) - torch.arange(N).float().''',
    reference_fn=lambda inputs, ws: [_sq_diff_ref(inputs[0]['N']) for _ in range(ws)],
    generate_test_case=_bcast_generate(_sq_diff_ref, 128, 1),
    call_candidate=_bcast_call,
    builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn='p167')},
))
