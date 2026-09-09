"""Round-23: 12 more no-comm _bcast problems.

Design: problems that combine multiple vectorized ops in ways that strat's
Phase-3 template enum tends to miss but kiss's freeform code composition
handles naturally. Focus:
- Nested modular arithmetic
- Multi-condition where
- Chained bitwise + arithmetic
- Boundary/edge patterns
"""
import torch
from .problems import CollectiveProblem, register_problem


def _bcast_gen(ref_fn, N, ndim=2, dtype=torch.float32):
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


_SIG = '''def evolved_{fn}(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_BASELINE = '''def evolved_{fn}(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
'''


def _register(name, dn, fn, sig_doc, ref, N, ndim):
    register_problem(CollectiveProblem(
        name=name, display_name=dn, evolved_fn_name=fn,
        signature=_SIG.format(fn=fn.replace('evolved_', '')),
        signature_doc=sig_doc,
        reference_fn=lambda inputs, ws, _r=ref: [_r(inputs[0]['N']) for _ in range(ws)],
        generate_test_case=_bcast_gen(ref, N, ndim),
        call_candidate=_bcast_call,
        builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn=fn.replace('evolved_', ''))},
    ))


# P_190 nested_pw_bcast: y[i] = (i%3+1) * (i%5+2)
def _p190(N):
    idx = torch.arange(N)
    return ((idx % 3 + 1) * (idx % 5 + 2)).to(torch.float32)
_register('nested_pw_bcast', 'Problem P_190', 'evolved_p190',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = (i%3 + 1) * (i%5 + 2).  Product of two piecewise terms.
Return shape (N,).''', _p190, 64, 1)


# P_191 chain_xor_bcast: y[i] = (i^(i>>2))
def _p191(N):
    idx = torch.arange(N)
    return torch.bitwise_xor(idx, idx >> 2).to(torch.float32)
_register('chain_xor_bcast', 'Problem P_191', 'evolved_p191',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = i XOR (i >> 2).
Return shape (N,).''', _p191, 128, 1)


# P_192 wave_bcast: y[i] = (i * 3 - N) % 11 - 5
def _p192(N):
    idx = torch.arange(N)
    return ((idx * 3 - N) % 11 - 5).to(torch.float32)
_register('wave_bcast', 'Problem P_192', 'evolved_p192',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = (i*3 - N) % 11 - 5.
Return shape (N,).''', _p192, 128, 1)


# P_193 three_way_bcast: y[i] = 0 if i<10 else (1 if i<20 else 2)
def _p193(N):
    idx = torch.arange(N)
    return torch.where(idx < 10, torch.zeros_like(idx),
                       torch.where(idx < 20, torch.ones_like(idx),
                                   torch.full_like(idx, 2))).to(torch.float32)
_register('three_way_bcast', 'Problem P_193', 'evolved_p193',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = 0 if i<10 else (1 if i<20 else 2). Three-way piecewise.
Return shape (N,).''', _p193, 64, 1)


# P_194 diag_bands_bcast: y[i,j] = ((i-j) // 3) % 4
def _p194(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (((ii - jj) // 3) % 4).to(torch.float32).contiguous()
_register('diag_bands_bcast', 'Problem P_194', 'evolved_p194',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = ((i-j) // 3) % 4. Diagonal bands.
Return shape (N, N).''', _p194, 32, 2)


# P_195 xor_add_bcast: y[i,j] = (i XOR j) + i
def _p195(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (torch.bitwise_xor(ii, jj) + ii).to(torch.float32).contiguous()
_register('xor_add_bcast', 'Problem P_195', 'evolved_p195',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) + i.
Return shape (N, N).''', _p195, 32, 2)


# P_196 boolean_grid_bcast: y[i,j] = 1 if (i&j)==0 else 0
def _p196(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (torch.bitwise_and(ii, jj) == 0).to(torch.float32).contiguous()
_register('boolean_grid_bcast', 'Problem P_196', 'evolved_p196',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = 1.0 if (i AND j) == 0 else 0.0.
Return shape (N, N).''', _p196, 32, 2)


# P_197 chained_mod_bcast: y[i] = ((i * 7) % 13) * ((i * 3) % 5)
def _p197(N):
    idx = torch.arange(N)
    return (((idx * 7) % 13) * ((idx * 3) % 5)).to(torch.float32)
_register('chained_mod_bcast', 'Problem P_197', 'evolved_p197',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = ((i*7) % 13) * ((i*3) % 5).
Return shape (N,).''', _p197, 64, 1)


# P_198 sign_mask_bcast: y[i,j] = (i-j) if i>j else 0
def _p198(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.where(ii > jj, ii - jj, torch.zeros_like(ii)).to(torch.float32).contiguous()
_register('sign_mask_bcast', 'Problem P_198', 'evolved_p198',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = i - j if i > j else 0. Sign-masked difference.
Return shape (N, N).''', _p198, 32, 2)


# P_199 pow_mod_bcast: y[i] = (2 ** (i % 6)) % 17
def _p199(N):
    idx = torch.arange(N)
    return ((2 ** (idx % 6)) % 17).to(torch.float32)
_register('pow_mod_bcast', 'Problem P_199', 'evolved_p199',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = (2^(i%6)) % 17.
Return shape (N,).''', _p199, 64, 1)


# P_200 concentric_bcast: y[i,j] = max(|i - N//2|, |j - N//2|)
def _p200(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    c = N // 2
    return torch.maximum((ii - c).abs(), (jj - c).abs()).to(torch.float32).contiguous()
_register('concentric_bcast', 'Problem P_200', 'evolved_p200',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = max(|i - N/2|, |j - N/2|). Concentric squares.
Return shape (N, N).''', _p200, 32, 2)


# P_201 diamond_bcast: y[i,j] = |i - N//2| + |j - N//2|
def _p201(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    c = N // 2
    return ((ii - c).abs() + (jj - c).abs()).to(torch.float32).contiguous()
_register('diamond_bcast', 'Problem P_201', 'evolved_p201',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = |i - N/2| + |j - N/2|. Diamond/L1 pattern.
Return shape (N, N).''', _p201, 32, 2)
