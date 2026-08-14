"""Round-24: 10 more problems targeting kiss's biggest wins.

Lessons: kiss wins BIG (>36×) on 2D bitwise + non-trivial vectorization.
Design 10 more in that sweet spot. Include some 3-dim outer-broadcast
where formula has 3 index axes.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _bcast_gen(ref_fn, N, ndim, dtype=torch.float32):
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


# P_210 xor_shl_bcast: y[i,j] = (i XOR j) << 1  (bitshift)
def _p210(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (torch.bitwise_xor(ii, jj) << 1).to(torch.float32).contiguous()
_register('xor_shl_bcast', 'Problem P_210', 'evolved_p210',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) << 1.
Return shape (N, N).''', _p210, 32, 2)


# P_211 xor_or_bcast: y[i,j] = (i XOR j) OR (i AND j)
def _p211(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_or(torch.bitwise_xor(ii, jj), torch.bitwise_and(ii, jj)).to(torch.float32).contiguous()
_register('xor_or_bcast', 'Problem P_211', 'evolved_p211',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) OR (i AND j).
Return shape (N, N).''', _p211, 32, 2)


# P_212 bit_hi_bcast: y[i,j] = (i XOR j) >> 3
def _p212(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (torch.bitwise_xor(ii, jj) >> 3).to(torch.float32).contiguous()
_register('bit_hi_bcast', 'Problem P_212', 'evolved_p212',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) >> 3.
Return shape (N, N).''', _p212, 32, 2)


# P_213 dilate_bcast: y[i,j] = 1 if (i%2==0 or j%2==0) else 0
def _p213(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii % 2 == 0) | (jj % 2 == 0)).to(torch.float32).contiguous()
_register('dilate_bcast', 'Problem P_213', 'evolved_p213',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = 1.0 if (i%2==0 OR j%2==0) else 0.0.
Return shape (N, N).''', _p213, 32, 2)


# P_214 pattern_stripe_bcast: y[i,j] = (i+2*j) & 0x7
def _p214(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_and(ii + 2 * jj, 0x7).to(torch.float32).contiguous()
_register('pattern_stripe_bcast', 'Problem P_214', 'evolved_p214',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i + 2*j) AND 0x7.
Return shape (N, N).''', _p214, 32, 2)


# P_215 wave2d_bcast: y[i,j] = (i * 3 + j * 5) % 11
def _p215(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii * 3 + jj * 5) % 11).to(torch.float32).contiguous()
_register('wave2d_bcast', 'Problem P_215', 'evolved_p215',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (3*i + 5*j) % 11.
Return shape (N, N).''', _p215, 32, 2)


# P_216 rev_shift_bcast: y[i,j] = (N - 1 - i) XOR j
def _p216(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_xor(N - 1 - ii, jj).to(torch.float32).contiguous()
_register('rev_shift_bcast', 'Problem P_216', 'evolved_p216',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (N - 1 - i) XOR j.
Return shape (N, N).''', _p216, 32, 2)


# P_217 clamp_bcast: y[i,j] = clamp((i-N//4) * (j-N//4), 0, 100)
def _p217(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    c = N // 4
    return torch.clamp((ii - c) * (jj - c), 0, 100).to(torch.float32).contiguous()
_register('clamp_bcast', 'Problem P_217', 'evolved_p217',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = clamp((i - N/4) * (j - N/4), 0, 100).
Return shape (N, N).''', _p217, 32, 2)


# P_218 popcount_ij_bcast: y[i,j] = popcount(i) + popcount(j)
def _p218(N):
    ref = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            ref[i, j] = bin(i).count('1') + bin(j).count('1')
    return ref
_register('popcount_ij_bcast', 'Problem P_218', 'evolved_p218',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = popcount(i) + popcount(j).
Return shape (N, N). Row and column popcount sum.''', _p218, 32, 2)


# P_219 gcd_lookup_bcast: y[i,j] = gcd(i+1, j+1)
def _p219(N):
    import math
    ref = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            ref[i, j] = math.gcd(i + 1, j + 1)
    return ref
_register('gcd_lookup_bcast', 'Problem P_219', 'evolved_p219',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=16.
Formula: y[i, j] = gcd(i + 1, j + 1).
Return shape (N, N). Greatest common divisor table.''', _p219, 16, 2)
