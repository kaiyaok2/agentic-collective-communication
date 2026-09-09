"""Round-22: 10 more 2D bcast problems targeting kiss's strength.

Lessons: kiss wins big on 2D matrix problems requiring broadcast.
Strat's Phase-3 handles simple 1D formulas well (const-fold) but often
misses 2D outer-product-like structures.
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


# P_180 tri_mask_bcast: y[i,j] = 1 if i >= j else 0  (lower triangular)
def _p180(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (ii >= jj).to(torch.float32).contiguous()
_register('tri_mask_bcast', 'Problem P_180', 'evolved_p180',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = 1.0 if i >= j else 0.0. Lower-triangular mask.
Return shape (N, N).''',
          _p180, 32, 2)


# P_181 mod_i_plus_j_bcast: y[i,j] = (i + 2*j) % 11
def _p181(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii + 2 * jj) % 11).to(torch.float32).contiguous()
_register('mod_i_plus_j_bcast', 'Problem P_181', 'evolved_p181',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i + 2*j) % 11.
Return shape (N, N).''',
          _p181, 32, 2)


# P_182 xor_mask_ij_bcast: y[i,j] = (i^j) & 0xF
def _p182(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_and(torch.bitwise_xor(ii, jj), 0xF).to(torch.float32).contiguous()
_register('xor_mask_ij_bcast', 'Problem P_182', 'evolved_p182',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) AND 0xF.
Return shape (N, N). Combines bitwise ops.''',
          _p182, 32, 2)


# P_183 sq_sum_ij_bcast: y[i,j] = i*i + j*j
def _p183(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (ii * ii + jj * jj).to(torch.float32).contiguous()
_register('sq_sum_ij_bcast', 'Problem P_183', 'evolved_p183',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = i*i + j*j.  Sum of squares.
Return shape (N, N).''',
          _p183, 32, 2)


# P_184 eq_mask_ij_bcast: y[i,j] = 1 if (i % 3) == (j % 5) else 0
def _p184(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii % 3) == (jj % 5)).to(torch.float32).contiguous()
_register('eq_mask_ij_bcast', 'Problem P_184', 'evolved_p184',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = 1.0 if (i%3) == (j%5) else 0.0.
Return shape (N, N).''',
          _p184, 32, 2)


# P_185 shifted_id_bcast: y[i,j] = 1 if j == (i+3) % N else 0
def _p185(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (jj == (ii + 3) % N).to(torch.float32).contiguous()
_register('shifted_id_bcast', 'Problem P_185', 'evolved_p185',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = 1.0 if j == (i+3) % N else 0.0. Shifted identity.
Return shape (N, N).''',
          _p185, 32, 2)


# P_186 abs_diff_ij_bcast: y[i,j] = |i - j| % 5
def _p186(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii - jj).abs() % 5).to(torch.float32).contiguous()
_register('abs_diff_ij_bcast', 'Problem P_186', 'evolved_p186',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = |i - j| % 5.
Return shape (N, N).''',
          _p186, 32, 2)


# P_187 poly_ij_bcast: y[i,j] = i*i - 2*j + 3
def _p187(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (ii * ii - 2 * jj + 3).to(torch.float32).contiguous()
_register('poly_ij_bcast', 'Problem P_187', 'evolved_p187',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = i^2 - 2*j + 3.
Return shape (N, N).''',
          _p187, 32, 2)


# P_188 hamming_mod_bcast: y[i,j] = popcount(i^j) % 4
def _p188(N):
    ii = torch.arange(N).unsqueeze(1)
    jj = torch.arange(N).unsqueeze(0)
    ref = torch.zeros(N, N, dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            ref[i, j] = bin(i ^ j).count('1') % 4
    return ref
_register('hamming_mod_bcast', 'Problem P_188', 'evolved_p188',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=16.
Formula: y[i, j] = popcount(i XOR j) % 4.
Return shape (N, N). Popcount + modulo.''',
          _p188, 16, 2)


# P_189 xor_min_bcast: y[i,j] = min(i XOR j, 7)
def _p189(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    xor = torch.bitwise_xor(ii, jj)
    return torch.clamp(xor, max=7).to(torch.float32).contiguous()
_register('xor_min_bcast', 'Problem P_189', 'evolved_p189',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = min(i XOR j, 7).
Return shape (N, N).''',
          _p189, 32, 2)
