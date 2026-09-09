"""Round-26: 10 more targeted kiss > strat problems.

Strategy: patterns that force multiple broadcasts + non-obvious combinators
that strat's Phase-3 templates may miss.
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


# P_230 xor_lookup_hi_bcast: y[i,j] = (i * 11 + j * 13) XOR ((i + j) >> 2)
def _p230(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_xor(ii * 11 + jj * 13, (ii + jj) >> 2).to(torch.float32).contiguous()
_register('xor_lookup_hi_bcast', 'Problem P_230', 'evolved_p230',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (11i + 13j) XOR ((i+j) >> 2). Nested bitwise + arith.
Return shape (N, N).''', _p230, 32, 2)


# P_231 outer_max_min_bcast: y[i,j] = max(i, j) - min(i, j)
def _p231(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (torch.maximum(ii, jj) - torch.minimum(ii, jj)).to(torch.float32).contiguous()
_register('outer_max_min_bcast', 'Problem P_231', 'evolved_p231',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = max(i, j) - min(i, j).
Return shape (N, N).''', _p231, 32, 2)


# P_232 xor_bit_low_bcast: y[i,j] = (i ^ j) & 0x1  (low bit of xor)
def _p232(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_and(torch.bitwise_xor(ii, jj), 0x1).to(torch.float32).contiguous()
_register('xor_bit_low_bcast', 'Problem P_232', 'evolved_p232',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) AND 1. Parity of xor.
Return shape (N, N).''', _p232, 32, 2)


# P_233 outer_bitxor_shr_bcast: y[i,j] = (i XOR j) >> ((i + j) & 3)  (data-dep shift)
def _p233(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    xor = torch.bitwise_xor(ii, jj)
    shift = torch.bitwise_and(ii + jj, 3)
    return (xor >> shift).to(torch.float32).contiguous()
_register('outer_bitxor_shr_bcast', 'Problem P_233', 'evolved_p233',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) >> ((i + j) AND 3).
Return shape (N, N). Data-dep bit shift.''', _p233, 32, 2)


# P_234 xor_add_bit_bcast: y[i,j] = ((i XOR j) + (i * j)) & 0xFF
def _p234(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_and(torch.bitwise_xor(ii, jj) + ii * jj, 0xFF).to(torch.float32).contiguous()
_register('xor_add_bit_bcast', 'Problem P_234', 'evolved_p234',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = ((i XOR j) + i*j) AND 0xFF.
Return shape (N, N).''', _p234, 32, 2)


# P_235 sq_xor_bcast: y[i] = (i * i) XOR i
def _p235(N):
    idx = torch.arange(N)
    return torch.bitwise_xor(idx * idx, idx).to(torch.float32)
_register('sq_xor_bcast', 'Problem P_235', 'evolved_p235',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = (i*i) XOR i.
Return shape (N,).''', _p235, 128, 1)


# P_236 sequential_mod_bcast: y[i] = ((i * 5 + 3) % (i + 1)) if i > 0 else 0
def _p236(N):
    idx = torch.arange(N)
    div = idx + 1
    return torch.where(idx > 0, (idx * 5 + 3) % div,
                        torch.zeros_like(idx)).to(torch.float32)
_register('sequential_mod_bcast', 'Problem P_236', 'evolved_p236',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = (5i + 3) % (i + 1) if i > 0 else 0.
Return shape (N,). Variable modulus per position.''', _p236, 64, 1)


# P_237 rev_seq_bcast: y[i] = N - 1 - i (reversed range)
def _p237(N):
    idx = torch.arange(N)
    return (N - 1 - idx).to(torch.float32)
_register('rev_seq_bcast', 'Problem P_237', 'evolved_p237',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = N - 1 - i. Reversed range.
Return shape (N,).''', _p237, 128, 1)


# P_238 xor_sq_bcast: y[i,j] = (i * i) XOR (j * j)
def _p238(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_xor(ii * ii, jj * jj).to(torch.float32).contiguous()
_register('xor_sq_bcast', 'Problem P_238', 'evolved_p238',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i^2) XOR (j^2).
Return shape (N, N).''', _p238, 32, 2)


# P_239 masked_max_bcast: y[i,j] = max(i, j) if abs(i-j) < 5 else 0
def _p239(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.where((ii - jj).abs() < 5, torch.maximum(ii, jj),
                        torch.zeros_like(ii)).to(torch.float32).contiguous()
_register('masked_max_bcast', 'Problem P_239', 'evolved_p239',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = max(i, j) if |i - j| < 5 else 0.
Return shape (N, N).''', _p239, 32, 2)
