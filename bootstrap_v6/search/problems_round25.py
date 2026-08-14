"""Round-25: 10 more no-comm _bcast problems — diverse formulas.

Focus on structural variety: 3-way (i,j,k) formulas, transposed vs
non-transposed 2D, sinusoidal/periodic patterns.
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


# P_220 xor_pow2_bcast: y[i,j] = (i XOR j) if (i XOR j) < 8 else 0
def _p220(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    xor = torch.bitwise_xor(ii, jj)
    return torch.where(xor < 8, xor, torch.zeros_like(xor)).to(torch.float32).contiguous()
_register('xor_pow2_bcast', 'Problem P_220', 'evolved_p220',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) if (i XOR j) < 8 else 0.
Return shape (N, N).''', _p220, 32, 2)


# P_221 outer_add_pow_bcast: y[i,j] = (i + j) ^ 2 (power)
def _p221(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii + jj) ** 2).to(torch.float32).contiguous()
_register('outer_add_pow_bcast', 'Problem P_221', 'evolved_p221',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i + j)^2.
Return shape (N, N).''', _p221, 32, 2)


# P_222 mod_grid_bcast: y[i,j] = (i * N + j) % 17
def _p222(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii * N + jj) % 17).to(torch.float32).contiguous()
_register('mod_grid_bcast', 'Problem P_222', 'evolved_p222',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i * N + j) % 17.  Linearized-then-modded.
Return shape (N, N).''', _p222, 32, 2)


# P_223 xor_add_mod_bcast: y[i,j] = ((i XOR j) + (i + j)) % 13
def _p223(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((torch.bitwise_xor(ii, jj) + ii + jj) % 13).to(torch.float32).contiguous()
_register('xor_add_mod_bcast', 'Problem P_223', 'evolved_p223',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = ((i XOR j) + (i + j)) % 13.
Return shape (N, N).''', _p223, 32, 2)


# P_224 mask_and_shift_bcast: y[i] = (i & 0xF) << 2
def _p224(N):
    idx = torch.arange(N)
    return (torch.bitwise_and(idx, 0xF) << 2).to(torch.float32)
_register('mask_and_shift_bcast', 'Problem P_224', 'evolved_p224',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = (i AND 0xF) << 2.
Return shape (N,).''', _p224, 128, 1)


# P_225 grid_step_bcast: y[i,j] = min(i, N-1-j)  (falling diagonal masked)
def _p225(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.minimum(ii, N - 1 - jj).to(torch.float32).contiguous()
_register('grid_step_bcast', 'Problem P_225', 'evolved_p225',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = min(i, N-1-j).
Return shape (N, N).''', _p225, 32, 2)


# P_226 xor_lookup_bcast: y[i,j] = (i * 5 + j * 7) XOR 13
def _p226(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return torch.bitwise_xor(ii * 5 + jj * 7, 13).to(torch.float32).contiguous()
_register('xor_lookup_bcast', 'Problem P_226', 'evolved_p226',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (5*i + 7*j) XOR 13.
Return shape (N, N).''', _p226, 32, 2)


# P_227 stairs_bcast: y[i,j] = (i // 4) + (j // 3)
def _p227(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii // 4) + (jj // 3)).to(torch.float32).contiguous()
_register('stairs_bcast', 'Problem P_227', 'evolved_p227',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i // 4) + (j // 3). Stair-step pattern.
Return shape (N, N).''', _p227, 32, 2)


# P_228 alt_xor_bcast: y[i,j] = (i XOR j) if (i XOR j) % 2 == 0 else 0
def _p228(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    xor = torch.bitwise_xor(ii, jj)
    return torch.where(xor % 2 == 0, xor, torch.zeros_like(xor)).to(torch.float32).contiguous()
_register('alt_xor_bcast', 'Problem P_228', 'evolved_p228',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) if (i XOR j) is even else 0.
Return shape (N, N).''', _p228, 32, 2)


# P_229 tanh_bcast: y[i,j] = clamp((i - N/2) * (j - N/2), -30, 30)
def _p229(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    c = N // 2
    return torch.clamp((ii - c) * (jj - c), -30, 30).to(torch.float32).contiguous()
_register('tanh_bcast', 'Problem P_229', 'evolved_p229',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = clamp((i - N/2) * (j - N/2), -30, 30).
Return shape (N, N).''', _p229, 32, 2)
