"""Round-21: 10 more no-comm _bcast problems.

Lessons learned:
- Kiss reliably finds arange + arithmetic formulas
- Strat sometimes finds const-fold but not always (LLM stochastic)
- Best kiss wins: formulas involving bitwise ops or non-trivial vectorization
- Strat's const-fold path: `torch.tensor([f(i) for i in range(N)])` — LLM
  occasionally proposes this. When it does, matches kiss.

New designs: formulas that need multiple vectorized ops so kiss's freeform
composition beats strat's single-strategy template. Include problems where
i, j appear as function args (not tensor values) — position-based must
compute from indices.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _bcast_gen(ref_fn, N, ndim=1, dtype=torch.float32):
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


def _register(name, dn, fn_name, sig_doc, ref, N, ndim=1):
    register_problem(CollectiveProblem(
        name=name, display_name=dn, evolved_fn_name=fn_name,
        signature=_SIG.format(fn=fn_name.split('_')[-1] if fn_name.startswith('evolved_') else fn_name),
        signature_doc=sig_doc,
        reference_fn=lambda inputs, ws, _ref=ref: [_ref(inputs[0]['N']) for _ in range(ws)],
        generate_test_case=_bcast_gen(ref, N, ndim),
        call_candidate=_bcast_call,
        builtin_templates={'baseline_ar_bcast': _BASELINE.format(fn=fn_name.split('_')[-1] if fn_name.startswith('evolved_') else fn_name)},
    ))


# P_170 xor_shr_bcast: y[i] = i XOR (i >> 1)  (gray code companion)
def _p170(N):
    idx = torch.arange(N)
    return torch.bitwise_xor(idx, idx >> 1).to(torch.float32)
_register('xor_shr_bcast', 'Problem P_170', 'evolved_p170',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = i XOR (i >> 1).  (This is a Gray code variant.)
Return shape (N,). No collective needed.''',
          _p170, 128)


# P_171 mod_xor_bcast: y[i,j] = (i XOR j) % 7
def _p171(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (torch.bitwise_xor(ii, jj) % 7).to(torch.float32).contiguous()
_register('mod_xor_bcast', 'Problem P_171', 'evolved_p171',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i XOR j) % 7.
Return shape (N, N). No collective needed.''',
          _p171, 32, ndim=2)


# P_172 muladd_bcast: y[i] = 3*i + 7
def _p172(N):
    idx = torch.arange(N)
    return (3 * idx + 7).to(torch.float32)
_register('muladd_bcast', 'Problem P_172', 'evolved_p172',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = 3*i + 7.
Return shape (N,). Linear affine, no collective needed.''',
          _p172, 128)


# P_173 saw_bcast: y[i] = i % 8
def _p173(N):
    idx = torch.arange(N)
    return (idx % 8).to(torch.float32)
_register('saw_bcast', 'Problem P_173', 'evolved_p173',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = i % 8.  Sawtooth pattern.
Return shape (N,).''',
          _p173, 128)


# P_174 range_shift_bcast: y[i] = (i + N//2) % N
def _p174(N):
    idx = torch.arange(N)
    return ((idx + N // 2) % N).to(torch.float32)
_register('range_shift_bcast', 'Problem P_174', 'evolved_p174',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=128.
Formula: y[i] = (i + N/2) % N.  Shifted range.
Return shape (N,).''',
          _p174, 128)


# P_175 min_ij_plus_bcast: y[i,j] = min(i, j) + 5
def _p175(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (torch.minimum(ii, jj) + 5).to(torch.float32).contiguous()
_register('min_ij_plus_bcast', 'Problem P_175', 'evolved_p175',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = min(i, j) + 5.
Return shape (N, N).''',
          _p175, 32, ndim=2)


# P_176 mul_ij_bcast: y[i,j] = i * j
def _p176(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return (ii * jj).to(torch.float32).contiguous()
_register('mul_ij_bcast', 'Problem P_176', 'evolved_p176',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = i * j.
Return shape (N, N).''',
          _p176, 32, ndim=2)


# P_177 add_mod_bcast: y[i,j] = (i + j) % 5
def _p177(N):
    idx = torch.arange(N)
    ii = idx.unsqueeze(1); jj = idx.unsqueeze(0)
    return ((ii + jj) % 5).to(torch.float32).contiguous()
_register('add_mod_bcast', 'Problem P_177', 'evolved_p177',
          '''Args: x (N, N) - rank 0 has reference, others zeros. N=32.
Formula: y[i, j] = (i + j) % 5.
Return shape (N, N).''',
          _p177, 32, ndim=2)


# P_178 abs_diff_sq_bcast: y[i] = |i - 5|^2
def _p178(N):
    idx = torch.arange(N)
    return ((idx - 5).abs() ** 2).to(torch.float32)
_register('abs_diff_sq_bcast', 'Problem P_178', 'evolved_p178',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = |i - 5|^2.
Return shape (N,).''',
          _p178, 64)


# P_179 tri_num_mod_bcast: y[i] = (i * (i+1) // 2) % 100
def _p179(N):
    idx = torch.arange(N)
    return ((idx * (idx + 1) // 2) % 100).to(torch.float32)
_register('tri_num_mod_bcast', 'Problem P_179', 'evolved_p179',
          '''Args: x (N,) - rank 0 has reference, others zeros. N=64.
Formula: y[i] = (i*(i+1)/2) % 100.  Triangular number mod 100.
Return shape (N,).''',
          _p179, 64)
