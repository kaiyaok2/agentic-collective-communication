"""V5 novel problems: sequences that don't have simple arithmetic closed-forms,
requiring lookup or unusual composition."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================================
# P_93 — gray_code: G(i) = i XOR (i >> 1) — classical binary-reflected Gray code
# ============================================================================
def _p93_ref(inputs, world_size):
    N = 128
    idx = torch.arange(N)
    ref = (idx ^ (idx >> 1)).float()
    return [ref.clone() for _ in range(world_size)]


def _p93_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 128
    idx = torch.arange(N)
    ref = (idx ^ (idx >> 1)).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p93_ref(inputs, world_size)}


def _p93_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P93_SIG = """\
def evolved_p93(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P93_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int (=128).
Formula: x[i] = i XOR (i >> 1) — the i-th binary-reflected Gray code.
Returns (N,) tensor identical on every rank.
"""

_P93_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p93(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="gray_code_bcast",
    display_name="Problem P_93",
    evolved_fn_name="evolved_p93",
    signature=_P93_SIG,
    signature_doc=_P93_DOC,
    reference_fn=_p93_ref,
    generate_test_case=_p93_generate,
    call_candidate=_p93_call,
    builtin_templates=_P93_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_94 — max_of_min_and_max: min(i,j) * max(i,j) + (i-j)^2 — compound formula
# ============================================================================
def _p94_ref(inputs, world_size):
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    mnj = torch.min(ii, jj); mxj = torch.max(ii, jj)
    ref = (mnj * mxj + (ii - jj) ** 2).float()
    return [ref.clone() for _ in range(world_size)]


def _p94_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    mnj = torch.min(ii, jj); mxj = torch.max(ii, jj)
    ref = (mnj * mxj + (ii - jj) ** 2).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N, N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p94_ref(inputs, world_size)}


def _p94_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P94_SIG = """\
def evolved_p94(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P94_DOC = """\
Args: x (N, N) — rank 0 has correct values, others zeros. N: int.
Formula: x[i, j] = min(i, j) * max(i, j) + (i - j) ** 2.
Returns (N, N) tensor identical on every rank.
"""

_P94_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p94(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="compound_ij_bcast",
    display_name="Problem P_94",
    evolved_fn_name="evolved_p94",
    signature=_P94_SIG,
    signature_doc=_P94_DOC,
    reference_fn=_p94_ref,
    generate_test_case=_p94_generate,
    call_candidate=_p94_call,
    builtin_templates=_P94_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_95 — perm_shuffle: p[i] = (2*i) % N — bijective permutation
# ============================================================================
def _p95_ref(inputs, world_size):
    N = 128
    idx = torch.arange(N)
    ref = ((2 * idx) % N).float()
    return [ref.clone() for _ in range(world_size)]


def _p95_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 128
    idx = torch.arange(N)
    ref = ((2 * idx) % N).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p95_ref(inputs, world_size)}


def _p95_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P95_SIG = """\
def evolved_p95(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P95_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int (=128).
Formula: x[i] = (2 * i) % N.
Returns (N,) tensor identical on every rank.
"""

_P95_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p95(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="perm_shuffle_bcast",
    display_name="Problem P_95",
    evolved_fn_name="evolved_p95",
    signature=_P95_SIG,
    signature_doc=_P95_DOC,
    reference_fn=_p95_ref,
    generate_test_case=_p95_generate,
    call_candidate=_p95_call,
    builtin_templates=_P95_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))
