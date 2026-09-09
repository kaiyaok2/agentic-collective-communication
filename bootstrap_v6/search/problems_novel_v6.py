"""V6 novel problems: deeper compositions, conditional formulas, non-standard
shapes. Designed to test whether kiss's freeform LLM code generation can
handle problems that strat's collective-first enumeration misses.

Focus: formulas that mix bitwise ops with arithmetic, conditional 2D masks,
higher-order polynomial patterns.
"""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================================
# P_96 — hamming_dist: h[i,j] = popcount(i XOR j)
# ============================================================================
def _p96_ref(inputs, world_size):
    N = 16
    ref = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            ref[i, j] = bin(i ^ j).count("1")
    return [ref.clone() for _ in range(world_size)]


def _p96_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 16
    ref = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            ref[i, j] = bin(i ^ j).count("1")
    inputs = [ref.clone() if r == 0 else torch.zeros(N, N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p96_ref(inputs, world_size)}


def _p96_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P96_SIG = """\
def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P96_DOC = """\
Args: x (N, N) — rank 0 has correct values, others zeros. N: int (=16).
Formula: x[i, j] = popcount(i XOR j) — Hamming distance between i and j interpreted as N-bit integers.
Returns (N, N) tensor identical on every rank.
"""

_P96_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p96(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="hamming_dist_bcast",
    display_name="Problem P_96",
    evolved_fn_name="evolved_p96",
    signature=_P96_SIG,
    signature_doc=_P96_DOC,
    reference_fn=_p96_ref,
    generate_test_case=_p96_generate,
    call_candidate=_p96_call,
    builtin_templates=_P96_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_97 — cond_mask: c[i,j] = 1 if (i^2 + j^2 <= N*N/4) else 0 — quadratic disk
# ============================================================================
def _p97_ref(inputs, world_size):
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = ((ii * ii + jj * jj) <= (N * N // 4)).float()
    return [ref.clone() for _ in range(world_size)]


def _p97_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = ((ii * ii + jj * jj) <= (N * N // 4)).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N, N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p97_ref(inputs, world_size)}


def _p97_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P97_SIG = """\
def evolved_p97(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P97_DOC = """\
Args: x (N, N) — rank 0 has correct values, others zeros. N: int (=32).
Formula: x[i, j] = 1 if (i*i + j*j) <= (N*N // 4) else 0 — quadratic disk mask.
Returns (N, N) tensor identical on every rank.
"""

_P97_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p97(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="quad_disk_bcast",
    display_name="Problem P_97",
    evolved_fn_name="evolved_p97",
    signature=_P97_SIG,
    signature_doc=_P97_DOC,
    reference_fn=_p97_ref,
    generate_test_case=_p97_generate,
    call_candidate=_p97_call,
    builtin_templates=_P97_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_98 — nested_mod: n[i] = ((i * 3 + 1) % (i % 7 + 2)) — nested modular
# ============================================================================
def _p98_ref(inputs, world_size):
    N = 64
    idx = torch.arange(N)
    ref = ((idx * 3 + 1) % (idx % 7 + 2)).float()
    return [ref.clone() for _ in range(world_size)]


def _p98_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 64
    idx = torch.arange(N)
    ref = ((idx * 3 + 1) % (idx % 7 + 2)).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p98_ref(inputs, world_size)}


def _p98_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P98_SIG = """\
def evolved_p98(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P98_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int (=64).
Formula: x[i] = (i * 3 + 1) % (i % 7 + 2) — nested modular expression.
Returns (N,) tensor identical on every rank.
"""

_P98_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p98(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="nested_mod_bcast",
    display_name="Problem P_98",
    evolved_fn_name="evolved_p98",
    signature=_P98_SIG,
    signature_doc=_P98_DOC,
    reference_fn=_p98_ref,
    generate_test_case=_p98_generate,
    call_candidate=_p98_call,
    builtin_templates=_P98_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_99 — piecewise: p[i] = i^2 if i < N/2 else (N - i)^2 — piecewise quadratic
# ============================================================================
def _p99_ref(inputs, world_size):
    N = 64
    idx = torch.arange(N)
    half = N // 2
    ref = torch.where(idx < half, idx * idx, (N - idx) * (N - idx)).float()
    return [ref.clone() for _ in range(world_size)]


def _p99_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 64
    idx = torch.arange(N)
    half = N // 2
    ref = torch.where(idx < half, idx * idx, (N - idx) * (N - idx)).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p99_ref(inputs, world_size)}


def _p99_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P99_SIG = """\
def evolved_p99(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P99_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int (=64).
Formula: x[i] = i*i if i < N/2 else (N-i)*(N-i) — piecewise quadratic reflecting at midpoint.
Returns (N,) tensor identical on every rank.
"""

_P99_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p99(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="piecewise_bcast",
    display_name="Problem P_99",
    evolved_fn_name="evolved_p99",
    signature=_P99_SIG,
    signature_doc=_P99_DOC,
    reference_fn=_p99_ref,
    generate_test_case=_p99_generate,
    call_candidate=_p99_call,
    builtin_templates=_P99_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_100 — sum_of_bits: s[i,j] = popcount(i) + popcount(j) — bit-count over grid
# ============================================================================
def _p100_ref(inputs, world_size):
    N = 32
    pc = torch.tensor([bin(int(i)).count("1") for i in range(N)]).float()
    ref = (pc.unsqueeze(1) + pc.unsqueeze(0))
    return [ref.clone() for _ in range(world_size)]


def _p100_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 32
    pc = torch.tensor([bin(int(i)).count("1") for i in range(N)]).float()
    ref = (pc.unsqueeze(1) + pc.unsqueeze(0))
    inputs = [ref.clone() if r == 0 else torch.zeros(N, N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p100_ref(inputs, world_size)}


def _p100_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P100_SIG = """\
def evolved_p100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P100_DOC = """\
Args: x (N, N) — rank 0 has correct values, others zeros. N: int (=32).
Formula: x[i, j] = popcount(i) + popcount(j) — sum of 1-bits of the row and column indices.
Returns (N, N) tensor identical on every rank.
"""

_P100_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="sum_popcount_bcast",
    display_name="Problem P_100",
    evolved_fn_name="evolved_p100",
    signature=_P100_SIG,
    signature_doc=_P100_DOC,
    reference_fn=_p100_ref,
    generate_test_case=_p100_generate,
    call_candidate=_p100_call,
    builtin_templates=_P100_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_101 — cross_xor: c[i, j] = (i XOR j) if (i + j) % 2 == 0 else 0
# ============================================================================
def _p101_ref(inputs, world_size):
    N = 16
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = torch.where((ii + jj) % 2 == 0, torch.bitwise_xor(ii, jj), torch.zeros_like(ii)).float()
    return [ref.clone() for _ in range(world_size)]


def _p101_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 16
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = torch.where((ii + jj) % 2 == 0, torch.bitwise_xor(ii, jj), torch.zeros_like(ii)).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N, N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p101_ref(inputs, world_size)}


def _p101_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P101_SIG = """\
def evolved_p101(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P101_DOC = """\
Args: x (N, N) — rank 0 has correct values, others zeros. N: int (=16).
Formula: x[i, j] = (i XOR j) if (i + j) % 2 == 0 else 0 — conditional XOR gated on parity.
Returns (N, N) tensor identical on every rank.
"""

_P101_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p101(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="cond_xor_bcast",
    display_name="Problem P_101",
    evolved_fn_name="evolved_p101",
    signature=_P101_SIG,
    signature_doc=_P101_DOC,
    reference_fn=_p101_ref,
    generate_test_case=_p101_generate,
    call_candidate=_p101_call,
    builtin_templates=_P101_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))
