"""Novel problems designed to test kiss > strat under strict no-leak/HW gate.

Design principle: formula is a COMPOSITION requiring 2+ layered arithmetic ops
that don't match a single template. Strat's "enumerate strategies" phase tends
to propose collective-only strategies (AR, AG+RS, permute) rather than novel
closed-forms. Kiss's ReAct can iteratively arrive at the closed-form by
paraphrasing and testing.

All problems: value-based signal used only to define the reference tensor's
value at position (i, j). No inputs affect output.
"""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================================
# P_87 — mod_arg_squared: (i^2) % K per position — quadratic composition
# ============================================================================
def _p87_ref(inputs, world_size):
    N, K = 32, 7
    ref = ((torch.arange(N) * torch.arange(N)) % K).float()
    return [ref.clone() for _ in range(world_size)]


def _p87_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N, K = 32, 7
    ref = ((torch.arange(N) * torch.arange(N)) % K).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N, "K": K} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {},
            "expected": _p87_ref(inputs, world_size)}


def _p87_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank_args["K"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P87_SIG = """\
def evolved_p87(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P87_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int. K: int (=7).
Formula: x[i] = (i * i) % K.
Returns (N,) tensor identical on every rank.
"""

_P87_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p87(x, N, K, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="mod_sq_bcast",
    display_name="Problem P_87",
    evolved_fn_name="evolved_p87",
    signature=_P87_SIG,
    signature_doc=_P87_DOC,
    reference_fn=_p87_ref,
    generate_test_case=_p87_generate,
    call_candidate=_p87_call,
    builtin_templates=_P87_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_88 — xor_grid: (i XOR j) grid — bitwise op on indices
# ============================================================================
def _p88_ref(inputs, world_size):
    N = 32
    ii = torch.arange(N).unsqueeze(1)
    jj = torch.arange(N).unsqueeze(0)
    ref = torch.bitwise_xor(ii, jj).float()
    return [ref.clone() for _ in range(world_size)]


def _p88_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = torch.bitwise_xor(ii, jj).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N, N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p88_ref(inputs, world_size)}


def _p88_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P88_SIG = """\
def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P88_DOC = """\
Args: x (N, N) — rank 0 has correct values, others zeros. N: int.
Formula: x[i, j] = i XOR j (bitwise XOR of the row and column indices).
Returns (N, N) tensor identical on every rank.
"""

_P88_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="xor_grid_bcast",
    display_name="Problem P_88",
    evolved_fn_name="evolved_p88",
    signature=_P88_SIG,
    signature_doc=_P88_DOC,
    reference_fn=_p88_ref,
    generate_test_case=_p88_generate,
    call_candidate=_p88_call,
    builtin_templates=_P88_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_89 — bit_count: popcount(i) per position — bitwise composition
# ============================================================================
def _p89_ref(inputs, world_size):
    N = 128
    idx = torch.arange(N)
    # Count number of 1-bits in each i
    ref = torch.tensor([bin(int(i)).count("1") for i in idx]).float()
    return [ref.clone() for _ in range(world_size)]


def _p89_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 128
    idx = torch.arange(N)
    ref = torch.tensor([bin(int(i)).count("1") for i in idx]).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p89_ref(inputs, world_size)}


def _p89_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P89_SIG = """\
def evolved_p89(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P89_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int (=128).
Formula: x[i] = popcount(i) = number of 1-bits in binary representation of i.
Returns (N,) tensor identical on every rank.
"""

_P89_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p89(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="popcount_bcast",
    display_name="Problem P_89",
    evolved_fn_name="evolved_p89",
    signature=_P89_SIG,
    signature_doc=_P89_DOC,
    reference_fn=_p89_ref,
    generate_test_case=_p89_generate,
    call_candidate=_p89_call,
    builtin_templates=_P89_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_90 — triangle_num: T(i) = i*(i+1)/2 per position
# ============================================================================
def _p90_ref(inputs, world_size):
    N = 64
    idx = torch.arange(N)
    ref = (idx * (idx + 1) // 2).float()
    return [ref.clone() for _ in range(world_size)]


def _p90_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 64
    idx = torch.arange(N)
    ref = (idx * (idx + 1) // 2).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p90_ref(inputs, world_size)}


def _p90_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P90_SIG = """\
def evolved_p90(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P90_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int (=64).
Formula: x[i] = i * (i + 1) / 2 (triangle numbers, integer arithmetic).
Returns (N,) tensor identical on every rank.
"""

_P90_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p90(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="triangle_num_bcast",
    display_name="Problem P_90",
    evolved_fn_name="evolved_p90",
    signature=_P90_SIG,
    signature_doc=_P90_DOC,
    reference_fn=_p90_ref,
    generate_test_case=_p90_generate,
    call_candidate=_p90_call,
    builtin_templates=_P90_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_91 — sign_alternate_2d: (-1)^(i+j) per position
# ============================================================================
def _p91_ref(inputs, world_size):
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = ((-1.0) ** (ii + jj).float()).float()
    return [ref.clone() for _ in range(world_size)]


def _p91_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 32
    ii = torch.arange(N).unsqueeze(1); jj = torch.arange(N).unsqueeze(0)
    ref = ((-1.0) ** (ii + jj).float()).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N, N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p91_ref(inputs, world_size)}


def _p91_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P91_SIG = """\
def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P91_DOC = """\
Args: x (N, N) — rank 0 has correct values, others zeros. N: int.
Formula: x[i, j] = (-1)^(i + j) — +1 on even-parity positions, -1 on odd-parity positions.
Returns (N, N) tensor identical on every rank.
"""

_P91_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="sign_alt_bcast",
    display_name="Problem P_91",
    evolved_fn_name="evolved_p91",
    signature=_P91_SIG,
    signature_doc=_P91_DOC,
    reference_fn=_p91_ref,
    generate_test_case=_p91_generate,
    call_candidate=_p91_call,
    builtin_templates=_P91_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))


# ============================================================================
# P_92 — bimodal_dist: (i - N//2)^2 (quadratic distance to center)
# ============================================================================
def _p92_ref(inputs, world_size):
    N = 128
    idx = torch.arange(N)
    ref = ((idx - N // 2) ** 2).float()
    return [ref.clone() for _ in range(world_size)]


def _p92_generate(world_size, pattern="uniform", shard_size=None, seed=0):
    N = 128
    idx = torch.arange(N)
    ref = ((idx - N // 2) ** 2).float()
    inputs = [ref.clone() if r == 0 else torch.zeros(N) for r in range(world_size)]
    per_rank_args = [{"x": inputs[r], "N": N} for r in range(world_size)]
    return {"per_rank_args": per_rank_args, "shared_args": {}, "expected": _p92_ref(inputs, world_size)}


def _p92_call(candidate_fn, rank_args, shared_args, rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=1):
    return candidate_fn(rank_args["x"], rank_args["N"], rank, ws, nd, cpd, xm_mock, torch_mock, num_nodes=num_nodes)


_P92_SIG = """\
def evolved_p92(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):"""
_P92_DOC = """\
Args: x (N,) — rank 0 has correct values, others zeros. N: int (=128).
Formula: x[i] = (i - N // 2) ** 2 (squared distance to the center index).
Returns (N,) tensor identical on every rank.
"""

_P92_BUILTINS = {
    "baseline_ar_bcast": '''\
def evolved_p92(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    return xm.all_reduce(xm.REDUCE_SUM, x)
''',
}

register_problem(CollectiveProblem(
    name="bimodal_dist_bcast",
    display_name="Problem P_92",
    evolved_fn_name="evolved_p92",
    signature=_P92_SIG,
    signature_doc=_P92_DOC,
    reference_fn=_p92_ref,
    generate_test_case=_p92_generate,
    call_candidate=_p92_call,
    builtin_templates=_P92_BUILTINS,
    optimization_hints="",
    public_api_code='',
    training_validation_code='',
))
