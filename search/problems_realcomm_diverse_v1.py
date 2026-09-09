"""Diverse Round 15 (V1): 4 problems from DIFFERENT algorithmic classes.

Design goal: after 18 sequential-AR-linearity wins, explore other
compiler-visible patterns Sorcar may find but strat's enumeration misses.
Each problem tests a distinct optimization axis.
"""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_4800: ar_output_reused — CSE across ARs
# Baseline calls AR(x) twice; correct answer needs only one.
# Sorcar can recognize the redundancy.
# ============================================================
def _p4800_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax * 2 + ax * 3).clone() for _ in range(world_size)]  # = 5 * ax

def _p4800_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4800_ref(per_rank_args, world_size)}

def _p4800_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4800_SIG = '''def evolved_p4800(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4800_DOC = '''Local x (N,). N=65536.
Compute: a = 2 * AR(x); b = 3 * AR(x); return a + b.
Both AR calls have the same input (x). Return (N,) tensor identical on every rank.'''

_P4800_BUILTINS = {'redundant_ar': '''def evolved_p4800(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = 2 * xm.all_reduce(xm.REDUCE_SUM, x)
    b = 3 * xm.all_reduce(xm.REDUCE_SUM, x)
    return a + b
'''}

register_problem(CollectiveProblem(
    name='ar_output_reused_chal',
    display_name='Problem P_4800',
    evolved_fn_name='evolved_p4800',
    signature=_P4800_SIG,
    signature_doc=_P4800_DOC,
    reference_fn=_p4800_ref,
    generate_test_case=_p4800_generate,
    call_candidate=_p4800_call,
    builtin_templates=_P4800_BUILTINS,
))


# ============================================================
# P_4801: ar_scaled_by_worldsize — REDUCE_MEAN vs REDUCE_SUM/W
# Baseline: AR(sum) then divide by W (a common pattern).
# Sorcar: use REDUCE_SUM once, apply /W locally (mathematically identical
# but may enable more compiler fusion if /W is done post-AR).
# Real edge: does the compiler treat these differently?
# ============================================================
def _p4801_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(ax / world_size).clone() for _ in range(world_size)]  # mean

def _p4801_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4801_ref(per_rank_args, world_size)}

def _p4801_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4801_SIG = '''def evolved_p4801(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4801_DOC = '''Local x (N,). N=65536.
Compute mean across ranks: baseline computes it via AR(SUM) + intermediate scale + AR(SUM) to "verify".
Correct answer: mean of x across all ranks = sum_r(x_r) / W.
Return (N,).'''

_P4801_BUILTINS = {'ar_double_verify': '''def evolved_p4801(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    # Pretend to compute mean via a redundant second reduction "for verification"
    scaled = ax / world_size
    verify = xm.all_reduce(xm.REDUCE_SUM, scaled) / world_size
    return verify
'''}

register_problem(CollectiveProblem(
    name='ar_scaled_by_worldsize_chal',
    display_name='Problem P_4801',
    evolved_fn_name='evolved_p4801',
    signature=_P4801_SIG,
    signature_doc=_P4801_DOC,
    reference_fn=_p4801_ref,
    generate_test_case=_p4801_generate,
    call_candidate=_p4801_call,
    builtin_templates=_P4801_BUILTINS,
))


# ============================================================
# P_4802: ag_slice_use — allgather then use only OWN slice
# Baseline: all_gather (bytes: W*N), then take own slice (rank * N)
# Sorcar: recognize no communication needed — you already have your slice.
# Real edge: does sorcar recognize this dead-communication pattern?
# ============================================================
def _p4802_ref(inputs, world_size):
    # Result: each rank sees own x scaled by 2 (deterministic ref)
    return [inp['x'] * 2 for inp in inputs]

def _p4802_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4802_ref(per_rank_args, world_size)}

def _p4802_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4802_SIG = '''def evolved_p4802(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4802_DOC = '''Local x (N,). N=65536.
Compute: y = 2 * x_r locally per rank (elementwise scale of own tensor).
NOTE: baseline uses all_gather + narrow(rank) which is wasteful — no data
from other ranks is actually used. Sorcar should recognize this.
Return (N,).'''

_P4802_BUILTINS = {'ag_then_own_slice': '''def evolved_p4802(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    scaled = 2 * x
    gathered = xm.all_gather(scaled, dim=0)  # shape (W*N,)
    own = gathered.narrow(0, rank * N, N)    # extract own slice — no cross-rank data used
    return own
'''}

register_problem(CollectiveProblem(
    name='ag_slice_use_chal',
    display_name='Problem P_4802',
    evolved_fn_name='evolved_p4802',
    signature=_P4802_SIG,
    signature_doc=_P4802_DOC,
    reference_fn=_p4802_ref,
    generate_test_case=_p4802_generate,
    call_candidate=_p4802_call,
    builtin_templates=_P4802_BUILTINS,
))


# ============================================================
# P_4803: ar_zero_input — AR of a zero tensor is zero
# Baseline: computes 2 ARs, one of which has all-zero input.
# Sorcar: recognize AR(0) = 0 without communication.
# Real edge: algebraic identity elimination.
# ============================================================
def _p4803_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [(3 * ax).clone() for _ in range(world_size)]  # zeros cancel out

def _p4803_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p4803_ref(per_rank_args, world_size)}

def _p4803_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P4803_SIG = '''def evolved_p4803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P4803_DOC = '''Local x (N,). N=65536.
Compute: return 3 * AR(x). Some baseline templates may express this as
 — the AR(zeros) is a dead call.
Return (N,) identical on every rank.'''

_P4803_BUILTINS = {'ar_with_dead_zero': '''def evolved_p4803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    zero = torch.zeros_like(x)
    az = xm.all_reduce(xm.REDUCE_SUM, zero)  # dead call: always zero
    return 3 * ax + az
'''}

register_problem(CollectiveProblem(
    name='ar_zero_input_chal',
    display_name='Problem P_4803',
    evolved_fn_name='evolved_p4803',
    signature=_P4803_SIG,
    signature_doc=_P4803_DOC,
    reference_fn=_p4803_ref,
    generate_test_case=_p4803_generate,
    call_candidate=_p4803_call,
    builtin_templates=_P4803_BUILTINS,
))
