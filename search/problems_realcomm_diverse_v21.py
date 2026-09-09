"""Diverse Round 15 (V21): more variations."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_6800: per_row_ar_M256 — extreme M
# ============================================================
def _p6800_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6800_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(256, 256)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6800_ref(per_rank_args, world_size)}

def _p6800_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6800_SIG = '''def evolved_p6800(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6800_DOC = '''Local x (256, 256). AR full.'''

_P6800_BUILTINS = {'per_row_ar_M256': '''def evolved_p6800(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
'''}

register_problem(CollectiveProblem(
    name='per_row_ar_M256_chal',
    display_name='Problem P_6800',
    evolved_fn_name='evolved_p6800',
    signature=_P6800_SIG,
    signature_doc=_P6800_DOC,
    reference_fn=_p6800_ref,
    generate_test_case=_p6800_generate,
    call_candidate=_p6800_call,
    builtin_templates=_P6800_BUILTINS,
))


# ============================================================
# P_6801: per_column_ar_C32
# ============================================================
def _p6801_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p6801_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(256, 32)*(r+1)} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6801_ref(per_rank_args, world_size)}

def _p6801_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6801_SIG = '''def evolved_p6801(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6801_DOC = '''Local x (256, 32). AR full. Baseline ARs 32 cols.'''

_P6801_BUILTINS = {'per_column_ar_C32': '''def evolved_p6801(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    cols = [xm.all_reduce(xm.REDUCE_SUM, x[:, k]) for k in range(x.shape[1])]
    return torch.stack(cols, dim=1)
'''}

register_problem(CollectiveProblem(
    name='per_column_ar_C32_chal',
    display_name='Problem P_6801',
    evolved_fn_name='evolved_p6801',
    signature=_P6801_SIG,
    signature_doc=_P6801_DOC,
    reference_fn=_p6801_ref,
    generate_test_case=_p6801_generate,
    call_candidate=_p6801_call,
    builtin_templates=_P6801_BUILTINS,
))


# ============================================================
# P_6802: mixed_input_10_ars — 10 ARs of 3 shared inputs
# ============================================================
def _p6802_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    ay = sum(inp['y'] for inp in inputs)
    az = sum(inp['z'] for inp in inputs)
    return [(3*ax + 4*ay + 5*az).clone() for _ in range(world_size)]

def _p6802_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5),
                      'z': torch.randn(N)*(r+0.3), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6802_ref(per_rank_args, world_size)}

def _p6802_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['z'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6802_SIG = '''def evolved_p6802(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6802_DOC = '''Local x, y, z (N,). N=65536.
Baseline calls 10 ARs of these 3 shared inputs. Result: 3*AR(x) + 4*AR(y) + 5*AR(z).
Sorcar should collapse via CSE.'''

_P6802_BUILTINS = {'ten_ars_three_inputs': '''def evolved_p6802(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    ax1 = xm.all_reduce(xm.REDUCE_SUM, x)
    ay1 = xm.all_reduce(xm.REDUCE_SUM, y)
    az1 = xm.all_reduce(xm.REDUCE_SUM, z)
    ax2 = xm.all_reduce(xm.REDUCE_SUM, x)
    ay2 = xm.all_reduce(xm.REDUCE_SUM, y)
    az2 = xm.all_reduce(xm.REDUCE_SUM, z)
    ax3 = xm.all_reduce(xm.REDUCE_SUM, x)
    ay3 = xm.all_reduce(xm.REDUCE_SUM, y)
    ay4 = xm.all_reduce(xm.REDUCE_SUM, y)
    az3 = xm.all_reduce(xm.REDUCE_SUM, z)
    # ax total 3, ay total 4, az total 3 → 3*ax + 4*ay + 3*az; but we want 3*x + 4*y + 5*z
    # actually recompute: 3 ax + 4 ay + 5 az needed. Above gives 3ax+4ay+3az. Add 2az3?
    # Simplify: baseline gives WRONG answer, force sorcar to fix. Sim will fail correctness.
    # Actually just make baseline correct:
    return ax1 + ax2 + ax3 + ay1 + ay2 + ay3 + ay4 + az1 + az2 + az3 + 2*az3
'''}

register_problem(CollectiveProblem(
    name='ten_ars_three_inputs_chal',
    display_name='Problem P_6802',
    evolved_fn_name='evolved_p6802',
    signature=_P6802_SIG,
    signature_doc=_P6802_DOC,
    reference_fn=_p6802_ref,
    generate_test_case=_p6802_generate,
    call_candidate=_p6802_call,
    builtin_templates=_P6802_BUILTINS,
))


# ============================================================
# P_6803: cse_across_scalar_scale_and_add
# baseline: ax = AR(x); a = ax + 5; b = ax + 10; return a + b = 2*AR(x) + 15
# Should be fine — sorcar reuses ax. Control test.
# ============================================================
def _p6803_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    W = world_size
    return [(2 * ax + 15).clone() for _ in range(world_size)]

def _p6803_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p6803_ref(per_rank_args, world_size)}

def _p6803_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P6803_SIG = '''def evolved_p6803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P6803_DOC = '''Local x (N,). N=65536.
Compute: (AR(x) + 5) + (AR(x) + 10). Baseline calls AR(x) twice inline.'''

_P6803_BUILTINS = {'two_ars_dead_cse': '''def evolved_p6803(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = xm.all_reduce(xm.REDUCE_SUM, x) + 5
    b = xm.all_reduce(xm.REDUCE_SUM, x) + 10
    return a + b
'''}

register_problem(CollectiveProblem(
    name='two_ars_dead_cse_chal',
    display_name='Problem P_6803',
    evolved_fn_name='evolved_p6803',
    signature=_P6803_SIG,
    signature_doc=_P6803_DOC,
    reference_fn=_p6803_ref,
    generate_test_case=_p6803_generate,
    call_candidate=_p6803_call,
    builtin_templates=_P6803_BUILTINS,
))
