"""Diverse Round 15 (V5): even more distinct classes."""
import torch
from .problems import CollectiveProblem, register_problem


# ============================================================
# P_5200: type_cast_around_ar — AR on f32 tensor derived from f16 input
# Baseline: cast input to f32, AR, cast back to f16 for output
# Sorcar: keep everything in f16 (2× less bandwidth)
# BUT: might lose precision. Sorcar should prefer f16 anyway for perf.
# ============================================================
def _p5200_ref(inputs, world_size):
    ax = sum(inp['x'].float() for inp in inputs).to(torch.float16)
    return [ax.clone() for _ in range(world_size)]

def _p5200_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 32768
    torch.manual_seed(seed)
    per_rank_args = [{'x': (torch.randn(N)*(r+1)).to(torch.float16), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5200_ref(per_rank_args, world_size)}

def _p5200_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5200_SIG = '''def evolved_p5200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5200_DOC = '''Local x (N,) fp16. N=32768.
Compute: y = AR(x) as fp16. Return (N,) fp16 identical on every rank.'''

_P5200_BUILTINS = {'cast_up_ar_down': '''def evolved_p5200(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    x_f32 = x.float()
    ax = xm.all_reduce(xm.REDUCE_SUM, x_f32)
    return ax.to(torch.float16)
'''}

register_problem(CollectiveProblem(
    name='fp16_upcast_ar_chal',
    display_name='Problem P_5200',
    evolved_fn_name='evolved_p5200',
    signature=_P5200_SIG,
    signature_doc=_P5200_DOC,
    reference_fn=_p5200_ref,
    generate_test_case=_p5200_generate,
    call_candidate=_p5200_call,
    builtin_templates=_P5200_BUILTINS,
))


# ============================================================
# P_5201: sequential_max_min — AR(MAX) then AR(MIN) — independent
# Baseline: both calls, 2 dispatches
# Sorcar: no way to fuse different ops. This is a control test.
# ============================================================
def _p5201_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    ys = [inp['y'] for inp in inputs]
    mx = xs[0]
    for x in xs[1:]: mx = torch.maximum(mx, x)
    mn = ys[0]
    for y in ys[1:]: mn = torch.minimum(mn, y)
    return [(mx + mn).clone() for _ in range(world_size)]

def _p5201_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'y': torch.randn(N)*(r+0.5), 'N': N}
                     for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5201_ref(per_rank_args, world_size)}

def _p5201_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['y'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5201_SIG = '''def evolved_p5201(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5201_DOC = '''Local x, y (N,). N=65536.
Compute: (max_r x_r) + (min_r y_r). Return (N,) identical on every rank.
Baseline uses redundant intermediate ARs. Sorcar should use just 1 MAX + 1 MIN.'''

_P5201_BUILTINS = {'max_min_with_extra': '''def evolved_p5201(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    mx = xm.all_reduce(xm.REDUCE_MAX, x)
    verify_mx = xm.all_reduce(xm.REDUCE_MAX, mx)     # dead
    mn = xm.all_reduce(xm.REDUCE_MIN, y)
    verify_mn = xm.all_reduce(xm.REDUCE_MIN, mn)     # dead
    return verify_mx + verify_mn
'''}

register_problem(CollectiveProblem(
    name='max_min_with_dead_chal',
    display_name='Problem P_5201',
    evolved_fn_name='evolved_p5201',
    signature=_P5201_SIG,
    signature_doc=_P5201_DOC,
    reference_fn=_p5201_ref,
    generate_test_case=_p5201_generate,
    call_candidate=_p5201_call,
    builtin_templates=_P5201_BUILTINS,
))


# ============================================================
# P_5202: ar_of_view — x.view(...) has same underlying storage
# Baseline: reshape x, AR, reshape back
# Sorcar: reshape is free, but the AR bytes don't change
# Weak test — mostly a compiler-fusion test.
# ============================================================
def _p5202_ref(inputs, world_size):
    ax = sum(inp['x'] for inp in inputs)
    return [ax.clone() for _ in range(world_size)]

def _p5202_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5202_ref(per_rank_args, world_size)}

def _p5202_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5202_SIG = '''def evolved_p5202(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5202_DOC = '''Local x (N,). N=65536.
Compute: AR(x). Return (N,) identical on every rank.
Baseline needlessly wraps AR with view/reshape ops.'''

_P5202_BUILTINS = {'ar_reshape_wrap': '''def evolved_p5202(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # baseline reshapes to (256, 256), AR, then reshapes back
    x2d = x.view(256, 256)
    ax2d = xm.all_reduce(xm.REDUCE_SUM, x2d)
    return ax2d.view(N)
'''}

register_problem(CollectiveProblem(
    name='ar_of_view_wrap_chal',
    display_name='Problem P_5202',
    evolved_fn_name='evolved_p5202',
    signature=_P5202_SIG,
    signature_doc=_P5202_DOC,
    reference_fn=_p5202_ref,
    generate_test_case=_p5202_generate,
    call_candidate=_p5202_call,
    builtin_templates=_P5202_BUILTINS,
))


# ============================================================
# P_5203: mixed_reduce_op — 2 different reduce ops on same tensor
# Baseline: AR(SUM) + AR(MAX) — 2 separate collectives
# Sorcar can't fuse different reduce ops, but might try stack trick.
# Genuinely different class — reduce-op family.
# ============================================================
def _p5203_ref(inputs, world_size):
    xs = [inp['x'] for inp in inputs]
    ax = sum(xs)
    mx = xs[0]
    for x in xs[1:]: mx = torch.maximum(mx, x)
    return [(ax + mx).clone() for _ in range(world_size)]

def _p5203_generate(world_size, pattern='uniform', shard_size=None, seed=0):
    N = 65536
    torch.manual_seed(seed)
    per_rank_args = [{'x': torch.randn(N)*(r+1), 'N': N} for r in range(world_size)]
    return {'per_rank_args': per_rank_args, 'shared_args': {},
            'expected': _p5203_ref(per_rank_args, world_size)}

def _p5203_call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
    return fn(args['x'], args['N'], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

_P5203_SIG = '''def evolved_p5203(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):'''
_P5203_DOC = '''Local x (N,). N=65536.
Compute: y = AR(SUM, x) + AR(MAX, x). Baseline adds a dead REDUCE_SUM verification
step. Sorcar should recognize and drop redundant AR(SUM).
Return (N,) identical on every rank.'''

_P5203_BUILTINS = {'sum_max_with_dead_sum': '''def evolved_p5203(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    verify_s = xm.all_reduce(xm.REDUCE_SUM, s) / world_size   # dead
    m = xm.all_reduce(xm.REDUCE_MAX, x)
    return verify_s + m
'''}

register_problem(CollectiveProblem(
    name='mixed_reduce_dead_sum_chal',
    display_name='Problem P_5203',
    evolved_fn_name='evolved_p5203',
    signature=_P5203_SIG,
    signature_doc=_P5203_DOC,
    reference_fn=_p5203_ref,
    generate_test_case=_p5203_generate,
    call_candidate=_p5203_call,
    builtin_templates=_P5203_BUILTINS,
))
