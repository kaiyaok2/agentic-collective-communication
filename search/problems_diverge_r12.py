"""Round 12 -- two further structural stressors on the confirmed depth lever.

Confirmed lever: overlay's enumerate-from-baseline + bounded R=3 refinement stays
trapped in DEEP dependent-collective framing (depth>=7 on best-of-8); kiss
collapses. r12 pushes on two orthogonal axes within the gate's depth<=8 budget:

DIR-S "mixed-primitive depth": alternate all_reduce and all_gather+local-sum
stages (both compute AR(SUM) but look like different primitives). Tests whether
overlay's trap depends on the chain being homogeneous, or persists when it must
recognize that two DIFFERENT-looking collectives are the same reduction.

DIR-T "multi-branch shared collapse": TWO independent deep sub-chains on
disjoint halves of x, each collapsing to its own AR, but the two ARs further
fuse into ONE all_reduce of the concatenation. Requires cross-branch reasoning
(recognize both branches reduce the same way -> stack + 1 AR). Overlay's
per-strategy refinement may collapse each branch to 1 AR (2 total) but miss the
cross-branch fusion to 1; kiss's open search may reach 1.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _reg(name, doc, ref_fn, gen_fn, builtin_code):
    sig = (f"def {name}_fn(x, rank, world_size, num_devices,\n"
           f"                 cores_per_device, xm, torch, num_nodes=1):")

    def _call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
        return fn(args["x"], r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

    register_problem(CollectiveProblem(
        name=name, display_name=name, evolved_fn_name=f"{name}_fn",
        signature=sig, signature_doc=doc, reference_fn=ref_fn,
        generate_test_case=gen_fn, call_candidate=_call,
        builtin_templates={name: builtin_code}))


def _gen_flat(world_size, N, seed):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# --- DIR-S: mixed AR / AG+sum depth chain (all == AR(SUM)) ---
def _mk_mixed_depth(name, depth, N=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        if st % 2 == 0:
            # all_gather + local sum of replicated/W == identity
            body += ["    g = xm.all_gather(s / W, dim=0)",
                     "    s = g.reshape(W, N).sum(dim=0)"]
        else:
            # plain identity reduce
            body += ["    s = xm.all_reduce(xm.REDUCE_SUM, s / W)"]
    body += ["    return s"]
    _reg(name, f"Local x ({N},). Return all-rank SUM. Baseline: {depth} "
         f"dependent stages ALTERNATING all_reduce and all_gather+local-sum "
         f"(both compute the same reduction). Fused optimum = 1 all_reduce.",
         _ref, _gen, "\n".join(body) + "\n")


# --- DIR-T: two independent deep sub-chains that further fuse ---
def _mk_multibranch(name, depth, half=256):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (2*half,)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, 2 * half, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    H = {half}; W = world_size",
            "    lo = x[:H]; hi = x[H:]",
            "    # branch A: deep identity chain on lo",
            "    a = xm.all_reduce(xm.REDUCE_SUM, lo)"]
    for _st in range(depth - 1):
        body += ["    a = xm.all_reduce(xm.REDUCE_SUM, a / W)"]
    body += ["    # branch B: deep identity chain on hi",
             "    b = xm.all_reduce(xm.REDUCE_SUM, hi)"]
    for _st in range(depth - 1):
        body += ["    b = xm.all_reduce(xm.REDUCE_SUM, b / W)"]
    body += ["    return torch.cat([a, b], dim=0)"]
    _reg(name, f"Local x (2*{half},). Return all-rank SUM. Baseline runs TWO "
         f"independent depth-{depth} reduce chains on the two halves. Optimum "
         f"fuses both into ONE all_reduce of the whole vector.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_mixed_depth("r12_mixed6", 6, 512)
    _mk_mixed_depth("r12_mixed8", 8, 512)
    _mk_multibranch("r12_branch4", 4, 256)
    _mk_multibranch("r12_branch4_big", 4, 1024)


register_all()
