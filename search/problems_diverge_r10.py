"""Round 10 -- does the depth-divergence GENERALIZE beyond the AR-sum fold?

r2/r9 established Sorcar>Overlay on DEEP all_reduce-SUM chains (linearity fold).
Open question: is the lever specific to sum-linearity, or does overlay's
bounded-refinement get trapped by ANY deep dependent-collective framing? r10
builds deep chains over DIFFERENT primitives / reduce ops whose optimum is still
a single collective, testing generality:

DIR-P "deep MAX chain": chain of D dependent all_reduce(REDUCE_MAX). max is
idempotent (max(max(x))==max(x)), so D stages collapse to 1 AR-MAX. Different
algebra from sum-linearity; tests whether overlay collapses idempotent depth.

DIR-Q "deep AG->reduce chain": each stage all_gathers then locally sums (==AR),
repeated with renormalize. Optimum is 1 all_reduce. Tests depth over a
gather-based framing rather than a reduce-based one.

DIR-R "deep RS+AG roundtrip chain": each stage reduce_scatters then all_gathers
(a redundant roundtrip == identity on replicated data), repeated D times, with a
real reduce up front. Optimum: 1 AR. Tests depth over a scatter/gather framing.
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


# --- DIR-P: deep MAX chain (idempotent -> collapses to 1 AR-MAX) ---
def _mk_deep_max(name, depth, N=1024):
    def _ref(inputs, world_size):
        st = torch.stack([inp['x'] for inp in inputs], dim=0)
        mx = torch.max(st, dim=0)[0]
        return [mx.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    W = world_size",
            "    m = xm.all_reduce(xm.REDUCE_MAX, x)"]
    for _st in range(depth - 1):
        body += ["    m = xm.all_reduce(xm.REDUCE_MAX, m)   # idempotent"]
    body += ["    return m"]
    _reg(name, f"Local x ({N},). Return elementwise all-rank MAX. Baseline "
         f"applies {depth} dependent all_reduce(MAX) stages (max is idempotent).",
         _ref, _gen, "\n".join(body) + "\n")


# --- DIR-Q: deep all_gather+local-sum chain (renormalized) ---
def _mk_deep_agsum(name, depth, N=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (N,)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    # stage 0: all_gather then local sum == AR(SUM, x)",
            "    g = xm.all_gather(x, dim=0)           # (W*N,)",
            "    s = g.reshape(W, N).sum(dim=0)        # (N,)"]
    for _st in range(depth - 1):
        body += [
            "    g = xm.all_gather(s / W, dim=0)       # replicated/W gathered",
            "    s = g.reshape(W, N).sum(dim=0)        # == s (identity)"]
    body += ["    return s"]
    _reg(name, f"Local x ({N},). Return all-rank SUM. Baseline does {depth} "
         f"dependent all_gather+local-sum stages (each after the first is an "
         f"identity roundtrip).", _ref, _gen, "\n".join(body) + "\n")


# --- DIR-R: deep reduce_scatter+all_gather roundtrip chain ---
def _mk_deep_rsag(name, depth, part=256):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (W*part,)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, world_size * part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)   # real reduce"]
    for _st in range(depth - 1):
        body += [
            "    rs = xm.reduce_scatter(xm.REDUCE_SUM, s / W, scatter_dim=0, shard_count=W)",
            "    s = xm.all_gather(rs, dim=0)          # RS+AG roundtrip (identity)"]
    body += ["    return s"]
    _reg(name, f"Local x (world*{part},), S={part}. Return all-rank SUM. "
         f"Baseline: a real reduce then {depth-1} reduce_scatter+all_gather "
         f"roundtrips (each an identity on replicated data).",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_deep_max("r10_maxchain6", 6, 1024)
    _mk_deep_max("r10_maxchain8", 8, 1024)
    _mk_deep_agsum("r10_agsum6", 6, 512)
    _mk_deep_agsum("r10_agsum8", 8, 512)
    # NOTE: DIR-R (deep RS+AG roundtrip) dropped — the /W-through-RS+AG
    # reconstruction is not fp32-exact under the gate (max_diff ~2.5), so its
    # baseline fails the fair gate. maxchain/agsum cover the generality test.


register_all()
