"""Round 18 -- probe the REVERSE lever surfaced by r14 zerosum.

r14 zerosum6/8 screened as a possible Overlay>Sorcar signal: overlay found the
1-AR fold (drop the zero-sum per-rank perturbation) while kiss stayed at ~2
collectives. Working hypothesis (H-REV): overlay's enumerate-K-strategies-from-
baseline NATURALLY lists "drop the additive per-rank term" as one candidate and
nails it; kiss's open ReAct may over-analyze the perturbation (try to preserve
its semantics) and not prove it cancels. I.e. the reverse lever = a fold that is
STRUCTURALLY OBVIOUS FROM THE BASELINE (a clean removable additive/multiplicative
term) rather than a deep distributive collapse.

If H-REV is right, these should reproduce/strengthen the reverse signal:
DIR-REV1 "single-stage obvious drop": ONE all_reduce with an added zero-sum
per-rank constant. Fold = drop the constant. Maximally obvious; no depth.
DIR-REV2 "scaled zero-sum": per-rank constant times a power-of-2, still sums to
0. Tests whether magnitude changes the reverse.
DIR-REV3 "additive then reduce, 3-deep": three stages each adding a zero-sum
term. If overlay's enumerate handles the multi-stage additive drop but kiss
doesn't, reverse strengthens with additive depth (opposite of the forward lever).
CONTROL DIR-REV4 "obvious multiplicative drop": AR with a per-rank multiply by
2^0=1 dressed as a[r] but all a[r]==1 -> literally identity scale. Both should
trivially drop it (tie) -- isolates whether it's the ZERO-SUM structure
specifically or any obvious removable term.

All zero-sum terms use d_r = float(2*r-(W-1)); sum_r d_r = 0 exactly (ints).
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


def _ref_sum(inputs, world_size):
    s = sum(inp['x'] for inp in inputs)
    return [s.clone() for _ in range(world_size)]


def _gen_factory(N):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref_sum(pra, world_size)}
    return _gen


def _mk(name, doc, N, body_lines):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size"] + body_lines + ["    return s"]
    _reg(name, doc, _ref_sum, _gen_factory(N), "\n".join(body) + "\n")


def _deep_zerosum(name, depth, N):
    """Depth-D zero-sum additive chain (matches r14 zerosum construction)."""
    body = ["    d = float(2 * rank - (W - 1))",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for _ in range(depth - 1):
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, (s / W) + d)"]
    _mk(name, f"Local x ({N},). Return all-rank SUM. Baseline runs {depth} "
        f"dependent all_reduce stages; each adds a per-rank perturbation before "
        f"reducing. Net = single all-rank SUM.", N, body)


def register_all():
    N = 512
    # Reproduce the r14 reverse signal at matched depths, plus a finer depth grid
    # to see whether the reverse GROWS or SHRINKS with additive depth.
    _deep_zerosum("r18_zs4", 4, N)
    _deep_zerosum("r18_zs6", 6, N)   # matches r14_zerosum6 (reverse screened 1.195)
    _deep_zerosum("r18_zs8", 8, N)   # matches r14_zerosum8 (reverse screened 1.273)
    # CONTROL: same depth-6 chain but MULTIPLICATIVE zero-net (scale/unscale) —
    # this is the r2 forward-lever construction. If zs6 reverses but this one
    # forwards, the additive-vs-multiplicative structure is what flips direction.
    body = ["    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    S = N // W if (N % W == 0) else N",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for _ in range(5):
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, s / W)"]
    _mk("r18_multctl6", f"Local x ({N},). Return all-rank SUM. Baseline: 6 "
        f"dependent identity all_reduce stages (divide by W each). Net = "
        f"single all-rank SUM.", N, body)


register_all()
