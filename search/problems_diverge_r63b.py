"""Round 63b -- FAMILY-4 TOP-UP (global-scalar coupling), pushing family-4 toward 10 problems.

Same mechanism as r63 (out = s + beta*stat(s), stat in {mean,sum}, broadcast to every
element; s = AR(SUM,x)). fam-4 only telescopes for LINEAR stats -- a uniform additive shift
c = beta*stat(s) is closed-form invertible only when stat is linear in s (mean/sum); nonlinear
stats such as mean(|s|) are NOT (adding a constant flips signs, so no closed-form recovery).
This battery therefore stays on mean/sum and widens the (beta, payload) net around the two
confirmed winners (gmean_b1.0_p2048, gsum_b0.02_p2048), since fam-4's best-of-N divergence is
weak and non-monotonic in beta.

Telescoping / inverse identical to r63:
    mean:  s = acc - beta*mean(acc)/(1+beta)
    sum:   s = acc - beta*sum(acc)/(1+beta*N)
naive plain AR fails. All new (beta, payload) combos are md5-distinct from every registered
problem (prescreen guard). Depth is NOT distinctness.
"""
import torch  # noqa: F401
from .problems import CollectiveProblem, register_problem

NBLOCK = 8


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


def _gen(world_size, seed, part, nblock=NBLOCK):
    torch.manual_seed(seed)
    N = nblock * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _stat_fwd(stat):
    return "s.mean()" if stat == "mean" else "s.sum()"


def _gc_code(name, part, depth, stat, beta):
    N = NBLOCK * part
    fwd = _stat_fwd(stat)
    if stat == "mean":
        inv = f"        s = acc - {beta} * acc.mean() / (1.0 + {beta})"
    else:
        inv = f"        s = acc - {beta} * acc.sum() / (1.0 + {beta} * {N})"
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; BETA = {beta}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += [f"    buf = s + BETA * ({fwd})",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
              "    acc = acc / W"]
        if not last:
            L += ["    if True:", inv]
        else:
            L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _gc_ref(part, stat, beta):
    N = NBLOCK * part

    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        g = (s.mean() if stat == "mean" else s.sum())
        out = s + beta * g
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, stat, beta, cue=True):
    ref = _gc_ref(part, stat, beta)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    statword = "arithmetic mean" if stat == "mean" else "sum"
    doc = (f"Local x ({NBLOCK}*{part},). {COUNT}"
           f"Final result = the elementwise SUM of x across ranks (call it s), plus "
           f"{beta} times the {statword} of s broadcast to every element (i.e. out = s + "
           f"{beta} * {statword}(s)). Every output element depends on a global reduction of s.")
    _reg(name, doc, ref, gen, _gc_code(name, part, depth, stat, beta))


def register_all():
    # widen the net near the two confirmed winners; all p2048 (winning payload) plus a
    # couple of alt payloads. betas chosen distinct from r63's {0.5,1.0,2.0 / 0.01,0.02}.
    _mk("r63b_gmean_b0p7_p2048_d8",  2048, 8, "mean", 0.7)
    _mk("r63b_gmean_b0p8_p2048_d8",  2048, 8, "mean", 0.8)
    _mk("r63b_gmean_b0p9_p2048_d8",  2048, 8, "mean", 0.9)
    _mk("r63b_gmean_b1p1_p2048_d8",  2048, 8, "mean", 1.1)
    _mk("r63b_gmean_b1p25_p2048_d8", 2048, 8, "mean", 1.25)
    _mk("r63b_gmean_b1p5_p2048_d8",  2048, 8, "mean", 1.5)
    _mk("r63b_gsum_b0p015_p2048_d8", 2048, 8, "sum", 0.015)
    _mk("r63b_gsum_b0p025_p2048_d8", 2048, 8, "sum", 0.025)
    _mk("r63b_gsum_b0p03_p2048_d8",  2048, 8, "sum", 0.03)
    _mk("r63b_gsum_b0p04_p2048_d8",  2048, 8, "sum", 0.04)


register_all()
