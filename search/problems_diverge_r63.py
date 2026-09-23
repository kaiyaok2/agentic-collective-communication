"""Round 63 -- FAMILY-4 direction B (NEW MECHANISM): GLOBAL-SCALAR coupling.

Families 1-3 are all DIAGONAL (output block b depends only on reduced block b). Direction A
(r62) is ROTATIONAL (block b depends on block b+k). Direction B is the structural OPPOSITE
of diagonal -- DENSE global coupling:

    out = s + beta * stat(s)          (stat in {mean, sum}, broadcast to every element)

where s = AR(SUM, x). Every output element depends on a GLOBAL reduction (mean/sum) of the
whole summed vector, so no per-block-independent strategy reproduces it -- overlay must
recognize a second reduction (over the already-reduced tensor) plus a broadcast add, on top
of the all_reduce.

Telescoping (exact, 1-AR-collapsible): each stage forms buf = s + beta*stat(s) and does a
redundant AR (buf is identical across ranks, so AR(SUM,buf)/W == buf); every non-final stage
inverts locally --
    mean:  mean(acc) = (1+beta)*mean(s)      => s = acc - beta*mean(acc)/(1+beta)
    sum:   sum(acc)  = (1+beta*N)*sum(s)      => s = acc - beta*sum(acc)/(1+beta*N)
so the depth-D chain nets to a SINGLE global-coupling application. naive plain AR(SUM,x)
fails the gate (beta*stat(s) != 0).

Distinctness: reference depends on (stat, beta, payload). gmean(beta) == gsum(beta/N) as an
output, but the registered betas are chosen so no gmean/gsum pair coincides; the prescreen
md5 cross-check (vs ALL registered problems) is the final guard. Depth is NOT distinctness.
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
    """Deep chain: add beta*stat(s) each stage, redundant AR, invert on all but last stage.
    Nets to out = s + beta*stat(s)."""
    N = NBLOCK * part
    fwd = _stat_fwd(stat)
    # local inverse of acc = s + beta*stat(s):
    if stat == "mean":
        inv = f"        s = acc - {beta} * acc.mean() / (1.0 + {beta})"
    else:  # sum
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
    _mk("r63_gmean_b0p5_p2048_d8", 2048, 8, "mean", 0.5)
    _mk("r63_gmean_b1p0_p2048_d8", 2048, 8, "mean", 1.0)
    _mk("r63_gmean_b2p0_p2048_d8", 2048, 8, "mean", 2.0)
    _mk("r63_gmean_b1p0_p1024_d8", 1024, 8, "mean", 1.0)
    _mk("r63_gmean_b1p0_p512_d8",  512,  8, "mean", 1.0)
    _mk("r63_gsum_b0p01_p2048_d8", 2048, 8, "sum", 0.01)
    _mk("r63_gsum_b0p02_p2048_d8", 2048, 8, "sum", 0.02)


register_all()
