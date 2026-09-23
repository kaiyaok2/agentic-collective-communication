"""Round 64 -- FAMILY-5 direction A (NEW MECHANISM): BIPARTITE CROSS-REGION coupling.

Distinct from every prior family: fam-1 per-shard scale, fam-2 routing count, fam-3
data-dependent diagonal, fam-4 (r63) single uniform global scalar. Here the B blocks split
into two regions A=[0,B/2) and B=[B/2,B); each region is shifted by the OTHER region's mean:

    out_A = s_A + beta * mean(s_B)        out_B = s_B + beta * mean(s_A)

where s = AR(SUM, x). The coupling is CROSS (A depends on B's reduction and vice-versa), a
2-region structure, not a single broadcast -- overlay must recognize two region means and a
crossed broadcast add, on top of the all_reduce.

Telescoping (exact, 1-AR-collapsible): each stage forms buf (cross-shift) and does a
redundant AR (buf identical across ranks => AR(SUM,buf)/W == buf); every non-final stage
inverts locally by solving the 2x2 linear system
    MA = mA + beta*mB,  MB = mB + beta*mA   =>   mA=(MA-beta*MB)/(1-beta^2), mB=(MB-beta*MA)/(1-beta^2)
so the depth-D chain nets to a SINGLE cross-coupling. naive plain AR fails (beta*mean != 0).
Requires beta != +/-1 (1-beta^2 != 0). Distinctness: (beta, payload); prescreen md5 guard.
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


def _code(name, part, depth, beta):
    H = (NBLOCK // 2) * part  # split index (region A = first half of the flat vector)
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; BETA = {beta}; H = {H}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    mA = s[:H].mean(); mB = s[H:].mean()",
              "    buf = s.clone()",
              "    buf[:H] = s[:H] + BETA * mB",
              "    buf[H:] = s[H:] + BETA * mA",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
              "    acc = acc / W"]
        if not last:
            L += ["    MA = acc[:H].mean(); MB = acc[H:].mean()",
                  "    den = 1.0 - BETA * BETA",
                  "    mAr = (MA - BETA * MB) / den; mBr = (MB - BETA * MA) / den",
                  "    nxt = acc.clone()",
                  "    nxt[:H] = acc[:H] - BETA * mBr",
                  "    nxt[H:] = acc[H:] - BETA * mAr",
                  "    s = nxt"]
        else:
            L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _ref(part, beta):
    H = (NBLOCK // 2) * part

    def _r(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        mA = s[:H].mean(); mB = s[H:].mean()
        out = s.clone()
        out[:H] = s[:H] + beta * mB
        out[H:] = s[H:] + beta * mA
        return [out.clone() for _ in range(world_size)]
    return _r


def _mk(name, part, depth, beta, cue=True):
    ref = _ref(part, beta)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), split into region A (first half) and region B "
           f"(second half). {COUNT}Final result = the elementwise SUM of x across ranks (s), "
           f"then region A is shifted by {beta} times the mean of region B and region B is "
           f"shifted by {beta} times the mean of region A (a cross-region coupling).")
    _reg(name, doc, ref, gen, _code(name, part, depth, beta))


def register_all():
    _mk("r64_bipart_b0p5_p2048_d8", 2048, 8, 0.5)
    _mk("r64_bipart_b0p3_p2048_d8", 2048, 8, 0.3)
    _mk("r64_bipart_b0p7_p2048_d8", 2048, 8, 0.7)
    _mk("r64_bipart_b0p5_p1024_d8", 1024, 8, 0.5)
    _mk("r64_bipart_b0p5_p512_d8",  512,  8, 0.5)


register_all()
