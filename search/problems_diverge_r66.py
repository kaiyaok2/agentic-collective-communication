"""Round 66 -- FAMILY-6 direction A (NEW MECHANISM): RANK-1 ORTHOGONAL-WALSH coupling.

Distinct from every prior family: fam-1 per-shard scale, fam-2 routing count, fam-3
data-dependent diagonal, fam-4 (r63) single uniform global scalar, fam-5 (r64/r65)
bipartite / global-norm. Here the reduced vector is shifted by a rank-1 outer product
built from two FIXED, mutually ORTHOGONAL sign vectors:

    out = s + beta * u * mean(v * s)

where s = AR(SUM, x), v is the period-2 sign pattern [+1,-1,+1,-1,...], u is the period-4
sign pattern [+1,+1,-1,-1,...], and v . u == 0. This is a structured rank-1 coupling: a
projection of s onto v, broadcast back through a DIFFERENT pattern u. It generalizes fam-4's
uniform coupling (u == v == ones, projection == mean) to a non-trivial projection/broadcast
pair -- overlay must recognize the two fixed patterns and the rank-1 update on top of the AR.

Telescoping (exact, 1-AR-collapsible): with A = I + beta * u v^T / N and v . u == 0, the
Sherman-Morrison inverse is exactly A^-1 = I - beta * u v^T / N (the denominator 1+beta*v.u/N
== 1). Each stage forms buf = A s and does a redundant AR (buf identical across ranks =>
AR(SUM,buf)/W == buf); every non-final stage inverts locally via s = acc - beta*u*mean(v*acc)
(mean(v*acc) == mean(v*s) because v.u == 0). So the depth-D chain nets to a SINGLE rank-1
update. naive plain AR fails (the coupling term is nonzero). The mean-projection (vs a raw
sum) keeps the coupling magnitude small so the redundant-AR round-trip is fp32-stable.
Distinctness: (beta, payload); prescreen md5 guard vs all registered.
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
    N = NBLOCK * part
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; BETA = {beta}; N = {N}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         "    idx = torch.arange(N)",
         "    v = (1 - 2 * (idx % 2)).to(s.dtype)",
         "    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    buf = s + BETA * u * (v * s).mean()",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
              "    acc = acc / W"]
        if not last:
            L += ["    s = acc - BETA * u * (v * acc).mean()"]
        else:
            L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _ref(part, beta):
    N = NBLOCK * part

    def _r(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        idx = torch.arange(N)
        v = (1 - 2 * (idx % 2)).to(s.dtype)
        u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
        out = s + beta * u * (v * s).mean()
        return [out.clone() for _ in range(world_size)]
    return _r


def _mk(name, part, depth, beta, cue=True):
    ref = _ref(part, beta)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},). {COUNT}Final result = the elementwise SUM of x "
           f"across ranks (s), plus {beta} times a fixed sign vector u (repeating pattern "
           f"[+1,+1,-1,-1]) scaled by the mean of the elementwise product of a fixed sign "
           f"vector v (repeating pattern [+1,-1]) with s.")
    _reg(name, doc, ref, gen, _code(name, part, depth, beta))


def register_all():
    _mk("r66_walsh_b0p5_p2048_d8", 2048, 8, 0.5)
    _mk("r66_walsh_b1p0_p2048_d8", 2048, 8, 1.0)
    _mk("r66_walsh_b2p0_p2048_d8", 2048, 8, 2.0)
    _mk("r66_walsh_b0p5_p1024_d8", 1024, 8, 0.5)
    _mk("r66_walsh_b1p0_p512_d8",  512,  8, 1.0)


register_all()
