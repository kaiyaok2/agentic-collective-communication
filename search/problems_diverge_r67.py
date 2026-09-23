"""Round 67 -- FAMILY-6 direction B (NEW MECHANISM): RANK-1 SELF-COUPLING (full Sherman-Morrison).

Companion to r66 (orthogonal-Walsh rank-1). Here the left and right vectors are the SAME
fixed sign pattern v (so v . v / N == 1, NON-orthogonal), giving a self-reinforcing rank-1
update whose fold requires the FULL Sherman-Morrison denominator:

    out = s + beta * v * mean(v * s)

where s = AR(SUM, x) and v is the period-2 sign pattern [+1,-1,+1,-1,...]. A projection of s
onto v is added back along v -- the v-component of s is amplified. Distinct from r66 (the
inverse there had a trivial denominator because v . u == 0); here the fold must carry the
1/(1+beta) factor.

Telescoping (exact, 1-AR-collapsible): with A = I + beta * v v^T / N and mean(v*v) == 1, the
Sherman-Morrison inverse is A^-1 = I - (beta/(1+beta)) * v v^T / N. Each stage forms buf = A s
and does a redundant AR (AR(SUM,buf)/W == buf); every non-final stage inverts locally via
s = acc - (beta/(1+beta))*v*mean(v*acc). So the depth-D chain nets to a SINGLE update.
naive plain AR fails. Requires 1+beta != 0 (positive beta safe). The mean-projection keeps the
coupling magnitude small so the redundant-AR round-trip is fp32-stable. Distinctness:
(beta, payload); prescreen md5 guard vs all registered.
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
         "    v = (1 - 2 * (idx % 2)).to(s.dtype)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    buf = s + BETA * v * (v * s).mean()",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
              "    acc = acc / W"]
        if not last:
            L += ["    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()"]
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
        out = s + beta * v * (v * s).mean()
        return [out.clone() for _ in range(world_size)]
    return _r


def _mk(name, part, depth, beta, cue=True):
    ref = _ref(part, beta)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},). {COUNT}Final result = the elementwise SUM of x "
           f"across ranks (s), plus {beta} times a fixed sign vector v (repeating pattern "
           f"[+1,-1]) scaled by the mean of the elementwise product of v with s (a "
           f"self-reinforcing projection onto v).")
    _reg(name, doc, ref, gen, _code(name, part, depth, beta))


def register_all():
    _mk("r67_vself_b0p5_p2048_d8", 2048, 8, 0.5)
    _mk("r67_vself_b1p0_p2048_d8", 2048, 8, 1.0)
    _mk("r67_vself_b2p0_p2048_d8", 2048, 8, 2.0)
    _mk("r67_vself_b0p5_p1024_d8", 1024, 8, 0.5)
    _mk("r67_vself_b1p0_p512_d8",  512,  8, 1.0)


register_all()
