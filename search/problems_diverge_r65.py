"""Round 65 -- FAMILY-5 direction B (NEW MECHANISM): MULTIPLICATIVE GLOBAL NORMALIZATION.

Distinct from fam-4 (r63), which ADDS a uniform global scalar (out = s + beta*stat(s)).
Here the whole reduced vector is DIVIDED by a global scalar built from its own magnitude:

    out = s / (1 + beta * mean(|s|))

where s = AR(SUM, x). Every element is scaled by one global normalizer -- a multiplicative
global coupling (vs fam-4's additive, fam-1's per-shard). overlay must recognize the global
abs-mean reduction and a broadcast divide, on top of the all_reduce.

Telescoping (exact, 1-AR-collapsible): each stage forms buf = s / (1 + beta*mean(|s|)) and
does a redundant AR (buf identical across ranks => AR(SUM,buf)/W == buf); every non-final
stage inverts locally. With A = mean(|acc|), M = mean(|s|): A = M/(1+beta*M) =>
M = A/(1-beta*A), g = 1+beta*M, s = acc*g. So the depth-D chain nets to a SINGLE
normalization. naive plain AR fails (dividing by 1+beta*mean(|s|) != identity). Requires
1-beta*A != 0 (small beta keeps it safe). Distinctness: (beta, payload); prescreen md5 guard.
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
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; BETA = {beta}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    g = 1.0 + BETA * s.abs().mean()",
              "    buf = s / g",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
              "    acc = acc / W"]
        if not last:
            L += ["    A = acc.abs().mean()",
                  "    M = A / (1.0 - BETA * A)",
                  "    gr = 1.0 + BETA * M",
                  "    s = acc * gr"]
        else:
            L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _ref(part, beta):
    def _r(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        g = 1.0 + beta * float(s.abs().mean())
        out = s / g
        return [out.clone() for _ in range(world_size)]
    return _r


def _mk(name, part, depth, beta, cue=True):
    ref = _ref(part, beta)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},). {COUNT}Final result = the elementwise SUM of x "
           f"across ranks (s), divided elementwise by the global scalar (1 + {beta} times the "
           f"mean of the absolute values of s).")
    _reg(name, doc, ref, gen, _code(name, part, depth, beta))


def register_all():
    _mk("r65_gnorm_b0p5_p2048_d8", 2048, 8, 0.5)
    _mk("r65_gnorm_b1p0_p2048_d8", 2048, 8, 1.0)
    _mk("r65_gnorm_b2p0_p2048_d8", 2048, 8, 2.0)
    _mk("r65_gnorm_b0p5_p1024_d8", 1024, 8, 0.5)
    _mk("r65_gnorm_b1p0_p512_d8",  512,  8, 1.0)


register_all()
