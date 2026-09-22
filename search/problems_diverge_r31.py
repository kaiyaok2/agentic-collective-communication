"""Round 31 -- INDEPENDENT DATA POINT (retry, well-formed): is the robust contiguous
rank-heterogeneous win tied to a[r]=1.0+0.5*(r%3), or a property of the pattern CLASS?

r29/r30 tried a[r]=2.0**((r%4)-1) (exponential levels 0.5,1,2,4) and found it a
DEGENERATE design -- the exponential description confuses BOTH agents (overlay
gate-fails, kiss doesn't fold), so it's uninformative. This round keeps the SAME
additive-linear form the agents parse cleanly but with a DIFFERENT, RICHER set of
constants: a[r] = 1.0 + 0.25*(r%5) -> FIVE distinct levels {1.0,1.25,1.5,1.75,2.0}
(0.25 = 2^-2 so fp32-exact scale AND unscale), vs the confirmed family's THREE levels
{1.0,1.5,2.0}. If this ALSO confirms robustly, the win is a property of "contiguous
rank-heterogeneous multiplicative depth", not of the specific 3-level constants.

  A. lin5_count8 : 5-level additive per-shard scale/unscale, narrated "8 dependent
     all_reduce operations". Predict: CONFIRMS (like r22/r23/r26 count8).
  B. lin5_res    : same code, result-only doc. Predict per L12: median traps,
     best-of-N may escape.

fp32-exact: 0.25 = 2^-2 exact; /W exact (W power of 2). a[r] in [1,2] so unscale
1/a[r] is well-conditioned.
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


def _gen_shards(world_size, seed, part):
    torch.manual_seed(seed)
    N = world_size * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# a[r] = 1.0 + 0.25*(r%5) -> {1.0,1.25,1.5,1.75,2.0}, additive-linear, fp32-exact.
def _lin5_code(name, part, depth=8):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.25*(r % 5) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _lin5_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.25 * (r % 5) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    ref = _lin5_ref(part)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': ref(pra, world_size)}

    RESULT = ("Final result = the summed vector with shard r scaled by "
              "a[r]=1.0+0.25*(r%5).")
    _reg("r31_lin5_count8",
         f"Local x (world*{part},), S={part}. The result is computed using 8 "
         f"dependent all_reduce operations. {RESULT}",
         ref, gen, _lin5_code("r31_lin5_count8", part))
    _reg("r31_lin5_res",
         f"Local x (world*{part},), S={part}. {RESULT}",
         ref, gen, _lin5_code("r31_lin5_res", part))


register_all()
