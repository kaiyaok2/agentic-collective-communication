"""Round 29 -- INDEPENDENT DATA POINT: is the robust contiguous-rank-heterogeneous
win tied to the specific a[r]=1.0+0.5*(r%3) pattern, or a property of the pattern
CLASS?

(An earlier affine construction here was invalid: after the first all_reduce every
rank holds the IDENTICAL vector, so a later AR(SUM,buf) = W*buf, not a sum over
distinct per-rank contributions -- so an additive zero-sum shift does NOT cancel in
later stages. That collapses back to plain scale/unscale, no new information. Dropped.)

Instead this round varies the HETEROGENEITY PATTERN on the confirmed contiguous
scale/unscale template: a[r] = 2.0**((r % 4) - 1) gives FOUR distinct exact levels
{0.5, 1.0, 2.0, 4.0} (all powers of 2 -> fp32-exact scale AND unscale) instead of the
usual THREE non-power-of-2 levels {1.0, 1.5, 2.0}. If this ALSO confirms robustly,
the win is a property of "contiguous rank-heterogeneous multiplicative depth", not of
the specific constants -> strengthens the 17 against a single-constant-artifact
objection.

  A. pow2_count8 : richer 4-level powers-of-2 per-shard scale/unscale, narrated
     "8 dependent all_reduce operations". Predict: CONFIRMS robustly.
  B. pow2_res    : same code, result-only doc. Predict per L12: median traps,
     best-of-N may escape (distributional).

fp32-exact: powers of 2 exact; /W exact (W power of 2).
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


# a[r] = 2.0**((r%4)-1) -> {0.5,1,2,4}, all powers of 2 (exact). Contiguous shards.
def _pow2_code(name, part, depth=8):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [2.0 ** ((r % 4) - 1) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / a[r]"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _pow2_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [2.0 ** ((r % 4) - 1) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    ref = _pow2_ref(part)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': ref(pra, world_size)}

    RESULT = ("Final result = the summed vector with shard r scaled by "
              "a[r]=2.0**((r%4)-1) (levels 0.5, 1, 2, 4).")
    _reg("r29_pow2_count8",
         f"Local x (world*{part},), S={part}. The result is computed using 8 "
         f"dependent all_reduce operations. {RESULT}",
         ref, gen, _pow2_code("r29_pow2_count8", part))
    _reg("r29_pow2_res",
         f"Local x (world*{part},), S={part}. {RESULT}",
         ref, gen, _pow2_code("r29_pow2_res", part))


register_all()
