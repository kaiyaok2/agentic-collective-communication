"""Round 36 -- DISTINCT MECHANISM: AFFINE (scale AND shift) per-shard chain.

Family-1 = pure per-shard MULTIPLICATIVE scale (confirms). r14 = pure per-shard ADDITIVE
zero-sum (Overlay MORE reliable, best-of-N tie). This round tests the COMBINATION, which
neither covers: a per-shard AFFINE map y = a[r]*block + c[r] applied before each
all_reduce(SUM) and inverted after. The question is whether affine behaves like the
multiplicative family (confirms), like the additive family (ties/reverse), or is its own
regime.

An affine map does NOT simply commute through AR(SUM): sum_r (a*x_r + c) = a*sum(x_r) + W*c
only when a,c are shared across the reduced ranks (they are -- functions of block index,
applied identically on every rank). So a D-deep chain of {apply affine; AR(SUM); apply
inverse affine} telescopes to ONE AR(SUM) plus a final affine -- but the bookkeeping mixes
a multiplicative factor (needs /a to invert) AND an additive term (needs W*c handling).
This is strictly richer than family-1 and than r14, and the inverse is more error-prone
(both a division and a subtraction, order-sensitive).

r36_affine8: per block r, a[r]=1.0+0.5*(r%3) (non-power-of-2, per r29 boundary),
c[r]=0.3+0.2*(r%2). Chain of 8 dependent AR(SUM), each affine/AR/inverse-affine, collapsing
to 1 AR(SUM) + a final affine.

If affine CONFIRMS like family-1 -> the trap is "any per-shard op with a multiplicative
component"; if it TIES like r14 -> the additive component dominates overlay's fold; either
outcome refines the family boundary with a genuinely mixed algebra (not a family-1 variant).

*_count8 = truthful count cue; *_res = result-only.

fp32: baseline genuine D=8 AR(SUM) affine chain (passes gate). Reference: after the first
AR every rank holds the full sum; the telescoping inverse-affine chain leaves net =
a[r]*sum_block + c[r] on each block r (final stage applies affine without inverting).
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


def _affine_code(name, part, depth=8):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size",
         "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
         "    c = [0.3 + 0.2*(r % 2) for r in range(W)]",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    buf = s.clone()",
              "    for r in range(W):",
              "        buf[r*S:(r+1)*S] = (a[r] * s[r*S:(r+1)*S] + c[r]) / W",
              "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            L += ["    for r in range(W):",
                  "        s[r*S:(r+1)*S] = (s[r*S:(r+1)*S] - c[r]) / a[r]"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _affine_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        out = s.clone()
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        c = [0.3 + 0.2 * (r % 2) for r in range(world_size)]
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part] + c[r]
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    ref = _affine_ref(part)

    def _mk_gen(ref):
        def gen(world_size, pattern='uniform', shard_size=None, seed=0):
            pra = _gen_shards(world_size, seed, part)
            return {'per_rank_args': pra, 'shared_args': {},
                    'expected': ref(pra, world_size)}
        return gen

    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    RES = ("Final result = the summed vector with a per-block AFFINE map applied: block r "
           "becomes a[r]*block + c[r], where a[r]=1.0+0.5*(r%3), c[r]=0.3+0.2*(r%2).")

    _reg("r36_affine8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{RES}",
         ref, _mk_gen(ref), _affine_code("r36_affine8_count8", part))
    _reg("r36_affine8_res",
         f"Local x (world*{part},), S={part}. {RES}",
         ref, _mk_gen(ref), _affine_code("r36_affine8_res", part))


register_all()
