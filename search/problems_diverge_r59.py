"""Round 59 -- FAMILY-1 EXPANSION (non-duplicate): per-shard MULTIPLICATIVE
scale/unscale diagonal, NEW scale patterns / depths / payloads.

Family-1 mechanism (unchanged): a depth-D chain that, each stage, applies a
per-shard STATIC multiplicative scale a[r] and undoes the previous stage's scale
before reducing again. The chain telescopes to ONE all_reduce(SUM,x) followed by
one per-shard multiply by a[r]. Overlay's enumerate-from-baseline stays pinned at
the D-deep chain; Kiss reads the gate error and folds to 1 AR + local diagonal.

These are DISTINCT problems from the confirmed r1/r2/r16/r20-23 set: the scale
vector a[] uses NEW residue cycles (r%5, r%7, r%4 with new coefficients) and NEW
(depth, payload) combinations not present in the existing family-1 catalog, so the
reference outputs (and md5 test cases) are distinct. Same trap algebra, new
instances -- no duplication of any existing (a-pattern, depth, payload) triple.
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


def _su_code(name, part, depth, a_expr):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            f"    a = [{a_expr} for r in range(W)]",
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


def _su_ref(part, a_pyfn):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [a_pyfn(r) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, a_expr, a_pyfn, adesc):
    ref = _su_ref(part, a_pyfn)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    doc = (f"Local x (world*{part},), S={part}. The result is computed as {depth} "
           f"dependent all_reduce stages: stage 1 reduces x, then each subsequent "
           f"stage applies the per-shard scale a[r]={adesc} and undoes the previous "
           f"stage's scale before reducing again. Final result = per-shard-scaled "
           f"all-rank SUM (shard r of the summed vector scaled by a[r]).")
    _reg(name, doc, ref, gen, _su_code(name, part, depth, a_expr))


def register_all():
    # NEW scale cycles + (depth, payload) combos not present in r1/r2/r16/r20-23.
    _mk("r59_su_a5_d8", 256, 8, "1.0 + 0.4*(r % 5)",
        lambda r: 1.0 + 0.4 * (r % 5), "1.0+0.4*(r%5)")
    _mk("r59_su_a4_d6_p512", 512, 6, "1.0 + 0.6*(r % 4)",
        lambda r: 1.0 + 0.6 * (r % 4), "1.0+0.6*(r%4)")
    _mk("r59_su_a7_d8", 256, 8, "1.0 + 0.3*(r % 7)",
        lambda r: 1.0 + 0.3 * (r % 7), "1.0+0.3*(r%7)")
    _mk("r59_su_a4_d8_p1024", 1024, 8, "1.0 + 0.6*(r % 4)",
        lambda r: 1.0 + 0.6 * (r % 4), "1.0+0.6*(r%4)")


register_all()
