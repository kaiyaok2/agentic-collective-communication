"""Round 9 -- DEPTH-SCALING SWEEP (the campaign's headline experiment).

r2 established the key result: divergence GROWS with baseline framing depth
(deep4 best-of-8 1.15x -> deep8 2.18x), because overlay's bounded R=3 refinement
stays trapped in the D-deep chain (8/8 seeds stuck at 8 collectives on deep8)
while kiss collapses to 2-3. r9 FILLS THE MONOTONIC CURVE at depths 5 and 7 (between r2's 4/6/8) plus
big-payload depth-7/8 variants, to establish the divergence-vs-depth trend.

Gate limit discovered here: the shared correctness gate resolves at most
resolve_passes=8 dependent-collective levels, so depth is CAPPED at 8 (depth>=10
returns an unresolved-tail error, NOT a real divergence). Deeper-than-8 framing
is therefore not testable on this gate without touching the shared scorer, which
the campaign will not do. Depth 8 is the deepest robust divergence point.

Uses the IDENTICAL _mk_deep_chain construction from r2 (mechanical per-shard
scale/unscale chain, fused optimum = 1 AR) so the only varying factor is depth.
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


def _mk_deep_chain(name, depth, part=256):
    """IDENTICAL scale/unscale construction to r2._mk_deep_chain (the
    divergence-producing lever). Gate note: the shared correctness gate resolves
    at most resolve_passes=8 dependent-collective levels, so depth is CAPPED at
    8 (depth>=10 returns an unresolved-tail error). r9 therefore FILLS THE
    MONOTONIC CURVE at depths 5 and 7 (between r2's 4/6/8) to establish the
    divergence-vs-depth trend within the gate's representable range."""
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W"]
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)"]
    body += ["    return s"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline: {depth} dependent "
         f"AR_SUM stages, each a mechanical per-shard scale/unscale; net result "
         f"= per-shard-scaled AR(x). Fused optimum is 1 AR.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_deep_chain("r9_deep5", 5)
    _mk_deep_chain("r9_deep7", 7)
    _mk_deep_chain("r9_deep8_big", 8, part=1024)
    _mk_deep_chain("r9_deep7_big", 7, part=1024)


register_all()
