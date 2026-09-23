"""Round 68 -- FAMILY-7 (NEW): rank-indexed BIDIAGONAL inter-shard coupling.

Conceptually distinct from every prior family: this is the first mechanism where an
output shard depends on ANOTHER rank's shard through a continuous rank-indexed
coefficient. fam-1 is a per-shard MULTIPLICATIVE diagonal (shards independent); fam-2
is rank->routing counts; fam-3/4/5/6 never mix shards. Here the final operator is a
unit-upper-bidiagonal matrix M on the reduced sum:

    out[shard r] = s[shard r] + b[r] * s[shard r+1]   (r < W-1),   out[shard W-1] = s[shard W-1]

with b[r] a rank-indexed coefficient (|b[r]|<1 keeps the back-substitution inverse
well-conditioned). M is unit-triangular -> invertible by local back-substitution, so the
depth-D chain telescopes to ONE all_reduce(SUM,x) followed by one local bidiagonal apply.
Each stage forms buf = M.s / W (buf identical across ranks -> AR(SUM,buf) == M.s) and undoes
the previous stage's M via back-substitution on all but the last stage. Plain AR(SUM,x)
returns s, not M.s -> trap holds.

Rank-DEPENDENT (like fam-1/2, which confirm reliably) but through OFF-DIAGONAL mixing rather
than a diagonal scale. MockTorch-traceable (indexing / add / mul / scalar loop). Distinct
(b-pattern, payload, depth) triples -> distinct reference md5 from every registered problem.
"""
import torch  # noqa: F401
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


def _bd_code(name, part, depth, b_expr):
    """Depth-D telescoping BASELINE (D dependent all_reduce stages)."""
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            f"    b = [{b_expr} for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s / W",
                 "    for r in range(W - 1):",
                 "        buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W - 2, -1, -1):",
                     "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _bd_ref(part, b_pyfn):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        b = [b_pyfn(r) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size - 1):
            out[r * part:(r + 1) * part] = (s[r * part:(r + 1) * part]
                                            + b[r] * s[(r + 1) * part:(r + 2) * part])
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, b_expr, b_pyfn, bdesc):
    ref = _bd_ref(part, b_pyfn)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    doc = (f"Local x (world*{part},), S={part}. The result is computed as {depth} dependent "
           f"all_reduce stages: stage 1 reduces x, then each subsequent stage applies the "
           f"rank-indexed bidiagonal coupling out[shard r] = s[shard r] + b[r]*s[shard r+1] "
           f"(b[r]={bdesc}; last shard unchanged) and undoes the previous stage's coupling "
           f"before reducing again. Final result = the bidiagonal-coupled all-rank SUM "
           f"(shard r of the summed vector plus b[r] times shard r+1).")
    _reg(name, doc, ref, gen, _bd_code(name, part, depth, b_expr))


def register_all():
    # deep d8 + heavy payloads (fam-1's proven strict-gate recipe), distinct b-patterns.
    _mk("r68_bidi_b02_d8_p2048", 2048, 8, "0.2 + 0.1*(r % 3)",
        lambda r: 0.2 + 0.1 * (r % 3), "0.2+0.1*(r%3)")
    _mk("r68_bidi_b03_d8_p2048", 2048, 8, "0.3 + 0.1*(r % 4)",
        lambda r: 0.3 + 0.1 * (r % 4), "0.3+0.1*(r%4)")
    _mk("r68_bidi_b025_d8_p1024", 1024, 8, "0.25 + 0.1*(r % 3)",
        lambda r: 0.25 + 0.1 * (r % 3), "0.25+0.1*(r%3)")
    _mk("r68_bidi_b02_d6_p2048", 2048, 6, "0.2 + 0.15*(r % 3)",
        lambda r: 0.2 + 0.15 * (r % 3), "0.2+0.15*(r%3)")
    _mk("r68_bidi_b03_d8_p1024", 1024, 8, "0.3 + 0.1*(r % 5)",
        lambda r: 0.3 + 0.1 * (r % 5), "0.3+0.1*(r%5)")


register_all()
