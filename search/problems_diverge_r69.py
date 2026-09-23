"""Round 69 -- FAMILY-8 (NEW): rank-indexed 2x2 block-rotation (Givens) coupling.

Conceptually distinct from every prior family. The final operator M on the reduced sum is
BLOCK-DIAGONAL ORTHOGONAL: adjacent shards are paired (2j, 2j+1) and each pair is mixed by a
rank-indexed 2x2 rotation of angle theta[j]:

    out[shard 2j]   = cos(th[j])*s[shard 2j] - sin(th[j])*s[shard 2j+1]
    out[shard 2j+1] = sin(th[j])*s[shard 2j] + cos(th[j])*s[shard 2j+1]

This is neither a diagonal scale (fam-1), a routing count (fam-2), an upper-triangular chain
(fam-7 couples r->r+1 one-directionally and unboundedly), a rank-1 outer product (fam-6), nor
a global norm (fam-5). M is orthogonal -> ALWAYS invertible (M^{-1} = rotation by -theta), so
the depth-D chain telescopes to ONE all_reduce(SUM,x) followed by one local block-rotation
apply. Each stage forms buf = M.s / W (buf identical across ranks -> AR(SUM,buf) == M.s) and
undoes the previous stage's rotation (rotate by -theta) on all but the last stage. Plain
AR(SUM,x) returns s, not M.s -> trap holds.

Rank-DEPENDENT (like fam-1/2/7, which confirm reliably) but through ORTHOGONAL pair-mixing
rather than a diagonal scale or a triangular chain. World sizes are even (4/8/224) so pairing
is exact. MockTorch-traceable (indexing / add / mul / scalar cos-sin constants). Distinct
(theta-pattern, payload, depth) triples -> distinct reference md5 from every registered problem.
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


def _br_code(name, part, depth, th_expr):
    """Depth-D telescoping BASELINE (D dependent all_reduce stages) with block-rotation M."""
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    import math",
            f"    S = {part}; W = world_size; P = W // 2",
            f"    th = [{th_expr} for j in range(P)]",
            "    c = [math.cos(t) for t in th]",
            "    sg = [math.sin(t) for t in th]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s / W",
                 "    for j in range(P):",
                 "        a = s[(2*j)*S:(2*j+1)*S]",
                 "        b = s[(2*j+1)*S:(2*j+2)*S]",
                 "        buf[(2*j)*S:(2*j+1)*S]   = (c[j]*a - sg[j]*b) / W",
                 "        buf[(2*j+1)*S:(2*j+2)*S] = (sg[j]*a + c[j]*b) / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for j in range(P):",
                     "        a = s[(2*j)*S:(2*j+1)*S].clone()",
                     "        b = s[(2*j+1)*S:(2*j+2)*S].clone()",
                     "        s[(2*j)*S:(2*j+1)*S]   = c[j]*a + sg[j]*b",
                     "        s[(2*j+1)*S:(2*j+2)*S] = -sg[j]*a + c[j]*b"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _br_ref(part, th_pyfn):
    import math

    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        P = world_size // 2
        out = s.clone()
        for j in range(P):
            th = th_pyfn(j)
            c, sg = math.cos(th), math.sin(th)
            a = s[(2 * j) * part:(2 * j + 1) * part].clone()
            b = s[(2 * j + 1) * part:(2 * j + 2) * part].clone()
            out[(2 * j) * part:(2 * j + 1) * part] = c * a - sg * b
            out[(2 * j + 1) * part:(2 * j + 2) * part] = sg * a + c * b
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, th_expr, th_pyfn, thdesc):
    ref = _br_ref(part, th_pyfn)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    doc = (f"Local x (world*{part},), S={part}. The result is computed as {depth} dependent "
           f"all_reduce stages: stage 1 reduces x, then each subsequent stage applies the "
           f"rank-indexed 2x2 block-rotation that mixes adjacent shard pairs (2j, 2j+1) by "
           f"angle theta[j]={thdesc} (out[2j]=cos*s[2j]-sin*s[2j+1]; out[2j+1]="
           f"sin*s[2j]+cos*s[2j+1]) and undoes the previous stage's rotation before reducing "
           f"again. Final result = the block-rotated all-rank SUM.")
    _reg(name, doc, ref, gen, _br_code(name, part, depth, th_expr))


def register_all():
    # deep d8 + heavy payloads (fam-1/7 proven strict-gate recipe), distinct theta-patterns.
    _mk("r69_rot_t03m3_d8_p2048", 2048, 8, "0.3 + 0.2*(j % 3)",
        lambda j: 0.3 + 0.2 * (j % 3), "0.3+0.2*(j%3)")
    _mk("r69_rot_t05m4_d8_p2048", 2048, 8, "0.5 + 0.15*(j % 4)",
        lambda j: 0.5 + 0.15 * (j % 4), "0.5+0.15*(j%4)")
    _mk("r69_rot_t04m3_d8_p1024", 1024, 8, "0.4 + 0.25*(j % 3)",
        lambda j: 0.4 + 0.25 * (j % 3), "0.4+0.25*(j%3)")


register_all()
