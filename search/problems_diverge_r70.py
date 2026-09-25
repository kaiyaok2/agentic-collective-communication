"""Round 70 -- FAMILY-9 (NEW): rank-indexed MONOMIAL-MAP coupling
(scaled modular-affine shard permutation).

Conceptually distinct from every prior family. The final operator M on the reduced sum is a
MONOMIAL MATRIX M = D.P: a fixed shard-block PERMUTATION P composed with a rank-indexed
diagonal scale D. Output shard k gathers from source shard perm(k) and scales it:

    out[shard k] = scale[k] * s[shard perm(k)],   perm(k) = (a*k + b) mod W

with gcd(a, W)=1 so perm is a bijection (exact for even/composite W incl. 224=2^5*7 when a is
odd and not a multiple of 7). This is NOT a diagonal scale (fam-1 keeps each shard in place),
NOT a routing count (fam-2), NOT an off-diagonal band (fam-7 bidiagonal / fam-8's dense 2x2
rotation) -- no prior family REORDERS the shard blocks. M is a monomial matrix -> ALWAYS
invertible (M^{-1} = P^{-1}.D^{-1}: inverse-permutation gather + reciprocal scale), so the
depth-D chain telescopes to ONE all_reduce(SUM,x) followed by one local scaled-gather. Each
stage forms buf = M.s / W (buf identical across ranks -> AR(SUM,buf) == M.s) and undoes the
previous stage's monomial map (inverse-perm scatter + divide by scale) on all but the last
stage. Plain AR(SUM,x) returns s, not M.s -> trap holds.

The 1-AR fold's local apply is a COMPUTED-INDEX gather across all W shard blocks with a
per-block scale (perm built from modular-affine arithmetic) -- indexing-intricate enough that a
single-pass implementation is easy to get subtly wrong (off-by-one, gather-vs-scatter
direction, mod arithmetic), while iterative refinement converges to the correct fold. Bounded
scale (|scale| in [0.4, 1.2]) keeps the depth-D baseline well-conditioned under the fp32 gate.
MockTorch-traceable (indexing / add / mul / scalar-list arithmetic). Distinct (a,b,scale,
payload) tuples -> distinct reference md5 from every registered problem.
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


def _mono_code(name, part, depth, a, boff, scale_expr):
    """Depth-D telescoping BASELINE (D dependent all_reduce stages) with monomial map M=D.P."""
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            f"    perm = [({a}*k + {boff}) % W for k in range(W)]",
            "    invperm = [0] * W",
            "    for k in range(W):",
            "        invperm[perm[k]] = k",
            f"    scale = [{scale_expr} for k in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        # buf = M.s / W  (identical across ranks -> AR(SUM,buf) == M.s)
        body += ["    buf = s / W",
                 "    for k in range(W):",
                 "        buf[k*S:(k+1)*S] = scale[k] * s[perm[k]*S:(perm[k]+1)*S] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            # undo previous M: recover pre-map s from current s (= M.s_prev)
            body += ["    tmp = s.clone()",
                     "    for k in range(W):",
                     "        s[perm[k]*S:(perm[k]+1)*S] = tmp[k*S:(k+1)*S] / scale[k]"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _mono_ref(part, a, boff, scale_pyfn):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        perm = [(a * k + boff) % world_size for k in range(world_size)]
        scale = [scale_pyfn(k) for k in range(world_size)]
        out = s.clone()
        for k in range(world_size):
            out[k * part:(k + 1) * part] = scale[k] * s[perm[k] * part:(perm[k] + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, a, boff, scale_expr, scale_pyfn, sdesc):
    ref = _mono_ref(part, a, boff, scale_pyfn)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    doc = (f"Local x (world*{part},), S={part}. The result is computed as {depth} dependent "
           f"all_reduce stages: stage 1 reduces x, then each subsequent stage applies the "
           f"rank-indexed monomial map out[shard k] = scale[k]*s[shard perm(k)] with "
           f"perm(k)=({a}*k+{boff}) mod W (a fixed shard-block permutation) and scale[k]={sdesc}, "
           f"then undoes the previous stage's map (inverse-permutation gather divided by scale) "
           f"before reducing again. Final result = the monomial-mapped all-rank SUM (each shard "
           f"k = scale[k] times shard perm(k) of the summed vector).")
    _reg(name, doc, ref, gen, _mono_code(name, part, depth, a, boff, scale_expr))


def register_all():
    # deep d8 + heavy payloads (fam-1/7 proven strict-gate recipe); a coprime to 224 (odd,
    # not a multiple of 7) so perm is a bijection at W in {4,8,224}; distinct (a,b,scale) tuples.
    _mk("r70_mono_a3b1_d8_p2048", 2048, 8, 3, 1, "0.5 + 0.3*(k % 3)",
        lambda k: 0.5 + 0.3 * (k % 3), "0.5+0.3*(k%3)")
    _mk("r70_mono_a5b2_d8_p2048", 2048, 8, 5, 2, "0.6 + 0.2*(k % 4)",
        lambda k: 0.6 + 0.2 * (k % 4), "0.6+0.2*(k%4)")
    _mk("r70_mono_a9b1_d8_p1024", 1024, 8, 9, 1, "0.4 + 0.25*(k % 3)",
        lambda k: 0.4 + 0.25 * (k % 3), "0.4+0.25*(k%3)")


register_all()
