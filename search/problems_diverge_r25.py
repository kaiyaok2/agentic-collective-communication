"""Round 25 -- VALIDATE L13: is it RANK-HETEROGENEITY (not the specific
contiguous scale/unscale code) that traps Overlay?

r24 sharpened L8 -> L13: homogeneous/globally-symmetric deep distributive
collapses (global-scalar homogeneity, permutation-invariance) FOLD for Overlay at
screen; only the per-shard scale/unscale chain -- where each shard is scaled by a
DIFFERENT rank-dependent a[r] so no single rank sees the collapse -- traps it. But
every confirmed win uses the SAME contiguous-block clone+slice code, so "rank-
heterogeneous multiplicative collapse" and "this specific code" are still
confounded. r25 de-confounds by keeping the collapse rank-heterogeneous +
multiplicative but changing HOW the heterogeneity is laid out, all result-only
docstrings:

  A. perm_scale8   : shard PERMUTED. Output shard perm[r] (perm = rotate-by-W//2, a
     fixed bijection) is scaled by a[r]. Rank-heterogeneous, multiplicative, non-
     locally-visible, but the shard<->scale mapping is a permutation, not identity.
  B. strided_scale8: shard STRIDED. "Shard r" = elements x[r::W] (interleaved, not a
     contiguous block); those elements scaled by a[r]. Same algebra, non-contiguous
     memory layout -- guards against the win being a contiguous-slice artifact.

Prediction under L13: BOTH confirm (rank-heterogeneous multiplicative depth is the
mechanism, independent of layout). If either FOLDS for overlay, L13 is too broad and
the win is code-shape-specific -> would REQUIRE narrowing the claim.

fp32-exact: a[r]=1.0+0.5*(r%3) exact; /W exact (W power of 2); permutation/stride
are pure index reorderings (exact).
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


# ---- A. permuted-shard scale/unscale -------------------------------------
# perm[r] = (r + W//2) % W is a bijection for any W. Output shard perm[r] scaled by
# a[r]. Deep chain scales, reduces, unscales, repeats -- collapse is invisible per
# rank because both the scale AND the target shard depend on rank.
def _perm_code(name, part, depth=8):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    perm = [(r + W // 2) % W for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        p = perm[r]",
                 "        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        p = perm[r]",
                     "        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _perm_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        perm = [(r + world_size // 2) % world_size for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            p = perm[r]
            out[p * part:(p + 1) * part] = a[r] * s[p * part:(p + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


# ---- B. strided-shard scale/unscale --------------------------------------
# "shard r" = x[r::W] (interleaved). Scale those elements by a[r]. Non-contiguous
# layout, same rank-heterogeneous multiplicative algebra.
def _strided_code(name, N, depth=8):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        buf[r::W] = a[r] * s[r::W] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r::W] = s[r::W] / max(a[r], 1e-9)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _strided_ref(N):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r::world_size] = a[r] * s[r::world_size]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk_gen(ref_fn):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0, _part=None, _N=None):
        part = _part if _part is not None else 256
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': ref_fn(pra, world_size)}
    return _gen


def register_all():
    part = 256

    # A. permuted-shard scale/unscale (N = world*part, contiguous shards, permuted map)
    perm_ref = _perm_ref(part)

    def perm_gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': perm_ref(pra, world_size)}

    _reg("r25_perm_scale8",
         f"Local x (world*{part},), S={part}. Return the summed vector with shard "
         f"perm[r] (perm = rotate-by-W//2) scaled by a[r]=1.0+0.5*(r%3).",
         perm_ref, perm_gen, _perm_code("r25_perm_scale8", part))

    # B. strided-shard scale/unscale. N must be world*part so x[r::W] partitions.
    def strided_gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)  # N = world*part
        N = world_size * part
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _strided_ref(N)(pra, world_size)}

    def strided_ref_ws(inputs, world_size):
        N = world_size * part
        return _strided_ref(N)(inputs, world_size)

    _reg("r25_strided_scale8",
         f"Local x (world*{part},). Return the summed vector with interleaved lane r "
         f"(elements x[r::world]) scaled by a[r]=1.0+0.5*(r%3).",
         strided_ref_ws, strided_gen, _strided_code("r25_strided_scale8", part))


register_all()
