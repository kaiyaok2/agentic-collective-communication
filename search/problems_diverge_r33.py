"""Round 33 -- SECOND-FAMILY PROBE: does a STRUCTURALLY DISTINCT collapse also trap
Overlay, or is the 18-win roster specific to per-shard DIAGONAL multiplicative scale?

All 18 robust wins are one mechanism: redundant AR(SUM) stages masked by a reversible
per-shard DIAGONAL scale (a[r]*shard), collapsible to 1 AR because the scale commutes
through the sum. Every TIE round (r8 telescope, r10 AG-roundtrip, r11 max-offset, r12
mixed-primitive, r14 additive-zero-sum) was a collapse Overlay's enumerate-from-baseline
DOES recognize. So a genuine SECOND family needs a DIFFERENT algebraic operator that is
still non-obvious to Overlay.

This round holds the proven AR/W skeleton byte-identical (so baselines pass the fp32
gate exactly like r22/r31) and swaps ONLY the per-stage reversible transform:

  A. shear8   -- NON-DIAGONAL LINEAR operator. Each stage couples adjacent shard PAIRS:
     even shard 2k <- (2k + 2k+1); odd shard unchanged. This is an elementary shear /
     row-operation matrix (off-diagonal coupling), NOT a diagonal scale. Its inverse is
     the exact integer un-shear (2k <- 2k - 2k+1). Chain of 8 dependent AR(SUM) stages,
     each shear/AR/unshear, nets to ONE AR + a final shear. **KEY PROBE.** If this
     CONFIRMS at best-of-16 -> a genuine SECOND family (non-diagonal linear collapse).
     If it TIES -> the win is specific to diagonal per-shard scaling.

  B. permscale8 -- MONOMIAL / ROUTING operator. Each stage scales shard r by a[r] AND
     moves it to position (r + W//2) % W (a half-rotation of shard positions). Combines
     routing with scale (a monomial matrix), distinct from pure diagonal. r24 showed
     Overlay FOLDS pure permutation-invariance; does permutation + HETEROGENEOUS scale
     re-trap it? Tests whether heterogeneity-in-general traps or multiplication only.

Both carry the TRUTHFUL count cue ("8 dependent all_reduce operations") -- the cue that
made every rank-heterogeneous win best-of-16-robust (r22/r26/r31). *_res variants
(result-only doc) are registered for the escapability contrast but are secondary.

fp32: reuses the SAME /W normalization the proven scale family relies on (gate is
tolerance-based, not bit-exact -- the proven family uses non-power-of-2 a[r]=1.5 and
passes). shear uses only integer combinations (add/sub) so it is STRICTLY cleaner than
the proven scale round-trip.
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


# ---------- A. shear (non-diagonal linear: couple adjacent shard pairs) ----------
def _shear_code(name, part, depth=8):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    buf = s.clone()",
              "    for k in range(W // 2):",
              "        a2 = 2*k; b2 = 2*k + 1",
              "        buf[a2*S:(a2+1)*S] = (s[a2*S:(a2+1)*S] + s[b2*S:(b2+1)*S]) / W",
              "        buf[b2*S:(b2+1)*S] = s[b2*S:(b2+1)*S] / W",
              "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            L += ["    for k in range(W // 2):",
                  "        a2 = 2*k; b2 = 2*k + 1",
                  "        s[a2*S:(a2+1)*S] = s[a2*S:(a2+1)*S] - s[b2*S:(b2+1)*S]"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _shear_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        out = s.clone()
        for k in range(world_size // 2):
            a2 = 2 * k; b2 = 2 * k + 1
            out[a2 * part:(a2 + 1) * part] = (s[a2 * part:(a2 + 1) * part]
                                              + s[b2 * part:(b2 + 1) * part])
            out[b2 * part:(b2 + 1) * part] = s[b2 * part:(b2 + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


# ---------- B. permscale (monomial: per-shard scale + half-rotation of positions) ----
def _permscale_code(name, part, depth=8):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size",
         "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
         "    p = [(r + W//2) % W for r in range(W)]",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    buf = s.clone()",
              "    for r in range(W):",
              "        buf[p[r]*S:(p[r]+1)*S] = a[r] * s[r*S:(r+1)*S] / W",
              "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            L += ["    tmp = s.clone()",
                  "    for r in range(W):",
                  "        tmp[r*S:(r+1)*S] = s[p[r]*S:(p[r]+1)*S] / a[r]",
                  "    s = tmp"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _permscale_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        p = [(r + world_size // 2) % world_size for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[p[r] * part:(p[r] + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    shear_ref = _shear_ref(part)
    perm_ref = _permscale_ref(part)

    def _mk_gen(ref):
        def gen(world_size, pattern='uniform', shard_size=None, seed=0):
            pra = _gen_shards(world_size, seed, part)
            return {'per_rank_args': pra, 'shared_args': {},
                    'expected': ref(pra, world_size)}
        return gen

    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    SHEAR_R = ("Final result = the summed vector where each even shard 2k is replaced "
               "by (shard 2k + shard 2k+1) and each odd shard is unchanged (a pairwise "
               "shear coupling adjacent shards).")
    PERM_R = ("Final result = the summed vector with shard r scaled by a[r]=1.0+0.5*"
              "(r%3) and moved to shard position (r + world//2) % world.")

    _reg("r33_shear8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{SHEAR_R}",
         shear_ref, _mk_gen(shear_ref), _shear_code("r33_shear8_count8", part))
    _reg("r33_shear8_res",
         f"Local x (world*{part},), S={part}. {SHEAR_R}",
         shear_ref, _mk_gen(shear_ref), _shear_code("r33_shear8_res", part))
    _reg("r33_permscale8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{PERM_R}",
         perm_ref, _mk_gen(perm_ref), _permscale_code("r33_permscale8_count8", part))
    _reg("r33_permscale8_res",
         f"Local x (world*{part},), S={part}. {PERM_R}",
         perm_ref, _mk_gen(perm_ref), _permscale_code("r33_permscale8_res", part))


register_all()
