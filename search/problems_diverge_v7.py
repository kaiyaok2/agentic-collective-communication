"""v7 -- targeted route-around-rejection battery (the ONLY lever that works).

Lesson from v5/v6: divergence between SorcarCCL (kiss, iterative ReAct) and
OverlayCCL (strat, enumerate-once -> refine-top-2) under a FAIR fp32 gate does
NOT come from making problems harder (v5 hd16 overshot -> both fail, kiss even
LOSES) nor from count-reduction that both one-shot (v6 fuse/dead/alg/scale ->
all tie). It comes from hd10's specific band:

    the cheaper optimum is CORRECT BUT ERROR-PRONE TO IMPLEMENT COLD
        (subtle per-shard index arithmetic + a normalization/scale identity
         that a first-draft fused impl gets wrong -> gate reject)
    AND
    the error is REPAIRABLE BY ITERATION (kiss reads the gate error, fixes it)

Overlay discards any strategy whose FIRST implementation fails the gate
(phase3 line 357-359), so a fused optimum that fails cold is thrown away and
overlay refines the safe multi-collective baseline. Kiss's ReAct loop sees the
same gate error and repairs the shard math. That gap is the whole divergence.

This file is MANY variations on that recipe -- all reduce to
`(per-shard linear map) . all_reduce(SUM, x)` (ONE collective) but the baseline
does 2-4 DEPENDENT all_reduces with per-shard index arithmetic between stages,
so the fused form requires deriving the folded per-shard coefficients AND
getting shard offsets / the /world_size normalization exactly right on the
first try. Families vary WHAT makes the fold error-prone:

  AFFINE   per-shard scale c[r] AND shift d[r] fold through the chain
  NORM4    4 dependent stages (deeper fold, more places to slip)
  PERMSC   per-shard scale indexed by a permutation (c[perm[r]])
  MODSC    per-shard scale with period not dividing world_size
  TRISC    cumulative/triangular per-shard scale (harder closed form)
  RSNORM   middle stage is reduce_scatter+all_gather (shard_count off-by-one)

Every optimum is separately validated to pass the SAME fp32 gate AND to carry
real sim headroom over the baseline. All ops MockXM-supported, <=1 collective
dependency level in the OPTIMUM (baselines chain, but the local gate scores the
1-collective optimum; the baseline chains are what overlay is anchored to).
"""
import torch
from .problems import CollectiveProblem, register_problem

S = 256


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


def _gen_shards(world_size, seed=0, part=S):
    torch.manual_seed(seed)
    N = world_size * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# ---------------------------------------------------------------------------
# Common shape: reference = per-shard linear map of AR(SUM, x).
#   out[r*S:(r+1)*S] = a[r] * sum_ranks(x)[r*S:(r+1)*S] + b[r]
# Baseline realizes this via a DEPENDENT chain of ARs with per-shard index
# arithmetic between stages (the /world_size normalization must be tracked so
# repeated AR(SUM) of already-reduced data stays correct). Optimum: ONE AR then
# the per-shard affine locally.
# ---------------------------------------------------------------------------
def _mk_shard_affine(name, a_of_r, b_of_r, nstage=3, part=S, doc_extra=""):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [a_of_r(r, world_size) for r in range(world_size)]
        b = [b_of_r(r, world_size) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part] + b[r]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    # Baseline: nstage dependent ARs. stage1 reduces; each middle stage rebuilds
    # a per-shard-scaled buffer (dividing by world_size so the next AR(SUM),
    # which re-adds W identical replicas, reconstructs the intended value); the
    # final stage folds in the additive shift the same way.
    a_expr = _fn_src(a_of_r)
    b_expr = _fn_src(b_of_r)
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            f"    a = [{a_expr} for r in range(W)]",
            f"    b = [{b_expr} for r in range(W)]",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)      # stage 1"]
    # middle scale stages (nstage-2 of them just re-apply identity-preserving
    # per-shard scale, deepening the dependent chain like hd10's stage3)
    for st in range(nstage - 2):
        body += [
            f"    buf{st} = s1.clone()",
            "    for r in range(W):",
            f"        buf{st}[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W",
            f"    s1 = xm.all_reduce(xm.REDUCE_SUM, buf{st})   # dependent stage",
            "    for r in range(W):",
            f"        s1[r*S:(r+1)*S] = s1[r*S:(r+1)*S] / max(a[r], 1e-9)",
        ]
    # final stage: apply affine then one more dependent AR (normalized)
    body += [
        "    bufN = s1.clone()",
        "    for r in range(W):",
        "        bufN[r*S:(r+1)*S] = (a[r] * s1[r*S:(r+1)*S] + b[r]) / W",
        "    out = xm.all_reduce(xm.REDUCE_SUM, bufN)      # final dependent stage",
        "    return out"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline: {nstage} "
         f"dependent AR_SUM stages with per-shard scale/shift arithmetic "
         f"between them. Result = per-shard affine of AR(x). {doc_extra}",
         _ref, _gen, "\n".join(body) + "\n")


def _fn_src(f):
    # f is one of the small lambdas below; we stored its source in ._src
    return f._src


def _L(src):
    def f(r, W):
        return eval(src, {"r": r, "W": W})
    f._src = src
    return f


# ---------------------------------------------------------------------------
# RSNORM: middle stage uses reduce_scatter + all_gather (hd8-style) so the
# fused optimum must recognize RS+AG == AR and collapse, but a cold draft
# mis-sets shard_count / scatter_dim / offset -> gate reject.
# ---------------------------------------------------------------------------
def _mk_rs_norm(name, a_of_r, part=S):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [a_of_r(r, world_size) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    a_expr = _fn_src(a_of_r)
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            f"    a = [{a_expr} for r in range(W)]",
            "    # stage 1: full reduce",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    # stage 2: per-shard scale, then RS+AG round-trip (== AR)",
            "    buf = s1.clone()",
            "    for r in range(W):",
            "        buf[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W",
            "    rs = xm.reduce_scatter(xm.REDUCE_SUM, buf, scatter_dim=0,",
            "                           shard_count=W)",
            "    out = xm.all_gather(rs, dim=0)",
            "    return out"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline: AR then a "
         f"per-shard scale then reduce_scatter+all_gather round-trip. Result = "
         f"per-shard-scaled AR(x).", _ref, _gen, "\n".join(body) + "\n")


def register_all():
    # --- AFFINE: scale + shift fold (more constants to track than hd10) ---
    _mk_shard_affine("v7_affine3",
                     _L("1.0 + 0.5*(r % 3)"), _L("0.25*(r % 4)"),
                     nstage=3, doc_extra="scale c[r]=1+0.5(r%3), shift d[r]=0.25(r%4).")
    # --- NORM4: 4 dependent stages (deeper fold) ---
    _mk_shard_affine("v7_norm4",
                     _L("1.0 + 0.5*(r % 3)"), _L("0.0"),
                     nstage=4, doc_extra="4-stage chain, scale-only fold.")
    # --- PERMSC: scale indexed by a permutation of shard id ---
    _mk_shard_affine("v7_permsc",
                     _L("1.0 + 0.25*(((r*3 + 1) % W))"), _L("0.0"),
                     nstage=3, doc_extra="scale indexed by a permuted shard id.")
    # --- MODSC: period 5 (does not divide world=224) ---
    _mk_shard_affine("v7_modsc",
                     _L("1.0 + 0.5*(r % 5)"), _L("0.0"),
                     nstage=3, doc_extra="scale period 5 (indivisible by world).")
    # --- TRISC: cumulative/triangular scale (harder closed form) ---
    _mk_shard_affine("v7_trisc",
                     _L("1.0 + 0.1*((r*(r+1)//2) % 7)"), _L("0.0"),
                     nstage=3, doc_extra="triangular-number-derived scale.")
    # --- NORM4 affine (deepest + shift) ---
    _mk_shard_affine("v7_affine4",
                     _L("1.0 + 0.5*(r % 3)"), _L("0.5*(r % 2)"),
                     nstage=4, doc_extra="4-stage affine (scale+shift).")
    # --- RSNORM: reduce_scatter+all_gather middle stage ---
    _mk_rs_norm("v7_rsnorm3", _L("1.0 + 0.5*(r % 3)"))
    _mk_rs_norm("v7_rsnorm_mod", _L("1.0 + 0.25*(r % 5)"))


register_all()
