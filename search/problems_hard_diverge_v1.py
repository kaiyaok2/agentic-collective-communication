"""VERY HARD divergence problems, batch v1.

Goal: problems where SorcarCCL (open ReAct loop, ~30 steps, sees gate
errors) genuinely beats OverlayCCL (enumerate K=5 -> implement -> refine
top-2 by sim, R=3) under an IDENTICAL correctness gate. The divergence
must come from the SEARCH SHAPE, not a gate asymmetry.

Search-shape levers these target:
  (L1) top-2 cut: the optimal seed looks mediocre after a cold one-shot
       implementation, so enumerate-refine discards it; only sustained
       iteration from that seed reaches the optimum.
  (L2) refinement depth: the optimum needs > R sequential dependent
       transforms from ANY seed; a fixed R=3 can't reach it.
  (L3) route-around-rejection: the correct optimum is tricky to implement
       (subtle broadcast / mixed-op / index math), so cold one-shot
       implementations fail the gate and OverlayCCL falls back to
       baseline, while an iterating agent repairs the error.
  (L4) composition: the optimum chains DISTINCT rewrite families in
       sequence (e.g. dead-branch elimination THEN linear fold THEN
       slab fusion); OverlayCCL enumerates one structural idea per
       strategy and does not compose three.

Every problem: baseline is textbook (many collectives), reference is the
mathematically exact value, and there EXISTS a correct low-collective
rewrite. No problem statement names its optimal rewrite. Coefficients
are chosen so the optimum is a genuine algebraic identity (exact in fp32;
the fair gate decides bf16 admissibility symmetrically for both sides).
"""
import torch
from .problems import CollectiveProblem, register_problem


def _reg(name, sig_args, doc, ref_fn, gen_fn, builtin_code, call_args=None):
    sig = (f"def {name}_fn({sig_args}, rank, world_size, num_devices,\n"
           f"                 cores_per_device, xm, torch, num_nodes=1):")

    def _call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
        vals = [args[a] for a in (call_args or [sig_args.split(",")[0].strip()])]
        return fn(*vals, r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

    register_problem(CollectiveProblem(
        name=name, display_name=name, evolved_fn_name=f"{name}_fn",
        signature=sig, signature_doc=doc, reference_fn=ref_fn,
        generate_test_case=gen_fn, call_candidate=_call,
        builtin_templates={name: builtin_code}))


# ---------------------------------------------------------------------------
# HD1 (L4 composition + L1): staged_dead_then_fold
# A chain of 12 scaled ARs of locally-derived payloads, but 6 of them are
# multiplied by coefficients that sum, in matched +/- pairs, to exactly
# zero (dead), and the surviving 6 are a linear combination that folds to
# ONE AR of a locally pre-combined payload. Optimum: prove the 6 dead
# terms cancel (F3) AND fold the rest (F1) -> 1 AR. A single structural
# idea ("stack them") reaches ~ (all 12) or (naive fold of 12) but not the
# dead-term elimination; you must SEE the cancellation, which needs
# iteration/inspection. Baseline: 12 ARs.
# ---------------------------------------------------------------------------
def _mk_staged_dead_then_fold(name, N=8192):
    # 12 payload transforms t_i(x); coef c_i. Dead pairs: (c_k, -c_k) on the
    # SAME transform so they cancel exactly. Survivors fold linearly.
    # transforms: t_i = (i+1) * x  (all locally-scaled versions of x)
    coefs = [2.0, -2.0, 5.0, 3.0, -3.0, 1.0, 7.0, -7.0, 4.0, 0.5, -0.5, 6.0]
    # net multiplier on AR(x):  sum_i c_i*(i+1)
    net = sum(c * (i + 1) for i, c in enumerate(coefs))

    def _ref(inputs, world_size):
        ax = sum(inp['x'] for inp in inputs)
        return [(net * ax).clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(N) * (0.5 + 0.1 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    coefs = {coefs!r}",
            "    acc = None",
            "    for i, c in enumerate(coefs):",
            "        term = c * xm.all_reduce(xm.REDUCE_SUM, (i + 1) * x)",
            "        acc = term if acc is None else acc + term",
            "    return acc"]
    _reg(name, "x", f"Local x ({N},). Baseline issues {len(coefs)} scaled "
         f"inline SUM-ARs of (i+1)*x. Many terms cancel or fold.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD2 (L3 tricky-impl + L2 depth): perblock_mixed_fold_3d
# A (16, 96) tensor. Baseline issues, per row m: AR_SUM(row) + 2*AR_MAX(row)
# - AR_MIN(row) => 3*16 = 48 collectives. Optimum: 3 full-tensor collectives
# (one SUM, one MAX, one MIN over the whole tensor) then combine with the
# right per-element broadcast. Tricky because MAX/MIN don't fold like SUM
# (can't stack a MAX and a SUM into one AR), and the row->full reshape must
# preserve semantics. Cold one-shot impls tend to get the reshape or the
# mixed-op combination wrong -> gate reject -> OverlayCCL top-2 may starve.
# ---------------------------------------------------------------------------
def _mk_perblock_mixed_fold(name, M=16, N=96):
    def _ref(inputs, world_size):
        xs = [inp['x'] for inp in inputs]
        s = sum(xs)
        mx = xs[0].clone(); mn = xs[0].clone()
        for x in xs[1:]:
            mx = torch.maximum(mx, x); mn = torch.minimum(mn, x)
        return [(s + 2.0 * mx - mn).clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(M, N) * (r + 1)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    out = []",
            "    for m in range(x.shape[0]):",
            "        s = xm.all_reduce(xm.REDUCE_SUM, x[m])",
            "        mx = xm.all_reduce(xm.REDUCE_MAX, x[m])",
            "        mn = xm.all_reduce(xm.REDUCE_MIN, x[m])",
            "        out.append(s + 2.0 * mx - mn)",
            "    return torch.stack(out, dim=0)"]
    _reg(name, "x", f"Local x ({M}, {N}). Baseline per row m: "
         f"AR_SUM(x[m]) + 2*AR_MAX(x[m]) - AR_MIN(x[m])  ({3*M} collectives).",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD3 (L1 deceptive-seed + L2 depth): reduce_scatter_ladder
# Baseline: y = AR(x) then take a rank-local slice of size N/world of y, then
# AR that slice again, etc. -- a 4-level ladder where each level's only
# consumer is a rank-local shard. Optimum collapses the whole ladder to a
# single reduce_scatter (the AR-then-local-slice identity) + minimal
# recombine, which the sim prices far below the 4 ARs. The FIRST structural
# strategy that reaches RS looks WORSE after one shot (RS edge cases), so
# top-2-by-initial-sim tends to keep the "stacked AR" seeds and refine those
# to a floor that's still above the RS optimum. Needs iteration to make RS
# correct AND cheap.
# ---------------------------------------------------------------------------
def _mk_rs_ladder(name, N=4096, levels=4):
    def _ref(inputs, world_size):
        # y = sum_i x_i over the full vector; the returned value is the full
        # reduced vector scaled by 'levels' worth of identity (each level is
        # AR of the same running sum on disjoint shards, recombined to full).
        ax = sum(inp['x'] for inp in inputs)
        return [ax.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        # N divisible by world for clean RS
        pra = [{'x': torch.randn(N) * (0.3 + 0.05 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    # Baseline: full AR, then re-AR the (now replicated) running",
            f"    # sum {levels - 1} more times. Since y is identical on all",
            "    # ranks, AR(y / world_size) == y, so each extra level is an",
            "    # identity. Net result == AR(x); the extra ARs are redundant.",
            f"    y = xm.all_reduce(xm.REDUCE_SUM, x)",
            f"    for _ in range({levels - 1}):",
            "        y = xm.all_reduce(xm.REDUCE_SUM, y / world_size)",
            "    return y"]
    _reg(name, "x", f"Local x ({N},), world divides N. Baseline: full AR then "
         f"{levels-1} redundant re-ARs of the same running sum. Result = AR(x).",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD4 (L4 composition, 3 families): dead_slab_linear_combo
# A (8, 1024) slab buffer. Baseline: for each of 8 slabs, compute
# c_i * AR(slab_i) + d_i * AR(slab_i)  where for HALF the slabs c_i + d_i = 0
# (dead slab -> contributes zero), and the surviving slabs fold to one AR
# each, and all surviving slabs share the SAME underlying full-buffer AR.
# Optimum requires: (F3) drop dead slabs, (F1) fold c_i+d_i per slab, (F7)
# fuse surviving slabs into one full-buffer AR + sliced scale. Three
# families composed. 16 ARs baseline -> 1 AR optimum.
# ---------------------------------------------------------------------------
def _mk_dead_slab_linear(name, n_slabs=8, slab_N=1024):
    N = n_slabs * slab_N
    # c_i, d_i per slab; dead when c+d==0
    cd = [(1.0, 1.0), (2.0, -2.0), (3.0, 1.0), (0.5, -0.5),
          (4.0, 2.0), (1.5, -1.5), (2.0, 3.0), (5.0, -5.0)]
    mult = [c + d for c, d in cd]  # per-slab net multiplier

    def _ref(inputs, world_size):
        ax = sum(inp['x'] for inp in inputs)
        parts = [mult[i] * ax[i * slab_N:(i + 1) * slab_N] for i in range(n_slabs)]
        return [torch.cat(parts).clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(N) * (0.4 + 0.05 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    n_slabs, slab_N = {n_slabs}, {slab_N}",
            f"    cd = {cd!r}",
            "    parts = []",
            "    for i in range(n_slabs):",
            "        c, d = cd[i]",
            "        seg = x[i*slab_N:(i+1)*slab_N]",
            "        a = c * xm.all_reduce(xm.REDUCE_SUM, seg)",
            "        b = d * xm.all_reduce(xm.REDUCE_SUM, seg)",
            "        parts.append(a + b)",
            "    return torch.cat(parts)"]
    _reg(name, "x", f"Local x ({N},) = {n_slabs} slabs of {slab_N}. Baseline: "
         f"per slab c*AR(seg)+d*AR(seg) ({2*n_slabs} ARs). Some slabs c+d=0.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD5 (L2 deep chain): telescoping_partial_sums
# Baseline computes 10 prefix all-reduces: p_k = AR(sum_{j<=k} (j+1)*x)
# for k=0..9, and returns sum_k (-1)^k * p_k. The alternating sign makes it
# telescope: most terms cancel, leaving a single AR of a specific linear
# combination. Reaching it needs recognizing the telescction (deep
# reasoning), not just "stack the 10 ARs". 10 ARs baseline -> 1 AR.
# ---------------------------------------------------------------------------
def _mk_telescoping(name, N=6144, K=10):
    # p_k uses payload q_k = sum_{j=0..k} (j+1)*x = T_k * x, T_k=(k+1)(k+2)/2
    # returns sum_k (-1)^k * AR(q_k) = AR( sum_k (-1)^k T_k * x ) = net*AR(x)
    T = [ (k+1)*(k+2)//2 for k in range(K) ]
    net = sum(((-1)**k) * T[k] for k in range(K))

    def _ref(inputs, world_size):
        ax = sum(inp['x'] for inp in inputs)
        return [(float(net) * ax).clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(N) * (0.2 + 0.03 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    K = {K}",
            "    acc = None",
            "    for k in range(K):",
            "        q = None",
            "        for j in range(k + 1):",
            "            term = (j + 1) * x",
            "            q = term if q is None else q + term",
            "        pk = xm.all_reduce(xm.REDUCE_SUM, q)",
            "        signed = pk if k % 2 == 0 else -pk",
            "        acc = signed if acc is None else acc + signed",
            "    return acc"]
    _reg(name, "x", f"Local x ({N},). Baseline: {K} prefix-sum ARs p_k with "
         f"alternating signs summed. K ARs.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


def register_all():
    _mk_staged_dead_then_fold("hd1_staged_dead_fold")
    _mk_perblock_mixed_fold("hd2_perblock_mixed_fold")
    _mk_rs_ladder("hd3_rs_ladder")
    _mk_dead_slab_linear("hd4_dead_slab_linear")
    _mk_telescoping("hd5_telescoping")


register_all()
