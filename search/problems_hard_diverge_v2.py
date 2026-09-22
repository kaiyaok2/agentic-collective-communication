"""VERY HARD divergence problems, batch v2 -- lever L3 (route-around-rejection).

v1 showed that when the optimum is ONE clean structural idea (batch the
collectives, fold the linear combo), OverlayCCL's enumerate-once step reaches
it just as reliably as kiss -> tie. The remaining search-shape lever is when
the optimum is CORRECT BUT ERROR-PRONE TO IMPLEMENT: subtle shard/index math
where a cold one-shot implementation usually produces a WRONG (gate-rejected)
candidate. OverlayCCL implements each of its K strategies exactly once; a
strategy that is right-in-concept but buggy-in-first-draft is discarded and
cannot re-enter the top-2. kiss sees the correctness error text and repairs
it over successive steps.

To make this a FAIR test (identical gate for both), the optimum must:
  - exist and pass the SAME fp32 gate,
  - be meaningfully cheaper than baseline,
  - require index/shard arithmetic that is easy to get wrong on the first
    try (off-by-one shard offsets, scatter_dim choice, ring-permute pairings,
    all_gather dim + local-slice recombine).

Each problem's baseline is a correct-but-expensive loop; the doc never names
the optimal rewrite.
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
# HD6 (L3): sharded_gram_diag
# Each rank holds a column block A_r of a tall matrix A = [A_0|...|A_{W-1}]
# (shape (D, C) per rank). The baseline computes the FULL Gram diagonal
# g = diag(A^T A) summed across ranks by: for each rank r, all_gather the
# column block, then locally accumulate. Result: g_j = sum over all columns
# j of (column_j . column_j). Optimum: each rank computes its LOCAL partial
# diag (A_r column norms) -- a (C,) vector -- then ONE all_reduce(SUM) over
# the stacked per-rank partials. The trick: the AR must reduce per-rank (C,)
# vectors that must be laid out so rank r's contribution lands in the r-th
# block of the (W*C,) output. Off-by-one in the block offset or the wrong
# concat order -> wrong answer -> gate reject. Baseline: W all_gathers of
# (D,C). Optimum: 1 AR of (W*C,).
# ---------------------------------------------------------------------------
def _mk_sharded_gram_diag(name, D=128, C=8):
    def _ref(inputs, world_size):
        # rank r contributes column-norms of its (D,C) block into block r of
        # a (world*C,) vector; summed across ranks (each rank has same layout
        # so the AR just sums, but only rank r's block is nonzero pre-scatter).
        out = torch.zeros(world_size * C)
        for r, inp in enumerate(inputs):
            A = inp['x']  # (D, C)
            out[r * C:(r + 1) * C] = (A * A).sum(dim=0)
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(D, C) * (0.5 + 0.05 * r)}
               for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    # Baseline: all_gather each rank's block, recompute the whole (W*C,).
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    D, C = {D}, {C}",
            "    # Gather every rank's (D,C) block, then build the (W*C,) diag.",
            "    g = xm.all_gather(x, dim=0)  # (W*D, C)",
            "    out_parts = []",
            "    for r in range(world_size):",
            "        block = g[r*D:(r+1)*D]        # (D, C)",
            "        out_parts.append((block*block).sum(dim=0))  # (C,)",
            "    return torch.cat(out_parts)       # (W*C,)"]
    _reg(name, "x", f"Local x is this rank's (D={D}, C={C}) column block. "
         f"Return (world*C,): block r = column-norms of rank r's block. "
         f"Baseline all_gathers the (D,C) blocks and recomputes.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD7 (L3 ring): ring_shifted_accumulate
# Baseline: y_r = sum_{k=0}^{W-1} x[(r-k) mod W] i.e. every rank ends up with
# the full sum of all ranks' vectors, computed by W sequential
# collective_permute steps around a ring (shift by 1 each step, accumulate).
# Result is just AR(SUM) but the baseline expresses it as a ring. Optimum:
# ONE all_reduce(SUM). The route-around-rejection twist: the *tempting*
# structural rewrite ("do it in fewer permute steps" / "recursive doubling")
# is easy to get wrong (pairing offsets, log2 rounds, non-power-of-2 world),
# so cold one-shot recursive-doubling impls fail the gate. The clean AR is
# correct but a refiner fixated on "improve the ring" may never switch
# families. Baseline: W collective_permutes. Optimum: 1 AR.
# ---------------------------------------------------------------------------
def _mk_ring_shifted_accumulate(name, N=2048):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    # Ring accumulate: rotate neighbor's buffer W-1 times, summing.",
            "    acc = x",
            "    buf = x",
            "    for step in range(world_size - 1):",
            "        pairs = [((i + 1) % world_size, i) for i in range(world_size)]",
            "        buf = xm.collective_permute(buf, pairs)",
            "        acc = acc + buf",
            "    return acc"]
    _reg(name, "x", f"Local x ({N},). Baseline ring-accumulates via "
         f"{'world-1'} collective_permute steps so every rank gets the "
         f"all-rank sum.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD8 (L3 reduce_scatter): blockwise_reduce_then_broadcast
# Baseline: full AR(SUM) of a (W*S,) buffer, then each rank keeps only its
# own S-sized shard and zero-fills the rest, then all_gather to reassemble.
# Net identity == AR(SUM). Optimum: reduce_scatter(SUM) to get each rank's
# shard directly, then all_gather -> 1 RS + 1 AG instead of 1 AR + 1 AG.
# BUT the scorer prices RS+AG as ~ the AR, so the REAL optimum is just the
# single AR (drop the shard/reassemble round-trip entirely). The trap:
# 'reduce_scatter' is an obvious enumerate strategy, and RS scatter_dim /
# shard_count off-by-one is a classic first-draft bug -> gate reject. A
# one-shot RS impl usually fails; the clean single-AR requires SEEING that
# the whole scatter/gather round trip is dead. Baseline: 1 AR + 1 AG.
# Optimum: 1 AR.
# ---------------------------------------------------------------------------
def _mk_blockwise_rs_bcast(name, S=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (W*S,)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        N = world_size * S
        pra = [{'x': torch.randn(N) * (0.25 + 0.03 * r)}
               for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}",
            "    # Full AR, keep only this rank's shard, zero the rest, then",
            "    # all_gather the shards back into the full (W*S,) vector.",
            "    full = xm.all_reduce(xm.REDUCE_SUM, x)   # (W*S,)",
            "    shard = full[rank*S:(rank+1)*S]          # (S,)",
            "    gathered = xm.all_gather(shard, dim=0)   # (W*S,)",
            "    return gathered"]
    _reg(name, "x", f"Local x ({'world*S'},), S={S}. Baseline: full AR, slice "
         f"this rank's S-shard, all_gather shards back to full. Result = AR(x).",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD9 (L3 mixed reduce + index): topk_mask_sum
# Baseline: AR(SUM) of x, AR(MAX) of x, then per element keep x_sum where
# x_max exceeds a per-position threshold derived from AR(MAX), else a scaled
# fallback -- expressed as an elementwise select over TWO collectives plus a
# redundant third AR(SUM) of the SAME x used only inside the fallback branch
# (which, algebraically, equals the first AR(SUM)). Optimum: recognize the
# 3rd AR is the 1st (CSE across a branch) -> 2 collectives. The trap: the
# branchy select with threshold indexing is error-prone; cold one-shot
# "simplifications" tend to change the select semantics -> gate reject.
# Baseline: 3 ARs. Optimum: 2 ARs.
# ---------------------------------------------------------------------------
def _mk_topk_mask_sum(name, N=3072):
    def _ref(inputs, world_size):
        xs = [inp['x'] for inp in inputs]
        s = sum(xs)
        mx = xs[0].clone()
        for x in xs[1:]:
            mx = torch.maximum(mx, x)
        thr = mx.mean()
        out = torch.where(mx > thr, s, 0.5 * s)
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(N) * (0.4 + 0.03 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    mx = xm.all_reduce(xm.REDUCE_MAX, x)",
            "    thr = mx.mean()",
            "    # Fallback branch re-reduces the SAME x (redundant).",
            "    s_fallback = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    return torch.where(mx > thr, s, 0.5 * s_fallback)"]
    _reg(name, "x", f"Local x ({N},). Baseline: AR_SUM, AR_MAX, then a "
         f"where(mx>mean, sum, 0.5*sum) using a SECOND AR_SUM in the "
         f"fallback. 3 collectives.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD10 (L3 deep + index): staged_shard_norm_chain
# A 3-stage dependent chain each of which is individually a shardable
# collective, but the naive baseline does a full AR at every stage:
#   stage1: s1 = AR(SUM, x)                 (W*S,)
#   stage2: per-shard scale by 1/||shard||  then AR(SUM) again
#   stage3: subtract shard-mean broadcast, AR(SUM) again
# The optimum fuses to a single AR of a locally pre-scaled payload, because
# stages 2 and 3 are rank-local transforms of already-reduced data that
# commute with the sum ONLY under a specific normalization identity. This
# needs > 3 dependent transforms to derive AND correct shard indexing; a
# fixed R=3 refinement budget from a non-fused seed can't reach it, and a
# cold one-shot fused impl usually mis-handles the per-shard norm.
# Baseline: 3 ARs. Optimum: 1 AR (+ local scale).
# ---------------------------------------------------------------------------
def _mk_staged_shard_norm(name, S=256):
    # We choose the identity so the three stages collapse to a known scalar
    # multiple of AR(x). stage2 multiplies by a FIXED per-shard constant c_r
    # (not data-dependent) and stage3 adds a FIXED per-shard constant d_r,
    # both applied AFTER reduction, replicated on all ranks -> they fold into
    # one linear map on AR(x). The 'norm' framing is a decoy; the constants
    # are deterministic from shard index.
    def _consts(world_size):
        c = [1.0 + 0.5 * (r % 3) for r in range(world_size)]  # per-shard scale
        return c

    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (W*S,)
        c = _consts(world_size)
        S = inputs[0]['x'].numel() // world_size
        out = s.clone()
        # stage2: scale shard r by c[r]; stage3: (identity here) -- net is a
        # per-shard scale of the reduced vector.
        for r in range(world_size):
            out[r * S:(r + 1) * S] = c[r] * s[r * S:(r + 1) * S]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        N = world_size * S
        pra = [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}",
            "    c = [1.0 + 0.5 * (r % 3) for r in range(world_size)]",
            "    # stage 1: reduce",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    # stage 2: rebuild a per-shard-scaled buffer, reduce again",
            "    buf2 = s1.clone()",
            "    for r in range(world_size):",
            "        buf2[r*S:(r+1)*S] = c[r] * s1[r*S:(r+1)*S] / world_size",
            "    s2 = xm.all_reduce(xm.REDUCE_SUM, buf2)",
            "    # stage 3: identity reduce (adds nothing but a dispatch)",
            "    s3 = xm.all_reduce(xm.REDUCE_SUM, s2 / world_size)",
            "    return s3"]
    _reg(name, "x", f"Local x ({'world*S'},), S={S}. Baseline: 3 dependent "
         f"AR_SUM stages with a per-shard scale in the middle. Result = "
         f"per-shard-scaled AR(x).",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


def register_all():
    _mk_sharded_gram_diag("hd6_sharded_gram_diag")
    # HD7 (ring_shifted_accum) intentionally NOT registered: its baseline
    # chains dependent collective_permute calls, which the mock's two-phase
    # collect/replay cannot resolve (a permute whose input is a prior
    # permute's result reads zeros during the collect phase). HD3 already
    # covers the ring->AR structural idea via redundant all_reduces.
    _mk_blockwise_rs_bcast("hd8_blockwise_rs_bcast")
    _mk_topk_mask_sum("hd9_topk_mask_sum")
    _mk_staged_shard_norm("hd10_staged_shard_norm")


register_all()
