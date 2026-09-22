"""VERY HARD divergence problems, batch v5 -- amplify the hd10 recipe.

hd10 was the ONE fair-gate divergence (1.19x). It won because its optimum
required THREE chained realizations that Sonnet 4.5 failed to get right on a
cold one-shot (all 4 of OverlayCCL's optimizing strategies failed the gate
with max_diff 3.7-4.7):

  1. all_reduce(SUM) of an ALREADY-REPLICATED tensor just multiplies by W;
  2. an interleaved `/W` cancels that W (so those stages are no-ops);
  3. the surviving effect is a per-BLOCK coefficient vector that must be
     reconstructed exactly (index r over blocks of size S).

The 12 tie problems each had a single clean optimum Sonnet one-shots. v5
keeps hd10's DNA -- a deep chain of scaled all_reduces whose true optimum is
`coeff_vector * all_reduce(SUM, x)` (ONE collective) -- but makes the fused
coefficient genuinely error-prone to reconstruct cold, and deepens the chain
so the sim headroom grows with the number of eliminated collectives:

  HD16 product_scale_chain   : coeff = PRODUCT of per-stage per-block scales
                               (5 ARs -> 1); trap = cumulative product index.
  HD17 intrablock_ramp_chain : coeff = c[r] * (1 + beta*i), i intra-block
                               (3 ARs -> 1); trap = tile-vs-repeat of a ramp.
  HD18 sign_telescope_chain  : coeff sign depends on BLOCK PARITY, magnitude
                               from a chain (4 ARs -> 1); trap = sign pattern.
  HD19 mixed_sum_max_chain   : SUM/MAX/SUM chain; MAX of a replicated tensor
                               is IDENTITY (x1), not W. The interleaved /W is
                               only cancelled by the SUM stages, so the "every
                               all_reduce scales by W" heuristic gives a factor
                               -W error (3 collectives -> 1). Its own family.
  HD20 modular_coeff_chain   : coeff[i] = base[i % m] with m NOT dividing S,
                               so the pattern is misaligned with block bounds
                               (3 ARs -> 1); trap = per-block reconstruction.

Every optimum is verified to exist and pass the SAME fp32 gate as the
baseline. All ops used are MockTorch-supported (arange, tensor, slice-assign,
elementwise, REDUCE_SUM/REDUCE_MAX).
"""
import torch
from .problems import CollectiveProblem, register_problem

S = 256  # block size (per rank there are W blocks of S -> vector is (W*S,))


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


def _gen_WS(world_size, seed=0):
    """Per-rank x of shape (W*S,)."""
    torch.manual_seed(seed)
    N = world_size * S
    pra = [{'x': torch.randn(N) * (0.25 + 0.03 * r)} for r in range(world_size)]
    return pra


def _stotal(inputs):
    """sum_r x_r  == all_reduce(SUM, x), shape (W*S,)."""
    return sum(inp['x'] for inp in inputs)


# ---------------------------------------------------------------------------
# HD16 -- product-of-scales chain. 5 all_reduces collapse to coeff * 1 AR,
# coeff[block r] = prod over 4 stages of that stage's per-block scale.
# ---------------------------------------------------------------------------
# stage scales a_j[r], j=0..3 (deterministic functions of r).
def _a16(j, r):
    return [1.0 + 0.3 * (r % 4),
            0.5 + 0.2 * ((r + 1) % 3),
            0.75 + 0.1 * (r % 5),
            1.25 - 0.05 * (r % 6)][j]


def _mk_hd16(name):
    def _ref(inputs, world_size):
        st = _stotal(inputs)                       # (W*S,)
        coeff = torch.empty(world_size * S)
        for r in range(world_size):
            p = 1.0
            for j in range(4):
                p *= _a16(j, r)
            coeff[r * S:(r + 1) * S] = p
        out = coeff * st
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_WS(world_size, seed)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}; W = world_size",
            "    A = [[1.0 + 0.3*(r%4) for r in range(W)],",
            "         [0.5 + 0.2*((r+1)%3) for r in range(W)],",
            "         [0.75 + 0.1*(r%5) for r in range(W)],",
            "         [1.25 - 0.05*(r%6) for r in range(W)]]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)          # (W*S,)",
            "    for j in range(4):",
            "        buf = s.clone()",
            "        for r in range(W):",
            "            buf[r*S:(r+1)*S] = A[j][r] * s[r*S:(r+1)*S] / W",
            "        s = xm.all_reduce(xm.REDUCE_SUM, buf)     # replicated -> W*buf",
            "    return s"]
    _reg(name,
         f"Local x is (world*S,), S={S}. Baseline: all_reduce(SUM), then 4 "
         f"stages that per-block scale by A[j][r], divide by world, and "
         f"all_reduce(SUM) again. Return the final (world*S,) vector "
         f"(identical on every rank).",
         _ref, _gen, "\n".join(body) + "\n")


# ---------------------------------------------------------------------------
# HD17 -- intra-block ramp. coeff[block r, pos i] = c[r] * (1 + beta*i).
# 3 all_reduces -> 1. Trap: reconstruct the ramp tiled over W blocks correctly.
# ---------------------------------------------------------------------------
def _mk_hd17(name, beta=0.01):
    def _ref(inputs, world_size):
        st = _stotal(inputs)
        ramp = 1.0 + beta * torch.arange(S, dtype=torch.float32)   # (S,)
        coeff = torch.empty(world_size * S)
        for r in range(world_size):
            c = 0.5 + 0.25 * (r % 4)
            coeff[r * S:(r + 1) * S] = c * ramp
        out = coeff * st
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_WS(world_size, seed)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}; W = world_size; beta = {beta}",
            "    ramp = 1.0 + beta * torch.arange(S, dtype=x.dtype)   # (S,)",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    for _stage in range(2):",
            "        buf = s.clone()",
            "        for r in range(W):",
            "            c = 0.5 + 0.25*(r%4)",
            "            buf[r*S:(r+1)*S] = c * ramp * s[r*S:(r+1)*S] / W",
            "        s = xm.all_reduce(xm.REDUCE_SUM, buf)",
            "    # two stages each multiply block r by (c[r]*ramp); the net",
            "    # per-block factor is (c[r]*ramp)**2 ... NO: stage2 re-reads",
            "    # s, so factor is applied ONCE per stage cumulatively.",
            "    return s"]
    # NOTE: with 2 identical stages the net factor is (c*ramp)^2. Encode that
    # in the reference so baseline == ref. Recompute ref accordingly below.

    def _ref2(inputs, world_size):
        st = _stotal(inputs)
        ramp = 1.0 + beta * torch.arange(S, dtype=torch.float32)
        coeff = torch.empty(world_size * S)
        for r in range(world_size):
            c = 0.5 + 0.25 * (r % 4)
            factor = (c * ramp) ** 2       # two cumulative stages
            coeff[r * S:(r + 1) * S] = factor
        out = coeff * st
        return [out.clone() for _ in range(world_size)]

    def _gen2(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_WS(world_size, seed)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref2(pra, world_size)}

    _reg(name,
         f"Local x is (world*S,), S={S}. Baseline: all_reduce(SUM), then 2 "
         f"stages that per-block scale by c[r]*(1+beta*i) (i = intra-block "
         f"position), divide by world, and all_reduce(SUM). Return the final "
         f"(world*S,) vector (identical on every rank).",
         _ref2, _gen2, "\n".join(body) + "\n")


# ---------------------------------------------------------------------------
# HD18 -- sign-telescope. coeff sign = block parity (+ even, - odd), magnitude
# from a 3-stage scaled chain. 4 all_reduces -> 1. Trap: the sign pattern.
# ---------------------------------------------------------------------------
def _mk_hd18(name):
    def _mag(r):
        return (1.0 + 0.2 * (r % 3)) * (0.8 + 0.1 * (r % 4)) * (1.1 - 0.05 * (r % 5))

    def _ref(inputs, world_size):
        st = _stotal(inputs)
        coeff = torch.empty(world_size * S)
        for r in range(world_size):
            sign = 1.0 if (r % 2 == 0) else -1.0
            coeff[r * S:(r + 1) * S] = sign * _mag(r)
        out = coeff * st
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_WS(world_size, seed)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}; W = world_size",
            "    M = [[1.0 + 0.2*(r%3) for r in range(W)],",
            "         [0.8 + 0.1*(r%4) for r in range(W)],",
            "         [1.1 - 0.05*(r%5) for r in range(W)]]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    for j in range(3):",
            "        buf = s.clone()",
            "        for r in range(W):",
            "            buf[r*S:(r+1)*S] = M[j][r] * s[r*S:(r+1)*S] / W",
            "        s = xm.all_reduce(xm.REDUCE_SUM, buf)",
            "    # apply the per-block sign (+ even blocks, - odd blocks)",
            "    buf = s.clone()",
            "    for r in range(W):",
            "        sgn = 1.0 if (r % 2 == 0) else -1.0",
            "        buf[r*S:(r+1)*S] = sgn * s[r*S:(r+1)*S]",
            "    return buf"]
    _reg(name,
         f"Local x is (world*S,), S={S}. Baseline: all_reduce(SUM), then 3 "
         f"stages that per-block scale by M[j][r], divide by world, and "
         f"all_reduce(SUM); finally negate the ODD blocks. Return the final "
         f"(world*S,) vector (identical on every rank).",
         _ref, _gen, "\n".join(body) + "\n")


# ---------------------------------------------------------------------------
# HD19 -- mixed SUM/MAX chain. The KILLER trap: MAX of a replicated tensor is
# IDENTITY (x1), NOT W. The /W inside the MAX stage is only cancelled by the
# SUM stages, so "every all_reduce scales by W" gives a factor error.
#   s1 = AR(SUM, x)                      = S_total          (genuine reduce)
#   buf= c[r]*s1/W ;  s2 = AR(MAX, buf)  = c[r]*s1/W        (MAX -> identity)
#   s3 = AR(SUM, s2)                     = W*s2 = c[r]*s1    (SUM -> xW)
# Net coeff[block r] = c[r]. Optimum: coeff * AR(SUM, x), ONE collective.
# ---------------------------------------------------------------------------
def _mk_hd19(name):
    def _c(r):
        return 1.0 + 0.5 * (r % 3)

    def _ref(inputs, world_size):
        st = _stotal(inputs)
        coeff = torch.empty(world_size * S)
        for r in range(world_size):
            coeff[r * S:(r + 1) * S] = _c(r)
        out = coeff * st
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_WS(world_size, seed)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}; W = world_size",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)         # S_total (W*S,)",
            "    buf = s1.clone()",
            "    for r in range(W):",
            "        c = 1.0 + 0.5*(r%3)",
            "        buf[r*S:(r+1)*S] = c * s1[r*S:(r+1)*S] / W",
            "    s2 = xm.all_reduce(xm.REDUCE_MAX, buf)       # replicated -> identity",
            "    s3 = xm.all_reduce(xm.REDUCE_SUM, s2)        # replicated -> W*s2",
            "    return s3"]
    _reg(name,
         f"Local x is (world*S,), S={S}. Baseline: all_reduce(SUM); per-block "
         f"scale by c[r] and divide by world; all_reduce(MAX); all_reduce(SUM). "
         f"Return the final (world*S,) vector (identical on every rank). Note "
         f"the reduce ops differ (SUM vs MAX).",
         _ref, _gen, "\n".join(body) + "\n")


# ---------------------------------------------------------------------------
# HD20 -- modular coefficient misaligned with block boundaries.
# coeff[i] = base[i % m] with m NOT dividing S. 3 ARs -> 1. Trap: the pattern
# must be tiled over the WHOLE (W*S,) vector, not reconstructed per block.
# ---------------------------------------------------------------------------
def _mk_hd20(name, m=7):
    base = [0.5, 1.0, 1.5, 0.75, 1.25, 0.9, 1.1]   # length m=7; 7 does NOT divide 256

    def _ref(inputs, world_size):
        st = _stotal(inputs)
        N = world_size * S
        idx = torch.arange(N) % m
        coeff = torch.tensor([base[int(k)] for k in idx], dtype=torch.float32)
        # two cumulative stages -> square
        out = (coeff ** 2) * st
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_WS(world_size, seed)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}; W = world_size; m = {m}",
            f"    base = {base}",
            "    N = W * S",
            "    idx = torch.arange(N) % m",
            "    coeff = torch.tensor([base[int(k)] for k in idx], dtype=x.dtype)",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    for _stage in range(2):",
            "        buf = coeff * s / W",
            "        s = xm.all_reduce(xm.REDUCE_SUM, buf)",
            "    return s"]
    _reg(name,
         f"Local x is (world*S,), S={S}. Baseline: all_reduce(SUM), then 2 "
         f"stages that scale position i by base[i % {m}] (a repeating pattern "
         f"whose period {m} does NOT divide S), divide by world, and "
         f"all_reduce(SUM). Return the final (world*S,) vector.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_hd16("hd16_product_scale_chain")
    # HD17 (intrablock_ramp_chain) NOT registered: its 1-collective optimum is
    # SLOWER in sim than the 3-AR baseline (the squared-ramp coefficient
    # construction costs more than the 2 eliminated collectives), so there is
    # no headroom to diverge on regardless of search shape.
    _mk_hd18("hd18_sign_telescope_chain")
    _mk_hd19("hd19_mixed_sum_max_chain")
    _mk_hd20("hd20_modular_coeff_chain")


register_all()
