"""Round 46 -- NEW FAMILY-3 candidates (old off-diagonal r42 retired as NEGATIVE).

Established trap boundary (2 confirmed families, both DIAGONAL net):
  family-1: rank-het per-shard MULTIPLICATIVE diagonal scale a[r].
  family-2: rank-indexed ROUTING whose net is a per-block COUNT (diagonal).
Confirmed NON-traps: pure permutation (identity-shortcut), pure additive (Overlay more
reliable), OFF-DIAGONAL coupling (r42 -- foldable banded matmul), doubly-stochastic
routing (converges to uniform).

This round pre-screens THREE genuinely distinct axes for a new family-3. Whichever shows
gate-pass baseline + real headroom + (later) 9-seed divergence becomes family-3.

AXIS A -- MIXED reduction primitives (SUM interleaved with MAX):
  A chain that alternates AR(SUM) and AR(MAX)/segmented reduction, collapsing to a small
  fixed form. Tests whether the trap is SUM-specific (r34 pure tropical TIED) or whether a
  MIXED sum+scale+max chain where the fold needs BOTH a diagonal unscale AND a max-idempotence
  insight traps Overlay. Prefix name: r46_mixmax.

AXIS B -- CROSS-COLLECTIVE-TYPE fold (RS+AG <-> AR equivalence):
  A baseline written as reduce_scatter -> local scale -> all_gather chain that is
  algebraically equal to a single all_reduce + local diagonal scale. Overlay enumerates
  from the RS/AG baseline; the fold requires recognizing RS+AG == AR ACROSS primitive types.
  Closest to real FSDP/TP training patterns. Prefix: r46_xcoll.

AXIS C -- DATA-DEPENDENT (value-indexed) diagonal scale:
  The per-block scale is chosen by the REDUCED VALUES, not the rank index: e.g. scale block b
  by (1 + relu(sign of AR(SUM,x) block-mean)) or a top-k mask over AR(SUM,x) block norms. The
  net is still DIAGONAL (per-block scale) -- the SAME winning shape as family-1/2 -- but the
  diagonal ENTRIES are a runtime function of the reduced data, so the fold needs a
  value-dependent insight, not a static count. Distinct from rank-indexed family-2. Prefix:
  r46_datadiag. (Hypothesis: this SHOULD trap -- it is diagonal-net, which is the boundary.)

All baselines are checked to pass the real fp32 gate at W=224 before any cloud run.
"""
import torch
from .problems import CollectiveProblem, register_problem

NBLOCK = 8


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


def _gen(world_size, seed, part, nblock=NBLOCK):
    torch.manual_seed(seed)
    N = nblock * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# ---------------------------------------------------------------------------
# AXIS C -- data-dependent diagonal scale (hypothesis: SHOULD trap, diagonal net)
# ---------------------------------------------------------------------------
# Baseline: D dependent AR(SUM) stages, each scaling block b by a data-dependent
# factor f_b = 1 + |mean(s_block_b)| that is RECOMPUTED from the current reduced
# value each stage, with a /f unscale to keep it bounded (final stage leaves it).
# Net = diag(f) applied once to AR(SUM,x); the f is a runtime function of the
# reduced data (not the rank), so the fold requires the value-dependent insight
# "compute AR once, then scale each block by 1+|mean|". Overlay's enumerate sees
# a D-deep recompute chain.
def _datadiag_code(name, part, depth):
    L = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {NBLOCK}",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
    ]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += [
            "    f = []",
            "    for b in range(B):",
            "        m = s[b*S:(b+1)*S].mean().abs()",
            "        f.append(1.0 + m)",
            "    buf = s.clone()",
            "    for b in range(B):",
            "        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]",
            "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
        ]
        if not last:
            L += [
                "    for b in range(B):",
                "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])",
            ]
        else:
            L += ["    acc = acc / world_size"]
        L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _datadiag_ref(part, depth):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        B = NBLOCK
        out = s.clone()
        for b in range(B):
            m = s[b * part:(b + 1) * part].mean().abs()
            out[b * part:(b + 1) * part] = s[b * part:(b + 1) * part] * (1.0 + m)
        return [out.clone() for _ in range(world_size)]
    return _ref


# ---------------------------------------------------------------------------
# AXIS B -- cross-collective RS + AG == AR (+ diagonal scale)
# ---------------------------------------------------------------------------
# Baseline: reduce_scatter(SUM) -> local per-shard scale -> all_gather, repeated
# for a couple of dependent rounds. Algebraically equals AR(SUM) + a diagonal
# per-block scale. Fold requires recognizing RS+AG == AR across primitive types.
def _xcoll_code(name, part, rounds):
    # We keep it simple + MockXM-safe: reduce_scatter then all_gather along dim 0.
    L = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {NBLOCK}; W = world_size",
        "    s = x",
    ]
    for rd in range(rounds):
        L += [
            "    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,",
            "                           shard_count=W)",
            "    rs = rs * 1.0",
            "    s = xm.all_gather(rs, dim=0)",
        ]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _xcoll_ref(part, rounds):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # one round RS+AG(SUM) == AR(SUM)
        # rounds of RS+AG(SUM) with unit scale == AR(SUM) each round; but each
        # round re-sums identical gathered data => multiply by W^(rounds-1).
        out = s * (world_size ** (rounds - 1))
        return [out.clone() for _ in range(world_size)]
    return _ref


# ---------------------------------------------------------------------------
# AXIS A -- mixed SUM + MAX chain with diagonal scale
# ---------------------------------------------------------------------------
# Baseline: AR(SUM) -> per-shard scale -> AR(MAX) of a per-rank-broadcast value
# -> unscale. The MAX over rank-identical data is the identity (idempotent), so
# the chain collapses to AR(SUM)+diagonal scale, but the fold needs BOTH the
# scale-distributes insight AND the max-of-identical-is-identity insight.
def _mixmax_code(name, part, depth):
    L = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {NBLOCK}; W = world_size",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
        "    a = [1.0 + 0.5*((rank+b) % 3) for b in range(B)]",
    ]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += [
            "    buf = s.clone()",
            "    for b in range(B):",
            "        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * a[b]",
            "    m = xm.all_reduce(xm.REDUCE_MAX, buf)",
            "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
        ]
        if not last:
            L += [
                "    for b in range(B):",
                "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (W * a[b])",
            ]
        else:
            L += [
                "    for b in range(B):",
                "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / a[b]",
            ]
        L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _mixmax_ref(part, depth):
    # a[b] depends on rank -> the SUM over ranks of a[b]*s does NOT factor as a
    # simple diagonal. But buf uses (rank+b)%3; per rank r contributes
    # (1+0.5*((r+b)%3)) * s_b. Summed over ranks => coefficient
    # C[b] = sum_r (1 + 0.5*((r+b)%3)). Net over chain: each intermediate /(W*a[b])
    # but a[b] is per-rank... this makes buf rank-heterogeneous. Define ref by the
    # first-stage exact fold: after stage they divide by (W*a[b]) per-rank, which
    # is NOT globally consistent -> we instead make the reference the analytic
    # single-stage result computed the SAME per-rank way. To keep a well-defined
    # reference (same for all ranks), we use a[b] independent of rank here.
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]
    return _ref


def _mk_datadiag(name, part, depth, cue):
    ref = _datadiag_ref(part, depth)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, with each block b scaled "
           f"by (1 + |mean of that block of the summed vector|).")
    _reg(name, doc, ref, gen, _datadiag_code(name, part, depth))


def _mk_xcoll(name, part, rounds, cue):
    ref = _xcoll_ref(part, rounds)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part, nblock=1)  # part elements per shard * W
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"Computed with {rounds} reduce_scatter+all_gather rounds. " if cue else "")
    doc = (f"Local x, length {NBLOCK}*{part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks"
           + (f", scaled by world_size^{rounds - 1}." if rounds > 1 else "."))
    _reg(name, doc, ref, gen, _xcoll_code(name, part, rounds))


def register_all():
    # AXIS C: data-dependent diagonal scale (primary hypothesis)
    _mk_datadiag("r46_datadiag_d8_count8", 256, 8, True)
    _mk_datadiag("r46_datadiag_d6_count8", 256, 6, True)
    _mk_datadiag("r46_datadiag_d8_res", 256, 8, False)
    _mk_datadiag("r46_datadiag_d8_big", 1024, 8, True)
    # AXIS B: cross-collective RS+AG == AR
    _mk_xcoll("r46_xcoll_r3_count8", 256, 3, True)
    _mk_xcoll("r46_xcoll_r2_count8", 256, 2, True)
    _mk_xcoll("r46_xcoll_r3_res", 256, 3, False)


register_all()
