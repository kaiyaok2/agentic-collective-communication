"""Round 47 -- FAMILY-3 battery: DATA-DEPENDENT (value-indexed) diagonal collapse
(+ a cross-collective RS+AG==AR robustness cluster).

Established before this round (2 confirmed families, both DIAGONAL net):
  family-1: rank-het per-shard MULTIPLICATIVE diagonal scale a[r] (STATIC constant).
  family-2: rank-INDEXED ROUTING net = per-block COUNT c[b] (STATIC, from routing overlap).

FAMILY-3 mechanism (NEW, distinct from both): the per-block diagonal scale is a
DATA-DEPENDENT function g(.) of the REDUCED VALUES -- computed at runtime from
AR(SUM,x), not from the rank index or a static constant. The baseline is a DEPTH-D
chain that RECOMPUTES g each stage and unscales, so the chain telescopes to a single
AR(SUM,x) followed by ONE application of g. Overlay's enumerate must recognize the
deep data-dependent recompute is stable across the chain (each intermediate returns
the plain sum) -- a value-dependent telescoping insight, distinct from family-1's
static-scale distribution and family-2's routing-count.

Pre-screen (r46, best-of-4, both pipelines Bedrock Sonnet-4.5):
  r46_datadiag_d8_count8 CONFIRMED even at bo4 (best 1.267/med 1.138/p .0069/CI[1.032,1.267]).
  r46_xcoll_r2_count8    CONFIRMED even at bo4 (best 1.116/p .0076/CI[1.075,1.116]).
  d6 / big / res weaker -> depth-8 + count-cue is the robust config, mirrors fam-1/2.

This round sweeps the DISTINCTNESS axes that don't duplicate each other:
  (C1) scale-FUNCTION g: mean-abs / relu-mean / norm-ratio / top-k SELECTION (MoE-like).
  (C2) DEPTH: d7 / d8 / d10.
  (C3) payload: 256 / 384.
  (C4) framing: count-cue vs result-only.
  (B)  cross-collective: RS+AG rounds == AR * W^(r-1) -- primitive-equivalence fold.

All baselines AND their ideal folds are gate-checked at W=224 before any cloud run.
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


# --------------------------------------------------------------------------
# The four data-dependent scale functions g(s) -> per-block factor f[b].
# Each is generated as inline code that runs on the reduced vector `s`, and
# mirrored EXACTLY in the reference so the gate is well-defined.
# --------------------------------------------------------------------------
TOPK = 3  # for the selection variant


def _g_code(kind, indent="    "):
    """Emit code computing list `f` (length B) from reduced vector `s`."""
    L = []
    if kind == "meanabs":
        L += [f"{indent}f = []",
              f"{indent}for b in range(B):",
              f"{indent}    f.append(1.0 + s[b*S:(b+1)*S].mean().abs())"]
    elif kind == "relu":
        L += [f"{indent}f = []",
              f"{indent}for b in range(B):",
              f"{indent}    mb = s[b*S:(b+1)*S].mean()",
              f"{indent}    f.append(1.0 + (mb if mb > 0 else mb*0.0))"]
    elif kind == "norm":
        L += [f"{indent}nb = [s[b*S:(b+1)*S].norm() for b in range(B)]",
              f"{indent}tot = sum(nb) + 1e-6",
              f"{indent}f = [1.0 + nb[b]/tot for b in range(B)]"]
    elif kind == "topk":
        L += [f"{indent}ma = [s[b*S:(b+1)*S].mean().abs() for b in range(B)]",
              f"{indent}order = sorted(range(B), key=lambda b: float(ma[b]), reverse=True)",
              f"{indent}sel = set(order[:{TOPK}])",
              f"{indent}f = [2.0 if b in sel else 1.0 for b in range(B)]"]
    else:
        raise ValueError(kind)
    return "\n".join(L)


def _g_ref(kind, s, part):
    """Python replica of g(s) -> factor tensor-safe list of floats."""
    B = NBLOCK
    if kind == "meanabs":
        return [1.0 + float(s[b * part:(b + 1) * part].mean().abs()) for b in range(B)]
    if kind == "relu":
        out = []
        for b in range(B):
            mb = float(s[b * part:(b + 1) * part].mean())
            out.append(1.0 + (mb if mb > 0 else 0.0))
        return out
    if kind == "norm":
        nb = [float(s[b * part:(b + 1) * part].norm()) for b in range(B)]
        tot = sum(nb) + 1e-6
        return [1.0 + nb[b] / tot for b in range(B)]
    if kind == "topk":
        ma = [float(s[b * part:(b + 1) * part].mean().abs()) for b in range(B)]
        order = sorted(range(B), key=lambda b: ma[b], reverse=True)
        sel = set(order[:TOPK])
        return [2.0 if b in sel else 1.0 for b in range(B)]
    raise ValueError(kind)


def _dd_code(name, part, depth, kind):
    L = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {NBLOCK}",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
    ]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += [_g_code(kind, indent="    ")]
        L += [
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


def _dd_ref(part, depth, kind):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        f = _g_ref(kind, s, part)
        out = s.clone()
        for b in range(NBLOCK):
            out[b * part:(b + 1) * part] = s[b * part:(b + 1) * part] * f[b]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk_dd(name, part, depth, kind, cue):
    ref = _dd_ref(part, depth, kind)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    gdesc = {
        "meanabs": "(1 + |mean of that block of the summed vector|)",
        "relu": "(1 + max(0, mean of that block of the summed vector))",
        "norm": "(1 + block_L2_norm / sum_of_all_block_L2_norms)",
        "topk": f"2.0 if the block is among the top-{TOPK} blocks by |mean|, else 1.0",
    }[kind]
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, with each block b "
           f"scaled by {gdesc}.")
    _reg(name, doc, ref, gen, _dd_code(name, part, depth, kind))


# --------------------------------------------------------------------------
# Cross-collective cluster (axis B): RS+AG rounds == AR * W^(rounds-1)
# --------------------------------------------------------------------------
def _xc_code(name, part, rounds):
    L = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        "    s = x",
    ]
    for _ in range(rounds):
        L += [
            "    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,",
            "                           shard_count=world_size)",
            "    rs = rs * 1.0",
            "    s = xm.all_gather(rs, dim=0)",
        ]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _xc_ref(part, rounds):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [(s * (world_size ** (rounds - 1))).clone() for _ in range(world_size)]
    return _ref


def _mk_xc(name, part, rounds, cue):
    ref = _xc_ref(part, rounds)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part, nblock=1)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"Computed with {rounds} reduce_scatter+all_gather rounds. " if cue else "")
    doc = (f"Local x, length {NBLOCK}*{part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks"
           + (f", scaled by world_size^{rounds - 1}." if rounds > 1 else "."))
    _reg(name, doc, ref, gen, _xc_code(name, part, rounds))


def register_all():
    # NOTE: pre-screen (prescreen_r47) dropped 4 candidates that FAIL the real
    # fp32 gate at W=224: norm/norm_res (.norm()+tensor-sum not traceable by
    # MockTorch -> TrackedTensor AttributeError) and meanabs_d10/topk_d10 (the
    # depth-10 recompute chain drifts past tolerance, max_diff 2.4-4.2). The 10
    # kept below all pass gate + fold with headroom 1.12-1.31x.
    # C1 -- scale-FUNCTION sweep at the confirmed depth 8 (count cue)
    _mk_dd("r47_dd_meanabs_d8", 256, 8, "meanabs", True)   # == r46 confirmed mechanism
    _mk_dd("r47_dd_relu_d8",    256, 8, "relu",    True)
    _mk_dd("r47_dd_topk_d8",    256, 8, "topk",    True)   # MoE-like top-k selection
    # C2 -- depth knob on the strongest function (same output as d8)
    _mk_dd("r47_dd_meanabs_d7",  256, 7,  "meanabs", True)
    # C3 -- payload
    _mk_dd("r47_dd_meanabs_d8_p384", 384, 8, "meanabs", True)
    _mk_dd("r47_dd_topk_d8_p384",    384, 8, "topk",    True)
    # C4 -- framing (result-only, same output as topk_d8)
    _mk_dd("r47_dd_topk_d8_res",    256, 8, "topk",    False)
    # B -- cross-collective RS+AG==AR cluster
    _mk_xc("r47_xc_r2", 256, 2, True)   # confirmed at bo4
    _mk_xc("r47_xc_r3", 256, 3, True)
    # r47_xc_r4 dropped: overlay HW-aborts at 224 ranks (reduce_scatter dim 256 not
    # divisible by shard_count 224) -> no fair warm-RT pair. Sim-pass / HW-abort.


register_all()
