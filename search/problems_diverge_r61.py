"""Round 61 -- FAMILY-3 EXPANSION (non-duplicate): DATA-DEPENDENT (value-indexed)
diagonal collapse, with NEW scale functions g(.) + new depths/payloads, plus a new
cross-collective RS+AG==AR point.

Family-3 mechanism (unchanged): the per-block diagonal factor f[b] is a DATA-DEPENDENT
function g(.) of the REDUCED values AR(SUM,x), recomputed each stage. The depth-D chain
recomputes g and unscales, telescoping to ONE AR(SUM,x) + one application of g. Overlay
must recognize the deep data-dependent recompute is stable across the chain (each
intermediate returns the plain sum); Kiss reaches the value-dependent telescope by
iteration.

DISTINCT from confirmed r47/r48 (meanabs/relu/topk/square/meansq/absdev/shift/halfabs).
NEW g-functions here:
  - var:      f[b] = 1 + (mean of squares - square of mean) of the block   [block variance]
  - soft:     f[b] = 1 + 0.5 * mean / (1 + |mean|)                          [saturating]
  - negrelu:  f[b] = 1 + max(0, -mean)                                      [negative-side relu]
plus NEW (proven-fn, depth/payload) combos not registered before: meansq@d6, meansq@p512,
relu@p384. All g's use only .mean()/.abs()/arith (MockTorch-traceable; .norm() is avoided
because it broke tracing in r47). The cross-collective point r61_xc_r5 extends the
RS+AG==AR ladder past the confirmed r47_xc_r4. No output duplicates an existing family-3
problem.
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


def _g_code(kind, indent="    "):
    L = []
    if kind == "var":
        L += [f"{indent}f = []",
              f"{indent}for b in range(B):",
              f"{indent}    sb = s[b*S:(b+1)*S]",
              f"{indent}    f.append(1.0 + (sb*sb).mean() - sb.mean()*sb.mean())"]
    elif kind == "soft":
        L += [f"{indent}f = []",
              f"{indent}for b in range(B):",
              f"{indent}    mb = s[b*S:(b+1)*S].mean()",
              f"{indent}    f.append(1.0 + 0.5 * mb / (1.0 + mb.abs()))"]
    elif kind == "negrelu":
        L += [f"{indent}f = []",
              f"{indent}for b in range(B):",
              f"{indent}    mb = -(s[b*S:(b+1)*S].mean())",
              f"{indent}    f.append(1.0 + (mb if mb > 0 else mb*0.0))"]
    elif kind == "meansq":
        L += [f"{indent}f = []",
              f"{indent}for b in range(B):",
              f"{indent}    sb = s[b*S:(b+1)*S]",
              f"{indent}    f.append(1.0 + (sb*sb).mean())"]
    elif kind == "relu":
        L += [f"{indent}f = []",
              f"{indent}for b in range(B):",
              f"{indent}    mb = s[b*S:(b+1)*S].mean()",
              f"{indent}    f.append(1.0 + (mb if mb > 0 else mb*0.0))"]
    else:
        raise ValueError(kind)
    return "\n".join(L)


def _g_ref(kind, s, part):
    B = NBLOCK
    out = []
    for b in range(B):
        sb = s[b * part:(b + 1) * part]
        if kind == "var":
            out.append(1.0 + float((sb * sb).mean()) - float(sb.mean()) * float(sb.mean()))
        elif kind == "soft":
            mb = float(sb.mean())
            out.append(1.0 + 0.5 * mb / (1.0 + abs(mb)))
        elif kind == "negrelu":
            mb = -float(sb.mean())
            out.append(1.0 + (mb if mb > 0 else 0.0))
        elif kind == "meansq":
            out.append(1.0 + float((sb * sb).mean()))
        elif kind == "relu":
            mb = float(sb.mean())
            out.append(1.0 + (mb if mb > 0 else 0.0))
        else:
            raise ValueError(kind)
    return out


def _dd_code(name, part, depth, kind):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += [_g_code(kind, indent="    ")]
        L += ["    buf = s.clone()",
              "    for b in range(B):",
              "        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            L += ["    for b in range(B):",
                  "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])"]
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
        "var": "(1 + block variance = mean of squares minus square of mean)",
        "soft": "(1 + 0.5*m/(1+|m|) where m is the block mean)",
        "negrelu": "(1 + max(0, -mean of that block))",
        "meansq": "(1 + mean of the squared entries of that block)",
        "relu": "(1 + max(0, mean of that block))",
    }[kind]
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, with each block b "
           f"scaled by {gdesc}.")
    _reg(name, doc, ref, gen, _dd_code(name, part, depth, kind))


def _xc_code(name, rounds):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         "    s = x"]
    for _ in range(rounds):
        L += ["    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scale=1.0, scatter_dim=0,",
              "                           shard_count=world_size)",
              "    rs = rs * 1.0",
              "    s = xm.all_gather(rs, dim=0)"]
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
           f"Final result = the elementwise SUM of x across ranks, scaled by world_size^{rounds - 1}.")
    _reg(name, doc, ref, gen, _xc_code(name, rounds))


def register_all():
    # DISTINCT-OUTPUT candidates only: the family-3 reference depends on (g-function,
    # payload), NOT on depth -- so depth-only variants share an md5 and are dropped.
    # Three NEW g-functions (var/soft/negrelu) x payloads + one proven-fn new payload.
    _mk_dd("r61_dd_var_d8",         256, 8, "var",     True)
    _mk_dd("r61_dd_soft_d8",        256, 8, "soft",    True)
    _mk_dd("r61_dd_negrelu_d8",     256, 8, "negrelu", True)
    _mk_dd("r61_dd_var_p384_d8",    384, 8, "var",     True)
    _mk_dd("r61_dd_soft_p512_d8",   512, 8, "soft",    True)
    _mk_dd("r61_dd_negrelu_p384_d8", 384, 8, "negrelu", True)
    _mk_dd("r61_dd_meansq_p512_d8", 512, 8, "meansq",  True)
    _mk_dd("r61_dd_relu_p384_d8",   384, 8, "relu",    True)


register_all()
