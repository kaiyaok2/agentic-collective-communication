"""Round 48 -- FAMILY-3 EXPANSION (more confirmed members on the confirmed axes).

Family-3 confirmed @ bo9 (r47fam3): data-dependent CONTINUOUS diagonal scale
g(reduced values) [relu_d8 1.267, meanabs_d7/d8] + cross-collective RS+AG==AR
[xc_r4 1.271]. Two sharp lessons from r47 drive this expansion:
  * CONTINUOUS scale fns trap; DISCRETE top-k selection ESCAPES (Overlay folds the
    stable mask). => only continuous g here.
  * Headroom is set by DEPTH (# dependent ARs folded away): d6~1.22, d7~1.27, d8~1.31;
    cross-collective r4~1.27, and grows with rounds. depth-10 DRIFTS past the fp32 gate
    (r47 dropped d10). => stay depth<=9, sweep more depths + deeper xcoll.
  * Which continuous fn Overlay can "see through" varies (relu 1.267 >> meanabs 1.10 at
    same depth). => add several DISTINCT continuous fns; the harder-to-recognize ones
    confirm more robustly.

New DISTINCT problems (all continuous, payload 256, depth<=9, gate+fold pre-screened):
  (F) scale-FUNCTION sweep at d8: square / meansq / mean-abs-deviation / shifted-abs / half-abs.
  (D) depth sweep on confirmed fns: relu@{d6,d9}, meanabs@{d6,d9}.
  (X) deeper cross-collective: xc_r5, xc_r6.
  (R) framing controls: square_d8_res, relu_d8_res.
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


# ---- continuous scale functions g(reduced s) -> per-block factor f[b] ------
def _g_code(kind):
    if kind == "square":
        return ["    f = [1.0 + (s[b*S:(b+1)*S].mean())**2 for b in range(B)]"]
    if kind == "meansq":
        return ["    f = [1.0 + (s[b*S:(b+1)*S]**2).mean() for b in range(B)]"]
    if kind == "absdev":
        return ["    f = []",
                "    for b in range(B):",
                "        blk = s[b*S:(b+1)*S]",
                "        f.append(1.0 + (blk - blk.mean()).abs().mean())"]
    if kind == "shift":
        return ["    f = [1.0 + (s[b*S:(b+1)*S].mean() - 0.1).abs() for b in range(B)]"]
    if kind == "halfabs":
        return ["    f = [1.0 + 0.5*s[b*S:(b+1)*S].mean().abs() for b in range(B)]"]
    if kind == "meanabs":
        return ["    f = [1.0 + s[b*S:(b+1)*S].mean().abs() for b in range(B)]"]
    if kind == "relu":
        return ["    f = []",
                "    for b in range(B):",
                "        mb = s[b*S:(b+1)*S].mean()",
                "        f.append(1.0 + (mb if mb > 0 else mb*0.0))"]
    raise ValueError(kind)


def _g_ref(kind, s, part):
    B = NBLOCK
    if kind == "square":
        return [1.0 + float(s[b * part:(b + 1) * part].mean()) ** 2 for b in range(B)]
    if kind == "meansq":
        return [1.0 + float((s[b * part:(b + 1) * part] ** 2).mean()) for b in range(B)]
    if kind == "absdev":
        out = []
        for b in range(B):
            blk = s[b * part:(b + 1) * part]
            out.append(1.0 + float((blk - blk.mean()).abs().mean()))
        return out
    if kind == "shift":
        return [1.0 + abs(float(s[b * part:(b + 1) * part].mean()) - 0.1) for b in range(B)]
    if kind == "halfabs":
        return [1.0 + 0.5 * abs(float(s[b * part:(b + 1) * part].mean())) for b in range(B)]
    if kind == "meanabs":
        return [1.0 + abs(float(s[b * part:(b + 1) * part].mean())) for b in range(B)]
    if kind == "relu":
        out = []
        for b in range(B):
            mb = float(s[b * part:(b + 1) * part].mean())
            out.append(1.0 + (mb if mb > 0 else 0.0))
        return out
    raise ValueError(kind)


def _dd_code(name, part, depth, kind):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += _g_code(kind)
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


_GDESC = {
    "square": "(1 + (mean of that block of the summed vector)^2)",
    "meansq": "(1 + mean of the squared entries of that block of the summed vector)",
    "absdev": "(1 + mean absolute deviation of that block of the summed vector)",
    "shift": "(1 + |mean of that block of the summed vector - 0.1|)",
    "halfabs": "(1 + 0.5*|mean of that block of the summed vector|)",
    "meanabs": "(1 + |mean of that block of the summed vector|)",
    "relu": "(1 + max(0, mean of that block of the summed vector))",
}


def _mk_dd(name, part, depth, kind, cue):
    ref = _dd_ref(part, depth, kind)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, with each block b "
           f"scaled by {_GDESC[kind]}.")
    _reg(name, doc, ref, gen, _dd_code(name, part, depth, kind))


# ---- cross-collective cluster (deeper) -------------------------------------
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


def _xc_ref(rounds):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [(s * (world_size ** (rounds - 1))).clone() for _ in range(world_size)]
    return _ref


def _mk_xc(name, part, rounds, cue):
    ref = _xc_ref(rounds)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part, nblock=1)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"Computed with {rounds} reduce_scatter+all_gather rounds. " if cue else "")
    doc = (f"Local x, length {NBLOCK}*{part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, "
           f"scaled by world_size^{rounds - 1}.")
    _reg(name, doc, ref, gen, _xc_code(name, rounds))


def register_all():
    # NOTE: pre-screen (prescreen_r48) dropped 4 gate-fails: relu_d9/meanabs_d9
    # (depth-9 recompute drifts past atol, max_diff ~2.3 -> depth ceiling is 8),
    # and xc_r5/xc_r6 (W^(rounds-1)=224^4 blows up float precision, max_diff
    # 481/1927 -> cross-collective can't exceed r4 at W=224). 9 kept below all
    # pass gate+fold with headroom 1.22-1.49x.
    # (F) new continuous scale functions at the strong depth 8
    _mk_dd("r48_dd_square_d8",  256, 8, "square",  True)
    _mk_dd("r48_dd_meansq_d8",  256, 8, "meansq",  True)   # hr 1.484 (heavy compute)
    _mk_dd("r48_dd_absdev_d8",  256, 8, "absdev",  True)   # hr 1.490 (heavy compute)
    _mk_dd("r48_dd_shift_d8",   256, 8, "shift",   True)
    _mk_dd("r48_dd_halfabs_d8", 256, 8, "halfabs", True)
    # (D) depth-knob variants of confirmed functions (shallower still traps?)
    _mk_dd("r48_dd_relu_d6",    256, 6, "relu",    True)
    _mk_dd("r48_dd_meanabs_d6", 256, 6, "meanabs", True)
    # (R) framing controls (result-only)
    _mk_dd("r48_dd_square_d8_res", 256, 8, "square", False)
    _mk_dd("r48_dd_relu_d8_res",   256, 8, "relu",   False)


register_all()
