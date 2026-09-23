"""Round 61b -- FAMILY-3 TOP-UP (non-duplicate): DATA-DEPENDENT (value-indexed)
diagonal collapse. Same telescoping mechanism as r47/r61 (a depth-D chain that
recomputes a data-dependent per-block factor g(.) of the reduced values and unscales,
folding to ONE AR(SUM,x) + one application of g), but chosen to hit the regime that
actually CONFIRMED in r61: LARGE-MAGNITUDE g-factors at HEAVY payloads. r61's small-factor
g's (var/soft/relu) tied because the fold headroom was tiny; its large-factor g's
(meansq, negrelu) confirmed. This batch stays in the large-factor regime.

g-functions (all MockTorch-traceable: only .mean()/.abs()/arith):
  - meansq   : 1 + mean(block^2)                 [proven r61 winner; NEW payloads only]
  - negrelu  : 1 + max(0, -mean(block))          [proven r61 winner; NEW payloads only]
  - quad     : 1 + mean(block^4)                 [NEW: very large 4th-moment factor]
  - meansq2  : 1 + 2*mean(block^2)               [NEW: distinct coefficient => distinct output]
  - absmean  : 1 + mean(|block|)                 [NEW: mean-of-abs, distinct from r47 meanabs=|mean|]
  - absmean2 : 1 + 2*mean(|block|)               [NEW: distinct coefficient]

Distinctness: the family-3 reference depends on (g-function, payload, NBLOCK), NOT on depth.
Every (g, payload) pair here is new vs the confirmed r47/r61 set: meansq/negrelu appear only
at payloads NOT previously registered (768/1024/512), and quad/meansq2/absmean/absmean2 are
new g-functions. The prescreen's md5 cross-check (vs all registered problems) is the guard.
"""
import torch  # noqa: F401
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
    L = [f"{indent}f = []", f"{indent}for b in range(B):",
         f"{indent}    sb = s[b*S:(b+1)*S]"]
    if kind == "meansq":
        L += [f"{indent}    f.append(1.0 + (sb*sb).mean())"]
    elif kind == "negrelu":
        L += [f"{indent}    mb = -(sb.mean())",
              f"{indent}    f.append(1.0 + (mb if mb > 0 else mb*0.0))"]
    elif kind == "quad":
        L += [f"{indent}    q = sb*sb",
              f"{indent}    f.append(1.0 + (q*q).mean())"]
    elif kind == "meansq2":
        L += [f"{indent}    f.append(1.0 + 2.0*(sb*sb).mean())"]
    elif kind == "absmean":
        L += [f"{indent}    f.append(1.0 + sb.abs().mean())"]
    elif kind == "absmean2":
        L += [f"{indent}    f.append(1.0 + 2.0*sb.abs().mean())"]
    else:
        raise ValueError(kind)
    return "\n".join(L)


def _g_ref(kind, s, part):
    out = []
    for b in range(NBLOCK):
        sb = s[b * part:(b + 1) * part]
        if kind == "meansq":
            out.append(1.0 + float((sb * sb).mean()))
        elif kind == "negrelu":
            mb = -float(sb.mean())
            out.append(1.0 + (mb if mb > 0 else 0.0))
        elif kind == "quad":
            q = sb * sb
            out.append(1.0 + float((q * q).mean()))
        elif kind == "meansq2":
            out.append(1.0 + 2.0 * float((sb * sb).mean()))
        elif kind == "absmean":
            out.append(1.0 + float(sb.abs().mean()))
        elif kind == "absmean2":
            out.append(1.0 + 2.0 * float(sb.abs().mean()))
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


def _mk_dd(name, part, depth, kind, cue=True):
    ref = _dd_ref(part, depth, kind)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    gdesc = {
        "meansq": "(1 + mean of the squared entries of that block)",
        "negrelu": "(1 + max(0, -mean of that block))",
        "quad": "(1 + mean of the 4th powers of that block)",
        "meansq2": "(1 + twice the mean of the squared entries of that block)",
        "absmean": "(1 + mean of the absolute values of that block)",
        "absmean2": "(1 + twice the mean of the absolute values of that block)",
    }[kind]
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, with each block b "
           f"scaled by {gdesc}.")
    _reg(name, doc, ref, gen, _dd_code(name, part, depth, kind))


def register_all():
    # Proven r61 winners at NEW payloads (distinct md5 via length).
    _mk_dd("r61b_dd_meansq_p768_d8", 768, 8, "meansq")
    _mk_dd("r61b_dd_meansq_p1024_d8", 1024, 8, "meansq")
    _mk_dd("r61b_dd_negrelu_p512_d8", 512, 8, "negrelu")
    _mk_dd("r61b_dd_negrelu_p768_d8", 768, 8, "negrelu")
    _mk_dd("r61b_dd_negrelu_p1024_d8", 1024, 8, "negrelu")
    # NEW large-factor g-functions at heavy payloads.
    _mk_dd("r61b_dd_quad_p512_d8", 512, 8, "quad")
    _mk_dd("r61b_dd_quad_p768_d8", 768, 8, "quad")
    _mk_dd("r61b_dd_meansq2_p512_d8", 512, 8, "meansq2")
    _mk_dd("r61b_dd_meansq2_p768_d8", 768, 8, "meansq2")
    _mk_dd("r61b_dd_absmean_p512_d8", 512, 8, "absmean")
    _mk_dd("r61b_dd_absmean_p768_d8", 768, 8, "absmean")
    _mk_dd("r61b_dd_absmean2_p512_d8", 512, 8, "absmean2")
    _mk_dd("r61b_dd_absmean2_p768_d8", 768, 8, "absmean2")


register_all()
