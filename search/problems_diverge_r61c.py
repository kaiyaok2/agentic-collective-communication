"""Round 61c -- FAMILY-3 TOP-UP #2 (non-duplicate): DATA-DEPENDENT diagonal collapse,
LARGER-COEFFICIENT large-magnitude g-factors. r61b confirmed the absmean family
(1 + mean(|block|)) at median ~1.18x but its best-of-N/CI were noisy, so only ~half of the
absmean variants cleared the strict gate. This batch pushes the same proven regime harder:
bigger coefficients (3x, 4x, 5x) grow the residual gap left when overlay only PARTIALLY
folds, giving each variant an independent, stronger shot at the best-of-N + Mann-Whitney +
bootstrap-CI gate. Also re-tries the proven absmean/absmean2 kinds at the heavier p1024.

g-functions (all MockTorch-traceable: only .mean()/.abs()/arith -- NO .max()/sqrt/**frac):
  - absmean3 : 1 + 3*mean(|block|)      [NEW coefficient => distinct output]
  - absmean4 : 1 + 4*mean(|block|)      [NEW coefficient]
  - meansq3  : 1 + 3*mean(block^2)      [NEW coefficient, distinct from meansq/meansq2]
  - absmean  : 1 + mean(|block|)        [proven kind; NEW payload p1024 only]
  - absmean2 : 1 + 2*mean(|block|)      [proven kind; NEW payload p1024 only]

Distinctness: reference depends on (g-function, payload, NBLOCK), NOT on depth. Every
(g, payload) pair is new vs the confirmed r47/r61/r61b set. The prescreen md5 cross-check
(vs ALL registered problems) is the final guard.
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
    if kind == "absmean3":
        L += [f"{indent}    f.append(1.0 + 3.0*sb.abs().mean())"]
    elif kind == "absmean4":
        L += [f"{indent}    f.append(1.0 + 4.0*sb.abs().mean())"]
    elif kind == "meansq3":
        L += [f"{indent}    f.append(1.0 + 3.0*(sb*sb).mean())"]
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
        if kind == "absmean3":
            out.append(1.0 + 3.0 * float(sb.abs().mean()))
        elif kind == "absmean4":
            out.append(1.0 + 4.0 * float(sb.abs().mean()))
        elif kind == "meansq3":
            out.append(1.0 + 3.0 * float((sb * sb).mean()))
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
        "absmean3": "(1 + three times the mean of the absolute values of that block)",
        "absmean4": "(1 + four times the mean of the absolute values of that block)",
        "meansq3": "(1 + three times the mean of the squared entries of that block)",
        "absmean": "(1 + mean of the absolute values of that block)",
        "absmean2": "(1 + twice the mean of the absolute values of that block)",
    }[kind]
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, then each summed block b "
           f"is multiplied by a data-dependent factor {gdesc}.")
    _reg(name, doc, ref, gen, _dd_code(name, part, depth, kind))


def register_all():
    _mk_dd("r61c_dd_absmean3_p512_d8", 512, 8, "absmean3")
    _mk_dd("r61c_dd_absmean3_p768_d8", 768, 8, "absmean3")
    _mk_dd("r61c_dd_absmean4_p512_d8", 512, 8, "absmean4")
    _mk_dd("r61c_dd_absmean4_p768_d8", 768, 8, "absmean4")
    _mk_dd("r61c_dd_meansq3_p512_d8", 512, 8, "meansq3")
    _mk_dd("r61c_dd_meansq3_p768_d8", 768, 8, "meansq3")
    _mk_dd("r61c_dd_absmean_p1024_d8", 1024, 8, "absmean")
    _mk_dd("r61c_dd_absmean2_p1024_d8", 1024, 8, "absmean2")


register_all()
