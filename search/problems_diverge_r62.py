"""Round 62 -- FAMILY-4 (NEW MECHANISM): POSITIONAL-SHIFT diagonal collapse.

Families 1-3 all fold a depth-D all_reduce chain to ONE AR(SUM,x) + a per-block op that
is DIAGONAL: output block b depends only on reduced block b (fam-1 static scale a[b],
fam-2 routing count c[b], fam-3 data-dependent factor f[b]). Family-4 breaks the diagonal:
the fold's local op is ROTATIONAL --

    out[b] = a[b] * s[(b + k) mod B]

where s = AR(SUM, x) reshaped to B blocks of size S, a[b] is a static per-block scale, and
k is a fixed positive block shift. Because the local op couples block b to a DIFFERENT
block (b+k), overlay's diagonal-oriented strategy enumeration must recognize the positional
gather, not just an elementwise scale.

Telescoping (still exact, still 1-AR-collapsible): each stage does a forward shift+scale
(buf[b] = a[b]*s[(b+k)%B]; AR(buf)/W == buf since buf is identical across ranks), and every
non-final stage undoes it locally (unscale + shift-back), so the depth-D chain nets to a
SINGLE forward shift+scale. naive plain AR(SUM,x) fails the gate (shifted+scaled != sum).

Distinctness: the reference depends on (a-pattern, k, B, payload) and, crucially, on k>=1
(k=0 would be fam-1). Depth is NOT used for distinctness. The prescreen md5 cross-check vs
all registered problems is the guard.
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


def _shift_code(name, part, depth, k, a_expr):
    """Deep chain: forward shift+scale each stage, undo (unscale+shift-back) on all
    but the last stage. Nets to out[b] = a[b] * s[(b+k) % B]."""
    B = NBLOCK
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {B}; K = {k}; W = world_size",
         f"    a = [{a_expr} for b in range(B)]",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        # forward: buf[b] = a[b] * s[(b+K) % B]; AR(buf)/W == buf (buf identical across ranks)
        L += ["    buf = s.clone()",
              "    for b in range(B):",
              "        src = (b + K) % B",
              "        buf[b*S:(b+1)*S] = a[b] * s[src*S:(src+1)*S]",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
              "    acc = acc / W"]
        if not last:
            # undo: recover s so the chain nets a single application.
            # acc[b] = a[b]*s_old[(b+K)%B]  =>  s_old[m] = acc[(m-K)%B] / a[(m-K)%B]
            L += ["    nxt = acc.clone()",
                  "    for m in range(B):",
                  "        j = (m - K) % B",
                  "        nxt[m*S:(m+1)*S] = acc[j*S:(j+1)*S] / max(a[j], 1e-9)",
                  "    s = nxt"]
        else:
            L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _shift_ref(part, k, a_pyfn):
    B = NBLOCK

    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [a_pyfn(b) for b in range(B)]
        out = s.clone()
        for b in range(B):
            src = (b + k) % B
            out[b * part:(b + 1) * part] = a[b] * s[src * part:(src + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, k, a_expr, a_pyfn, adesc, cue=True):
    ref = _shift_ref(part, k, a_pyfn)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = the elementwise SUM of x across ranks, then each output block b "
           f"is set to {adesc} times the summed block at position (b+{k}) mod {NBLOCK} "
           f"(a fixed cyclic block shift by {k}).")
    _reg(name, doc, ref, gen, _shift_code(name, part, depth, k, a_expr))


# static per-block scale patterns (b in 0..7)
def _a_lin(b):
    return 1.0 + 0.5 * (b % 3)


def _a_alt(b):
    return 1.0 + 0.25 * (b % 4)


def register_all():
    _mk("r62_shift_k1_p512_d8",  512,  8, 1, "1.0 + 0.5*(b % 3)",  _a_lin, "(1 + 0.5*(b mod 3))")
    _mk("r62_shift_k1_p1024_d8", 1024, 8, 1, "1.0 + 0.5*(b % 3)",  _a_lin, "(1 + 0.5*(b mod 3))")
    _mk("r62_shift_k1_p2048_d8", 2048, 8, 1, "1.0 + 0.5*(b % 3)",  _a_lin, "(1 + 0.5*(b mod 3))")
    _mk("r62_shift_k2_p1024_d8", 1024, 8, 2, "1.0 + 0.5*(b % 3)",  _a_lin, "(1 + 0.5*(b mod 3))")
    _mk("r62_shift_k2_p2048_d8", 2048, 8, 2, "1.0 + 0.5*(b % 3)",  _a_lin, "(1 + 0.5*(b mod 3))")
    _mk("r62_shift_k3_p1024_d8", 1024, 8, 3, "1.0 + 0.5*(b % 3)",  _a_lin, "(1 + 0.5*(b mod 3))")
    _mk("r62_shift_k3_p2048_d8", 2048, 8, 3, "1.0 + 0.5*(b % 3)",  _a_lin, "(1 + 0.5*(b mod 3))")
    _mk("r62_shift_k1_alt_p2048_d8", 2048, 8, 1, "1.0 + 0.25*(b % 4)", _a_alt, "(1 + 0.25*(b mod 4))")
    _mk("r62_shift_k2_alt_p2048_d8", 2048, 8, 2, "1.0 + 0.25*(b % 4)", _a_alt, "(1 + 0.25*(b mod 4))")


register_all()
