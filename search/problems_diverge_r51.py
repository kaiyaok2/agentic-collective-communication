"""Round 51 -- FAM-4 RESCUE: DEGENERATE (NOT a trap). Kept as a documented negative.

VERDICT (2026-09-21, faithful pre-screen at W=224): this design does NOT trap and was NOT
run on cloud. The zero-sum additive bias beta[r]=(r-(W-1)/2)*G satisfies sum_r beta[r]=0, so
the bias CANCELS in every AR(SUM). The intermediate buffers are only *transiently*
heterogeneous -- the reduction erases the heterogeneity in the same step -- so the entire
depth-D chain collapses algebraically to `W * all_reduce(SUM, x)`. Confirmed empirically: the
one-line fold `return world_size * xm.all_reduce(xm.REDUCE_SUM, x)` PASSES the fp32 gate
(sim 5160us, dispatch floor). Overlay's obvious guess therefore succeeds -> no trap -> would
tie by construction. LAW: additive per-rank injection that cancels globally can never trap;
cancellation IS the fold. A trap needs a per-rank factor that SURVIVES the reduction
(multiplicative fam-1, selective fam-2, dependent-diagonal fam-3). Original design note below.

---
Original (pre-verdict) design intent -- DEPENDENT rank-heterogeneous accumulation chain.

r49/r50 (parallel fusion) TIED: a parallel algebraic identity (AR-linearity, concat)
is one-shot-foldable by BOTH pipelines. To make a trap we need Overlay's obvious guess
to FAIL the fp32 gate. This round combines the two ingredients that DO trap:
  (i) rank-HETEROGENEOUS intermediates (so AR is a genuine reduction, not identity), and
  (ii) a DEPENDENT chain (so the fold requires a non-obvious telescoping insight),
but on a NEW axis vs fam-1/2/3: a per-rank ADDITIVE bias injected INTO the chain that
cancels globally.

Mechanism: D dependent stages. At stage t, each rank adds a rank-dependent bias
beta[r] = (r - (W-1)/2) * gamma  (a ZERO-SUM-over-ranks additive term, per block),
then AR(SUM). Because sum_r beta[r] = 0, the biases CANCEL in the reduction, so the
net is still AR(SUM,x) telescoped -- but the INTERMEDIATE buffers are rank-heterogeneous
(each rank holds x_r + beta[r]), so Overlay's "post-AR data is identical -> subsequent
AR is identity" shortcut is FALSE, and the naive AR-only guess also fails because the
per-stage scale c!=1. To keep it bounded and gate-passing we scale by a per-block
STATIC factor a[b] each stage and unscale by /(W*a[b]) -- same telescoping skeleton as
fam-3 but with the ADDITIVE zero-sum injection as the distinct trap ingredient.

NOTE: r14 (pure additive zero-sum) TIED earlier, but that was PARALLEL/single-stage and
Overlay was MORE reliable. Here the additive term is embedded in a DEPTH-D dependent
chain with a per-block scale, matching the fam-3 depth-8 structure that traps. This
tests whether zero-sum additive injection + depth is a NEW trap or still escapes.

All SUM -> gate-safe. Pre-screened at W=224 before any cloud run.
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


def _gen(world_size, seed, part):
    torch.manual_seed(seed)
    N = NBLOCK * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _code(name, part, depth, gamma):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}; W = world_size; G = {gamma}",
         "    a = [1.0 + 0.25*(b % 4) for b in range(B)]",
         "    beta = (rank - (W - 1)/2.0) * G",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += ["    buf = s.clone()",
              "    for b in range(B):",
              "        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * a[b] + beta",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            L += ["    for b in range(B):",
                  "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (W * a[b])"]
        else:
            L += ["    for b in range(B):",
                  "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / a[b]"]
        L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _ref(part, depth, gamma):
    def _r(inputs, world_size):
        # beta[r] = (r-(W-1)/2)*G sums to 0 over r, so each stage's AR(SUM, s*a+beta)
        # = a*(sum_r s) + sum_r beta = a*(W*s_shared) ... but s starts as AR(SUM,x)
        # which is already shared across ranks. Compute exactly by simulation.
        W = world_size
        xs = [inp['x'] for inp in inputs]
        s_shared = sum(xs)  # AR(SUM,x); identical on every rank
        B = NBLOCK
        a = [1.0 + 0.25 * (b % 4) for b in range(B)]
        betas = [(r - (W - 1) / 2.0) * gamma for r in range(W)]
        s = [s_shared.clone() for _ in range(W)]  # per-rank copies (all equal here)
        for st in range(depth - 1):
            last = (st == depth - 2)
            bufs = []
            for r in range(W):
                buf = s[r].clone()
                for b in range(B):
                    buf[b * part:(b + 1) * part] = s[r][b * part:(b + 1) * part] * a[b] + betas[r]
                bufs.append(buf)
            acc = sum(bufs)  # AR(SUM)
            newvals = []
            for r in range(W):
                v = acc.clone()
                if not last:
                    for b in range(B):
                        v[b * part:(b + 1) * part] = v[b * part:(b + 1) * part] / (W * a[b])
                else:
                    for b in range(B):
                        v[b * part:(b + 1) * part] = v[b * part:(b + 1) * part] / a[b]
                newvals.append(v)
            s = newvals
        return [s[r].clone() for r in range(W)]
    return _r


def _mk(name, part, depth, gamma, cue):
    ref = _ref(part, depth, gamma)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Each stage adds a rank-dependent bias (rank-(W-1)/2)*{gamma} that cancels across "
           f"ranks, with per-block scale a[b]=1+0.25*(b mod 4). Final result = the elementwise "
           f"SUM of x across ranks, with block b scaled by a[b].")
    _reg(name, doc, ref, gen, _code(name, part, depth, gamma))


def register_all():
    _mk("r51_zbias_d8_g05",  256, 8, 0.05, True)
    _mk("r51_zbias_d8_g10",  256, 8, 0.10, True)
    _mk("r51_zbias_d6_g05",  256, 6, 0.05, True)
    _mk("r51_zbias_d8_g05_res", 256, 8, 0.05, False)


register_all()
