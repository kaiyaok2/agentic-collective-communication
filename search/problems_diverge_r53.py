"""Round 53 -- FAMILY-4 CANDIDATE: MAX/MIN-semiring telescoping (non-linear reduction axis).

RECON (this session): collective_permute is HARD-REJECTED at world_size>64 (SIGABRT-known on
this cluster) so the permute/send axis is physically blocked at W=224. The surviving new axis
is the NON-LINEAR reduction (REDUCE_MAX / REDUCE_MIN), which no confirmed family touches -- all
of fam-1/2/3 live in the SUM-linear world where Overlay folds via linearity/additivity.

KEY ASYMMETRY that makes this a genuinely distinct trap. SUM is NOT idempotent: AR(SUM) of
replicated data = W*data, so fam-1's per-shard scale/unscale chain LOOKS like real depth-D work.
MAX IS idempotent: AR(MAX) of rank-identical data = data, so a naive MAX chain visibly collapses
(the r10/r39 identity-shortcut Overlay reliably finds). To trap on MAX we must keep the data
RANK-HETEROGENEOUS between MAXes. The clean way: inject a per-RANK additive shift beta[r] each
stage. Under MAX a per-rank additive shift SURVIVES the reduction as the CONSTANT max_r(beta[r])
-- a max-plus-semiring identity, NOT the SUM-linearity every confirmed family uses.

Mechanism (max_shift): D dependent stages. cur starts as x (rank-heterogeneous). Each stage:
  buf_r = cur_r + beta[r];   m = AR(MAX, buf);   cur = m   (rank-identical after stage 1)
- Stage 1: m1[i] = max_r(x_r[i] + beta[r])   -- GENUINE: beta is inside the max (survives),
  argmax rank varies per element, so it is NOT max_r(x_r)+const. Overlay's "drop beta" guess is
  WRONG -> fails fp32 gate.
- Stages 2..D: cur is rank-identical (=m1 broadcast), so m_{t+1}[i] = m_t[i] + max_r(beta[r]).
  These telescope: mD = m1 + (D-1)*max_r(beta[r]).
So the FOLD = ONE AR(MAX, x + beta[r]) then add the local constant (D-1)*max_r(beta[r]). D
collectives -> 1. The fold needs the max-plus insight "a per-rank additive shift on rank-identical
data pulls out of MAX as max_r(beta)". Overlay's enumerate (built for SUM-linearity) does not
reach it; it either stays pinned at D AR(MAX) or fails the gate with the drop-beta guess.

MIN analog (min_shift): symmetric with REDUCE_MIN and min_r(beta[r]).

All ops are MAX/MIN + local constant adds -> gate-exact (no float-precision drift). beta[r] is a
per-RANK scalar (the fam-1-analog surviving factor). Pre-screened at W=224 before any cloud run.
"""
import torch
from .problems import CollectiveProblem, register_problem


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


def _gen(world_size, seed, N):
    torch.manual_seed(seed)
    # distinct per rank; different argmax rank per element so beta genuinely matters
    return [{'x': torch.randn(N) * (0.7 + 0.03 * r)} for r in range(world_size)]


# beta[r] as a per-rank scalar. GAMMA controls spread; pattern is non-monotone-in-magnitude
# so max_r(beta) is a genuine reduction (not just beta[W-1]).
def _beta_expr(gamma):
    # beta[r] = gamma * ((r * 37) % world_size - world_size/2) / world_size  (scaled, in ~[-g/2,g/2])
    return f"({gamma} * (((rank * 37) % world_size) - world_size/2.0) / world_size)"


def _beta_vals(gamma, W):
    return [gamma * (((r * 37) % W) - W / 2.0) / W for r in range(W)]


# ---------- max_shift ----------
def _mk_max(name, N, depth, gamma, cue, reduce_kind="MAX"):
    is_max = (reduce_kind == "MAX")

    def _ref(inputs, world_size):
        W = world_size
        betas = _beta_vals(gamma, W)
        xs = [inp['x'] for inp in inputs]
        # stage 1: elementwise max/min over r of (x_r + beta[r])
        stk = torch.stack([xs[r] + betas[r] for r in range(W)], dim=0)  # (W,N)
        m1 = stk.max(dim=0).values if is_max else stk.min(dim=0).values
        Bconst = max(betas) if is_max else min(betas)
        out = m1 + (depth - 1) * Bconst
        return [out.clone() for _ in range(W)]

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, N)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}
    red = "xm.REDUCE_MAX" if is_max else "xm.REDUCE_MIN"
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    beta = {_beta_expr(gamma)}",
         "    cur = x + beta",
         f"    for _t in range({depth}):",
         f"        cur = xm.all_reduce({red}, cur)",
         "        if _t < %d:" % (depth - 1),
         "            cur = cur + beta",
         "    return cur"]
    kind = "MAX" if is_max else "MIN"
    COUNT = (f"Computed with {depth} dependent all_reduce({kind}) operations. " if cue else "")
    doc = (f"Local x ({N},), distinct per rank. {COUNT}Each rank adds a per-rank shift "
           f"beta[r]={gamma}*(((r*37) mod W) - W/2)/W before each all_reduce({kind}); the "
           f"per-rank shift is re-applied between stages. Return the depth-{depth} result.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


def register_all():
    # MAX depth sweep
    _mk_max("r53_maxshift_d8_n1024", 1024, 8, 2.0, True, "MAX")
    _mk_max("r53_maxshift_d6_n1024", 1024, 6, 2.0, True, "MAX")
    _mk_max("r53_maxshift_d4_n1024", 1024, 4, 2.0, True, "MAX")
    # gamma sweep at d8
    _mk_max("r53_maxshift_d8_g1", 1024, 8, 1.0, True, "MAX")
    _mk_max("r53_maxshift_d8_g4", 1024, 8, 4.0, True, "MAX")
    # MIN analog
    _mk_max("r53_minshift_d8_n1024", 1024, 8, 2.0, True, "MIN")
    _mk_max("r53_minshift_d6_n1024", 1024, 6, 2.0, True, "MIN")
    # framing controls (result-only, no count cue)
    _mk_max("r53_maxshift_d8_res", 1024, 8, 2.0, False, "MAX")
    _mk_max("r53_minshift_d8_res", 1024, 8, 2.0, False, "MIN")


register_all()
