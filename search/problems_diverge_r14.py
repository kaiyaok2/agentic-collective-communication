"""Round 14 -- POSITIVE confirmation of L8 via a NEW distributive family.

L8 (from r10/r11): divergence requires the fused optimum to rest on a GLOBALLY-
distributive collapse (invisible from the baseline's local framing), not a
locally-visible identity. r2/r9's scale/unscale confirmed; r10 re-max, r11
max-offset (locally-visible) tied. To make L8 falsifiable in the WINNING
direction (not just via null results), this round builds two distributive
collapses that are STRUCTURALLY DIFFERENT from scale/unscale. If L8 is right,
they should CONFIRM (Sorcar>Overlay); if they tie, L8 is too narrow and must be
revised.

DIR-X "zero-sum perturbation chain": each of D dependent AR(SUM) stages adds a
per-RANK vector delta_r that is constructed so sum_r delta_r == 0 exactly. After
the all_reduce the perturbation vanishes, so the net over the whole chain is
still a single AR(SUM, x). But the collapse requires recognizing a GLOBAL
property (the deltas cancel across ranks) that is invisible from any single
rank's local code -- exactly the L8 condition, and NOT a scale/unscale.
deltas are +/- powers of 2 arranged in cancelling pairs -> fp32-exact.

DIR-Y "linear-combination accumulation": s_{k+1} = AR(SUM, x/W + s_k/W) with a
final correction. Each stage folds the previous running result back in with the
fresh input; by linearity of AR(SUM) the whole chain telescopes to one weighted
all_reduce. Distributive (rests on linearity), structurally distinct from a pure
per-shard scale. All coefficients are powers of 2 -> fp32-exact.
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


def _gen_flat(world_size, N, seed):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# --- DIR-X: zero-sum perturbation chain (global cancellation) ---
def _mk_zerosum_chain(name, depth, N=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    # Each stage: rank r adds delta_r = (rank - (W-1)/2 scaled to integer pairs)
    # arranged so sum_r delta_r == 0. Use delta_r = 2.0*(rank) - (W-1) times a
    # power-of-2 unit vector? Simpler exact construction: d_r = float(2*r - (W-1)).
    # sum_r (2*r-(W-1)) = 2*(W-1)W/2 - W(W-1) = W(W-1)-W(W-1) = 0. Exact ints.
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    # per-rank additive perturbation that sums to exactly 0 across ranks",
            "    d = float(2 * rank - (W - 1))",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        # add d locally, reduce, subtract the reduced perturbation (which is 0).
        # Written so it looks like genuine dependent work: perturb -> reduce.
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, (s / W) + d)"]
    body += ["    return s"]
    _reg(name, f"Local x ({N},). Return all-rank SUM. Baseline runs {depth} "
         f"dependent all_reduce stages; each adds a per-rank perturbation before "
         f"reducing. Net result = single all-rank SUM of x.",
         _ref, _gen, "\n".join(body) + "\n")


# --- DIR-Y: linear-combination accumulation (telescoping by linearity) ---
def _mk_lincomb_chain(name, depth, N=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    # s0 = AR(x). Then s_{k+1} = AR( x/W + (s_k - AR(x))/W )? Keep it exact and
    # net-identity: s_{k+1} = AR( s_k * 0.5 / W + x_frac ) with x_frac chosen so
    # the fixed point is AR(x). Simplest exact telescoping: repeatedly average s
    # with a re-reduced x, all powers of 2. Net stays AR(SUM, x).
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    xr = x  # keep fresh local input"]
    for st in range(depth - 1):
        # s = 0.5*s + 0.5*AR(x) but AR(x) recomputed via reduce of xr; halving is
        # power-of-2 exact. Net after chain = AR(SUM, x) (geometric fixed point,
        # exact because both terms equal AR(x) at every step).
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, (xr * 0.5) + (s * 0.5) / W)"]
    body += ["    return s"]
    _reg(name, f"Local x ({N},). Return all-rank SUM. Baseline runs {depth} "
         f"dependent all_reduce stages folding the running result with the fresh "
         f"input via a linear combination. Net result = single all-rank SUM of x.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_zerosum_chain("r14_zerosum6", 6, 512)
    _mk_zerosum_chain("r14_zerosum8", 8, 512)
    _mk_lincomb_chain("r14_lincomb6", 6, 512)
    _mk_lincomb_chain("r14_lincomb8", 8, 512)


register_all()
