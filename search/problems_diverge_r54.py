"""Round 54 -- FAMILY-4 STRENGTHEN: per-BLOCK max-plus telescoping with awkward unshift.

r53 (scalar max-plus shift) gave only a WEAK/stochastic best-of-4 signal: overlay reaches the
"pull an additive constant out of MAX" semiring identity on some seeds, so the trap is soft.
Lesson from fam-1: the robust trap makes the NET fold constant non-obvious via a per-BLOCK,
non-pow2 scale/unscale that keeps the baseline bounded and looks like genuine depth-D work.

This round ports that structure to the max-plus semiring. B blocks; per-rank, per-block additive
shift beta[r,b], and a per-BLOCK unshift u[b] subtracted every stage (the max-plus analog of
fam-1's /a). D dependent stages:
  stage 1: cur = x + beta[rank, :]                      (rank-heterogeneous)
           m   = AR(MAX, cur)                            (genuine: argmax rank varies per elem)
           cur = m - u   (per-block unshift; keeps values bounded, looks like real work)
  stage t>1: cur = cur + beta[rank, :]; m = AR(MAX, cur); cur = m - u
Because after stage 1 cur is rank-identical, each later stage contributes
  max_r(cur_r + beta[r,b]) = cur[b] + max_r(beta[r,b])   then - u[b].
So the NET per block b telescopes to:
  out[b] = m1[b] + (D-1) * (max_r(beta[r,b]) - u[b])
The fold = ONE AR(MAX, x + beta[rank]) then add the per-block vector
  (D-1) * (colmax(beta)[b] - u[b]).
To fold, Overlay must (1) see beta pulls out of MAX per block, (2) compute the per-block max over
ranks, (3) subtract the awkward per-block u[b], (4) scale by (D-1) -- a multi-step non-obvious
fold, like fam-1. Its naive "drop beta / one AR(MAX)" guess is WRONG (fails fp32 gate).

u[b] is deliberately NOT equal to colmax(beta)[b] and NOT a power of two, so the net per-block
constant is an awkward nonzero vector (mirrors r31 non-pow2 > r29 pow2). All MAX/MIN + local adds
=> gate-exact. Pre-screened at W=224 before any cloud run.
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
    return [{'x': torch.randn(N) * (0.7 + 0.03 * r)} for r in range(world_size)]


# beta[r,b] as a per-(rank,block) scalar; non-monotone in r so colmax is a genuine reduction.
# u[b] a per-block awkward (non-pow2) unshift.
def _beta_code(gamma):
    # returns python building a length-N (=B*S) vector `beta` given rank, S, B.
    return [f"    G = {gamma}",
            "    beta = torch.zeros(B*S)",
            "    for b in range(B):",
            "        val = G * ((((rank * 37) + b * 13) % world_size) - world_size/2.0) / world_size",
            "        beta[b*S:(b+1)*S] = val"]


def _u_code():
    return ["    u = torch.zeros(B*S)",
            "    for b in range(B):",
            "        u[b*S:(b+1)*S] = 0.1 + 0.07 * ((b * 5) % 6)"]


def _beta_vals(gamma, W, B):
    return [[gamma * ((((r * 37) + b * 13) % W) - W / 2.0) / W for b in range(B)] for r in range(W)]


def _u_vals(B):
    return [0.1 + 0.07 * ((b * 5) % 6) for b in range(B)]


def _mk(name, part, depth, gamma, cue, is_max=True):
    def _ref(inputs, world_size):
        W = world_size
        B = NBLOCK
        S = part
        beta = _beta_vals(gamma, W, B)      # [W][B]
        u = _u_vals(B)                      # [B]
        xs = [inp['x'] for inp in inputs]
        # stage 1: per-element extremum over r of (x_r + beta[r,b(elem)])
        shifted = []
        for r in range(W):
            v = xs[r].clone()
            for b in range(B):
                v[b*S:(b+1)*S] = v[b*S:(b+1)*S] + beta[r][b]
            shifted.append(v)
        stk = torch.stack(shifted, dim=0)   # (W, N)
        m1 = stk.max(dim=0).values if is_max else stk.min(dim=0).values
        # net per-block constant
        out = m1.clone()
        for b in range(B):
            col = [beta[r][b] for r in range(W)]
            colext = max(col) if is_max else min(col)
            out[b*S:(b+1)*S] = out[b*S:(b+1)*S] + (depth - 1) * (colext - u[b])
        return [out.clone() for _ in range(W)]

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}
    red = "xm.REDUCE_MAX" if is_max else "xm.REDUCE_MIN"
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}"]
    L += _beta_code(gamma)
    L += _u_code()
    L += ["    cur = x + beta",
          f"    for _t in range({depth}):",
          f"        m = xm.all_reduce({red}, cur)",
          "        if _t < %d:" % (depth - 1),
          "            cur = m - u + beta",
          "        else:",
          "            cur = m",
          "    return cur"]
    kind = "MAX" if is_max else "MIN"
    COUNT = (f"Computed with {depth} dependent all_reduce({kind}) operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}, distinct per rank. {COUNT}"
           f"Each stage adds a per-(rank,block) shift beta[r,b] before all_reduce({kind}) and "
           f"subtracts a per-block unshift u[b] after; the shift is re-applied each stage. Return "
           f"the depth-{depth} result.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


def register_all():
    # MAX depth sweep, per-block structure
    _mk("r54_bmax_d8_p256", 256, 8, 2.0, True, True)
    _mk("r54_bmax_d6_p256", 256, 6, 2.0, True, True)
    _mk("r54_bmax_d4_p256", 256, 4, 2.0, True, True)
    # gamma sweep at d8
    _mk("r54_bmax_d8_g3", 256, 8, 3.0, True, True)
    _mk("r54_bmax_d8_g5", 256, 8, 5.0, True, True)
    # payload
    _mk("r54_bmax_d8_p512", 512, 8, 2.0, True, True)
    # MIN analog
    _mk("r54_bmin_d8_p256", 256, 8, 2.0, True, False)
    # framing control
    _mk("r54_bmax_d8_res", 256, 8, 2.0, False, True)


register_all()
