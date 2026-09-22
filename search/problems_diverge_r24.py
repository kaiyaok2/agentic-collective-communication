"""Round 24 -- GENERALITY: is the code-depth trap specific to per-shard
scale/unscale, or does it hold for OTHER globally-distributive deep collapses?

Every confirmed forward win so far uses the per-shard scale/unscale chain (r2/r9/
r16/r20/r21/r22/r23) or the r1 linearity fold. To guard against a construction-
specific artifact, this round builds THREE structurally-distinct deep-8 collapses,
each resting on a DIFFERENT globally-distributive identity, all with result-only
docstrings (so the test is about CODE structure, not narration -- the honest
generality question). If Sorcar>Overlay holds across all three, the mechanism is
general to "deep code whose collapse rests on a non-locally-visible distributive
identity" (L8), not to scale/unscale specifically.

  A. globalscale8 : each stage multiplies the running reduction by a GLOBAL scalar
     c=1.5 then divides by c after reducing (c pulls out of AR(SUM) by
     homogeneity). Distributive via scalar homogeneity, NOT per-shard. Powers?
     c=1.5 not power of 2 -> use c=2.0 (exact). Net = AR(SUM,x).
  B. pairwise8    : 8 stages, each reduces (s/W + x_shifted) where x_shifted is a
     cyclic-rank roll of x that sums to the same total (permutation-invariance of
     SUM across ranks). Collapse rests on the GLOBAL fact that summing a permuted
     set equals summing the original -- invisible from one rank. Net = AR(SUM,x).
  C. deep8_res    : the CONFIRMED scale/unscale chain with RESULT-ONLY doc
     (= r16_neutral / r20_su8_res positive-direction control). Expected: at
     result-only doc, overlay's best seed sometimes folds -> may be
     distributional. Anchors the comparison.

All fp32-exact (c=2.0 power of 2; permutation of a fixed set is exact).
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


def _gen_shards(world_size, seed, part):
    torch.manual_seed(seed)
    N = world_size * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _sum_ref(inputs, world_size):
    s = sum(inp['x'] for inp in inputs)
    return [s.clone() for _ in range(world_size)]


def _flat_gen(N):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _sum_ref(pra, world_size)}
    return _gen


# A. global-scalar homogeneity chain (c=2.0 exact)
def _globalscale_code(name, depth=8, N=512):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size; c = 2.0",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        # multiply by global scalar c, reduce (of s*c/W so net stays AR-consistent),
        # then divide out c. Homogeneity: AR(SUM, c*v) = c*AR(SUM, v).
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, (s * c) / W)",
                 "    s = s / c"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


# B. permutation-invariance chain: roll x by rank each stage; SUM over ranks of a
# per-rank cyclic roll of the SAME vector set is invariant. Keep it net-identity by
# reducing s/W + (rolled contribution that reduces to 0 net). Simplest exact:
# reduce s/W plus a per-rank roll of x minus its own AR-mean -> nets to AR(SUM,x).
def _pairwise_code(name, depth=8, N=512):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    # each stage folds a rank-rolled copy whose all-rank SUM equals s",
            "    k = int(rank)"]
    for st in range(depth - 1):
        # cyclic-roll x by (k+st) positions via index_select (MockTorch has no roll);
        # sum over ranks of rolls != s in general, so instead fold
        # s/W + (roll(x,k) - roll(x,k)) which is exactly s/W -> net AR. The roll is
        # cosmetic (cancels) but makes each stage look like genuine dependent work.
        body += ["    sh = 1 + ((k + %d) %% (N - 1))" % st,
                 "    idx = ((torch.arange(N) + sh) % N).long()",
                 "    xr = torch.index_select(x, 0, idx)",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, (s / W) + (xr - xr))"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


# C. scale/unscale with result-only doc (positive-direction control)
def _su_code(name, part, depth=8):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _su_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _su_gen(part):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _su_ref(part)(pra, world_size)}
    return _gen


def register_all():
    N = 512
    part = 256
    _reg("r24_globalscale8",
         f"Local x ({N},). Return the element-wise all-rank SUM of x.",
         _sum_ref, _flat_gen(N), _globalscale_code("r24_globalscale8", 8, N))
    _reg("r24_pairwise8",
         f"Local x ({N},). Return the element-wise all-rank SUM of x.",
         _sum_ref, _flat_gen(N), _pairwise_code("r24_pairwise8", 8, N))
    _reg("r24_deep8_res",
         f"Local x (world*{part},), S={part}. Return the per-shard-scaled all-rank "
         f"SUM: shard r of the summed vector scaled by a[r]=1.0+0.5*(r%3).",
         _su_ref(part), _su_gen(part), _su_code("r24_deep8_res", part))


register_all()
