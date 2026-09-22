"""v6 -- BROAD, DIVERSE divergence battery (many families, not one).

Design principle (corrected from v5): don't chase "harder", chase STRUCTURAL
ANCHORING across DIVERSE framings. OverlayCCL enumerates K=5 strategies from
the baseline's framing, implements each ONCE, discards any that fail the gate,
and refines the top-2. It diverges from SorcarCCL (open ReAct that reframes
freely and repairs across gate errors) when the baseline framing biases all K
strategies away from the true optimum.

The reliable, gate-valid headroom lever locally is COLLECTIVE-COUNT REDUCTION
(the local gate resolves only ~1 level of collective dependency, so deep
dependent chains and mis-scored collective_permute belong to the cluster
track). This file exercises count-reduction through MANY DISTINCT SEMANTIC
FRAMINGS so "does overlay's enumerate miss the collapse" is tested across
families, not one:

  FUSE   independent per-segment reductions  -> ONE reduction of the whole
  DEAD   live reduction + k DEAD reductions  -> ONE reduction (drop dead)
  CSE    the same reduction computed twice   -> ONE reduction (reuse)
  ALG    a*AR(u) + b*AR(v) (+ ...)           -> ONE AR of the combination
  A2A    transpose via all_gather + slice    -> ONE all_to_all
  SCALE  AR(x)*c1 + AR(x)*c2 across segments  -> ONE AR then local scale

Every optimum is separately verified to pass the SAME fp32 gate AND to have
real sim headroom over the baseline. All ops are MockTorch/MockXM-supported
and stay within 1 collective-dependency level.
"""
import torch
from .problems import CollectiveProblem, register_problem

S = 256


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


def _gen_flat(world_size, N, seed=0):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (0.25 + 0.03 * r)} for r in range(world_size)]


# ===========================================================================
# FUSE -- x packs `nparts` independent (S,) sub-buffers. Baseline all_reduces
# each separately (nparts collectives). Optimum: ONE all_reduce of the whole.
# ===========================================================================
def _mk_fuse(name, nparts, part=S):
    def _ref(inputs, world_size):
        st = sum(inp['x'] for inp in inputs)
        return [st.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, nparts * part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    part = {part}; nparts = {nparts}",
            "    outs = []",
            "    for j in range(nparts):",
            "        seg = x[j*part:(j+1)*part]",
            "        outs.append(xm.all_reduce(xm.REDUCE_SUM, seg))",
            "    return torch.cat(outs, dim=0)"]
    _reg(name, f"x packs {nparts} independent (={part}) sub-buffers; return "
         f"the all-rank sum of each, concatenated. Baseline all_reduces each "
         f"sub-buffer separately.", _ref, _gen, "\n".join(body) + "\n")


# ===========================================================================
# DEAD -- x packs a live (S,) half then `ndead` dead (S,) segments. Each dead
# segment is all_reduced and scaled by a per-rank coefficient that is 0 on
# every rank. Baseline all_reduces 1 live + ndead dead. Optimum: ONE AR.
# ===========================================================================
def _mk_dead(name, ndead, part=S):
    def _ref(inputs, world_size):
        live = sum(inp['x'][:part] for inp in inputs)
        return [live.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, (1 + ndead) * part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    part = {part}; ndead = {ndead}; W = world_size",
            "    live = x[:part]",
            "    acc = xm.all_reduce(xm.REDUCE_SUM, live)",
            "    for j in range(ndead):",
            "        seg = x[(j+1)*part:(j+2)*part]",
            "        dr = xm.all_reduce(xm.REDUCE_SUM, seg)",
            "        coeff = float((rank % W) - rank)   # == 0 on every rank",
            "        acc = acc + coeff * dr",
            "    return acc"]
    _reg(name, f"x packs a live (={part}) half then {ndead} more (={part}) "
         f"segments; return the all-rank sum of the live half. Each extra "
         f"segment is scaled by a per-rank coefficient.",
         _ref, _gen, "\n".join(body) + "\n")


# ===========================================================================
# CSE -- the reference is 2*AR(x); baseline all_reduces x TWICE (two named
# temporaries) and adds them. Optimum: ONE AR, then double locally.
# ===========================================================================
def _mk_cse(name, part=S):
    def _ref(inputs, world_size):
        st = sum(inp['x'] for inp in inputs)
        return [(2.0 * st).clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    a = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    b = xm.all_reduce(xm.REDUCE_SUM, x)   # identical to a",
            "    return a + b"]
    _reg(name, f"part={part}: return 2x the all-rank sum of x. Baseline "
         f"all_reduces x twice and adds the two results.",
         _ref, _gen, "\n".join(body) + "\n")


# ===========================================================================
# ALG -- x packs u,v (and w). Reference = a*sum(u) + b*sum(v) (+ c*sum(w)).
# Baseline all_reduces each buffer separately then combines. Optimum: form the
# linear combination LOCALLY (a*u + b*v + ...) and all_reduce ONCE (linearity).
# ===========================================================================
def _mk_alg(name, coeffs, part=S):
    n = len(coeffs)

    def _ref(inputs, world_size):
        segs = [sum(inp['x'][j*part:(j+1)*part] for inp in inputs) for j in range(n)]
        out = sum(coeffs[j] * segs[j] for j in range(n))
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, n * part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    part = {part}; coeffs = {list(coeffs)}",
            "    acc = None",
            "    for j, c in enumerate(coeffs):",
            "        seg = x[j*part:(j+1)*part]",
            "        r = xm.all_reduce(xm.REDUCE_SUM, seg)",
            "        acc = c*r if acc is None else acc + c*r",
            "    return acc"]
    _reg(name, f"x packs {n} (={part}) sub-buffers u_0..u_{n-1}; return "
         f"sum_j coeffs[j]*sum_ranks(u_j) with coeffs={list(coeffs)}. Baseline "
         f"all_reduces each sub-buffer separately then combines.",
         _ref, _gen, "\n".join(body) + "\n")


# ===========================================================================
# A2A -- transpose across ranks. Each rank holds a (W,K) tile; rank r wants,
# for each source s, source s's row r. Baseline all_gathers full tiles (W*W*K)
# and slices; optimum is ONE all_to_all.
# ===========================================================================
def _mk_a2a(name, K):
    def _ref(inputs, world_size):
        return [torch.stack([inputs[s]['x'][r] for s in range(world_size)], dim=0)
                for r in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(world_size, K) * (0.3 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    K = {K}; W = world_size",
            "    g = xm.all_gather(x, dim=0)          # (W*W, K)",
            "    rows = []",
            "    for s in range(W):",
            "        rows.append(g[s*W:(s+1)*W][rank])",
            "    return torch.stack(rows, dim=0)"]
    _reg(name, f"Transpose across ranks, K={K}: each rank holds a (world,K) "
         f"tile; rank r returns (world,K) stacking, over sources s, source s's "
         f"row r. Baseline all_gathers full tiles then selects.",
         _ref, _gen, "\n".join(body) + "\n")


# ===========================================================================
# SCALE -- reference = sum_j c_j * sum_ranks(x)  where every segment is the
# SAME buffer x (not distinct). Baseline all_reduces x once PER coefficient
# (nc collectives) then scales+adds. Optimum: AR(x) once, multiply by sum(c_j).
# ===========================================================================
def _mk_scale(name, coeffs, part=S):
    def _ref(inputs, world_size):
        st = sum(inp['x'] for inp in inputs)
        out = sum(coeffs) * st
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    coeffs = {list(coeffs)}",
            "    acc = None",
            "    for c in coeffs:",
            "        r = xm.all_reduce(xm.REDUCE_SUM, x)   # same x each time",
            "        acc = c*r if acc is None else acc + c*r",
            "    return acc"]
    _reg(name, f"part={part}: return (sum of coeffs)={sum(coeffs)} times the "
         f"all-rank sum of x. Baseline all_reduces x once per coefficient "
         f"({len(coeffs)} times) and accumulates.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    # FUSE family (independent segments -> 1)
    _mk_fuse("v6_fuse3", 3)
    _mk_fuse("v6_fuse6", 6)
    _mk_fuse("v6_fuse10", 10)
    # DEAD family (live + k dead -> 1)
    _mk_dead("v6_dead1", 1)
    _mk_dead("v6_dead3", 3)
    _mk_dead("v6_dead5", 5)
    # CSE family (duplicate reduction -> 1)
    _mk_cse("v6_cse2")
    # ALG family (linear combination fused by linearity)
    _mk_alg("v6_alg2", (2.0, -1.0))
    _mk_alg("v6_alg3", (1.5, -0.5, 2.0))
    _mk_alg("v6_alg5", (1.0, -1.0, 2.0, 0.5, -1.5))
    # A2A family (transpose reframe)
    _mk_a2a("v6_a2a_k128", 128)
    _mk_a2a("v6_a2a_k256", 256)
    # SCALE family (same buffer reduced per coefficient -> 1)
    _mk_scale("v6_scale3", (1.0, 2.0, 3.0))
    _mk_scale("v6_scale6", (1.0, -1.0, 2.0, 0.5, 1.5, -0.5))


register_all()
