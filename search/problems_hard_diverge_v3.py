"""VERY HARD divergence problems, batch v3 -- maximal L3 stress.

v1/v2 finding so far: under a fair (identical) gate, OverlayCCL's
enumerate+implement step one-shots the optimum whenever the optimum is a
single clean idea OR a clean algebraic fold -> tie with kiss. The ONLY
structural asymmetry left to exploit is OverlayCCL's hard rule (phase3
line 357-359): a strategy whose FIRST implementation fails the correctness
gate is discarded and can never enter the refine top-2. kiss, by contrast,
sees the correctness error text and repairs the same idea across steps.

So v3 problems are built so the UNIQUE cheap optimum requires error-prone
index/shard arithmetic (2D all_gather dim + interleaved recombine;
reduce_scatter scatter_dim/shard_count; strided block reassembly) while
every clean-to-implement alternative stays at/near baseline cost. If a
cold one-shot of the tricky optimum usually fails the gate, OverlayCCL
keeps only the expensive-but-correct family and refines that; kiss can
iterate the tricky family to correctness.

These are the hardest to implement of all three batches. All optima are
verified to exist and pass the SAME fp32 gate.
"""
import torch
from .problems import CollectiveProblem, register_problem


def _reg(name, sig_args, doc, ref_fn, gen_fn, builtin_code, call_args=None):
    sig = (f"def {name}_fn({sig_args}, rank, world_size, num_devices,\n"
           f"                 cores_per_device, xm, torch, num_nodes=1):")

    def _call(fn, args, s, r, w, nd, cpd, xm, tm, num_nodes=1):
        vals = [args[a] for a in (call_args or [sig_args.split(",")[0].strip()])]
        return fn(*vals, r, w, nd, cpd, xm, tm, num_nodes=num_nodes)

    register_problem(CollectiveProblem(
        name=name, display_name=name, evolved_fn_name=f"{name}_fn",
        signature=sig, signature_doc=doc, reference_fn=ref_fn,
        generate_test_case=gen_fn, call_candidate=_call,
        builtin_templates={name: builtin_code}))


# ---------------------------------------------------------------------------
# HD11 (L3, interleaved all_gather recombine): transpose_gather_interleave
# Each rank holds a (R, C) tile. The global result is the column-major
# INTERLEAVE of all ranks' tiles: out[i*W + r] row = rank r's row i. Baseline
# does R separate all_gathers (one per row index, gathering that row from all
# ranks) and stacks. Optimum: ONE all_gather of the whole (R,C) tile (giving
# (W*R, C) in RANK-major order) then a strided reshape/permute to convert
# rank-major -> row-major interleave. The reshape is the classic
# (W,R,C)->(R,W,C)->(R*W,C) transpose that is trivial to get backwards.
# Baseline: R all_gathers. Optimum: 1 all_gather + reshape.
# ---------------------------------------------------------------------------
def _mk_transpose_gather_interleave(name, R=6, C=64):
    def _ref(inputs, world_size):
        # out row (i*W + r) = inputs[r]['x'][i]
        out = torch.empty(R * world_size, C)
        for r, inp in enumerate(inputs):
            tile = inp['x']  # (R, C)
            for i in range(R):
                out[i * world_size + r] = tile[i]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(R, C) * (0.5 + 0.03 * r)}
               for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    R, C = {R}, {C}",
            "    # Baseline: gather each row index separately across ranks,",
            "    # then interleave rank-by-rank into the output.",
            "    rows = []",
            "    for i in range(R):",
            "        gi = xm.all_gather(x[i:i+1], dim=0)   # (W, C): row i of every rank",
            "        rows.append(gi)",
            "    # rows[i][r] == rank r's row i; output row i*W+r == rows[i][r]",
            "    out = torch.empty(R * world_size, C)",
            "    for i in range(R):",
            "        for r in range(world_size):",
            "            out[i*world_size + r] = rows[i][r]",
            "    return out"]
    _reg(name, "x", f"Local x is this rank's (R={R}, C={C}) tile. Return "
         f"(R*world, C) where output row i*world+r == rank r's row i "
         f"(row-major interleave). Baseline gathers each row index separately.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD12 (L3, reduce_scatter scatter_dim): colsum_scatter_2d
# Each rank holds a (W, K) tile; stack across ranks is a (W, W, K) cube
# conceptually. The result each rank needs: rank r receives the SUM over all
# ranks of their r-th ROW, i.e. out_r = sum_over_ranks tile[r]. Baseline:
# all_reduce the full (W, K) tile, then each rank slices its own row r ->
# (W, K) AR then index. Optimum: reduce_scatter(SUM, scatter_dim=0) over the
# (W,K) tile so rank r directly gets row r's reduction -> (1, K) or (K,). The
# scatter_dim + shard_count + output-shape handling is a first-draft trap;
# and the reference wants a (K,) not (1,K), so squeeze errors abound.
# Baseline: 1 AR (of W*K) + slice. Optimum: 1 reduce_scatter (of W*K -> K).
# The RS is priced below the full-AR-then-slice by the sim.
# ---------------------------------------------------------------------------
def _mk_colsum_scatter_2d(name, K=256):
    def _ref(inputs, world_size):
        # tiles stacked: T[r] = inputs[r]['x'] shape (W, K).
        # rank r result = sum_over_s inputs[s]['x'][r]  -> (K,)
        W = world_size
        S = torch.zeros(W, K)
        for s in range(W):
            S += inputs[s]['x']          # elementwise (W,K)
        # rank r wants row r
        return [S[r].clone() for r in range(W)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(world_size, K) * (0.3 + 0.02 * r)}
               for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    # x is this rank's (W, K) tile. Full-reduce, then keep row rank.",
            "    full = xm.all_reduce(xm.REDUCE_SUM, x)   # (W, K)",
            "    return full[rank]                        # (K,)"]
    _reg(name, "x", f"Local x is this rank's (world, K={K}) tile. Rank r must "
         f"return (K,) = sum over ranks of their row r. Baseline all_reduces "
         f"the full tile then slices row rank.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


# ---------------------------------------------------------------------------
# HD13 (L3, strided block reassembly): blockdiag_gather
# Each rank holds a (B,) vector. The global result is a (W*B,) vector that
# is the all-rank SUM, but reordered by a bit-reversal-like stride: output
# position p maps to source position perm(p) where perm interleaves even/odd
# halves. Baseline: AR(SUM) full then a python loop scatter into perm. The
# reduction is trivial (AR) but the PERMUTATION is the trap -- easy to invert
# the mapping. Optimum: AR(SUM) + a single vectorized index_select with the
# correct perm (no loop). Same collective count as baseline (1 AR) but the
# baseline's python loop over W*B elements is the cost; the optimum vectorizes
# it. Distinguishes "get the perm right in one vectorized op" (error-prone)
# from "loop it" (correct but slow). Baseline: 1 AR + O(W*B) py loop.
# Optimum: 1 AR + 1 index_select.
# ---------------------------------------------------------------------------
def _mk_blockdiag_gather(name, B=32):
    def _perm(n):
        # interleave even then odd indices: [0,2,4,...,1,3,5,...]
        ev = list(range(0, n, 2))
        od = list(range(1, n, 2))
        return ev + od

    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (W*B,)
        n = s.numel()
        perm = _perm(n)
        out = s[torch.tensor(perm)]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        N = world_size * B
        pra = [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)      # (W*B,)",
            "    n = s.shape[0]",
            "    # Reorder: even indices first, then odd indices.",
            "    out = torch.empty_like(s)",
            "    j = 0",
            "    for i in range(0, n, 2):",
            "        out[j] = s[i]; j += 1",
            "    for i in range(1, n, 2):",
            "        out[j] = s[i]; j += 1",
            "    return out"]
    _reg(name, "x", f"Local x ({'world*B'},), B={B}. Return AR(SUM) reordered "
         f"so all even indices come first, then all odd indices. Baseline "
         f"reorders with a python loop.",
         _ref, _gen, "\n".join(body) + "\n", call_args=["x"])


def register_all():
    _mk_transpose_gather_interleave("hd11_transpose_gather_interleave")
    _mk_colsum_scatter_2d("hd12_colsum_scatter_2d")
    # HD13 (blockdiag_gather) NOT registered: its baseline needs
    # torch.empty_like + scalar item-assignment, which MockTorch does not
    # implement, and it does not reduce collective count (1 AR either way),
    # so it is a weak divergence lever anyway.


register_all()
