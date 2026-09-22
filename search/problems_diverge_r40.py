"""Round 40 -- SECOND FAMILY: rank-indexed ROUTING (mask-overlap count scale).

Motivation (from the r37/r39 post-mortem). A PURE-PERMUTATION collapse can never
trap Overlay: after the first all_reduce every rank holds IDENTICAL data, so any
subsequent all_reduce(SUM, permute(s))/W is a LOCALLY-VISIBLE identity ("keep the
permutes local"). The permutation's complexity is irrelevant -- the fold is trivially
visible because the inter-collective buffer is rank-identical. Family-1 traps ONLY
because a per-shard MULTIPLICATIVE scale makes the inter-collective buffer
RANK-HETEROGENEOUS, so the identity-shortcut is FALSE and the collapse (scale
distributes through SUM) is a genuine non-local insight.

This family makes the inter-collective state rank-heterogeneous WITHOUT any
multiplicative constant, via RANK-INDEXED ROUTING:

    s = AR(SUM, x)                          # rank-identical full sum
    for stage in 1..D-1:                    # D-1 further dependent AR(SUM)
        # rank r masks a CONTIGUOUS WINDOW of L blocks starting at (r + off) % B,
        # zeroing all other blocks. THE WINDOW DEPENDS ON `rank` -> each rank
        # contributes a DIFFERENT masked buffer -> the buffers are genuinely
        # rank-heterogeneous, so this AR(SUM) is a REAL reduction, not an identity.
        buf_r = mask_r(rank) * s
        acc   = AR(SUM, buf_r)              # == c * s  where c[b] = #ranks covering b
        s     = acc / c                     # unscale by the per-block overlap count
    return s                                # == c * (AR(SUM, x))  [final stage: no unscale]

The overlap count c[b] = #{ranks r : block b in window_r} is a FIXED integer vector
that depends only on (W, B, L, off) -- NOT on the data. So the whole D-deep chain
folds to:  1 AR(SUM)  +  a local per-block multiply by c  (final stage leaves the
count applied; intermediate stages cancel via /c). There is NO scalar constant
written in the code: the coefficients EMERGE from routing-overlap counts. That is a
structurally DISTINCT algebra from family-1 (which writes an explicit a[r] diagonal
scale): here the "scale" is the SUM OF RANK-INDEXED ROUTING PATTERNS.

Why Overlay cannot fold it from the baseline framing: a single rank only ever holds
its own masked copy mask_r*s; the reduced value c*s is NOT a local function of any one
rank's buffer, so "AR-of-identical-data = identity" is FALSE. Recognizing the fold
requires summing the D-1 rank-indexed masks across ALL ranks into the count vector c
-- a non-local, count-theoretic insight, exactly the family-1 difficulty but reached
through routing rather than an explicit scale. Overlay's enumerate-from-baseline +
bounded R=3 refine stays pinned near the D-deep chain; Kiss's open ReAct reads the
gate error and iterates to the count fold.

Numeric: the per-block net over the chain is exactly c[b] (bounded, O(W)), because
each intermediate stage unscales by c[b] and only the final stage leaves it applied
-- so the baseline passes the fp32 gate (atol=1e-5) just like family-1's scale/unscale
skeleton. (An un-normalized variant would blow up as c^(D-1); the /c unscale keeps it
bounded, mirroring family-1's /max(a,eps).)

Axes probed (mirroring family-1's robustness sweep):
  - depth D in {4,6,8}                (the divergence has a depth threshold ~6-7)
  - window L in {2,3,4}               (controls how non-uniform c is; L must give a
                                       non-constant c, else it degenerates to a global
                                       scalar that BOTH fold -- the r24 tie class)
  - truthful count cue vs result-only framing (count cue made family-1 bo16-robust)
  - payload part in {256,1024}        (headroom, does not change the structure)

Blocks: B=8 contiguous segments of `part` each (independent of world size). The window
wraps modulo B. Masking built with explicit slice writes (MockTorch has no boolean-mask
gather on blocks; per-block slice assignment is exact).
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


def _gen_shards(world_size, seed, part):
    torch.manual_seed(seed)
    N = NBLOCK * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _count_vec(world_size, L, off):
    """c[b] = number of ranks whose length-L window (start (r+off)%B) covers block b."""
    B = NBLOCK
    c = [0] * B
    for r in range(world_size):
        start = (r + off) % B
        for j in range(L):
            c[(start + j) % B] += 1
    return c


def _route_code(name, part, depth, L, off):
    L_lines = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {NBLOCK}; W = world_size; L = {L}; OFF = {off}",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
        "    # per-block overlap count c[b] = #ranks whose window covers block b",
        "    c = [0]*B",
        "    for r in range(W):",
        "        st = (r + OFF) % B",
        "        for j in range(L):",
        "            c[(st + j) % B] += 1",
    ]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L_lines += [
            "    # rank-indexed routing: THIS rank masks its own length-L window",
            "    start = (rank + OFF) % B",
            "    keep = set((start + j) % B for j in range(L))",
            "    buf = torch.zeros_like(s)",
            "    for b in range(B):",
            "        if b in keep:",
            "            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]",
            "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
        ]
        if not last:
            L_lines += [
                "    for b in range(B):",
                "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]",
            ]
        L_lines += ["    s = acc"]
    L_lines += ["    return s"]
    return "\n".join(L_lines) + "\n"


def _route_ref(part, depth, L, off):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        c = _count_vec(world_size, L, off)
        B = NBLOCK
        out = s.clone()
        for b in range(B):
            out[b * part:(b + 1) * part] = c[b] * s[b * part:(b + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, L, off, count_cue):
    ref = _route_ref(part, depth, L, off)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': ref(pra, world_size)}

    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. "
             if count_cue else "")
    RES = (f"Final result = the elementwise SUM of x across ranks, with each of the "
           f"{NBLOCK} blocks scaled by the number of ranks whose length-{L} contiguous "
           f"window (window start = (rank+{off}) mod {NBLOCK}) covers that block.")
    doc = f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}{RES}"
    _reg(name, doc, ref, gen, _route_code(name, part, depth, L, off))


def register_all():
    # off=2 so the window wrap gives a genuinely non-uniform c on W=7,B=8
    # (c ~= [2,2,3,3,3,3,3,2] for L=3): non-constant => not a global scalar (avoids
    # the r24 permutation-invariance tie), but every block is covered (no zeros).
    OFF = 2

    # --- depth sweep at L=3 (count cue) ---
    _mk("r40_route_d4_L3_count8",  256, 4, 3, OFF, True)
    _mk("r40_route_d6_L3_count8",  256, 6, 3, OFF, True)
    _mk("r40_route_d8_L3_count8",  256, 8, 3, OFF, True)

    # --- window sweep at D=8 (count cue) ---
    _mk("r40_route_d8_L2_count8",  256, 8, 2, OFF, True)
    _mk("r40_route_d8_L4_count8",  256, 8, 4, OFF, True)

    # --- payload sweep at D=8,L=3 (count cue) ---
    _mk("r40_route_d8_L3_big",    1024, 8, 3, OFF, True)
    _mk("r40_route_d6_L3_big",    1024, 6, 3, OFF, True)

    # --- result-only framing (no count cue) across depth/window ---
    _mk("r40_route_d8_L3_res",     256, 8, 3, OFF, False)
    _mk("r40_route_d6_L3_res",     256, 6, 3, OFF, False)
    _mk("r40_route_d8_L2_res",     256, 8, 2, OFF, False)
    _mk("r40_route_d8_L4_res",     256, 8, 4, OFF, False)
    _mk("r40_route_d4_L3_res",     256, 4, 3, OFF, False)

    # --- extra count-cue depth points for the threshold curve ---
    _mk("r40_route_d5_L3_count8",  256, 5, 3, OFF, True)
    _mk("r40_route_d7_L3_count8",  256, 7, 3, OFF, True)
    _mk("r40_route_d8_L3_count8b", 512, 8, 3, OFF, True)


register_all()
