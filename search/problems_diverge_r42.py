"""Round 42 -- THIRD FAMILY PROBE: rank-indexed OFF-DIAGONAL coupling (banded/Toeplitz net).

Families found so far both have a DIAGONAL net:
  - family-1: rank-het per-shard MULTIPLICATIVE diagonal scale a[r].
  - family-2 (r40): rank-indexed ROUTING whose net is a diagonal per-block COUNT c[b].
The unifying principle is rank-HETEROGENEOUS inter-collective state that Overlay's
enumerate-from-baseline cannot fold; the two families reach it via two algebras, but BOTH
nets are diagonal.

This round tests a genuinely DISTINCT algebra: an OFF-DIAGONAL (banded, Toeplitz-like)
net that is NOT doubly-stochastic (so it does NOT converge to the uniform-average that
makes conservative routing/rotation TIE -- see the r37/r39 rotation ties and the circulant
screen). The collapse is a fixed BLOCK-COUPLING matrix, not a per-block scale.

Mechanism:
    s = AR(SUM, x)                          # rank-identical full sum
    for stage in 1..D-1:                    # D-1 further dependent AR(SUM)
        # rank r contributes a FULL copy of s, EXCEPT at its own block b=r%B it adds the
        # left-neighbor block:  buf_r[b] = s[b] (+ s[b-1] only when b == r%B).
        # The coupling LOCATION depends on `rank`, so the buffers are rank-heterogeneous
        # and AR(SUM) is a genuine reduction (Overlay's identity-shortcut is FALSE).
        acc[b] = sum_r buf_r[b] = W*s[b] + (#ranks with r%B==b) * s[b-1]
        s = acc / W                          # normalize to stay bounded (final stage: no /W)
    return s

Net over the chain = M^(D-1) / W^(D-2) applied block-wise, where M = W*I + P and P is the
subdiagonal coupling (P[b, b-1] = #ranks whose index maps to block b). M is diagonal-DOMINANT
and lower-bidiagonal (a Toeplitz band), NOT a scale and NOT doubly-stochastic: its powers keep
diagonal 1 with a GROWING off-diagonal band, so it never degenerates to the global average.
The optimum is 1 AR(SUM) + a local banded matmul by the precomputed net; a single rank only
sees its own coupled copy, so the net is NOT a local function of any one buffer -- folding it
requires the non-local insight of summing the rank-indexed couplings into the band. Overlay's
naive "AR-only" or "add left neighbor once" guesses FAIL the fp32 gate (verified: AR-only
max_diff ~29; the coupling is ~93% of the signal). NO multiplicative constant and NO per-block
scale: this is a pure off-diagonal LINEAR-COUPLING algebra.

Numeric: /W each intermediate stage keeps the net bounded (values O(30) at D=8), so the
baseline passes the fp32 gate (atol=1e-5), mirroring family-1/family-2's normalization.

If r42 CONFIRMS at strict best-of-8 -> a THIRD family (off-diagonal coupling), distinct from
both diagonal families. If it TIES -> off-diagonal collapses are foldable by Overlay and the
trap needs a diagonal (scale/count) net.

Blocks: B=8 contiguous segments of `part`. Coupling is block b <- block (b-1)%B. Built with
explicit slice reads/writes (MockTorch-safe; no banded-matmul helper needed in the baseline).
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


def _coupling_vec(world_size):
    """p[b] = number of ranks whose index maps to block b (r % B). This is the
    per-block strength of the b <- b-1 subdiagonal coupling."""
    B = NBLOCK
    p = [0] * B
    for r in range(world_size):
        p[r % B] += 1
    return p


def _couple_code(name, part, depth):
    L = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {NBLOCK}; W = world_size",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
    ]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += [
            "    # rank-indexed subdiagonal coupling: THIS rank adds its left-neighbor",
            "    # block only at its own block index (b == rank % B).",
            "    buf = s.clone()",
            "    b = rank % B",
            "    buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] + s[((b-1) % B)*S:((b-1) % B + 1)*S]",
            "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
        ]
        if not last:
            L += ["    acc = acc / W"]
        L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _couple_ref(part, depth):
    def _ref(inputs, world_size):
        import numpy as _np
        s = sum(inp['x'] for inp in inputs)
        B = NBLOCK
        W = world_size
        p = _coupling_vec(W)
        # M = W*I + subdiagonal(p): M[b,b]=W, M[b,(b-1)%B]+=p[b]
        M = _np.zeros((B, B))
        for b in range(B):
            M[b, b] = W
            M[b, (b - 1) % B] += p[b]
        # chain net: stages 1..depth-1, each M then /W except last (no /W)
        # => net = M^(depth-1) / W^(depth-2)
        net = _np.linalg.matrix_power(M, depth - 1) / (W ** (depth - 2))
        out = torch.zeros_like(s)
        for b in range(B):
            for k in range(B):
                w = net[b, k]
                if w != 0.0:
                    out[b * part:(b + 1) * part] += float(w) * s[k * part:(k + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, count_cue):
    ref = _couple_ref(part, depth)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': ref(pra, world_size)}

    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. "
             if count_cue else "")
    RES = (f"Final result = a fixed banded linear combination of the {NBLOCK} blocks of "
           f"the elementwise SUM of x across ranks, where each block b receives block b "
           f"plus a subdiagonal coupling from block b-1 accumulated over the chain.")
    doc = f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}{RES}"
    _reg(name, doc, ref, gen, _couple_code(name, part, depth))


def register_all():
    # depth sweep (count cue) -- expect a depth threshold like family-1/family-2
    _mk("r42_couple_d8_count8", 256, 8, True)
    _mk("r42_couple_d6_count8", 256, 6, True)
    _mk("r42_couple_d4_count8", 256, 4, True)
    _mk("r42_couple_d8_big",   1024, 8, True)
    # result-only framing
    _mk("r42_couple_d8_res",    256, 8, False)
    _mk("r42_couple_d6_res",    256, 6, False)
    # extra depth points for the threshold curve
    _mk("r42_couple_d7_count8", 256, 7, True)
    _mk("r42_couple_d5_count8", 256, 5, True)


register_all()
