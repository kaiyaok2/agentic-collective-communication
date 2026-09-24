"""Round 71 -- FAMILY-1' (crash-free reformulation of the retired world-scaled
family-1): rank-indexed real-WEIGHTED per-block MULTIPLICATIVE diagonal, FIXED-BLOCK
(world-independent op count).

The retired family-1 (r59) applied a per-shard scale with a `for r in range(W)`
tensor-slice loop over W=224 shards x depth-8 -> the HLO op count scaled with the world
size, blew up, and CRASHED Overlay at 224 ranks (an unfair Phase-4 gate gap that made
its RT "wins" crash artifacts). Here the operator is FIXED-BLOCK (B <= 16 blocks): EVERY
tensor-slice loop runs over the fixed block count B, never over world_size, so the op
count is world-independent and neither pipeline crashes at 224 ranks. World-dependence
survives only inside a cheap integer/scalar loop that accumulates the length-B emergent
weight vector A (no tensor ops).

Mechanism (mirrors the proven crash-free family-2 skeleton; DISTINCT operator): each
stage, rank r masks a rank-DEPENDENT window of blocks and weights the kept blocks by a
rank-DEPENDENT real weight w[r], then all_reduce(SUM)s. Because every rank holds the
same reduced vector s, the reduced value at block b is A[b]*s[b] where
A[b] = sum of w[r] over ranks whose window covers b -- a REAL-valued emergent diagonal.
(Family-2 accumulates an INTEGER overlap count; using real per-rank weights yields a
genuinely different diagonal and a distinct reference md5.) The depth-D chain telescopes
to ONE all_reduce(SUM,x) + a local per-block multiply by A. Overlay cannot fold from the
baseline framing -- A is non-local, a single rank only ever holds its own masked+weighted
copy -- so it stays pinned at the D-deep chain; Kiss iterates to the weighted-diagonal
fold. B in {5,6} does NOT divide the scoring world 224 (=2^5*7), keeping A non-uniform.
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


def _gen_shards(world_size, seed, part, nblock):
    torch.manual_seed(seed)
    N = nblock * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _keep_py(spec, rank):
    B = spec["B"]; off = spec["off"]; L = spec["L"]; stride = spec.get("stride", 1)
    start = (rank + off) % B
    return set((start + stride * j) % B for j in range(L))


def _wpy(spec, rank):
    # wexpr uses the placeholder {v}; the reference weights each rank by its OWN rank.
    return float(eval(spec["wexpr"].format(v="rank"), {"rank": rank}))


def _A_vec(spec, world_size):
    B = spec["B"]
    A = [0.0] * B
    for r in range(world_size):
        w = _wpy(spec, r)
        for b in _keep_py(spec, r):
            A[b] += w
    return A


def _code(name, spec, depth):
    part = spec["part"]; B = spec["B"]; off = spec["off"]
    L = spec["L"]; stride = spec.get("stride", 1); wexpr = spec["wexpr"]
    Lc = [f"def {name}_fn(x, rank, world_size, num_devices,",
          "                 cores_per_device, xm, torch, num_nodes=1):",
          f"    S = {part}; W = world_size",
          "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
          f"    A = [0.0]*{B}",
          "    for r in range(W):",
          f"        w = {wexpr.format(v='r')}",
          f"        st = (r + {off}) % {B}",
          f"        for j in range({L}):",
          f"            A[(st + {stride}*j) % {B}] += w"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        Lc += [f"    start = (rank + {off}) % {B}",
               f"    keep = set((start + {stride}*j) % {B} for j in range({L}))",
               f"    w = {wexpr.format(v='rank')}",
               "    buf = torch.zeros_like(s)",
               f"    for b in range({B}):",
               "        if b in keep:",
               "            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]",
               "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            Lc += [f"    for b in range({B}):",
                   "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]"]
        Lc += ["    s = acc"]
    Lc += ["    return s"]
    return "\n".join(Lc) + "\n"


def _ref(spec, depth):
    part = spec["part"]; B = spec["B"]

    def ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        A = _A_vec(spec, world_size)
        out = s.clone()
        for b in range(B):
            out[b * part:(b + 1) * part] = A[b] * s[b * part:(b + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return ref


def _mk(name, spec, depth, cue):
    ref = _ref(spec, depth)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, spec["part"], spec["B"])
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. "
             if cue else "")
    doc = (f"Local x ({spec['B']}*{spec['part']},), {spec['B']} blocks of {spec['part']}. "
           f"{COUNT}Final result = the elementwise SUM of x across ranks, with each block "
           f"scaled by the total of the rank-dependent real weights whose rank-dependent "
           f"block window covers that block.")
    _reg(name, doc, ref, gen, _code(name, spec, depth))


def register_all():
    # DISTINCT-OUTPUT candidates. Real emergent diagonal A[b] depends on (B, off, L,
    # stride, weight expr) and world size -- NOT on depth. B in {5,6} does not divide 224
    # so A stays non-uniform. Windows cover >= B-2 blocks so ws=4 keeps every A[b] > 0
    # (the /A unscale stays inside the fp32 gate). Weight exprs are strictly positive.
    _mk("r71_wdiag_b5_L3_w1_d8",  {"B": 5, "off": 0, "L": 3, "stride": 1, "part": 2048, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r71_wdiag_b5_L4_w1_d8",  {"B": 5, "off": 0, "L": 4, "stride": 1, "part": 2048, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r71_wdiag_b6_L4_w1_d8",  {"B": 6, "off": 0, "L": 4, "stride": 1, "part": 2048, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r71_wdiag_b6_L5_w1_d8",  {"B": 6, "off": 0, "L": 5, "stride": 1, "part": 2048, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r71_wdiag_b5_L3_w2_d8",  {"B": 5, "off": 1, "L": 3, "stride": 1, "part": 2048, "wexpr": "0.4 + 0.03*({v} % 5)"}, 8, True)
    _mk("r71_wdiag_b6_L4_w2_d8",  {"B": 6, "off": 2, "L": 4, "stride": 1, "part": 2048, "wexpr": "0.4 + 0.03*({v} % 5)"}, 8, True)
    _mk("r71_wdiag_b5_s2_L3_d8",  {"B": 5, "off": 0, "L": 3, "stride": 2, "part": 2048, "wexpr": "0.6 + 0.02*({v} % 7)"}, 8, True)
    _mk("r71_wdiag_b6_s2_L4_d8",  {"B": 6, "off": 0, "L": 4, "stride": 2, "part": 2048, "wexpr": "0.6 + 0.02*({v} % 7)"}, 8, True)
    _mk("r71_wdiag_b5_L4_w3_d8",  {"B": 5, "off": 0, "L": 4, "stride": 1, "part": 1024, "wexpr": "0.5 + 0.05*({v} % 3)"}, 8, True)
    _mk("r71_wdiag_b6_L5_w3_d8",  {"B": 6, "off": 0, "L": 5, "stride": 1, "part": 1024, "wexpr": "0.5 + 0.05*({v} % 3)"}, 8, True)


register_all()
