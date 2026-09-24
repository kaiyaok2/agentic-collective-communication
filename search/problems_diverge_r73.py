"""Round 73 -- FAMILY-8 (crash-free reformulation of the retired world-scaled monomial
family, prev family-9 / r70): rank-indexed real-weighted per-block MONOMIAL-MAP -- a
FIXED block permutation composed with an emergent diagonal scale, FIXED-BLOCK
(world-independent op count).

The retired monomial family (r70) built its block map with a `for r in range(W)`
tensor-slice loop over W=224 shards -> HLO op count scaled with world size, blew up, and
CRASHED Overlay at 224 ranks (an unfair Phase-4 gate gap). Here the operator is
FIXED-BLOCK (B <= 16 blocks): EVERY tensor-slice loop runs over the fixed block count B,
never over world_size, so the op count is world-independent and neither pipeline crashes
at 224 ranks. World-dependence survives only inside a cheap integer/scalar loop that
accumulates the length-B emergent scale vector A (no tensor ops).

Mechanism (distinct operator vs family-1' diagonal and family-7' bidiagonal): the
result at block b is a single monomial A[b]*s[sigma(b)], where sigma(b) = (a*b + c) % B
is a FIXED block permutation (gcd(a,B)=1, so sigma is a bijection) and A[b] is the total
of the rank-dependent real weights whose rank-dependent window covers block b -- a
REAL-valued emergent diagonal. Because sigma is a bijection with A[b] != 0, the map is
strictly invertible (rec[sigma(b)] = acc[b]/A[b]), so the depth-D baseline telescopes to
ONE all_reduce + a local monomial apply: each non-last stage applies the monomial map
then inverts it (a fixed-B scalar loop), i.e. the identity on s; only the final stage
applies it without inverting. Overlay cannot fold from the baseline framing -- A is
non-local and the permutation shuffles blocks, so a single rank only ever holds its own
masked+weighted copy -- so it stays pinned at the D-deep chain; Sorcar iterates to the
monomial fold. B in {5,6} does NOT divide the scoring world 224 (=2^5*7), keeping A
non-uniform.
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


def _sigma(spec):
    B = spec["B"]; a = spec["a"]; c = spec["c"]
    return [(a * b + c) % B for b in range(B)]


def _keep_py(spec, rank):
    B = spec["B"]; off = spec["off"]; L = spec["L"]; stride = spec.get("stride", 1)
    start = (rank + off) % B
    return set((start + stride * j) % B for j in range(L))


def _A_vec(spec, world_size):
    B = spec["B"]
    A = [0.0] * B
    for r in range(world_size):
        w = float(eval(spec["wexpr"].format(v="rank"), {"rank": r}))
        for b in _keep_py(spec, r):
            A[b] += w
    return A


def _code(name, spec, depth):
    part = spec["part"]; B = spec["B"]; off = spec["off"]
    L = spec["L"]; stride = spec.get("stride", 1); wexpr = spec["wexpr"]
    sig = _sigma(spec)
    Lc = [f"def {name}_fn(x, rank, world_size, num_devices,",
          "                 cores_per_device, xm, torch, num_nodes=1):",
          f"    S = {part}; W = world_size",
          f"    SIG = {sig}",
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
               "            jb = SIG[b]",
               "            buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]",
               "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            # invert the monomial map: rec[sigma(b)] = acc[b]/A[b]
            Lc += ["    rec = acc.clone()",
                   f"    for b in range({B}):",
                   "        jb = SIG[b]",
                   "        rec[jb*S:(jb+1)*S] = acc[b*S:(b+1)*S] / A[b]",
                   "    s = rec"]
        else:
            Lc += ["    s = acc"]
    Lc += ["    return s"]
    return "\n".join(Lc) + "\n"


def _ref(spec, depth):
    part = spec["part"]; B = spec["B"]
    sig = _sigma(spec)

    def ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        A = _A_vec(spec, world_size)
        out = s.clone()
        for b in range(B):
            jb = sig[b]
            out[b * part:(b + 1) * part] = A[b] * s[jb * part:(jb + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return ref


def _mk(name, spec, depth, cue):
    ref = _ref(spec, depth)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, spec["part"], spec["B"])
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. "
             if cue else "")
    sig = _sigma(spec)
    doc = (f"Local x ({spec['B']}*{spec['part']},), {spec['B']} blocks of {spec['part']}. "
           f"{COUNT}Final result = a per-block MONOMIAL map of the elementwise SUM of x "
           f"across ranks: block b = A[b]*s[sigma(b)] with the fixed block permutation "
           f"sigma = {sig}, where A[b] is the total of the rank-dependent real weights "
           f"whose rank-dependent block window covers block b.")
    _reg(name, doc, ref, gen, _code(name, spec, depth))


def register_all():
    # DISTINCT-OUTPUT monomial candidates. sigma(b)=(a*b+c)%B is a bijection (gcd(a,B)=1).
    # Windows cover >= B-1 blocks so A[b] > 0 everywhere (the /A invert stays inside the
    # fp32 gate at ws=4). B in {5,6} does not divide 224 so A stays non-uniform.
    _mk("r73_mono_b5_a2_L4_w1_d8",
        {"B": 5, "part": 2048, "a": 2, "c": 0, "off": 0, "L": 4, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r73_mono_b5_a3_L4_w1_d8",
        {"B": 5, "part": 2048, "a": 3, "c": 1, "off": 0, "L": 4, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r73_mono_b6_a5_L5_w1_d8",
        {"B": 6, "part": 2048, "a": 5, "c": 0, "off": 0, "L": 5, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r73_mono_b5_a2_L5_w1_d8",
        {"B": 5, "part": 2048, "a": 2, "c": 2, "off": 0, "L": 5, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r73_mono_b5_a3_L4_w2_d8",
        {"B": 5, "part": 2048, "a": 3, "c": 0, "off": 1, "L": 4, "wexpr": "0.4 + 0.03*({v} % 5)"}, 8, True)
    _mk("r73_mono_b6_a5_L5_w2_d8",
        {"B": 6, "part": 2048, "a": 5, "c": 2, "off": 2, "L": 5, "wexpr": "0.4 + 0.03*({v} % 5)"}, 8, True)
    _mk("r73_mono_b5_a2_L4_w3_d8",
        {"B": 5, "part": 1024, "a": 2, "c": 1, "off": 0, "L": 4, "wexpr": "0.6 + 0.02*({v} % 7)"}, 8, True)
    _mk("r73_mono_b6_a5_L5_w3_d8",
        {"B": 6, "part": 1024, "a": 5, "c": 1, "off": 0, "L": 5, "wexpr": "0.6 + 0.02*({v} % 7)"}, 8, True)
    _mk("r73_mono_b6_a5_L6_w1_d8",
        {"B": 6, "part": 2048, "a": 5, "c": 0, "off": 0, "L": 6, "wexpr": "0.5 + 0.02*{v}"}, 8, True)
    _mk("r73_mono_b5_a3_L5_w1_d8",
        {"B": 5, "part": 2048, "a": 3, "c": 0, "off": 0, "L": 5, "wexpr": "0.5 + 0.05*({v} % 3)"}, 8, True)
    _mk("r73_mono_b6_a5_L5_w4_d8",
        {"B": 6, "part": 2048, "a": 5, "c": 3, "off": 1, "L": 5, "wexpr": "0.45 + 0.02*({v} % 4)"}, 8, True)
    _mk("r73_mono_b5_a2_L4_w4_d8",
        {"B": 5, "part": 2048, "a": 2, "c": 3, "off": 2, "L": 4, "wexpr": "0.55 + 0.02*({v} % 6)"}, 8, True)


register_all()
