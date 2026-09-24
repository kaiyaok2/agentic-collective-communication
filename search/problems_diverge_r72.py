"""Round 72 -- FAMILY-7' (crash-free reformulation of the retired world-scaled
bidiagonal family-7 / r68): rank-indexed real-weighted per-block LOWER-BIDIAGONAL
coupling, FIXED-BLOCK (world-independent op count).

The retired family-7 (r68) coupled neighbouring shards with a `for r in range(W)`
tensor-slice loop over W=224 shards -> HLO op count scaled with world size, blew up,
and CRASHED Overlay at 224 ranks (an unfair Phase-4 gate gap). Here the operator is
FIXED-BLOCK (B <= 16 blocks): EVERY tensor-slice loop runs over the fixed block count
B, never over world_size, so the op count is world-independent and neither pipeline
crashes at 224 ranks. World-dependence survives only inside cheap integer/scalar loops
that accumulate the length-B emergent diagonal A and sub-diagonal C (no tensor ops).

Mechanism (distinct operator vs family-1' diagonal): each stage, rank r masks a
rank-DEPENDENT window of blocks and adds a real diagonal weight wd[r] onto block b and
a real sub-diagonal weight wo[r] coupling s[b-1] into block b, then all_reduce(SUM)s.
Because every rank holds the same reduced vector s, the reduced value at block b is
A[b]*s[b] + C[b]*s[b-1], where A[b] = sum of wd[r] over ranks whose diagonal window
covers b and C[b] = sum of wo[r] over ranks whose sub-diagonal window covers b (b>=1;
block 0 has no in-coupling, so the emergent matrix M is LOWER-bidiagonal and strictly
invertible by forward substitution). The depth-D chain telescopes to ONE all_reduce +
a local bidiagonal apply: each non-last stage applies M then inverts it by forward
substitution (a fixed-B scalar loop), so it is the identity on s; only the final stage
applies M without inverting. Overlay cannot fold from the baseline framing -- A and C
are non-local, a single rank only holds its own masked+weighted copy -- so it stays
pinned at the D-deep chain; Sorcar iterates to the bidiagonal fold. B in {5,6} does NOT
divide the scoring world 224 (=2^5*7), keeping A and C non-uniform.
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


def _keep_py(start_off, L, B, rank):
    start = (rank + start_off) % B
    return set((start + j) % B for j in range(L))


def _AC_vec(spec, world_size):
    B = spec["B"]
    A = [0.0] * B
    C = [0.0] * B
    for r in range(world_size):
        wd = float(eval(spec["wdexpr"].format(v="rank"), {"rank": r}))
        wo = float(eval(spec["woexpr"].format(v="rank"), {"rank": r}))
        for b in _keep_py(spec["offd"], spec["Ld"], B, r):
            A[b] += wd
        for b in _keep_py(spec["offo"], spec["Lo"], B, r):
            if b >= 1:               # lower-bidiagonal: block 0 has no in-coupling
                C[b] += wo
    return A, C


def _code(name, spec, depth):
    part = spec["part"]; B = spec["B"]
    offd = spec["offd"]; Ld = spec["Ld"]; wdexpr = spec["wdexpr"]
    offo = spec["offo"]; Lo = spec["Lo"]; woexpr = spec["woexpr"]
    Lc = [f"def {name}_fn(x, rank, world_size, num_devices,",
          "                 cores_per_device, xm, torch, num_nodes=1):",
          f"    S = {part}; W = world_size",
          "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
          f"    A = [0.0]*{B}",
          f"    C = [0.0]*{B}",
          "    for r in range(W):",
          f"        wd = {wdexpr.format(v='r')}",
          f"        wo = {woexpr.format(v='r')}",
          f"        sd = (r + {offd}) % {B}",
          f"        for j in range({Ld}):",
          f"            A[(sd + j) % {B}] += wd",
          f"        so = (r + {offo}) % {B}",
          f"        for j in range({Lo}):",
          f"            b = (so + j) % {B}",
          "            if b >= 1:",
          "                C[b] += wo"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        Lc += [f"    sd = (rank + {offd}) % {B}",
               f"    kd = set((sd + j) % {B} for j in range({Ld}))",
               f"    so = (rank + {offo}) % {B}",
               f"    ko = set((so + j) % {B} for j in range({Lo}))",
               f"    wd = {wdexpr.format(v='rank')}",
               f"    wo = {woexpr.format(v='rank')}",
               "    buf = torch.zeros_like(s)",
               f"    for b in range({B}):",
               "        if b in kd:",
               "            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]",
               "        if b in ko and b >= 1:",
               "            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]",
               "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            # forward-substitution invert of the lower-bidiagonal M -> recover s
            Lc += ["    rec = acc.clone()",
                   "    rec[0:S] = acc[0:S] / A[0]",
                   f"    for b in range(1, {B}):",
                   "        rec[b*S:(b+1)*S] = (acc[b*S:(b+1)*S] - C[b] * rec[(b-1)*S:b*S]) / A[b]",
                   "    s = rec"]
        else:
            Lc += ["    s = acc"]
    Lc += ["    return s"]
    return "\n".join(Lc) + "\n"


def _ref(spec, depth):
    part = spec["part"]; B = spec["B"]

    def ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        A, C = _AC_vec(spec, world_size)
        out = s.clone()
        for b in range(B):
            blk = A[b] * s[b * part:(b + 1) * part]
            if b >= 1:
                blk = blk + C[b] * s[(b - 1) * part:b * part]
            out[b * part:(b + 1) * part] = blk
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
           f"{COUNT}Final result = the elementwise SUM of x across ranks passed through a "
           f"per-block LOWER-BIDIAGONAL map: block b = A[b]*s[b] + C[b]*s[b-1] (block 0 = "
           f"A[0]*s[0]), where A[b] and C[b] are the totals of the rank-dependent real "
           f"weights whose rank-dependent windows cover block b on the diagonal and "
           f"sub-diagonal respectively.")
    _reg(name, doc, ref, gen, _code(name, spec, depth))


def register_all():
    # DISTINCT-OUTPUT bidiagonal candidates. Diagonal windows cover >= B-1 blocks so
    # A[b] > 0 everywhere (forward-sub division stays inside the fp32 gate at ws=4).
    # Sub-diagonal weights are small (well-conditioned M). B in {5,6} does not divide 224.
    _mk("r72_bidiag_b5_Ld4_Lo2_w1_d8",
        {"B": 5, "part": 2048, "offd": 0, "Ld": 4, "wdexpr": "0.8 + 0.02*{v}",
         "offo": 1, "Lo": 2, "woexpr": "0.15 + 0.01*({v} % 5)"}, 8, True)
    _mk("r72_bidiag_b6_Ld5_Lo2_w1_d8",
        {"B": 6, "part": 2048, "offd": 0, "Ld": 5, "wdexpr": "0.8 + 0.02*{v}",
         "offo": 1, "Lo": 2, "woexpr": "0.15 + 0.01*({v} % 5)"}, 8, True)
    _mk("r72_bidiag_b5_Ld4_Lo2_w2_d8",
        {"B": 5, "part": 2048, "offd": 1, "Ld": 4, "wdexpr": "0.7 + 0.03*({v} % 5)",
         "offo": 2, "Lo": 2, "woexpr": "0.2 + 0.01*({v} % 7)"}, 8, True)
    _mk("r72_bidiag_b6_Ld5_Lo2_w2_d8",
        {"B": 6, "part": 2048, "offd": 2, "Ld": 5, "wdexpr": "0.7 + 0.03*({v} % 5)",
         "offo": 3, "Lo": 2, "woexpr": "0.2 + 0.01*({v} % 7)"}, 8, True)
    _mk("r72_bidiag_b5_Ld4_Lo2_w3_d8",
        {"B": 5, "part": 1024, "offd": 0, "Ld": 4, "wdexpr": "0.9 + 0.04*({v} % 3)",
         "offo": 1, "Lo": 2, "woexpr": "0.25 + 0.01*({v} % 3)"}, 8, True)
    _mk("r72_bidiag_b6_Ld5_Lo3_w3_d8",
        {"B": 6, "part": 1024, "offd": 0, "Ld": 5, "wdexpr": "0.9 + 0.04*({v} % 3)",
         "offo": 1, "Lo": 3, "woexpr": "0.25 + 0.01*({v} % 3)"}, 8, True)
    _mk("r72_bidiag_b5_Ld5_Lo2_w1_d8",
        {"B": 5, "part": 2048, "offd": 0, "Ld": 5, "wdexpr": "0.8 + 0.02*{v}",
         "offo": 1, "Lo": 2, "woexpr": "0.18 + 0.01*({v} % 5)"}, 8, True)
    _mk("r72_bidiag_b6_Ld6_Lo2_w1_d8",
        {"B": 6, "part": 2048, "offd": 0, "Ld": 6, "wdexpr": "0.8 + 0.02*{v}",
         "offo": 1, "Lo": 2, "woexpr": "0.18 + 0.01*({v} % 5)"}, 8, True)
    _mk("r72_bidiag_b6_Ld5_Lo4_w1_d8",
        {"B": 6, "part": 2048, "offd": 0, "Ld": 5, "wdexpr": "0.85 + 0.02*{v}",
         "offo": 1, "Lo": 4, "woexpr": "0.15 + 0.01*({v} % 5)"}, 8, True)
    _mk("r72_bidiag_b5_Ld4_Lo2_w4_d8",
        {"B": 5, "part": 2048, "offd": 2, "Ld": 4, "wdexpr": "0.75 + 0.02*({v} % 4)",
         "offo": 3, "Lo": 2, "woexpr": "0.22 + 0.01*({v} % 6)"}, 8, True)


register_all()
