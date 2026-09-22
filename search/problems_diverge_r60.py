"""Round 60 -- FAMILY-2 EXPANSION (non-duplicate): rank-indexed ROUTING, per-block
overlap-COUNT scale, with NEW window structures.

Family-2 mechanism (unchanged): each stage, rank r masks a rank-DEPENDENT set of
blocks (its "window") and all_reduce(SUM)s the masked buffer. The reduced value is
c[b]*s[b] where c[b] = #ranks whose window covers block b -- a STATIC integer count
that EMERGES from routing overlap (no explicit scalar in the code). The depth-D chain
telescopes to ONE all_reduce(SUM,x) + a local per-block multiply by the count vector c.
Overlay cannot fold it from the baseline framing (a single rank only holds its own
masked copy; the count is non-local), so it stays pinned at the D-deep chain; Kiss
iterates to the count fold.

DISTINCT from the confirmed r40/r43 set (which all use a CONTIGUOUS length-L window,
start=(r+off)%B, B=8, off=2). Here we introduce NEW window SHAPES that produce
DIFFERENT count vectors c and DIFFERENT reference outputs (distinct md5):
  - strided window (step k): keep {(start + k*j) % B : j<L}
  - reverse start:           start = (off - r) % B
  - double window:           two disjoint sub-windows per rank
  - rank-dependent length:   L_r = Lmin + (r % Lspan)
  - spread start:            start = (2*r + off) % B
  - larger block count:      B = 16
No (shape, B, off, L, depth, payload) tuple here duplicates an existing family-2
problem. The count vector c stays bounded and the /c unscale keeps the baseline inside
the fp32 gate, mirroring the confirmed family-2 skeleton.
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


# We keep code + reference in lock-step by generating both from one spec dict.
def _emit_keep(spec):
    """Emit python lines building set `keep` from `rank`, using B/off/L/etc."""
    B = spec["B"]; off = spec["off"]
    L = spec["L"]; stride = spec.get("stride", 1)
    shape = spec["shape"]
    lines = [f"    B = {B}; OFF = {off}"]
    if shape == "spread":
        lines += [f"    start = (2*rank + OFF) % B"]
    elif shape == "reverse":
        lines += [f"    start = (OFF - rank) % B"]
    else:
        lines += [f"    start = (rank + OFF) % B"]
    if shape == "double":
        L2 = spec["L2"]
        lines += [f"    keep = set((start + j) % B for j in range({L2}))",
                  f"    keep |= set((start + B//2 + j) % B for j in range({L2}))"]
    elif shape == "varL":
        lines += [f"    Lr = {spec['lmin']} + (rank % {spec['lspan']})",
                  f"    keep = set((start + {stride}*j) % B for j in range(Lr))"]
    else:  # contiguous(stride=1) / strided(stride>1) / spread / reverse
        lines += [f"    keep = set((start + {stride}*j) % B for j in range({L}))"]
    return lines


def _keep_py(spec, rank):
    B = spec["B"]; off = spec["off"]; L = spec["L"]; stride = spec.get("stride", 1)
    shape = spec["shape"]
    if shape == "spread":
        start = (2 * rank + off) % B
    elif shape == "reverse":
        start = (off - rank) % B
    else:
        start = (rank + off) % B
    if shape == "double":
        L2 = spec["L2"]
        keep = set((start + j) % B for j in range(L2))
        keep |= set((start + B // 2 + j) % B for j in range(L2))
    elif shape == "varL":
        Lr = spec["lmin"] + (rank % spec["lspan"])
        keep = set((start + stride * j) % B for j in range(Lr))
    else:
        keep = set((start + stride * j) % B for j in range(L))
    return keep


def _count_vec(spec, world_size):
    B = spec["B"]
    c = [0] * B
    for r in range(world_size):
        for b in _keep_py(spec, r):
            c[b] += 1
    return c


def _route_code(name, spec, depth):
    part = spec["part"]; B = spec["B"]
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
         f"    c = [0]*{B}",
         "    for r in range(W):"]
    # build count via the same keep-set logic, indented under the r-loop
    keep_lines_r = _emit_keep({**spec, }) if False else None
    # inline count builder (rank -> r)
    off = spec["off"]; stride = spec.get("stride", 1); shape = spec["shape"]
    if shape == "spread":
        L += ["        st = (2*r + OFF0) % B0".replace("OFF0", str(off)).replace("B0", str(B))]
    elif shape == "reverse":
        L += ["        st = (OFF0 - r) % B0".replace("OFF0", str(off)).replace("B0", str(B))]
    else:
        L += ["        st = (r + OFF0) % B0".replace("OFF0", str(off)).replace("B0", str(B))]
    if shape == "double":
        L2 = spec["L2"]
        L += [f"        ks = set((st + j) % {B} for j in range({L2}))",
              f"        ks |= set((st + {B}//2 + j) % {B} for j in range({L2}))"]
    elif shape == "varL":
        L += [f"        Lr = {spec['lmin']} + (r % {spec['lspan']})",
              f"        ks = set((st + {stride}*j) % {B} for j in range(Lr))"]
    else:
        L += [f"        ks = set((st + {stride}*j) % {B} for j in range({spec['L']}))"]
    L += ["        for b in ks:",
          "            c[b] += 1"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        L += _emit_keep(spec)
        L += ["    buf = torch.zeros_like(s)",
              f"    for b in range({B}):",
              "        if b in keep:",
              "            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]",
              "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            L += [f"    for b in range({B}):",
                  "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]"]
        L += ["    s = acc"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _route_ref(spec, depth):
    part = spec["part"]; B = spec["B"]

    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        c = _count_vec(spec, world_size)
        out = s.clone()
        for b in range(B):
            out[b * part:(b + 1) * part] = c[b] * s[b * part:(b + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, spec, depth, cue):
    ref = _route_ref(spec, depth)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, spec["part"], spec["B"])
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. "
             if cue else "")
    doc = (f"Local x ({spec['B']}*{spec['part']},), {spec['B']} blocks of {spec['part']}. "
           f"{COUNT}Final result = the elementwise SUM of x across ranks, with each block "
           f"scaled by the number of ranks whose rank-dependent block window covers it.")
    _reg(name, doc, ref, gen, _route_code(name, spec, depth))


def register_all():
    # DISTINCT-OUTPUT candidates. The family-2 reference (count vector c) depends on
    # (B, off, L, shape), NOT on depth -- so we vary those, never depth alone.
    # Crucially B in {5,6} does NOT divide the scoring world size 224 (= 2^5*7), so the
    # per-block counts stay NON-uniform at W=224 -> genuinely distinct outputs (md5),
    # unlike B=8/16 which divide 224 into a uniform constant. Distinct from the confirmed
    # r40 (B=8) and r43 (B=9..12) sets. Window covers >= B-3 blocks so ws=4 is fully
    # covered (every c[b] > 0, keeping the /c unscale inside the fp32 gate).
    _mk("r60_b5_L2_d8",     {"shape": "contig", "B": 5, "off": 0, "L": 2, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b5_L3_d8",     {"shape": "contig", "B": 5, "off": 0, "L": 3, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b5_o1_L2_d8",  {"shape": "contig", "B": 5, "off": 1, "L": 2, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b6_L3_d8",     {"shape": "contig", "B": 6, "off": 0, "L": 3, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b6_L4_d8",     {"shape": "contig", "B": 6, "off": 0, "L": 4, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b6_o2_L3_d8",  {"shape": "contig", "B": 6, "off": 2, "L": 3, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b6_L5_d8",     {"shape": "contig", "B": 6, "off": 0, "L": 5, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b6_varL_d8",   {"shape": "varL", "B": 6, "off": 0, "lmin": 3, "lspan": 2, "L": 3, "stride": 1, "part": 256}, 8, True)
    _mk("r60_b5_varL_d8",   {"shape": "varL", "B": 5, "off": 0, "lmin": 2, "lspan": 2, "L": 2, "stride": 1, "part": 256}, 8, True)


register_all()
