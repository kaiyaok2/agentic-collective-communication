"""Round 43 -- FAMILY-2 EXPANSION: distinct rank-indexed ROUTING variants.

Family-2 (r40) confirmed 4 members (all D=8, L=3). This round widens the family to
15 members along axes that produce GENUINELY DIFFERENT computations (different
reference outputs and/or tensor sizes) -- NOT docstring/constant duplicates -- while
keeping the SAME trap mechanism and the D=8, L=3 trap knobs fixed.

Mechanism (unchanged from r40): each of D-1 dependent stages, rank r masks a
rank-indexed WINDOW of L blocks (start (r+OFF)%B, optional stride), zeroing the rest;
the window DEPENDS ON `rank`, so the inter-collective buffers are genuinely
RANK-HETEROGENEOUS -> AR(SUM) is a real reduction, not an identity -> Overlay's
"AR-of-identical-data = identity" shortcut is FALSE. Net over the chain = per-block
scale by c[b] = #ranks whose window covers b (each intermediate stage /c to stay
bounded; final stage leaves it applied).

IMPORTANT count-vector note (corrects the r40 docstring): the gate runs at world_size
W = num_nodes*32 = 224. Because 224 is divisible by B=8, c degenerates to a UNIFORM
scalar [84,84,...] at B=8 -- so the r40 win is driven by the depth-8 rank-indexed
MASKING chain (rank-heterogeneous intermediate buffers), NOT by block-heterogeneity of
the net. To get a genuinely NON-UNIFORM net at W=224 we vary B to values 224 is NOT
divisible by (B in {6,9,10,11,12} -> heterogeneous c, verified). Those variants are
therefore structurally distinct: different net vectors AND different tensor lengths.

Distinctness axes (all keep D=8, L=3, and the same masking mechanism):
  - BLOCK-COUNT B in {6,9,10,11,12}: non-uniform c at W=224 + different tensor length
    (B*part). Strongest distinctness axis -- different reference output per B.
  - PAYLOAD part in {256,384}: different tensor size (avoid 1024, which hurt r40's
    fold reliability per-seed).
  - ROUTING TOPOLOGY: contiguous window vs STRIDED window (keep every-2nd block) ->
    different overlap-count vector, different reference output.
  - FRAMING: truthful op-count cue vs result-only docstring.

All baselines are numerically checked to pass the fp32 gate (atol=1e-5) at W=224
before launch (see _selfcheck at bottom / the r43 pre-screen).
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


def _window_blocks(start, L, B, stride):
    """The set of block indices this rank keeps: L blocks from `start`, step `stride`."""
    return set((start + stride * j) % B for j in range(L))


def _count_vec(world_size, B, L, off, stride):
    """c[b] = #ranks whose (start=(r+off)%B, step=stride, len=L) window covers b."""
    c = [0] * B
    for r in range(world_size):
        for b in _window_blocks((r + off) % B, L, B, stride):
            c[b] += 1
    return c


def _route_code(name, part, depth, B, L, off, stride):
    Ls = [
        f"def {name}_fn(x, rank, world_size, num_devices,",
        "                 cores_per_device, xm, torch, num_nodes=1):",
        f"    S = {part}; B = {B}; W = world_size; L = {L}; OFF = {off}; STR = {stride}",
        "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
        "    # per-block overlap count c[b] = #ranks whose window covers block b",
        "    c = [0]*B",
        "    for r in range(W):",
        "        st = (r + OFF) % B",
        "        for j in range(L):",
        "            c[(st + STR*j) % B] += 1",
    ]
    for st in range(depth - 1):
        last = (st == depth - 2)
        Ls += [
            "    # rank-indexed routing: THIS rank masks its own length-L window",
            "    start = (rank + OFF) % B",
            "    keep = set((start + STR*j) % B for j in range(L))",
            "    buf = torch.zeros_like(s)",
            "    for b in range(B):",
            "        if b in keep:",
            "            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]",
            "    acc = xm.all_reduce(xm.REDUCE_SUM, buf)",
        ]
        if not last:
            Ls += [
                "    for b in range(B):",
                "        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]",
            ]
        Ls += ["    s = acc"]
    Ls += ["    return s"]
    return "\n".join(Ls) + "\n"


def _route_ref(part, depth, B, L, off, stride):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        c = _count_vec(world_size, B, L, off, stride)
        out = s.clone()
        for b in range(B):
            out[b * part:(b + 1) * part] = c[b] * s[b * part:(b + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, depth, B, L, off, stride, count_cue):
    ref = _route_ref(part, depth, B, L, off, stride)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part, B)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': ref(pra, world_size)}

    COUNT = (f"The result is computed using {depth} dependent all_reduce operations. "
             if count_cue else "")
    topo = ("contiguous" if stride == 1 else f"stride-{stride}")
    RES = (f"Final result = the elementwise SUM of x across ranks, with each of the "
           f"{B} blocks scaled by the number of ranks whose length-{L} {topo} "
           f"window (window start = (rank+{off}) mod {B}) covers that block.")
    doc = f"Local x ({B}*{part},), {B} blocks of {part}. {COUNT}{RES}"
    _reg(name, doc, ref, gen, _route_code(name, part, depth, B, L, off, stride))


OFF = 2


def register_all():
    D = 8; L = 3  # trap knobs held fixed

    # --- BLOCK-COUNT sweep (non-uniform c at W=224; different tensor length) ---
    _mk("r43_route_B6_d8_L3_count8",  256, D, 6,  L, OFF, 1, True)
    _mk("r43_route_B9_d8_L3_count8",  256, D, 9,  L, OFF, 1, True)
    _mk("r43_route_B10_d8_L3_count8", 256, D, 10, L, OFF, 1, True)
    _mk("r43_route_B11_d8_L3_count8", 256, D, 11, L, OFF, 1, True)
    _mk("r43_route_B12_d8_L3_count8", 256, D, 12, L, OFF, 1, True)

    # --- PAYLOAD sweep (different tensor size; 384 is in the working range) ---
    _mk("r43_route_B8_p384_count8",   384, D, 8,  L, OFF, 1, True)
    _mk("r43_route_B10_p384_count8",  384, D, 10, L, OFF, 1, True)

    # --- ROUTING TOPOLOGY (strided window -> different overlap counts) ---
    _mk("r43_route_B8_strided_count8",  256, D, 8,  L, OFF, 2, True)
    _mk("r43_route_B10_strided_count8", 256, D, 10, L, OFF, 2, True)

    # --- FRAMING (result-only doc, on heterogeneous-B variants) ---
    _mk("r43_route_B10_d8_L3_res", 256, D, 10, L, OFF, 1, False)
    _mk("r43_route_B12_d8_L3_res", 256, D, 12, L, OFF, 1, False)


register_all()
