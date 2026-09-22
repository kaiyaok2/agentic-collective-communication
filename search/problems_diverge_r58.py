"""Round 58 -- r56 EXPANSION battery (group-wise AR family candidates).

Prepared while the r56/r57 best-of-4 screens run, so that a confirmed family can be expanded to
~10 members immediately. Variants along orthogonal knobs:

- grpstr / grpcont: strided (r % NG) vs contiguous (r // (W//NG)) group topology. Same telescoping
  algebra, different fold constants + different md5s.
- grponly: ALL stages group-wise (no global mix). After stage 1 the chain is a per-GROUP scalar
  recurrence v_j = A_j^(t) m1_j, so the FOLD = ONE group-AR + a GROUP-INDEXED local constant
  A_{g(rank)}^(D-1) -- the fold constant itself tempts a per-group Python loop. 1 collective.
- depth (d6/d8/d10) and payload (n1024/n8192) sweeps on the alternating structure.
- gswap: a[r] on global stages and b[r] on group stages (swapped roles).

All SUM-linear, NG=4 (divides every test world size 4/8/16/224). Gate-exact. Pre-screened at
W=224 before any cloud run.
"""
import torch
from .problems import CollectiveProblem, register_problem

NG = 4


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


def _a(r):
    return 1.0 + 0.5 * ((r * 13) % 5) / 5.0


def _b(r):
    return 1.0 + 0.25 * ((r * 7) % 9) / 9.0


def _groups(W, strided):
    if strided:
        return [[r for r in range(W) if r % NG == j] for j in range(NG)]
    blk = W // NG
    return [list(range(j * blk, (j + 1) * blk)) for j in range(NG)]


def _gid(r, W, strided):
    return (r % NG) if strided else (r // (W // NG))


def _mk(name, N, depth, strided=True, grponly=False, swap=False, cue=True):
    def _ref(inputs, world_size):
        W = world_size
        xs = [inp['x'] for inp in inputs]
        groups = _groups(W, strided)
        fa, fb = (_b, _a) if swap else (_a, _b)
        cur = [xs[r].clone() for r in range(W)]
        for t in range(depth):
            if grponly or t % 2 == 0:
                nxt = [None] * W
                for G in groups:
                    s = sum((fa(r) * cur[r] for r in G), torch.zeros(N))
                    for r in G:
                        nxt[r] = s.clone()
                cur = nxt
            else:
                s = sum((fb(r) * cur[r] for r in range(W)), torch.zeros(N))
                cur = [s.clone() for _ in range(W)]
        return cur

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(N) * (0.5 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    ga = "1.0 + 0.5 * ((rank * 13) % 5) / 5.0"
    gb = "1.0 + 0.25 * ((rank * 7) % 9) / 9.0"
    if swap:
        ga, gb = gb, ga
    grp_expr = ("[[r for r in range(W) if r % NG == j] for j in range(NG)]" if strided else
                "[list(range(j*(W//NG), (j+1)*(W//NG))) for j in range(NG)]")
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; NG = {NG}; D = {depth}",
         f"    groups = {grp_expr}",
         f"    a = {ga}",
         f"    b = {gb}",
         "    cur = x",
         "    for t in range(D):"]
    if grponly:
        L += ["        cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)"]
    else:
        L += ["        if t % 2 == 0:",
              "            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)",
              "        else:",
              "            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)"]
    L += ["    return cur"]
    topo = "strided (rank mod NG)" if strided else "contiguous (rank // (W/NG))"
    kind = ("group-wise only" if grponly else "alternating group-wise/global")
    COUNT = f"Computed with {depth} {kind} all_reduce(SUM) stages. " if cue else ""
    doc = (f"Local x ({N},), distinct per rank. Ranks partitioned into {NG} {topo} groups. "
           f"{COUNT}Group stages all_reduce a[r]*cur within groups"
           + ("" if grponly else "; global stages all_reduce b[r]*cur")
           + f". a[r]={ga.replace('rank','r')}, b[r]={gb.replace('rank','r')}. "
           f"Return the depth-{depth} result.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


def register_all():
    # group topology
    _mk("r58_grpcont_d8_n1024", 1024, 8, strided=False)
    # all-group-stage chain (fold constant is GROUP-indexed)
    _mk("r58_grponly_d8_n1024", 1024, 8, grponly=True)
    _mk("r58_grponly_d6_n1024", 1024, 6, grponly=True)
    _mk("r58_grponly_d8_cont", 1024, 8, strided=False, grponly=True)
    # depth / payload
    _mk("r58_hier_d10_n1024", 1024, 10)
    _mk("r58_hier_d8_n8192", 8192, 8)
    # swapped factors
    _mk("r58_gswap_d8_n1024", 1024, 8, swap=True)
    # framing control
    _mk("r58_grponly_d8_res", 1024, 8, grponly=True, cue=False)


register_all()
