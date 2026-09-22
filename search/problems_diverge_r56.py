"""Round 56 -- FAMILY-4 CANDIDATE: hierarchical GROUP-wise all_reduce telescoping.

Corrected trap law (2026-09-21 audit of fam-1/fam-3 confirmed wins): the divergence lever is
EFFICIENCY -- overlay writes CORRECT but per-rank Python slice-assign loops and never collapses the
depth-D collective chain; kiss vectorizes the per-index factor and collapses the chain. A 4th
family therefore needs a NEW per-index structure that (i) tempts the per-rank loop and (ii) has a
deep telescoping chain kiss can collapse.

NEW AXIS (untouched by fam-1/2/3): the GROUPS argument of all_reduce. Ranks are partitioned into
NG=4 fixed groups (rank % 4). The baseline is a depth-D alternating chain:
  odd stage t: m = AR(SUM, a[r] * cur, groups=G)   -- group-wise partial sums; result is
               group-constant but HETEROGENEOUS ACROSS groups (defeats the identity shortcut).
  even stage t: m = AR(SUM, b[r] * cur)             -- global mix of the 4 group values.
a[r] = 1 + 0.5*((r*13) % 5)/5 and b[r] = 1 + 0.25*((r*7) % 9)/9 are per-RANK factors (the loop
temptation), applied via per-rank scalar multiply before each collective.

TELESCOPING: after stage 1, cur is group-constant. For a group-constant vector v with value v_j on
group j:
  group stage: AR_g(SUM, a*v)  -> value_j = (sum_{r in G_j} a[r]) * v_j = A_j * v_j  (diagonal in j)
  global stage: AR(SUM, b*v)   -> sum_j (sum_{r in G_j} b[r]) * v_j = <B, v> (rank-uniform)
So the entire depth-D tail collapses to a 4x4 DIAGONAL+RANK-1 product on the group-value vector --
computable LOCALLY from the group sums. The FOLD = ONE group AR (m1_j = sum_{r in G_j} a[r] x_r)
+ ONE global AR to disseminate cross-group values + local closed-form constants. D collectives -> 2.

Overlay temptation: the hierarchical group/global alternation looks structurally irreducible (its
"hierarchical two-level aggregation" strategy keeps every stage), and a[r]/b[r] invite per-rank
Python loops. Kiss's algebraic collapse gives the fold.

All SUM + local scalar multiplies => fp32-gate-exact (atol 1e-3 absorbs reassociation).
Pre-screened at W=224 before any cloud run. NG=4 divides every test world size (4/8/16/224).
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


def _mk(name, N, depth, cue):
    def _ref(inputs, world_size):
        W = world_size
        xs = [inp['x'] for inp in inputs]
        groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]
        cur = [xs[r].clone() for r in range(W)]
        for t in range(depth):
            if t % 2 == 0:  # group-wise
                nxt = [None] * W
                for j, G in enumerate(groups):
                    s = sum((_a(r) * cur[r] for r in G), torch.zeros(N))
                    for r in G:
                        nxt[r] = s.clone()
                cur = nxt
            else:           # global
                s = sum((_b(r) * cur[r] for r in range(W)), torch.zeros(N))
                cur = [s.clone() for _ in range(W)]
        return cur

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(N) * (0.5 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; NG = {NG}; D = {depth}",
         "    groups = [[r for r in range(W) if r % NG == j] for j in range(NG)]",
         "    a = 1.0 + 0.5 * ((rank * 13) % 5) / 5.0",
         "    b = 1.0 + 0.25 * ((rank * 7) % 9) / 9.0",
         "    cur = x",
         "    for t in range(D):",
         "        if t % 2 == 0:",
         "            cur = xm.all_reduce(xm.REDUCE_SUM, a * cur, groups=groups)",
         "        else:",
         "            cur = xm.all_reduce(xm.REDUCE_SUM, b * cur)",
         "    return cur"]
    COUNT = (f"Computed with {depth} alternating group-wise/global all_reduce(SUM) stages. "
             if cue else "")
    doc = (f"Local x ({N},), distinct per rank. Ranks are partitioned into {NG} groups by rank "
           f"mod {NG}. {COUNT}Odd stages all_reduce a[r]*cur within each group (a[r]=1+0.5*((r*13)"
           f"mod 5)/5); even stages all_reduce b[r]*cur globally (b[r]=1+0.25*((r*7) mod 9)/9). "
           f"Return the depth-{depth} result.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


def register_all():
    _mk("r56_hier_d8_n1024", 1024, 8, True)
    _mk("r56_hier_d6_n1024", 1024, 6, True)
    _mk("r56_hier_d4_n1024", 1024, 4, True)
    _mk("r56_hier_d8_n4096", 4096, 8, True)
    _mk("r56_hier_d8_res", 1024, 8, False)
    _mk("r56_hier_d6_res", 1024, 6, False)


register_all()
