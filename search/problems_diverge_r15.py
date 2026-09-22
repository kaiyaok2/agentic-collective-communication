"""Round 15 -- PARTIAL-COLLAPSE gradient: does the gap need optimum==1 AR?

Every confirmed win so far collapses a deep chain to a SINGLE all_reduce. Open
question for the mechanism: is Sorcar's edge "removes ALL redundant collectives"
or "removes MORE redundant collectives than overlay"? This round builds baselines
whose FUSED OPTIMUM still contains a genuine collective (cannot go below 2),
because the computation truly needs two DIFFERENT irreducible reductions plus a
removable deep chain wrapped around them.

DIR-Z "wrapped-double-reduce": the answer is AR(SUM,x) + AR(MAX,x) (two
irreducible distinct reductions == 2 collective floor). The baseline computes the
SUM via a deep D-stage scale/unscale chain (collapsible to 1) then the MAX via a
single AR. Optimum = 2 collectives (1 SUM + 1 MAX). Overlay may stay trapped in
the deep SUM chain (2 + (D-1) collectives); kiss should collapse the SUM chain to
1, reaching the 2-collective floor. If Sorcar still wins here, the edge is
"collapses removable depth" regardless of whether the floor is 1 or 2 -- a
cleaner, more general statement of the mechanism than "reaches a single AR."
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


def _gen_flat(world_size, N, seed):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _mk_wrapped_double(name, depth, N=512):
    def _ref(inputs, world_size):
        stk = torch.stack([inp['x'] for inp in inputs], dim=0)
        ssum = stk.sum(dim=0)
        smax = stk.max(dim=0).values
        out = ssum + smax
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    # SUM via a deep collapsible scale/unscale chain",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, s / W)"]
    body += ["    # MAX is a genuinely distinct irreducible reduction",
             "    m = xm.all_reduce(xm.REDUCE_MAX, x)",
             "    return s + m"]
    _reg(name, f"Local x ({N},). Return (all-rank SUM) + (all-rank MAX), "
         f"element-wise. Baseline computes the SUM via {depth} dependent "
         f"all_reduce stages, then the MAX via one all_reduce. The two reductions "
         f"are distinct (cannot be merged).",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_wrapped_double("r15_wrap4", 4, 512)
    _mk_wrapped_double("r15_wrap6", 6, 512)
    _mk_wrapped_double("r15_wrap8", 8, 512)
    _mk_wrapped_double("r15_wrap6_big", 6, 4096)


register_all()
