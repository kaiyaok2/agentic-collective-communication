"""Round 19 -- GENERALIZE the r16 framing effect across families.

r16 proved (byte-identical code, docstring-only variation) that Overlay's trap on
the deep-8 multiplicative chain is DESCRIPTION-driven: narrating the 8 stages
pins overlay at baseline; a result-only or fusion-hint doc frees it. This round
tests whether the description-narration effect GENERALIZES to other collapsible
families, i.e. whether "narrate the procedure" is a universal Overlay trap or
specific to the scale/unscale chain.

For two DIFFERENT collapsible baselines, we emit a matched pair:
  *_narr : docstring narrates the multi-step procedure (predicted: overlay trapped)
  *_res  : docstring states only the result       (predicted: overlay finds fold)

FAMILY A -- linearity fold (r1-style): baseline sums x via a 4-way split-and-
partial-reduce then recombines; optimum = 1 AR. (payload split, all pow-2 exact)
FAMILY B -- redundant-recompute: baseline computes AR(x) three times and averages
(identity); optimum = 1 AR.

If narr diverges and res ties for BOTH families, the framing effect is general.
If only the scale/unscale family responds, r16's effect is construction-specific.
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


def _ref_sum(inputs, world_size):
    s = sum(inp['x'] for inp in inputs)
    return [s.clone() for _ in range(world_size)]


def _gen_factory(N):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref_sum(pra, world_size)}
    return _gen


# FAMILY A: split into 4 quarters, reduce each separately, concat. Optimum = 1 AR.
def _famA_code(name, N):
    q = N // 4
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; Q = {q}",
            "    p0 = xm.all_reduce(xm.REDUCE_SUM, x[:Q])",
            "    p1 = xm.all_reduce(xm.REDUCE_SUM, x[Q:2*Q])",
            "    p2 = xm.all_reduce(xm.REDUCE_SUM, x[2*Q:3*Q])",
            "    p3 = xm.all_reduce(xm.REDUCE_SUM, x[3*Q:])",
            "    return torch.cat([p0, p1, p2, p3], dim=0)"]
    return "\n".join(body) + "\n"


# FAMILY B: recompute AR(x) three times and average (identity). Optimum = 1 AR.
def _famB_code(name, N):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    a = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    b = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    c = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    return (a + b + c) / 3.0"]
    return "\n".join(body) + "\n"


def register_all():
    N = 512
    # FAMILY A pair
    cA = _famA_code("r19_Anarr", N)
    _reg("r19_Anarr",
         f"Local x ({N},). The result is computed by SPLITTING x into 4 quarters, "
         f"running a separate all_reduce on each quarter, then concatenating the "
         f"4 reduced quarters back together in order.",
         _ref_sum, _gen_factory(N), cA)
    _reg("r19_Ares",
         f"Local x ({N},). Return the element-wise all-rank SUM of x.",
         _ref_sum, _gen_factory(N), _famA_code("r19_Ares", N))
    # FAMILY B pair
    _reg("r19_Bnarr",
         f"Local x ({N},). The result is computed by performing THREE independent "
         f"all_reduce passes over x and averaging the three reduced results for "
         f"numerical robustness.",
         _ref_sum, _gen_factory(N), _famB_code("r19_Bnarr", N))
    _reg("r19_Bres",
         f"Local x ({N},). Return the element-wise all-rank SUM of x.",
         _ref_sum, _gen_factory(N), _famB_code("r19_Bres", N))


register_all()
