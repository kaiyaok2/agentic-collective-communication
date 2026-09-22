"""Round 6 -- misleading-hint lever (attack the signature_doc framing).

Both pipelines read the problem's signature_doc. OverlayCCL's enumerate step is
especially anchored to it (it reasons "from the baseline framing" to produce K
strategies). If the doc strongly suggests a plausible-but-suboptimal STRUCTURE,
overlay's K candidates may cluster around the hinted structure, while kiss's
open ReAct is freer to ignore the hint after seeing sim feedback.

DIR-J "decoy doc": the reference is simply all_reduce(SUM, x) (optimum = 1 AR).
But the doc describes it as a "two-phase hierarchical reduction" (intra-node
then inter-node) and the baseline implements that literally with 2 grouped ARs.
The hint pushes toward keeping the 2-phase structure; the optimum is a single
flat AR. If overlay stays 2-phase and kiss flattens, divergence; if both
flatten, tie (=> hint doesn't bind, parity is robust).

DIR-K "false-dependency doc": doc frames stage2 as depending on stage1's
reduced value, but algebraically stage2 is independent and both fold into one
AR. The narrative dependency is the decoy.
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


def _gen_flat(world_size, N, seed, scale=0.3):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (scale + 0.02 * r)} for r in range(world_size)]


# --- DIR-J: hierarchical-reduction decoy (optimum = flat AR) ---
def _mk_hier_decoy(name, N=1024, group=4):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size; g = {group}",
            "    # phase 1: reduce within groups of g",
            "    intra_groups = [list(range(i, min(i+g, W))) for i in range(0, W, g)]",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x, groups=intra_groups)",
            "    # phase 2: reduce across group leaders (all ranks) -- but s1 now",
            "    # holds per-group partial sums replicated within each group, so a",
            "    # second AR over complementary groups double-counts; correct it",
            "    # by dividing by g before the inter-group reduce.",
            "    lead_groups = [list(range(j, W, g)) for j in range(g)]",
            "    s2 = xm.all_reduce(xm.REDUCE_SUM, s1 / 1.0, groups=lead_groups)",
            "    return s2"]
    _reg(name, f"Local x ({N},). Compute the all-rank SUM as a TWO-PHASE "
         f"HIERARCHICAL reduction: first reduce within node-groups of {group}, "
         f"then reduce across groups. Return the full all-rank sum.",
         _ref, _gen, "\n".join(body) + "\n")


# --- DIR-K: false narrative dependency (two folds into one AR) ---
def _mk_false_dep(name, part=256):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(world_size * part) * (0.3 + 0.02 * r)}
               for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    # stage 1: reduce the raw payload",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    # stage 2: 'depends on' s1 -- rebuild scaled buffer, reduce again",
            "    buf = s1.clone()",
            "    for r in range(W):",
            "        buf[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W",
            "    s2 = xm.all_reduce(xm.REDUCE_SUM, buf)",
            "    return s2"]
    _reg(name, f"Local x (world*{part},), S={part}. Stage 2's per-shard "
         f"scaling DEPENDS ON the reduced result of stage 1, so two dependent "
         f"all_reduces are required. Result = per-shard-scaled AR(x).",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_hier_decoy("r6_hier_g4", 1024, group=4)
    _mk_hier_decoy("r6_hier_g8", 1024, group=8)
    _mk_false_dep("r6_falsedep_s256", 256)
    _mk_false_dep("r6_falsedep_s1024", 1024)


register_all()
