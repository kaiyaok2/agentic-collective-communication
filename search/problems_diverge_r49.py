"""Round 49 -- FAM-4 candidate: STACKED MULTI-STATISTIC AR fusion.

Distinct from confirmed families:
  fam-1 static per-RANK multiplicative scale; fam-2 rank-indexed routing count;
  fam-3 data-dependent continuous scale + cross-primitive RS+AG==AR.

fam-4 mechanism: the baseline issues K SEPARATE all_reduce(SUM) collectives, each on a
DIFFERENT elementwise function of x (x, x^2, |x|, relu(x), ...), and returns them
concatenated -- the classic LayerNorm/BatchNorm/moment-statistics pattern (need sum,
sum-of-squares, ... across ranks). Because AR(SUM) is elementwise-linear over the shard
axis, AR(SUM, cat[f0(x), f1(x), ...]) == cat[AR(SUM,f0(x)), AR(SUM,f1(x)), ...], so the K
collectives FUSE into ONE AR of the concatenated inputs, split afterward. The trap is a
COLLECTIVE-COUNT fold by INPUT CONCATENATION across different nonlinear functions --
NOT a scale (fam-1/3), NOT routing (fam-2), NOT a cross-primitive identity (fam-3 xcoll).
Multi-output. Real: fused moment stats in normalization layers.

All SUM primitives -> gate-safe. Baseline + fold both pass fp32 gate at W=224 (pre-screened).
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


def _gen(world_size, seed, N):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# The K statistic functions, as (code-expr-on t, python-fn) pairs.
_FUNCS = {
    "sum":    ("t",            lambda t: t),
    "sq":     ("t*t",          lambda t: t * t),
    "abs":    ("t.abs()",      lambda t: t.abs()),
    "relu":   ("t.clamp(min=0.0)", lambda t: t.clamp(min=0.0)),
    "cube":   ("t*t*t",        lambda t: t * t * t),
}


def _stat_code(name, N, keys):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         "    t = x",
         "    parts = []"]
    for k in keys:
        expr = _FUNCS[k][0]
        L += [f"    parts.append(xm.all_reduce(xm.REDUCE_SUM, {expr}))"]
    L += ["    return torch.cat(parts)"]
    return "\n".join(L) + "\n"


def _stat_ref(keys):
    def _ref(inputs, world_size):
        xs = [inp['x'] for inp in inputs]
        s = sum(xs)  # elementwise sum across ranks of x
        # careful: each stat is AR(SUM, f(x)) = sum_r f(x_r), NOT f(sum_r x_r)
        parts = []
        for k in keys:
            f = _FUNCS[k][1]
            parts.append(sum(f(x) for x in xs))
        out = torch.cat(parts)
        return [out.clone() for _ in range(world_size)]
    return _ref


def _mk(name, N, keys, cue):
    ref = _stat_ref(keys)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, N)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    names = {"sum": "the sum", "sq": "the sum of squares", "abs": "the sum of absolute values",
             "relu": "the sum of relu", "cube": "the sum of cubes"}
    listed = ", ".join(names[k] for k in keys)
    COUNT = (f"Computed with {len(keys)} separate all_reduce operations. " if cue else "")
    doc = (f"Local x, length {N}. {COUNT}"
           f"Final result = the concatenation over ranks of [{listed}] of x "
           f"(each reduced elementwise across ranks), in that order (length {len(keys)}*{N}).")
    _reg(name, doc, ref, gen, _stat_code(name, N, keys))


def register_all():
    # K-sweep (number of fused statistics), payload 512
    _mk("r49_stat_k2",  512, ["sum", "sq"], True)              # mean/var pattern
    _mk("r49_stat_k3",  512, ["sum", "sq", "abs"], True)
    _mk("r49_stat_k4",  512, ["sum", "sq", "abs", "relu"], True)
    _mk("r49_stat_k5",  512, ["sum", "sq", "abs", "relu", "cube"], True)
    # payload sweep at k3
    _mk("r49_stat_k3_p256", 256, ["sum", "sq", "abs"], True)
    _mk("r49_stat_k3_p1024", 1024, ["sum", "sq", "abs"], True)
    # framing controls (result-only)
    _mk("r49_stat_k3_res", 512, ["sum", "sq", "abs"], False)
    _mk("r49_stat_k4_res", 512, ["sum", "sq", "abs", "relu"], False)


register_all()
