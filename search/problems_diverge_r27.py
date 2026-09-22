"""Round 27 -- does the COUNT cue also trap the GLOBALLY-SYMMETRIC r24 families?

r24: global-scalar homogeneity (globalscale8) and permutation-invariance (pairwise8)
BOTH folded for overlay at screen (ratio 1.0) under result-only docstrings —
overlay factors out the single global scalar / sees the roll cancel. r26 then showed
a TRUTHFUL count cue RE-ANCHORS overlay onto genuinely-deep-but-fusible code (the
strided layout). Open question (L14 boundary): is a globally-symmetric collapse
"escapable because the CODE is trivially foldable" (then a count cue should NOT
rescue it — overlay folds anyway), or "escapable because the result-only description
gave overlay nothing to anchor to" (then the count cue SHOULD trap it, like r26)?

This distinguishes L14 from L13/L13'. Both r24 families, deep-8 code UNCHANGED, only
the docstring gains the count assertion "8 dependent all_reduce operations":

  A. globalscale_count8 : global-scalar homogeneity chain + count cue.
  B. pairwise_count8     : permutation-invariance chain + count cue.

Predictions:
  - If count RESCUES both -> L14 dominates: description re-anchoring works regardless
    of whether the collapse is symmetric; the r24 nulls were description artifacts,
    not code-structure facts. (Would broaden the confirmed regime substantially.)
  - If count does NOT rescue (overlay still folds) -> the symmetry escapability is a
    CODE-STRUCTURE fact (overlay's fold is triggered by seeing a single global factor
    / a cancelling term, independent of the count claim). This would BOUND L14 to
    "re-anchors fusible-but-rank-heterogeneous code" and preserve L13 (symmetric
    collapses are code-escapable). Either way it sharpens the mechanism.

fp32-exact: c=2.0 (power of 2); index reorderings exact.
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


def _sum_ref(inputs, world_size):
    s = sum(inp['x'] for inp in inputs)
    return [s.clone() for _ in range(world_size)]


def _flat_gen(N):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _sum_ref(pra, world_size)}
    return _gen


# global-scalar homogeneity chain (identical code to r24_globalscale8)
def _globalscale_code(name, depth=8, N=512):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size; c = 2.0",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, (s * c) / W)",
                 "    s = s / c"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


# permutation-invariance chain (identical code to r24_pairwise8)
def _pairwise_code(name, depth=8, N=512):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    k = int(rank)"]
    for st in range(depth - 1):
        body += ["    sh = 1 + ((k + %d) %% (N - 1))" % st,
                 "    idx = ((torch.arange(N) + sh) % N).long()",
                 "    xr = torch.index_select(x, 0, idx)",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, (s / W) + (xr - xr))"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def register_all():
    N = 512
    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    _reg("r27_globalscale_count8",
         f"Local x ({N},). {COUNT}Final result = the element-wise all-rank SUM of x.",
         _sum_ref, _flat_gen(N), _globalscale_code("r27_globalscale_count8", 8, N))
    _reg("r27_pairwise_count8",
         f"Local x ({N},). {COUNT}Final result = the element-wise all-rank SUM of x.",
         _sum_ref, _flat_gen(N), _pairwise_code("r27_pairwise_count8", 8, N))


register_all()
