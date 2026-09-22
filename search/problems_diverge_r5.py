"""Round 5 -- compositional redundancy (needs MULTIPLE independent edits).

Lesson so far: single-collapse problems tend to tie (both one-shot the one edit;
seed noise on error-prone ones). New hypothesis: overlay refines the top-2
strategies for a FIXED R=3 rounds, and each refinement tends to make ONE
structural improvement. If reaching the optimum requires COMPOSING several
INDEPENDENT collapses (drop redundant AR #1, fuse scale, collapse RS+AG, drop
redundant AR #2), a bounded refinement budget may land a partial win while kiss
(30 free steps) composes all of them.

DIR-I "stacked redundancies": baseline = real AR + (redundant identity AR) +
per-shard scale via another AR + (redundant RS+AG round-trip). Four independent
things to remove/fuse; optimum = 1 AR + local scale. Deterministic, gate-clean
at every partial stage (so overlay CAN make progress but may plateau).
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


def _gen_shards(world_size, seed, part):
    torch.manual_seed(seed)
    N = world_size * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _mk_stacked(name, part=256, n_extra_ar=1, rs_roundtrip=True):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)          # (1) real reduce"]
    # (2) per-shard scale via a dependent AR
    body += [
        "    buf = s.clone()",
        "    for r in range(W):",
        "        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W",
        "    s = xm.all_reduce(xm.REDUCE_SUM, buf)        # (2) scale+reduce"]
    # (3..) redundant identity ARs
    for i in range(n_extra_ar):
        body += [f"    s = xm.all_reduce(xm.REDUCE_SUM, s / W)      # redundant AR {i}"]
    # (last) redundant RS+AG round-trip
    if rs_roundtrip:
        body += [
            "    rs = xm.reduce_scatter(xm.REDUCE_SUM, s, scatter_dim=0, shard_count=W)",
            "    s = xm.all_gather(rs / W, dim=0)             # redundant RS+AG"]
    body += ["    return s"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline stacks several "
         f"independent redundancies (a real reduce, a per-shard scale reduce, "
         f"redundant identity reduces, and a reduce_scatter+all_gather "
         f"round-trip). Result = per-shard-scaled AR(x).",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_stacked("r5_stacked2", 256, n_extra_ar=1, rs_roundtrip=False)
    _mk_stacked("r5_stacked3", 256, n_extra_ar=1, rs_roundtrip=True)
    _mk_stacked("r5_stacked4", 256, n_extra_ar=2, rs_roundtrip=True)
    _mk_stacked("r5_stacked4_big", 1024, n_extra_ar=2, rs_roundtrip=True)


register_all()
