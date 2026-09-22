"""Round 4 -- reduce_scatter fusion / redundant-gather layout family.

Distinct structure from folds & framing: the baseline moves MORE bytes than
necessary because of a layout choice, and the optimum is a different collective
that moves less. Deterministic, gate-clean in both forms (no error-prone math),
so divergence (if any) is attributable to whether overlay's enumerate reaches
the lower-traffic primitive.

DIR-G "AR->RS+AG collapse": baseline does all_reduce(SUM) then keeps only this
rank's shard (the caller only needs the sharded result), i.e. AR then slice.
The optimum is reduce_scatter(SUM) directly (no full replication). But the
reference OUTPUT is the full replicated vector, so actually the collapse target
is AR itself vs a redundant AR+AG. We instead make the baseline do
reduce_scatter THEN all_gather THEN a redundant second all_gather; optimum drops
the redundant AG. Both gate-clean.

DIR-H "double gather": baseline all_gathers x, then all_gathers a slice of the
result again (redundant re-gather of already-replicated data). Optimum: single
all_gather. Tests whether overlay spots a redundant collective on
already-replicated data as readily as kiss.
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


# --- DIR-G: reduce_scatter + all_gather + redundant all_gather ---
def _mk_rs_ag_redundant(name, part=256):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)  # (W*part,)
        return [s.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, world_size * part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    rs = xm.reduce_scatter(xm.REDUCE_SUM, x, scatter_dim=0,",
            "                           shard_count=W)      # (S,)",
            "    g1 = xm.all_gather(rs, dim=0)              # (W*S,) == AR(x)",
            "    # redundant re-gather of an already-replicated vector",
            "    g2 = xm.all_gather(g1[rank*S:(rank+1)*S], dim=0)",
            "    return g2"]
    _reg(name, f"Local x (world*{part},), S={part}. Return all-rank SUM. "
         f"Baseline: reduce_scatter, all_gather, then a redundant second "
         f"all_gather of its own shard.", _ref, _gen, "\n".join(body) + "\n")


# --- DIR-H: double all_gather (redundant re-gather) ---
def _mk_double_gather(name, N=512):
    def _ref(inputs, world_size):
        # full concatenation of all ranks' x
        cat = torch.cat([inp['x'] for inp in inputs], dim=0)
        return [cat.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    g = xm.all_gather(x, dim=0)                # (W*N,)",
            "    # redundant: re-gather this rank's own slice back to full",
            "    mine = g[rank*N:(rank+1)*N]",
            "    g2 = xm.all_gather(mine, dim=0)            # (W*N,) identical",
            "    return g2"]
    _reg(name, f"Local x ({N},). Return the concatenation of all ranks' x. "
         f"Baseline all_gathers, slices its own part, then all_gathers again.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_rs_ag_redundant("r4_rsag_s256", 256)
    _mk_rs_ag_redundant("r4_rsag_s1024", 1024)
    _mk_double_gather("r4_dblgather_n512", 512)
    _mk_double_gather("r4_dblgather_n2048", 2048)


register_all()
