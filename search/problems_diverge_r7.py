"""Round 7 -- primitive-discovery lever (optimum needs a NON-OBVIOUS collective).

Lesson so far: when the optimum is "collapse N ARs into 1", both pipelines find
it (fold/count families tie). New hypothesis: overlay's enumerate step reasons
"from the baseline framing" — if the baseline is written with all_gather +
heavy local reshuffle, overlay's K strategies may all stay in the all_gather
family, whereas the true optimum is a DIFFERENT primitive (all_to_all or
collective_permute) that moves far fewer bytes. kiss's open ReAct, seeing sim
feedback, can jump primitives mid-search.

DIR-L "transpose via all_to_all": baseline all_gathers the full (W, S) matrix on
every rank then each rank keeps column `rank` (a transpose/shuffle). The optimum
is a single all_to_all that routes each rank's contribution directly — moves
S bytes/rank instead of W*S. Deterministic and gate-clean in both forms.

DIR-M "rotate via collective_permute": baseline all_gathers then slices the
neighbor's shard (a ring rotation). Optimum: collective_permute with a ring
pairing — one point-to-point exchange instead of a full gather.
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


# --- DIR-L: all_to_all transpose (baseline over-gathers) ---
def _mk_a2a_transpose(name, S=256):
    def _ref(inputs, world_size):
        # rank r contributes row r = x_r (S,). Output on rank r = column r
        # gathered across all rows: out_r[j] = x_j[r-th shard]. We define shards
        # of size S//world? Keep simple: out on rank r = [x_0[r], x_1[r], ...]?
        # Use block transpose: each rank's x is (world*Sblk,) split into W blocks;
        # after a2a rank r holds block r from every rank.
        W = world_size
        Sblk = S
        # inputs[k]['x'] is (W*Sblk,). block j of rank k = x_k[j*Sblk:(j+1)*Sblk]
        outs = []
        for r in range(W):
            parts = [inputs[k]['x'][r * Sblk:(r + 1) * Sblk] for k in range(W)]
            outs.append(torch.cat(parts, dim=0))  # (W*Sblk,)
        return outs

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        N = world_size * S
        pra = [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    Sblk = {S}; W = world_size",
            "    # baseline: all_gather the full matrix, then locally pick this",
            "    # rank's block from every contributor and concatenate.",
            "    g = xm.all_gather(x, dim=0)   # (W*W*Sblk,), x is (W*Sblk,)",
            "    parts = []",
            "    for k in range(W):",
            "        base = k * (W * Sblk)",
            "        parts.append(g[base + rank*Sblk : base + (rank+1)*Sblk])",
            "    return torch.cat(parts, dim=0)   # (W*Sblk,)"]
    _reg(name, f"Local x (world*{S},) holds W blocks of {S}. Produce the block "
         f"transpose: rank r's output concatenates block r taken from every "
         f"rank. (Equivalent to an all_to_all of blocks.)",
         _ref, _gen, "\n".join(body) + "\n")


# --- DIR-M: ring rotation (baseline over-gathers) ---
def _mk_ring_rotate(name, S=256, shift=1):
    def _ref(inputs, world_size):
        W = world_size
        # output on rank r = x of rank (r - shift) mod W
        return [inputs[(r - shift) % W]['x'].clone() for r in range(W)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        pra = [{'x': torch.randn(S) * (0.3 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {S}; W = world_size; shift = {shift}",
            "    # baseline: all_gather everyone's x, then slice the source rank.",
            "    g = xm.all_gather(x, dim=0)   # (W*S,)",
            "    src = (rank - shift) % W",
            "    return g[src*S:(src+1)*S]"]
    _reg(name, f"Local x ({S},). Output = x from rank (rank-{shift}) mod W "
         f"(a ring rotation by {shift}). Baseline all_gathers then slices.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_a2a_transpose("r7_a2a_s128", 128)
    _mk_a2a_transpose("r7_a2a_s256", 256)
    _mk_ring_rotate("r7_ring_s256", 256, shift=1)
    _mk_ring_rotate("r7_ring_s1024", 1024, shift=1)


register_all()
