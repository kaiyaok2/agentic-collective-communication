"""Round 55 -- FAMILY-4 CANDIDATE (non-AR primitive): large-payload all_to_all transpose.

Lessons applied (this session, from the cost-model + confirmed-win audit):
- The divergence lever in fam-1/2/3 is EFFICIENCY, not a correctness-gate failure: overlay writes
  CORRECT but per-rank/per-block Python slice-assign loops that stay at baseline cost, while Sorcar
  vectorizes + collapses. MAX/MIN tied because a broadcast-add shift does not tempt the Python loop.
- collective_permute is HARD-REJECTED at world_size>64 (SIGABRT) -> dead axis at W=224.
- all_to_all SURVIVES ws>64 and records ZERO payload bytes in the cost model (dispatch-only, no
  bandwidth floor, no HBM-peak term), whereas all_gather records INPUT bytes AND the subsequent
  local `cat` of the W*N gathered matrix trips the quadratic HBM-peak penalty at large N.

Structure (block transpose): each rank holds x (W*S,) = W blocks of S. The block-transpose output
on rank r concatenates block r taken from every rank: out_r = cat_k x_k[r*S:(r+1)*S]. This is
EXACTLY one all_to_all of blocks.

Two ways to compute it:
  (A) OPTIMUM (Sorcar target): xm.all_to_all(x, split=0, concat=0, split_count=W). ONE collective,
      records 0 bytes -> dispatch-floor cost, NO bandwidth floor, NO HBM-peak penalty.
  (B) BASELINE / overlay temptation: all_gather the full matrix g (W*W*S,) then locally slice+cat
      this rank's block from every contributor. Moves W* more bytes AND materializes the W*W*S
      gathered tensor -> bandwidth floor + quadratic HBM-peak penalty at large S.

At small S both are dispatch-floor-bound (r7 showed ~1.05 headroom). This round SCALES S so the
all_gather's byte-cost + HBM-peak dominate, opening a real gap ONLY reachable by switching primitive
to all_to_all. The trap: overlay's enumerate, anchored on the all_gather-framed baseline, tends to
stay in the all_gather family and merely tidy the local slice/cat (still gathers W* bytes), whereas
Sorcar's open search can jump to all_to_all on seeing the sim gap.

All ops are pure data movement (gate-exact, no float drift). Pre-screened at W=224 before cloud run.
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


def _mk_a2a_transpose(name, S, cue=True):
    def _ref(inputs, world_size):
        W = world_size
        outs = []
        for r in range(W):
            outs.append(torch.cat([inputs[k]['x'][r * S:(r + 1) * S] for k in range(W)], dim=0))
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
            "    g = xm.all_gather(x, dim=0)   # (W*W*Sblk,)",
            "    parts = []",
            "    for k in range(W):",
            "        base = k * (W * Sblk)",
            "        parts.append(g[base + rank*Sblk : base + (rank+1)*Sblk])",
            "    return torch.cat(parts, dim=0)   # (W*Sblk,)"]
    A2A = ("Equivalent to a single all_to_all of blocks. " if cue else "")
    doc = (f"Local x (world*{S},) holds W blocks of {S}. Produce the block transpose: rank r's "
           f"output concatenates block r taken from every rank. {A2A}")
    _reg(name, doc, _ref, _gen, "\n".join(body) + "\n")


def register_all():
    # payload sweep: gap widens with S as all_gather byte-cost + HBM-peak dominate
    _mk_a2a_transpose("r55_a2a_s1024", 1024)
    _mk_a2a_transpose("r55_a2a_s2048", 2048)
    _mk_a2a_transpose("r55_a2a_s4096", 4096)
    _mk_a2a_transpose("r55_a2a_s8192", 8192)
    _mk_a2a_transpose("r55_a2a_s16384", 16384)
    _mk_a2a_transpose("r55_a2a_s32768", 32768)
    # framing controls (no all_to_all cue in the doc)
    _mk_a2a_transpose("r55_a2a_s8192_res", 8192, cue=False)
    _mk_a2a_transpose("r55_a2a_s16384_res", 16384, cue=False)


register_all()
