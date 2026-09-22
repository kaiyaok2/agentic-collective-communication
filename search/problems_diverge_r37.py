"""Round 37 -- DISTINCT MECHANISM: group-theoretic (block-rotation) collapse.

Family-1's collapse is arithmetic (scalar field). r35 is combinatorial (partition of
unity). This round tests a GROUP-THEORETIC collapse: the reversible per-stage op is a
BLOCK-ROTATION (a permutation of shard positions), and a D-deep chain of rotations
composes to a SINGLE net rotation (the symmetric group's composition law), so the whole
chain collapses to 1 all_reduce(SUM) + one net block-rotation.

Distinct from r24_pairwise (which TIED): there the roll was a SELF-CANCELLING pair
(rotate then immediately un-rotate = identity), locally visible as a no-op. Here each
stage rotates by a DIFFERENT non-cancelling amount sh[k], and NO adjacent pair cancels;
the collapse is only visible once you compose all D shifts into (sum_k sh[k]) mod W --
a non-locally-visible group identity (L8). The reduction between rotations is a genuine
AR(SUM); rotations commute through it (rotating shard positions then summing = summing
then rotating, since SUM is order-independent over the reduced ranks), which is exactly
why the chain telescopes.

r37_rotsum8: stage k rotates blocks by sh[k]=(k+1) positions, AR(SUM), (no inverse between
stages -- the net rotation is intended). Composes to a single AR(SUM) then rotate-by-
(sum_k (k+1)) mod W. MockTorch lacks roll, so cyclic block-rotation is built via
index_select on a rotated block-index vector.

If rotsum CONFIRMS -> the trap extends to group-theoretic collapses (a genuinely distinct
family). If it TIES -> collapse must be in the scalar/arithmetic domain (overlay sees
permutation structure).

*_count8 truthful count cue; *_res result-only.

fp32: baseline genuine D=8 AR(SUM)+rotate chain (passes gate). Reference: sum across ranks,
then permute the block order by the net cyclic shift (sum_{k=1..D} k) mod W.
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


def _rotate_blocks_code():
    # emit a helper (inline) that rotates block order by `sh` using index_select
    return ("        idx = torch.arange(W*S)\n"
            "        bidx = ((torch.arange(W) + sh) % W).long()\n"
            "        parts = [s[bidx[j]*S:(bidx[j]+1)*S] for j in range(W)]\n"
            "        s = torch.cat(parts, dim=0)\n")


def _rotsum_code(name, part, depth=8):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        sh = st + 1
        L += [f"    sh = {sh} % W",
              "    bidx = ((torch.arange(W) + sh) % W).long()",
              "    parts = [s[bidx[j]*S:(bidx[j]+1)*S] for j in range(W)]",
              "    buf = torch.cat(parts, dim=0)",
              "    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _rotsum_ref(part, depth=8):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        net = sum(range(1, depth)) % world_size  # shifts applied on stages 1..depth-1
        bidx = [((j + net) % world_size) for j in range(world_size)]
        parts = [s[bidx[j] * part:(bidx[j] + 1) * part] for j in range(world_size)]
        out = torch.cat(parts, dim=0)
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    ref = _rotsum_ref(part)

    def _mk_gen(ref):
        def gen(world_size, pattern='uniform', shard_size=None, seed=0):
            pra = _gen_shards(world_size, seed, part)
            return {'per_rank_args': pra, 'shared_args': {},
                    'expected': ref(pra, world_size)}
        return gen

    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    RES = ("Final result = the elementwise SUM of x across ranks, with the block order "
           "cyclically rotated by the net shift (sum of per-stage rotations) mod world.")

    _reg("r37_rotsum8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{RES}",
         ref, _mk_gen(ref), _rotsum_code("r37_rotsum8_count8", part))
    _reg("r37_rotsum8_res",
         f"Local x (world*{part},), S={part}. {RES}",
         ref, _mk_gen(ref), _rotsum_code("r37_rotsum8_res", part))


register_all()
