"""Round 38 -- SECOND-FAMILY ROBUSTNESS: a DIFFERENT group action (reflection).

r37 (cyclic block-rotation chain composing to one net rotation) CONFIRMED at best-of-8 --
a candidate GROUP-THEORETIC second family with NO multiplicative scale. To test whether
that family is robust (not r37-cyclic-specific), r38 uses a DIFFERENT group action from
the same symmetric group: BLOCK REVERSAL (a reflection / involution), composed with a
per-stage cyclic offset so consecutive stages don't trivially cancel.

Mechanism: stage k applies a block-order permutation pi_k = reverse-then-rotate-by-k,
AR(SUM), (no inverse between stages). The composition of D such permutations is a single
net permutation pi = pi_{D-1} o ... o pi_1 (symmetric-group product), so the chain
collapses to 1 AR(SUM) + one net block-permutation. Distinct from r37: reflections are
involutions (order 2), so the group structure is dihedral, not cyclic -- a different
non-locally-visible composition identity.

If r38 CONFIRMS -> the group-theoretic family is robust across group actions (strong
second-family evidence). If r38 TIES while r37 held -> r37's win was cyclic-specific
(overlay can fuse reflections but not rotations), which is itself an interesting boundary.

Reflection built via index_select on a reversed block-index vector (MockTorch has no flip).
Reference composes the exact net permutation in numpy-free torch.

*_count8 truthful count cue; *_res result-only.

fp32: baseline genuine D=8 AR(SUM)+permute chain (passes gate). Reference: sum across
ranks, then apply the net block permutation (composition of the D per-stage reverse-rotate
maps), each stage's /W keeping the repeated AR(SUM) over identical data an exact identity.
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


def _perm_stage(world_size, k):
    # block index map for stage k: reverse then rotate-by-k.  new_pos j <- src block idx
    # reversal: rev[j] = W-1-j ; then rotate: rr[j] = rev[(j + k) % W]
    W = world_size
    return [((W - 1 - ((j + k) % W)) % W) for j in range(W)]


def _reflect_code(name, part, depth=8):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        k = st + 1
        L += [f"    k = {k} % W",
              "    bidx = [((W - 1 - ((j + k) % W)) % W) for j in range(W)]",
              "    parts = [s[bidx[j]*S:(bidx[j]+1)*S] for j in range(W)]",
              "    buf = torch.cat(parts, dim=0)",
              "    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _reflect_ref(part, depth=8):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        W = world_size
        # compose the D-1 per-stage block permutations (stages 1..depth-1)
        cur = list(range(W))  # cur[newpos] = source block currently at newpos
        for st in range(depth - 1):
            k = (st + 1) % W
            stage = [((W - 1 - ((j + k) % W)) % W) for j in range(W)]
            cur = [cur[stage[j]] for j in range(W)]
        parts = [s[cur[j] * part:(cur[j] + 1) * part] for j in range(W)]
        out = torch.cat(parts, dim=0)
        return [out.clone() for _ in range(W)]
    return _ref


def register_all():
    part = 256
    ref = _reflect_ref(part)

    def _mk_gen(ref):
        def gen(world_size, pattern='uniform', shard_size=None, seed=0):
            pra = _gen_shards(world_size, seed, part)
            return {'per_rank_args': pra, 'shared_args': {},
                    'expected': ref(pra, world_size)}
        return gen

    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    RES = ("Final result = the elementwise SUM of x across ranks, with the block order "
           "transformed by the composition of per-stage reverse-then-rotate permutations.")

    _reg("r38_reflect8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{RES}",
         ref, _mk_gen(ref), _reflect_code("r38_reflect8_count8", part))
    _reg("r38_reflect8_res",
         f"Local x (world*{part},), S={part}. {RES}",
         ref, _mk_gen(ref), _reflect_code("r38_reflect8_res", part))


register_all()
