"""Round 39 -- TUNED group-theoretic (pure-permutation) collapse: NO multiplicative scale.

r37 (cyclic block-rotation chain) confirmed at best-of-8 but ESCAPED at best-of-16:
its net permutation is a SINGLE cyclic shift by (sum_k sh[k]) mod W == 28, a trivial
closed-form scalar. So Overlay's best-of-16 draw eventually one-shots the fold
(`torch.cat([s[c:], s[:c]])`). That made r37 a draw-fragile win, not a robust family.

This round REMOVES the closed form. Each stage composes the per-stage rotation-by-k with
a FIXED, NON-AFFINE, non-involutive block permutation PI (an explicit length-B table whose
inverse differs from itself and which is not a stride/affine map). The composition of D such
maps is a generic element of the symmetric group S_B with NO scalar closed form -- the only
way to fold the chain is to COMPOSE the D per-stage index arrays step by step (integer-exact)
and gather once. There is no "shift by c" and no "apply PI once" shortcut; a wrong
composition order or a forward/inverse mix-up fails the fp32 correctness gate.

Mechanism (mirrors r37's identity so the chain is a genuine D=8 AR collapse, but with a
non-abelian, non-closed-form net permutation):
    s = AR(SUM, x)                      # every rank now holds the full sum
    for k in 1..D-1:                    # D-1 further dependent AR(SUM)
        bidx_k = compose(rotate_by_k, PI)   # a genuine block permutation of s
        buf    = gather blocks of s by bidx_k
        s      = AR(SUM, buf) / W        # AR over identical-across-ranks data => identity
    return s                            # == AR(SUM,x) with blocks permuted by the NET map
The optimum is 1 AR(SUM) + apply the NET block permutation (composition of the D-1 stage
maps). Because PI is non-affine the net map has no closed form -- Overlay's enumerate-from-
baseline guesses a shift/single-PI fold, fails the gate, discards it, and stays pinned at the
deep baseline; Kiss's ReAct reads the gate error and iterates the index composition to
correctness. NO per-shard scale anywhere: this is a PURE-PERMUTATION collapse.

If r39 CONFIRMS at best-of-8 AND leads on average sim -> a genuine SECOND divergence family
(group-theoretic / pure-permutation, distinct from family-1's multiplicative per-shard scale).
If it TIES -> even a non-closed-form permutation is foldable by both, and no second family
exists.

Blocks: B=8 (independent of world size), P=256 each; block permutation reorders the 8
contiguous segments of the length-B*P vector. Built with torch.cat on gathered slices
(MockTorch has no roll/index_select-on-blocks helper; explicit slice+cat is exact).

*_count8 truthful count cue; *_res result-only.

fp32: baseline is a genuine D=8 AR(SUM)+permute chain (passes gate). Reference: sum across
ranks, then apply the exact net block permutation (composition of the D-1 stage maps).
"""
import torch
from .problems import CollectiveProblem, register_problem


# Fixed base permutation of B=8 blocks: non-affine (not j->a*j+b mod 8), non-involutive
# (PI != PI^{-1}), no fixed cyclic structure. Chosen so composition has no closed form.
PI = [3, 0, 5, 7, 1, 6, 2, 4]
NBLOCK = 8


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
    N = NBLOCK * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


def _stage_bidx(k):
    # stage k: rotate the B blocks by k, THEN apply the fixed non-affine PI.
    # returns bidx s.t. output block j is sourced from input block bidx[j].
    B = NBLOCK
    rot = [((j + k) % B) for j in range(B)]      # rotate-by-k
    return [rot[PI[j]] for j in range(B)]        # then PI (non-commuting compose)


def _perm_code(name, part, depth=8):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}; W = world_size",
         f"    PI = {PI}",
         "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        k = st + 1
        L += [f"    k = {k} % B",
              "    rot = [((j + k) % B) for j in range(B)]",
              "    bidx = [rot[PI[j]] for j in range(B)]",
              "    parts = [s[bidx[j]*S:(bidx[j]+1)*S] for j in range(B)]",
              "    buf = torch.cat(parts, dim=0)",
              "    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W"]
    L += ["    return s"]
    return "\n".join(L) + "\n"


def _perm_ref(part, depth=8):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        B = NBLOCK
        # compose the D-1 per-stage block permutations (stages 1..depth-1) in order.
        # cur[newpos] = source block currently at newpos
        cur = list(range(B))
        for st in range(depth - 1):
            k = (st + 1) % B
            stage = _stage_bidx(k)
            cur = [cur[stage[j]] for j in range(B)]
        parts = [s[cur[j] * part:(cur[j] + 1) * part] for j in range(B)]
        out = torch.cat(parts, dim=0)
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 4096  # large payload so 8 AR >> 1 AR: the fold headroom is decisive, not floor-thin
    ref = _perm_ref(part)

    def _mk_gen(ref):
        def gen(world_size, pattern='uniform', shard_size=None, seed=0):
            pra = _gen_shards(world_size, seed, part)
            return {'per_rank_args': pra, 'shared_args': {},
                    'expected': ref(pra, world_size)}
        return gen

    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    RES = ("Final result = the elementwise SUM of x across ranks, with the 8 blocks "
           "reordered by the composition of per-stage (rotate-by-k then fixed permutation) "
           "block maps.")

    _reg("r39_permsum8_count8",
         f"Local x (8*{part},), 8 blocks of {part}. {COUNT}{RES}",
         ref, _mk_gen(ref), _perm_code("r39_permsum8_count8", part))
    _reg("r39_permsum8_res",
         f"Local x (8*{part},), 8 blocks of {part}. {RES}",
         ref, _mk_gen(ref), _perm_code("r39_permsum8_res", part))


register_all()
