"""Round 52 -- FAMILY-4 CANDIDATE: collective_permute COMPOSITION (non-AR axis).

Lesson from the completed 3-family law: a trap needs Overlay's obvious one-shot guess to
FAIL the fp32 gate. The r39 close showed a pure permutation BETWEEN all_reduce collectives
is trivially foldable because post-AR data is rank-IDENTICAL (identity-shortcut). But
xm.collective_permute moves DISTINCT per-rank tensors -- there is NO replication, so each
stage is a genuine data movement and the identity-shortcut cannot apply. Folding a depth-D
permute chain requires actually COMPOSING the routing (rank-index arithmetic), which is
error-prone and non-local. This is a NEW primitive axis (no reduction at all).

collective_permute semantics (from correctness_test executor): pairs are (src, dst) meaning
rank `src` sends its tensor to rank `dst`; after the op rank r holds the tensor that started
on the unique src with dst==r. A "rotation by +k in rank space" is pairs=[(s,(s+k)%W) for s],
after which rank r holds x_orig[(r-k)%W].

Mechanisms screened here (all pure routing -> gate-exact, no float drift):
  * rot_net:  D rotations by k_t; net = single rotation by K=sum(k_t)%W -> fold to 1 permute.
  * rot_id:   D rotations with sum(k_t)%W == 0; net = IDENTITY -> fold to ZERO comm (`return x`).
              The trap: prove a depth-D chain of genuine, non-trivial data movements COMPOSES
              to a no-op. Overlay's enumerate sees D real permutes; Sorcar can iterate.
  * permrot:  D rotations interleaved with a fixed non-affine rank permutation PI applied each
              stage; net = a single composed permutation (non-closed-form) -> fold to 1 permute.

All routing is WITHIN a single logical ring over the 224 ranks (no cross-node ring split at
the pair level -- the sim only SIGABRTs cross-node ring *patterns* declared as hardware probes,
not correctness-level pair lists). Pre-screened before any cloud run.
"""
import torch
from .problems import CollectiveProblem, register_problem

# a fixed non-affine permutation of a period; expanded to W by blocks (see _pi_perm).
_PI8 = [3, 0, 5, 7, 1, 6, 2, 4]


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
    # each rank a DISTINCT tensor so permutation composition actually matters
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r) + 0.01 * r} for r in range(world_size)]


def _pi_perm(W):
    # a fixed permutation of range(W): apply _PI8 within each block of 8, identity if W%8.
    if W % 8 != 0:
        return list(range(W))
    p = list(range(W))
    for base in range(0, W, 8):
        for i in range(8):
            p[base + i] = base + _PI8[i]
    return p


# ---------- rot_net: D rotations, net = single rotation by K ----------
def _mk_rot_net(name, N, shifts, cue):
    K = sum(shifts)

    def _ref(inputs, world_size):
        Kn = K % world_size
        return [inputs[(r - Kn) % world_size]['x'].clone() for r in range(world_size)]

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, N)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; shifts = {list(shifts)}",
         "    cur = x",
         "    for k in shifts:",
         "        pairs = [(s, (s + k) % W) for s in range(W)]",
         "        cur = xm.collective_permute(cur, pairs=pairs)",
         "    return cur"]
    COUNT = (f"Computed with {len(shifts)} collective_permute rotations. " if cue else "")
    doc = (f"Local x ({N},), distinct per rank. {COUNT}Return the tensor obtained by rotating "
           f"the per-rank tensors around the {N}-length ring of ranks; net result on rank r is "
           f"the original tensor of rank (r - {K % 224}) mod W.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


# ---------- rot_id: D rotations summing to 0 mod W -> identity ----------
def _mk_rot_id(name, N, shifts, cue):
    assert sum(shifts) % 224 == 0, "shifts must sum to 0 mod 224 (W) for identity"

    def _ref(inputs, world_size):
        Kn = sum(shifts) % world_size
        return [inputs[(r - Kn) % world_size]['x'].clone() for r in range(world_size)]

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, N)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; shifts = {list(shifts)}",
         "    cur = x",
         "    for k in shifts:",
         "        pairs = [(s, (s + k) % W) for s in range(W)]",
         "        cur = xm.collective_permute(cur, pairs=pairs)",
         "    return cur"]
    COUNT = (f"Computed with {len(shifts)} collective_permute rotations. " if cue else "")
    doc = (f"Local x ({N},), distinct per rank. {COUNT}Return the per-rank tensors after a "
           f"sequence of ring rotations whose shifts sum to a multiple of the world size.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


# ---------- permrot: D (rotation then fixed PI) -> single composed permutation ----------
def _mk_permrot(name, N, shifts, cue):
    def _net_perm(W):
        # start: pos[r] = r  (rank r's slot). Apply, per stage: rotation by k then PI.
        # We track, for each final rank r, which ORIGINAL rank's tensor it holds.
        # Model forward: after rotation by k, rank r holds src (r-k)%W. After PI (pairs
        # (s, PI[s])), rank PI[s] holds what was on s -> rank r holds what was on PI^{-1}(r).
        pi = _pi_perm(W)
        pinv = [0] * W
        for s in range(W):
            pinv[pi[s]] = s
        # holder[r] = original rank whose tensor rank r currently holds
        holder = list(range(W))
        for k in shifts:
            # rotation by k: newholder[r] = holder[(r-k)%W]
            holder = [holder[(r - k) % W] for r in range(W)]
            # PI: newholder[r] = holder[pinv[r]]
            holder = [holder[pinv[r]] for r in range(W)]
        return holder

    def _ref(inputs, world_size):
        holder = _net_perm(world_size)
        return [inputs[holder[r]]['x'].clone() for r in range(world_size)]

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, N)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    W = world_size; shifts = {list(shifts)}",
         "    PI8 = [3, 0, 5, 7, 1, 6, 2, 4]",
         "    pi = list(range(W))",
         "    if W % 8 == 0:",
         "        for base in range(0, W, 8):",
         "            for i in range(8):",
         "                pi[base + i] = base + PI8[i]",
         "    cur = x",
         "    for k in shifts:",
         "        pairs = [(s, (s + k) % W) for s in range(W)]",
         "        cur = xm.collective_permute(cur, pairs=pairs)",
         "        pairs2 = [(s, pi[s]) for s in range(W)]",
         "        cur = xm.collective_permute(cur, pairs=pairs2)",
         "    return cur"]
    COUNT = (f"Computed with {2*len(shifts)} collective_permute operations. " if cue else "")
    doc = (f"Local x ({N},), distinct per rank. {COUNT}Return the per-rank tensors after a "
           f"sequence of (ring rotation, fixed block permutation) stages; the net is a single "
           f"permutation of the ranks' tensors.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


def register_all():
    # net single rotation (K != 0): D-chain folds to 1 permute
    _mk_rot_net("r52_rotnet_d8_n1024", 1024, [3, 5, 2, 7, 1, 6, 4, 3], True)
    _mk_rot_net("r52_rotnet_d6_n1024", 1024, [3, 5, 2, 7, 1, 6], True)
    _mk_rot_net("r52_rotnet_d8_res", 1024, [3, 5, 2, 7, 1, 6, 4, 3], False)
    # net identity (sum of shifts % 224 == 0): D-chain folds to ZERO comm
    _mk_rot_id("r52_rotid_d8_n1024", 1024, [32, 48, 16, 64, 8, 24, 16, 16], True)  # sum=224
    _mk_rot_id("r52_rotid_d6_n1024", 1024, [32, 48, 16, 64, 8, 56], True)          # sum=224
    _mk_rot_id("r52_rotid_d8_res", 1024, [32, 48, 16, 64, 8, 24, 16, 16], False)
    # composed permutation (rotation + fixed PI): folds to 1 permute, non-closed-form
    _mk_permrot("r52_permrot_d4_n1024", 1024, [3, 5, 2, 7], True)
    _mk_permrot("r52_permrot_d4_res", 1024, [3, 5, 2, 7], False)


register_all()
