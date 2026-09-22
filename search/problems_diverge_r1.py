"""Round 1 -- two directions.

DIR-A "payload scaling of the error-prone fold": hd10's fused optimum saves
(nstage-1) all_reduces of size world*S. At S=256 the fixed ~5160us dispatch
floor dominates, so headroom is only ~1.05-1.5x. If we SCALE S, each AR becomes
bandwidth-bound and the multi-AR baseline's cost grows faster than the 1-AR
optimum -> headroom should widen. Question: does bigger payload push the
SYMMETRIC best-of-K divergence to statistical significance, or does the
per-seed stochasticity (which system nails the error-prone fold) wash it out
regardless of headroom?

DIR-B "decoy optimum": exploit overlay's enumerate-once-discard rule from a
DIFFERENT angle than hd10. Here the MOST OBVIOUS optimization is subtly WRONG
(passes a naive eyeball check but fails the fp32 gate), while the correct
optimum is a small variation. Overlay enumerates K strategies from the baseline
framing; the obvious-but-wrong fold is the top idea, gets implemented once,
fails the gate, is discarded -> overlay refines the safe baseline. Kiss sees
the gate error on the obvious fold and iterates to the corrected variation.
The decoy: fold a per-shard scale but with an OFF-BY-ONE in the normalization
(dividing by world_size once too few/many across dependent stages) that a cold
draft reproduces but the gate catches.
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


# --- DIR-A: payload-scaled staged_shard affine fold (scale-only) ---
def _mk_scaled_fold(name, part, nstage=3):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(nstage - 2):
        body += [
            f"    buf{st} = s1.clone()",
            "    for r in range(W):",
            f"        buf{st}[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W",
            f"    s1 = xm.all_reduce(xm.REDUCE_SUM, buf{st})",
            "    for r in range(W):",
            f"        s1[r*S:(r+1)*S] = s1[r*S:(r+1)*S] / max(a[r], 1e-9)"]
    body += [
        "    bufN = s1.clone()",
        "    for r in range(W):",
        "        bufN[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W",
        "    out = xm.all_reduce(xm.REDUCE_SUM, bufN)",
        "    return out"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline: {nstage} "
         f"dependent AR_SUM stages with per-shard scale between them. "
         f"Result = per-shard-scaled AR(x).", _ref, _gen, "\n".join(body) + "\n")


# --- DIR-B: decoy optimum (obvious fold is off-by-one on normalization) ---
# The reference is a per-shard scale of AR(x). The baseline chains TWO ARs where
# the middle rebuild divides by W; the SUBTLE point is that stage-3 does NOT
# re-divide (it's an identity-add stage), so a cold "just fuse the scales" draft
# that divides by W once too many produces W-fold-too-small output -> gate fail.
def _mk_decoy_fold(name, part=512):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref(pra, world_size)}

    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)",
            "    # stage 2: scale, normalize by W, reduce again",
            "    buf = s1.clone()",
            "    for r in range(W):",
            "        buf[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W",
            "    s2 = xm.all_reduce(xm.REDUCE_SUM, buf)",
            "    # stage 3: IDENTITY re-add (already scaled) -- do NOT re-divide",
            "    buf3 = s2.clone() / W",
            "    s3 = xm.all_reduce(xm.REDUCE_SUM, buf3)",
            "    return s3"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline: 3 dependent "
         f"AR_SUM stages; a per-shard scale is applied in stage 2 and stages "
         f"are re-normalized by world_size to keep AR(SUM) of already-reduced "
         f"data consistent. Result = per-shard-scaled AR(x).",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    # DIR-A payload scaling (scale-only fold, 3 stages)
    _mk_scaled_fold("r1_fold_s256", 256)
    _mk_scaled_fold("r1_fold_s1024", 1024)
    _mk_scaled_fold("r1_fold_s4096", 4096)
    _mk_scaled_fold("r1_fold_s256_deep", 256, nstage=4)
    _mk_scaled_fold("r1_fold_s4096_deep", 4096, nstage=4)
    # DIR-B decoy optimum
    _mk_decoy_fold("r1_decoy_s512", 512)
    _mk_decoy_fold("r1_decoy_s4096", 4096)


register_all()
