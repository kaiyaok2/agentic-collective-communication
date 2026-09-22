"""Round 2 -- structurally-robust levers (not per-seed noise).

Lessons applied: L3 (error-prone fold is seed-stochastic -> best-of-N ties),
L4 (depth widens headroom, payload does not). So target overlay's STRUCTURAL
constraints rather than one-shot difficulty:

DIR-C "depth vs refinement budget": a chain of D dependent all_reduces where
each stage is an EASY per-shard scale, but there are MANY of them. The fused
optimum is 1 AR. Overlay enumerates from the baseline and refines top-2 for a
FIXED R=3 rounds; if collapsing D stages needs more than a couple of edits (or
if a partial collapse to D/2 stages is the "obvious" first refinement), a bounded
budget may plateau at a partial collapse while kiss (30 steps) drives to 1 AR.
Deterministic (no tricky math), so if it ties it's a clean negative; if it
diverges it's robust.

DIR-E "framing lock-in": the baseline computes AR(SUM, x) then a per-shard
scale (already a valid, gate-passing, MODERATELY fast form). An EASY refinement
(drop a redundant final identity AR) is obvious and overlay will take it -> a
correct-but-still-2-AR form. The FAST optimum (1 AR) requires recognizing that
the per-shard scale commutes so the middle AR is removable too -- a DIFFERENT
structural edit. If overlay locks onto the easy refinement framing and its R=3
budget plateaus at 2 AR, while kiss reframes to 1 AR, that is a robust divergence
grounded in overlay's bounded-refinement design, not seed luck.
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


# --- DIR-C: D-deep mechanical scale chain (fused = 1 AR) ---
def _mk_deep_chain(name, depth, part=256):
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

    # baseline: depth dependent ARs. stage 1 reduces; stages 2..depth each apply
    # the a[r] scale once and undo the previous stage's scale so the net over
    # the whole chain is exactly one application of a[r]. Each stage divides by
    # W to keep AR(SUM) of replicated data consistent.
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += [
            "    buf = s.clone()",
            "    for r in range(W):"]
        if last:
            # final stage: leave the a[r] scale applied (net result)
            body += ["        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W"]
        else:
            # intermediate: apply a[r] then immediately plan to undo next stage
            body += ["        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W"]
        body += [f"    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)"]
    body += ["    return s"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline: {depth} dependent "
         f"AR_SUM stages, each a mechanical per-shard scale/unscale; net result "
         f"= per-shard-scaled AR(x). Fused optimum is 1 AR.",
         _ref, _gen, "\n".join(body) + "\n")


# --- DIR-E: framing lock-in (easy refinement = 2 AR; fast optimum = 1 AR) ---
def _mk_framing_lockin(name, part=512):
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

    # baseline has 3 ARs: (1) real reduce, (2) a scale+reduce that is the crux,
    # (3) a REDUNDANT identity AR that is the obvious thing to drop. Dropping (3)
    # is a trivial refinement -> lands at 2 AR. Removing (2) requires the
    # commutation insight (the per-shard scale can be applied AFTER (1) locally),
    # a different edit that a lock-in on "drop redundant ops" may miss.
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s1 = xm.all_reduce(xm.REDUCE_SUM, x)          # real reduce",
            "    buf = s1.clone()",
            "    for r in range(W):",
            "        buf[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W",
            "    s2 = xm.all_reduce(xm.REDUCE_SUM, buf)        # scale+reduce (crux)",
            "    s3 = xm.all_reduce(xm.REDUCE_SUM, s2 / W)     # REDUNDANT identity",
            "    return s3"]
    _reg(name, f"Local x (world*{part},), S={part}. Baseline: 3 dependent "
         f"AR_SUM stages (one is a redundant identity reduce). Result = "
         f"per-shard-scaled AR(x).", _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_deep_chain("r2_deep4", 4)
    _mk_deep_chain("r2_deep6", 6)
    _mk_deep_chain("r2_deep8", 8)
    _mk_deep_chain("r2_deep6_big", 6, part=1024)
    _mk_framing_lockin("r2_lockin_s512", 512)
    _mk_framing_lockin("r2_lockin_s2048", 2048)


register_all()
