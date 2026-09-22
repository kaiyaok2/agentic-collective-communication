"""Round 16 -- CAUSAL test: is the lever the CODE depth or the DESCRIPTION framing?

Confirmed mechanism: overlay enumerates strategies FROM THE BASELINE FRAMING and
stays trapped in deep chains. But "baseline framing" has two parts overlay sees:
(a) the baseline CODE (deep chain), (b) the problem DESCRIPTION (docstring).
This round isolates which one drives the trap by holding the computation fixed
and varying ONLY the docstring.

All three problems compute the SAME thing (per-shard-scaled AR(SUM,x)) via the
SAME deep-8 scale/unscale chain that r2_deep8/r9 confirmed as a 2.2-2.5x Sorcar
win. The ONLY difference is the signature_doc:

  r16_neutral : describes the RESULT only ("return per-shard-scaled all-rank sum")
                -- no mention of stages. Both systems free to find 1 AR.
  r16_deepdoc : explicitly describes the DEEP baseline ("computed as 8 dependent
                stages, each ...") -- anchors the reader to the chain.
  r16_hintdoc : describes the result AND drops the fusion hint ("note: the whole
                chain is equivalent to a single all_reduce").

PREDICTION: if the trap is CODE-driven, all three diverge equally (~2.2x) since
the baseline code is identical. If it is DESCRIPTION-driven, r16_deepdoc diverges
most (overlay anchored to stages), r16_hintdoc least (overlay handed the fusion),
r16_neutral in between. The delta across the three ISOLATES the framing effect --
a sharper causal claim than "deep code traps overlay."
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


def _build_deep8_code(name, part, depth=8):
    """Identical construction to r2_deep8 (the confirmed 2.2x win)."""
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _ref_factory(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _gen_factory(part):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _ref_factory(part)(pra, world_size)}
    return _gen


def register_all():
    part = 256
    code_neutral = _build_deep8_code("r16_neutral", part)
    code_deepdoc = _build_deep8_code("r16_deepdoc", part)
    code_hintdoc = _build_deep8_code("r16_hintdoc", part)

    DOC_NEUTRAL = (f"Local x (world*{part},), S={part}. Return the per-shard-scaled "
                   f"all-rank SUM: shard r of the summed vector scaled by "
                   f"a[r]=1.0+0.5*(r%3).")
    DOC_DEEPDOC = (f"Local x (world*{part},), S={part}. The result is computed as "
                   f"8 dependent all_reduce stages: stage 1 reduces x, then each "
                   f"subsequent stage applies the per-shard scale a[r] and undoes "
                   f"the previous stage's scale before reducing again. Final "
                   f"result = per-shard-scaled all-rank SUM.")
    DOC_HINTDOC = (f"Local x (world*{part},), S={part}. Return the per-shard-scaled "
                   f"all-rank SUM (shard r scaled by a[r]=1.0+0.5*(r%3)). NOTE: the "
                   f"multi-stage reduce chain is mathematically equivalent to a "
                   f"SINGLE all_reduce followed by a local per-shard scale.")

    _reg("r16_neutral", DOC_NEUTRAL, _ref_factory(part), _gen_factory(part), code_neutral)
    _reg("r16_deepdoc", DOC_DEEPDOC, _ref_factory(part), _gen_factory(part), code_deepdoc)
    _reg("r16_hintdoc", DOC_HINTDOC, _ref_factory(part), _gen_factory(part), code_hintdoc)


register_all()
