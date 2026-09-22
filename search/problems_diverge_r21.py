"""Round 21 -- DECOMPOSE the narration trap: is it the COUNT or the PROCEDURE?

r16/r20 proved narration is causal (byte-identical code, docstring-only) on the
scale/unscale family: full narration ("8 dependent all_reduce stages; each applies
the per-shard scale and undoes the previous stage's scale before reducing again")
pins overlay at baseline (confirms ~1.9x); result-only ties. But the full
narration bundles TWO cues:
  (a) the COLLECTIVE COUNT   -- "computed using 8 all_reduce operations"
  (b) the PER-STAGE PROCEDURE -- "each applies the per-shard scale then reduces"

This round separates them on the CONFIRMED su family (so best-of-N is a clean
signal), byte-identical deep-8 code across all four, docstring-only variation:

  su8_full     : count + procedure  (= r20_su8_narr; positive control ~1.9x)
  su8_countonly: names the count 8, NO procedure narration
  su8_proconly : narrates the per-stage procedure, NEVER states the number 8
  su8_res      : result only        (negative control; ties)

PREDICTION space:
  - If COUNT drives the trap: countonly ~ full (both confirm), proconly ~ res (tie).
  - If PROCEDURE drives it:   proconly ~ full,               countonly ~ res.
  - If BOTH needed:           only full confirms; the two partials tie.
  - If EITHER suffices:       both partials confirm like full.
Any of these sharpens r16's causal claim from "narration traps" to "SPECIFICALLY
the <count|procedure> cue traps overlay's enumerate-from-baseline."
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


def _su_code(name, part, depth=8):
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


def _su_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * s[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


def _su_gen(part):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': _su_ref(part)(pra, world_size)}
    return _gen


def register_all():
    part = 256
    RESULT = (f"shard r of the summed vector scaled by a[r]=1.0+0.5*(r%3)")

    DOC_FULL = (f"Local x (world*{part},), S={part}. The result is computed as 8 "
                f"dependent all_reduce stages: stage 1 reduces x, then each "
                f"subsequent stage applies the per-shard scale a[r] and undoes the "
                f"previous stage's scale before reducing again. Final result = "
                f"per-shard-scaled all-rank SUM ({RESULT}).")
    DOC_COUNT = (f"Local x (world*{part},), S={part}. The result is computed using "
                 f"8 dependent all_reduce operations. Final result = per-shard-"
                 f"scaled all-rank SUM ({RESULT}).")
    DOC_PROC = (f"Local x (world*{part},), S={part}. The result is computed by a "
                f"chain of dependent all_reduce stages: an initial reduce of x, "
                f"then each subsequent stage applies the per-shard scale a[r] and "
                f"undoes the previous stage's scale before reducing again. Final "
                f"result = per-shard-scaled all-rank SUM ({RESULT}).")
    DOC_RES = (f"Local x (world*{part},), S={part}. Return the per-shard-scaled "
               f"all-rank SUM: {RESULT}.")

    _reg("r21_su8_full", DOC_FULL, _su_ref(part), _su_gen(part), _su_code("r21_su8_full", part))
    _reg("r21_su8_countonly", DOC_COUNT, _su_ref(part), _su_gen(part), _su_code("r21_su8_countonly", part))
    _reg("r21_su8_proconly", DOC_PROC, _su_ref(part), _su_gen(part), _su_code("r21_su8_proconly", part))
    _reg("r21_su8_res", DOC_RES, _su_ref(part), _su_gen(part), _su_code("r21_su8_res", part))


register_all()
