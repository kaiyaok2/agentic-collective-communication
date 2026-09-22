"""Round 23 -- ADVERSARIAL: does overlay anchor to the STATED count or the CODE?

L11 (r21/r22): overlay's trap tracks the collective COUNT asserted in the docstring
(count1 frees, count8 traps, count16 traps max). But in every prior round the code
actually HAD the deep chain, so "anchors to stated count" and "anchors to code
depth" were confounded whenever both agreed. This round DE-CONFOUNDS them by making
the docstring's asserted count DISAGREE with the code:

  A. minimal_says8 : code is the ALREADY-MINIMAL single all_reduce (1 collective),
     but the docstring claims "computed using 8 dependent all_reduce operations".
     - If overlay anchors to the STATED count -> it may hallucinate/over-engineer
       or freeze -> kiss (reading the trivial code) folds trivially -> Sorcar wins
       (a NEW mechanism: false-high count over minimal code).
     - If overlay anchors to the CODE -> it sees 1 collective, both stay at floor
       -> tie. (This would BOUND L11 to "count cue only bites when code backs it".)
  B. deep8_says1  : code is the full deep-8 chain, docstring claims "a SINGLE
     all_reduce" (false-LOW count, mirror of r22 count1 but now the claim is a LIE).
     - If the count cue dominates -> the "1" frees overlay (folds) -> tie (like r22
       count1). - If code depth dominates when it contradicts a low count -> overlay
       still trapped by the 8-deep code -> Sorcar wins. Distinguishes "count cue"
       from "count cue only when truthful".

Controls: minimal_res (minimal code, result-only doc -> tie floor) and
deep8_count8 (= r22 positive control, both agree at 8).

This is the decisive test of WHAT overlay anchors to when description and code
DISAGREE -- the sharpest possible probe of the L11 mechanism.
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


def _su_deep_code(name, part, depth=8):
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


def _su_minimal_code(name, part):
    # The already-optimal single-AR implementation of the SAME per-shard-scaled sum.
    return (f"def {name}_fn(x, rank, world_size, num_devices,\n"
            f"                 cores_per_device, xm, torch, num_nodes=1):\n"
            f"    S = {part}; W = world_size\n"
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]\n"
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)\n"
            "    out = s.clone()\n"
            "    for r in range(W):\n"
            "        out[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S]\n"
            "    return out\n")


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
    RESULT = "shard r of the summed vector scaled by a[r]=1.0+0.5*(r%3)"

    # A. minimal code, docstring LIES that it uses 8 collectives
    _reg("r23_minimal_says8",
         f"Local x (world*{part},), S={part}. The result is computed using 8 "
         f"dependent all_reduce operations. Final result = per-shard-scaled "
         f"all-rank SUM ({RESULT}).",
         _su_ref(part), _su_gen(part), _su_minimal_code("r23_minimal_says8", part))
    # A-control: minimal code, result-only doc
    _reg("r23_minimal_res",
         f"Local x (world*{part},), S={part}. Return the per-shard-scaled all-rank "
         f"SUM ({RESULT}).",
         _su_ref(part), _su_gen(part), _su_minimal_code("r23_minimal_res", part))
    # B. deep-8 code, docstring LIES that it is a single collective
    _reg("r23_deep8_says1",
         f"Local x (world*{part},), S={part}. The result is computed using a SINGLE "
         f"all_reduce operation. Final result = per-shard-scaled all-rank SUM "
         f"({RESULT}).",
         _su_ref(part), _su_gen(part), _su_deep_code("r23_deep8_says1", part))
    # B-control: deep-8 code, truthful count8 (= r22 positive control)
    _reg("r23_deep8_count8",
         f"Local x (world*{part},), S={part}. The result is computed using 8 "
         f"dependent all_reduce operations. Final result = per-shard-scaled "
         f"all-rank SUM ({RESULT}).",
         _su_ref(part), _su_gen(part), _su_deep_code("r23_deep8_count8", part))


register_all()
