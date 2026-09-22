"""Round 26 -- does NARRATION rescue each layout into a CONFIRMED win? (L11/L12 x L13')

r25 (result-only docstrings): perm_scale8 traps overlay's MEDIAN (1.866) but not
best-of-N; strided_scale8 FOLDS even at screen (transparent interleaved layout).
L11/L12: a narrated/counted docstring pins overlay's best-of-N when it AGREES with
the code. This round adds the count cue ("8 dependent all_reduce operations") to
BOTH r25 layouts, testing two predictions:

  A. perm_count8    : narration over CONTIGUOUS permuted deep-8 code. Predict:
     CONFIRMS best-of-N (like r22/r23 su8_count8) -> a 17th confirmed win, and proves
     the confirmed mechanism is not identity-map-specific (permuted contiguous shards
     work too).
  B. strided_count8 : narration over the TRANSPARENT interleaved deep-8 code.
     Predict per L12: narration does NOT rescue it -> overlay still folds (the code's
     real transparency dominates the description). If it DID confirm, that would show
     description can override code transparency -> would REVISE L12. Decisive test of
     whether narration reinforces vs overrides code structure on the layout axis.

Controls carried by r25 (result-only) results. fp32-exact as r25.
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


# permuted contiguous shards (identical code to r25_perm_scale8)
def _perm_code(name, part, depth=8):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    perm = [(r + W // 2) % W for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        p = perm[r]",
                 "        buf[p*S:(p+1)*S] = a[r] * s[p*S:(p+1)*S] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        p = perm[r]",
                     "        s[p*S:(p+1)*S] = s[p*S:(p+1)*S] / max(a[r], 1e-9)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _perm_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        perm = [(r + world_size // 2) % world_size for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            p = perm[r]
            out[p * part:(p + 1) * part] = a[r] * s[p * part:(p + 1) * part]
        return [out.clone() for _ in range(world_size)]
    return _ref


# strided/interleaved lanes (identical code to r25_strided_scale8)
def _strided_code(name, depth=8):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            "    W = world_size",
            "    a = [1.0 + 0.5*(r % 3) for r in range(W)]",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        last = (st == depth - 2)
        body += ["    buf = s.clone()",
                 "    for r in range(W):",
                 "        buf[r::W] = a[r] * s[r::W] / W",
                 "    s = xm.all_reduce(xm.REDUCE_SUM, buf)"]
        if not last:
            body += ["    for r in range(W):",
                     "        s[r::W] = s[r::W] / max(a[r], 1e-9)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _strided_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        a = [1.0 + 0.5 * (r % 3) for r in range(world_size)]
        out = s.clone()
        for r in range(world_size):
            out[r::world_size] = a[r] * s[r::world_size]
        return [out.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    COUNT = "The result is computed using 8 dependent all_reduce operations. "

    perm_ref = _perm_ref(part)

    def perm_gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': perm_ref(pra, world_size)}

    _reg("r26_perm_count8",
         f"Local x (world*{part},), S={part}. {COUNT}Final result = the summed vector "
         f"with shard perm[r] (perm = rotate-by-W//2) scaled by a[r]=1.0+0.5*(r%3).",
         perm_ref, perm_gen, _perm_code("r26_perm_count8", part))

    strided_ref = _strided_ref(part)

    def strided_gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_shards(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {},
                'expected': strided_ref(pra, world_size)}

    _reg("r26_strided_count8",
         f"Local x (world*{part},). {COUNT}Final result = the summed vector with "
         f"interleaved lane r (elements x[r::world]) scaled by a[r]=1.0+0.5*(r%3).",
         strided_ref, strided_gen, _strided_code("r26_strided_count8"))


register_all()
