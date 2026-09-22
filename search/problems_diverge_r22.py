"""Round 22 -- STRESS-TEST the count-cue lever (L11) across truth-value & family.

r21 established (L11): overlay's enumerate anchors to the STATED collective COUNT.
Naming "8 all_reduce operations" pins it at baseline; narrating the procedure
without a count mostly lets it fold. This round probes the count cue three ways,
all on byte-identical deep-8 code (su family unless noted), docstring-only:

  A. su8_count1   : docstring truthfully states the OPTIMUM count ("equivalent to
                    a SINGLE all_reduce"). Predict: FREES overlay (ties) -- a count
                    assertion of 1 should un-anchor it (mirror of r16_hintdoc).
  B. su8_count16  : docstring OVERSTATES the count ("16 dependent all_reduce
                    operations" -- code actually has 8). Predict: traps overlay AT
                    LEAST as hard as count8; tests whether a larger asserted count
                    deepens the trap or whether 8 already saturates it.
  C. zs8_count8   : the COUNT cue on the ADDITIVE zero-sum family (r20 zs8_narr was
                    distributional-only). Isolates whether the count lever is
                    family-independent at the DISTRIBUTIONAL level, and whether --
                    unlike full narration -- a bare count assertion can push the
                    additive family to survive best-of-N. Predict: median forward
                    (like zs8_narr) but best-of-N still ties (additive escapable).

Controls: su8_count8 (= r21 countonly positive control) and zs8_res (negative).
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


def _gen_flat(world_size, N, seed):
    torch.manual_seed(seed)
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


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


def _zs_code(name, depth=8, N=512):
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    N = {N}; W = world_size",
            "    d = float(2 * rank - (W - 1))",
            "    s = xm.all_reduce(xm.REDUCE_SUM, x)"]
    for st in range(depth - 1):
        body += ["    s = xm.all_reduce(xm.REDUCE_SUM, (s / W) + d)"]
    body += ["    return s"]
    return "\n".join(body) + "\n"


def _zs_ref(inputs, world_size):
    s = sum(inp['x'] for inp in inputs)
    return [s.clone() for _ in range(world_size)]


def _zs_gen(N=512):
    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, N, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _zs_ref(pra, world_size)}
    return _gen


def register_all():
    part = 256
    N = 512
    SU_RESULT = "shard r of the summed vector scaled by a[r]=1.0+0.5*(r%3)"

    # A. truthful minimal count -> should FREE overlay
    _reg("r22_su8_count1",
         f"Local x (world*{part},), S={part}. The result is computed using a SINGLE "
         f"all_reduce operation. Final result = per-shard-scaled all-rank SUM "
         f"({SU_RESULT}).",
         _su_ref(part), _su_gen(part), _su_code("r22_su8_count1", part))
    # B. overstated count (code has 8; doc says 16)
    _reg("r22_su8_count16",
         f"Local x (world*{part},), S={part}. The result is computed using 16 "
         f"dependent all_reduce operations. Final result = per-shard-scaled "
         f"all-rank SUM ({SU_RESULT}).",
         _su_ref(part), _su_gen(part), _su_code("r22_su8_count16", part))
    # positive control (= r21 countonly)
    _reg("r22_su8_count8",
         f"Local x (world*{part},), S={part}. The result is computed using 8 "
         f"dependent all_reduce operations. Final result = per-shard-scaled "
         f"all-rank SUM ({SU_RESULT}).",
         _su_ref(part), _su_gen(part), _su_code("r22_su8_count8", part))
    # C. count cue on the additive zero-sum family
    _reg("r22_zs8_count8",
         f"Local x ({N},). The result is computed using 8 dependent all_reduce "
         f"operations. Final result = all-rank SUM of x.",
         _zs_ref, _zs_gen(N), _zs_code("r22_zs8_count8", 8, N))


register_all()
