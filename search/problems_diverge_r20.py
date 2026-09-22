"""Round 20 -- does the r16 NARRATION trap OVERRIDE algebraic structure?

r16 proved (byte-identical code, docstring-only) that narrating the deep chain
pins overlay at baseline on the SCALE/UNSCALE family. r19 tried to generalize but
used SHALLOW families (headroom 1.08-1.12) where overlay one-shots regardless of
doc -> confounded null. This round fixes that with a clean 2x2 at DEEP headroom:

  FAMILY su (scale/unscale, depth 8)  -- r16's CONFIRMED forward-divergent family.
  FAMILY zs (zero-sum additive, depth 8) -- r14's family that REVERSES
        distributionally (overlay's enumerate naturally drops the additive
        perturbation; kiss over-analyzes and stays at ~2 collectives on ~half
        its seeds -> overlay more reliable on NEUTRAL doc, L9).

  x _narr : docstring NARRATES the 8 dependent stages (anchors to the chain)
  x _res  : docstring states the RESULT only (both free to fold)

Byte-identical code within each family; only the docstring differs.

PREDICTIONS:
  su_narr  -> forward divergence (replicate r16_deepdoc, ~1.9x)          [pos ctrl]
  su_res   -> tie (replicate r16_neutral)                               [pos ctrl]
  zs_res   -> tie on best-of-N (r14/r18: distributional reverse only)   [known]
  zs_narr  -> THE NOVEL TEST. If narration OVERRIDES the additive structure,
              overlay gets trapped by the narrated chain and zs FORWARDS (Sorcar
              wins) despite zs normally favoring overlay. If algebra dominates,
              zs_narr stays tied/reverse. Either outcome is a sharp finding:
              framing>algebra, or algebra caps the framing trap.
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


# ---- FAMILY su: scale/unscale (identical construction to r2_deep8 / r16) ----
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


# ---- FAMILY zs: zero-sum additive chain (identical to r14_zerosum8) ----
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

    # scale/unscale docstrings
    SU_RES = (f"Local x (world*{part},), S={part}. Return the per-shard-scaled "
              f"all-rank SUM: shard r of the summed vector scaled by "
              f"a[r]=1.0+0.5*(r%3).")
    SU_NARR = (f"Local x (world*{part},), S={part}. The result is computed as 8 "
               f"dependent all_reduce stages: stage 1 reduces x, then each "
               f"subsequent stage applies the per-shard scale a[r] and undoes the "
               f"previous stage's scale before reducing again. Final result = "
               f"per-shard-scaled all-rank SUM.")
    _reg("r20_su8_res", SU_RES, _su_ref(part), _su_gen(part), _su_code("r20_su8_res", part))
    _reg("r20_su8_narr", SU_NARR, _su_ref(part), _su_gen(part), _su_code("r20_su8_narr", part))

    # zero-sum additive docstrings
    ZS_RES = (f"Local x ({N},). Return the element-wise all-rank SUM of x.")
    ZS_NARR = (f"Local x ({N},). The result is computed as 8 dependent all_reduce "
               f"stages: stage 1 reduces x, then each subsequent stage adds a "
               f"per-rank perturbation d=2*rank-(W-1) to the running result before "
               f"reducing again. Final result = all-rank SUM of x.")
    _reg("r20_zs8_res", ZS_RES, _zs_ref, _zs_gen(N), _zs_code("r20_zs8_res", 8, N))
    _reg("r20_zs8_narr", ZS_NARR, _zs_ref, _zs_gen(N), _zs_code("r20_zs8_narr", 8, N))


register_all()
