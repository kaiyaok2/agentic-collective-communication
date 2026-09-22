"""Round 11 -- mechanism-confirmation: genuine-looking dependent work over a
NON-SUM primitive (MAX).

Refined L7 (from r8 vs r2): depth diverges on BEST-of-N only when each stage
looks like GENUINE, non-obviously-removable dependent work. r2's scale/unscale
achieves this for SUM (overlay stays trapped even on its best seed). r10 tests
generality but with OBVIOUSLY-removable stages (re-max, AG roundtrip), which the
refined lesson predicts will TIE on best-of-N. r11 is the CONFIRMATORY test: a
MAX chain where each stage does genuine-looking work -- add a per-shard offset
before the max and subtract it after -- so the stage is NOT an obvious identity,
yet the whole chain still collapses to a single all_reduce(MAX) plus a local
per-shard affine. If Sorcar wins best-of-8 here, the depth-trap generalizes
beyond sum-linearity to the algebra of max.

Construction: define y_r = x_r (shifted). Baseline maintains running m; each
stage adds offset o (broadcast), all_reduce(MAX), subtracts o -> identity on the
max, but looks like real dependent work. Final: per-shard scale a[r] (power of
2, fp-exact). Optimum: 1 all_reduce(MAX) + local per-shard scale.
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


def _mk_max_offset_chain(name, depth, part=256):
    """Deep MAX chain with genuine-looking add-offset/max/sub-offset stages
    (each an identity on the max but not obviously so), then a final power-of-2
    per-shard scale. Optimum = 1 AR-MAX + local scale."""
    def _ref(inputs, world_size):
        st = torch.stack([inp['x'] for inp in inputs], dim=0)
        mx = torch.max(st, dim=0)[0]                      # (W*part,)
        a = [2.0 ** (r % 3) for r in range(world_size)]
        out = mx.clone()
        for r in range(world_size):
            out[r * part:(r + 1) * part] = a[r] * mx[r * part:(r + 1) * part]
        return [out.clone() for _ in range(world_size)]

    def _gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen_flat(world_size, world_size * part, seed)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    # per-stage offset o_st (a scalar broadcast). add before max, subtract after:
    # max(v + o) - o == max(v). Powers of two so fp-exact.
    body = [f"def {name}_fn(x, rank, world_size, num_devices,",
            "                 cores_per_device, xm, torch, num_nodes=1):",
            f"    S = {part}; W = world_size",
            "    a = [2.0 ** (r % 3) for r in range(W)]",
            "    m = xm.all_reduce(xm.REDUCE_MAX, x)"]
    for st in range(depth - 1):
        o = float(2 ** (st % 4))    # 1,2,4,8 cycling; exact in fp32
        body += [
            f"    m = m + {o}",
            "    m = xm.all_reduce(xm.REDUCE_MAX, m)",
            f"    m = m - {o}"]
    body += ["    out = m.clone()",
             "    for r in range(W):",
             "        out[r*S:(r+1)*S] = a[r] * m[r*S:(r+1)*S]",
             "    return out"]
    _reg(name, f"Local x (world*{part},), S={part}. Return per-shard-scaled "
         f"elementwise all-rank MAX (scale a[r]=2**(r%3)). Baseline: {depth} "
         f"dependent all_reduce(MAX) stages, each wrapped in an add/subtract "
         f"offset. Fused optimum is 1 AR-MAX + local scale.",
         _ref, _gen, "\n".join(body) + "\n")


def register_all():
    _mk_max_offset_chain("r11_maxoff4", 4)
    _mk_max_offset_chain("r11_maxoff6", 6)
    _mk_max_offset_chain("r11_maxoff8", 8)
    _mk_max_offset_chain("r11_maxoff6_big", 6, part=1024)


register_all()
