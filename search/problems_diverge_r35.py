"""Round 35 -- DISTINCT MECHANISM: idempotent MASKED partial-reduction collapse.

Family-1's collapse is ARITHMETIC (per-shard multiplicative scale distributes through
SUM). This round tests a structurally different collapse: SUPPORT-DISJOINTNESS /
idempotence. NOT a scale, NOT an offset -- a masking algebra.

Mechanism: a D-deep chain of all_reduce(SUM), where stage k first ZEROES every block
except block k (a disjoint support mask), then all_reduces. Because the masks partition
the vector into disjoint supports, summing the D masked-and-reduced stages recovers
exactly the full all_reduce(SUM) of the original -- i.e. the entire D-collective chain
collapses to ONE all_reduce(SUM). The collapse rests on: sum_k mask_k * AR(x) = AR(x)
when the mask_k partition the index set. This is a PARTITION-OF-UNITY identity, distinct
from both the multiplicative-scale family (r1/r2) and the additive zero-sum family (r14).

r35_masksum8 accumulates the per-block contributions:
    acc = 0
    for k in range(D):                  # D dependent all_reduce(SUM)
        m = zeros_like(x); m[block k] = x[block k]
        acc = acc + all_reduce(SUM, m)  # each AR contributes only block k of the total
    return acc                          # == all_reduce(SUM, x), computed in D collectives
The optimum is a SINGLE all_reduce(SUM, x) (blocks partition -> masking is redundant).

Distinct from r10 (re-max idempotence, which TIED): there the redundancy was a
locally-visible repeat of the SAME op; here each stage masks a DIFFERENT disjoint block,
so no single stage looks redundant -- the redundancy is only visible GLOBALLY once you
see the masks tile the whole vector (an L8-style non-locally-visible collapse). This is
the sharpest non-arithmetic distinct-family probe.

*_count8 carries the truthful count cue; *_res is result-only. Awkward-free here because
the collapse is combinatorial, not arithmetic -- no non-power-of-2 concern.

fp32: baseline is a genuine D=8 AR(SUM) accumulation (passes gate, sim ~12k). Reference =
plain elementwise sum across ranks (each block's partial equals that block of the full
sum; the disjoint masks reassemble the full sum exactly).
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


def _masksum_code(name, part, depth=8):
    # depth stages, each masks a disjoint block-group and accumulates its all_reduce.
    # blocks 0..W-1 partitioned into `depth` groups by index % depth.
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; W = world_size; D = {depth}",
         "    acc = torch.zeros_like(x)",
         "    for k in range(D):",
         "        m = torch.zeros_like(x)",
         "        for r in range(W):",
         "            if r % D == k:",
         "                m[r*S:(r+1)*S] = x[r*S:(r+1)*S]",
         "        acc = acc + xm.all_reduce(xm.REDUCE_SUM, m)",
         "    return acc"]
    return "\n".join(L) + "\n"


def _masksum_ref(part):
    def _ref(inputs, world_size):
        s = sum(inp['x'] for inp in inputs)
        return [s.clone() for _ in range(world_size)]
    return _ref


def register_all():
    part = 256
    ref = _masksum_ref(part)

    def _mk_gen(ref):
        def gen(world_size, pattern='uniform', shard_size=None, seed=0):
            pra = _gen_shards(world_size, seed, part)
            return {'per_rank_args': pra, 'shared_args': {},
                    'expected': ref(pra, world_size)}
        return gen

    COUNT = "The result is computed using 8 dependent all_reduce operations. "
    RES = ("Final result = the elementwise SUM of x across all ranks (each block's "
           "contribution is accumulated over disjoint masked partitions).")

    _reg("r35_masksum8_count8",
         f"Local x (world*{part},), S={part}. {COUNT}{RES}",
         ref, _mk_gen(ref), _masksum_code("r35_masksum8_count8", part))
    _reg("r35_masksum8_res",
         f"Local x (world*{part},), S={part}. {RES}",
         ref, _mk_gen(ref), _masksum_code("r35_masksum8_res", part))


register_all()
