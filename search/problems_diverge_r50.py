"""Round 50 -- FAM-5 candidate: AR-LINEARITY additive fusion (per-ELEMENT weights).

Distinct from confirmed families:
  fam-1 per-RANK scale a[r] (rank-heterogeneous, needs /a to invert);
  fam-2 rank-indexed routing; fam-3 data-dependent scale + cross-primitive.

fam-5 mechanism: the baseline computes a SUM of K independent all_reduce(SUM) terms,
where term k is AR(SUM, w_k * x) for a fixed per-ELEMENT weight vector w_k that is the
SAME on every rank (rank-HOMOGENEOUS -- this is the key distinction from fam-1's per-rank
a[r]). By linearity of AR(SUM) over the shard axis AND over scalar/elementwise
multiply: sum_k AR(SUM, w_k * x) == AR(SUM, (sum_k w_k) * x). So K collectives + K
weightings FUSE to ONE AR of the pre-summed-weight-times-x. Overlay's enumerate sees K
distinct weighted reductions; the fold requires the linearity insight
"pull the weight-sum inside a single AR". Trap = COLLECTIVE-COUNT fold by INPUT
SUMMATION with rank-HOMOGENEOUS per-element weights -- neither a per-rank scale (fam-1)
nor concatenation (fam-4). Real: multiple weighted-residual all-reduces in a fused
gradient/feature accumulation.

To make the fold non-trivial the weights are DEPTH-CHAINED: stage t reduces
AR(SUM, w_t * s) and accumulates, so a naive "it's just K*AR(x)" guess FAILS the gate
(the w_t differ). All SUM -> gate-safe; baseline + fold pre-screened at W=224.
"""
import torch
from .problems import CollectiveProblem, register_problem

NBLOCK = 8


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


def _gen(world_size, seed, part):
    torch.manual_seed(seed)
    N = NBLOCK * part
    return [{'x': torch.randn(N) * (0.3 + 0.02 * r)} for r in range(world_size)]


# per-element weight for term k: block b gets constant (1 + 0.1*((k+b) % 5)).
# rank-INDEPENDENT (same on every rank) -- the fam-5 distinction. w_k is a length-N
# vector built from a per-block pattern; sum_k w_k is a fixed length-N vector.
def _wcode(indent):
    return [f"{indent}wk = torch.ones(B*S)",
            f"{indent}for b in range(B):",
            f"{indent}    wk[b*S:(b+1)*S] = 1.0 + 0.1*((k + b) % 5)"]


def _lin_code(name, part, K):
    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {part}; B = {NBLOCK}",
         "    acc = None",
         f"    for k in range({K}):"]
    L += _wcode("        ")
    L += ["        term = xm.all_reduce(xm.REDUCE_SUM, wk * x)",
          "        acc = term if acc is None else acc + term",
          "    return acc"]
    return "\n".join(L) + "\n"


def _lin_ref(part, K):
    def _ref(inputs, world_size):
        xs = [inp['x'] for inp in inputs]
        B = NBLOCK
        N = B * part
        acc = torch.zeros(N)
        for k in range(K):
            wk = torch.ones(N)
            for b in range(B):
                wk[b * part:(b + 1) * part] = 1.0 + 0.1 * ((k + b) % 5)
            # AR(SUM, wk*x) = wk * sum_r x_r  (wk rank-independent)
            acc = acc + wk * sum(xs)
        return [acc.clone() for _ in range(world_size)]
    return _ref


def _mk(name, part, K, cue):
    ref = _lin_ref(part, K)

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        pra = _gen(world_size, seed, part)
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': ref(pra, world_size)}
    COUNT = (f"Computed as a sum of {K} separate all_reduce operations. " if cue else "")
    doc = (f"Local x ({NBLOCK}*{part},), {NBLOCK} blocks of {part}. {COUNT}"
           f"Final result = sum over k=0..{K-1} of (w_k elementwise-times the across-rank "
           f"SUM of x), where w_k[block b] = 1 + 0.1*((k+b) mod 5).")
    _reg(name, doc, ref, gen, _lin_code(name, part, K))


def register_all():
    # K-sweep (number of linear terms fused), payload 256
    _mk("r50_lin_k4",  256, 4, True)
    _mk("r50_lin_k6",  256, 6, True)
    _mk("r50_lin_k8",  256, 8, True)
    _mk("r50_lin_k10", 256, 10, True)
    # payload sweep at k8
    _mk("r50_lin_k8_p384", 384, 8, True)
    _mk("r50_lin_k8_p512", 512, 8, True)
    # framing controls
    _mk("r50_lin_k8_res",  256, 8, False)
    _mk("r50_lin_k6_res",  256, 6, False)


register_all()
