"""Round 57 -- FAMILY-4 CANDIDATE: cross-primitive equivalence via all_to_all.

fam-3 confirmed that Overlay misses the RS+AG == AR cross-collective identity at depth 4 (xc_r4,
8 collectives -> 1). This round probes a mechanistically DIFFERENT equivalence: the
transpose-reduction identity
    all_to_all(x) -> local per-shard sum  ==  reduce_scatter(SUM, x)
so each baseline stage (a2a + local sum + all_gather) == ONE all_reduce. The baseline burns TWO
collectives per stage; depth D -> 2*D collectives; the ideal fold is ONE global AR + local scalar.

Stage semantics (x local (W*S,), W blocks of S):
    y = all_to_all(a[rank]*cur)          # rank r receives block r from every rank
    z = sum over the W received blocks   # shard r of SUM_k a[k]*cur_k
    cur = all_gather(z) / u              # reassemble the full vector; u = 0.9*A_tot (awkward,
                                         # non-pow2 unscale keeps values bounded, fam-1 recipe)
Telescoping: after stage 1 cur is rank-identical, so each later stage multiplies by A_tot/u where
A_tot = sum_k a[k]. Final = (1/0.9)^(D-1) * SUM_k a[k] x_k / (0.9*A_tot).
FOLD = ONE all_reduce(SUM, a[rank]*x) + local scalar multiply. 2*D collectives -> 1.

Per-rank a[r] = 1 + 0.5*((r*11) % 7)/7 is the loop-temptation factor; the "drop a[r]" guess fails
the fp32 gate. Distinct surface from fam-1/2/3: the trap is a PRIMITIVE-equivalence insight
(a2a+sum == RS, RS+AG == AR), not a diagonal-factor fold alone. All SUM-linear => gate-exact.
Pre-screened at W=224 before any cloud run.
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


def _a(r):
    return 1.0 + 0.5 * ((r * 11) % 7) / 7.0


def _mk(name, S, depth, cue):
    def _ref(inputs, world_size):
        W = world_size
        xs = [inp['x'] for inp in inputs]
        A_tot = sum(_a(k) for k in range(W))
        u = 0.9 * A_tot
        s1 = sum((_a(k) * xs[k] for k in range(W)), torch.zeros(W * S))
        out = (s1 / u) * ((A_tot / u) ** (depth - 1))
        return [out.clone() for _ in range(W)]

    def gen(world_size, pattern='uniform', shard_size=None, seed=0):
        torch.manual_seed(seed)
        N = world_size * S
        pra = [{'x': torch.randn(N) * (0.4 + 0.02 * r)} for r in range(world_size)]
        return {'per_rank_args': pra, 'shared_args': {}, 'expected': _ref(pra, world_size)}

    L = [f"def {name}_fn(x, rank, world_size, num_devices,",
         "                 cores_per_device, xm, torch, num_nodes=1):",
         f"    S = {S}; W = world_size; D = {depth}",
         "    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0",
         "    A_tot = sum(1.0 + 0.5*((k*11) % 7)/7.0 for k in range(W))",
         "    u = 0.9 * A_tot",
         "    cur = x",
         "    for _t in range(D):",
         "        y = xm.all_to_all(a * cur, split_dimension=0, concat_dimension=0,",
         "                          split_count=W)",
         "        z = torch.sum(y.reshape(W, S), dim=0)",
         "        cur = xm.all_gather(z, dim=0) / u",
         "    return cur"]
    COUNT = (f"Computed with {depth} stages, each all_to_all + local shard-sum + all_gather. "
             if cue else "")
    doc = (f"Local x (world*{S},) = W blocks of {S}, distinct per rank. {COUNT}Each stage scales "
           f"by a[r]=1+0.5*((r*11) mod 7)/7, all_to_all-transposes blocks, sums the W received "
           f"blocks into this rank's shard, all_gathers the shards, then divides by 0.9*sum(a). "
           f"Return the depth-{depth} result.")
    _reg(name, doc, _ref, gen, "\n".join(L) + "\n")


def register_all():
    _mk("r57_a2asum_d6_s256", 256, 6, True)
    _mk("r57_a2asum_d4_s256", 256, 4, True)
    _mk("r57_a2asum_d3_s256", 256, 3, True)
    _mk("r57_a2asum_d2_s256", 256, 2, True)
    _mk("r57_a2asum_d6_s64", 64, 6, True)
    _mk("r57_a2asum_d4_res", 256, 4, False)
    _mk("r57_a2asum_d6_res", 256, 6, False)


register_all()
